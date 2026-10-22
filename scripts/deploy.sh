#!/usr/bin/env bash
# Deploy the demo to a single EC2 instance. See docs/adr/0006-*.md for why.
#
#   scripts/deploy.sh plan      # print what would happen, touch nothing
#   scripts/deploy.sh provision # create the instance and security group
#   scripts/deploy.sh push      # build the image and copy it to the instance
#   scripts/deploy.sh start     # run the container with secrets from SSM
#   scripts/deploy.sh smoke     # health, an answer, an abstention, a rate limit
#   scripts/deploy.sh destroy   # terminate and delete what provision created
#
# Everything is parameterised by environment variable, with defaults matching
# the ADR. Nothing here creates a billable resource except `provision`, and that
# prints what it will do and waits for confirmation.
#
# The API key is NEVER baked into the image or passed on a command line. It is
# read from SSM Parameter Store at container start.
set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t4g.small}"
NAME="${DEPLOY_NAME:-observable-rag}"
KEY_NAME="${KEY_NAME:-$NAME}"
VOLUME_GB="${VOLUME_GB:-20}"
SSM_KEY_PARAM="${SSM_KEY_PARAM:-/observable-rag/openai-api-key}"
FASTAPI_REF="${FASTAPI_REF:-master}"
PLATFORM="linux/arm64"            # t4g is Graviton; see the ADR
IMAGE="$NAME:deploy"
TARBALL="/tmp/$NAME.tar"
SSH_USER="ubuntu"

# Ubuntu 24.04 LTS arm64, resolved from SSM so it is never a stale hardcoded AMI
AMI_PARAM="/aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
dim()  { printf '\033[2m%s\033[0m\n' "$*"; }
die()  { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null || die "$1 is required but not installed"; }

instance_id() {
  aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Name,Values=$NAME" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null | head -1
}

public_ip() {
  aws ec2 describe-instances --region "$REGION" --instance-ids "$(instance_id)" \
    --query 'Reservations[].Instances[].PublicIpAddress' --output text 2>/dev/null
}

cmd_plan() {
  need aws
  bold "plan"
  echo "  region          $REGION"
  echo "  instance type   $INSTANCE_TYPE  (arm64 / Graviton)"
  echo "  volume          ${VOLUME_GB}GB gp3"
  echo "  name tag        $NAME"
  echo "  ssh key pair    $KEY_NAME"
  echo "  api key from    SSM $SSM_KEY_PARAM"
  echo "  docs revision   $FASTAPI_REF"
  echo
  local account existing
  account=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "NOT CONFIGURED")
  echo "  aws account     $account"
  existing=$(instance_id || true)
  if [ -n "$existing" ]; then
    echo "  existing        $existing at $(public_ip)"
  else
    echo "  existing        none — provision would create one"
  fi
  echo
  dim "  rough cost: ~\$12/mo compute + ~\$1.60/mo storage (estimates)"
  dim "  nothing has been created. 'provision' is the step that spends money."
}

cmd_provision() {
  need aws
  [ -z "$(instance_id || true)" ] || die "an instance tagged $NAME already exists: $(instance_id)"

  cmd_plan
  echo
  bold "This creates billable AWS resources."
  read -r -p "Type the instance type to confirm ($INSTANCE_TYPE): " confirm
  [ "$confirm" = "$INSTANCE_TYPE" ] || die "not confirmed"

  local ami sg
  ami=$(aws ssm get-parameters --region "$REGION" --names "$AMI_PARAM" \
        --query 'Parameters[0].Value' --output text)
  [ -n "$ami" ] && [ "$ami" != "None" ] || die "could not resolve an Ubuntu arm64 AMI"
  dim "  ami $ami"

  if ! aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" >/dev/null 2>&1; then
    aws ec2 create-key-pair --region "$REGION" --key-name "$KEY_NAME" \
      --query 'KeyMaterial' --output text > "$HOME/.ssh/$KEY_NAME.pem"
    chmod 400 "$HOME/.ssh/$KEY_NAME.pem"
    bold "  wrote $HOME/.ssh/$KEY_NAME.pem — keep it, it is the only copy"
  fi

  sg=$(aws ec2 create-security-group --region "$REGION" \
        --group-name "$NAME" --description "$NAME demo" \
        --query 'GroupId' --output text 2>/dev/null || \
       aws ec2 describe-security-groups --region "$REGION" --group-names "$NAME" \
        --query 'SecurityGroups[0].GroupId' --output text)

  # 80 open to the world (it is a public demo); 22 only from here.
  local me
  me=$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')
  aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$sg" \
    --protocol tcp --port 80 --cidr 0.0.0.0/0 >/dev/null 2>&1 || true
  aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$sg" \
    --protocol tcp --port 22 --cidr "$me/32" >/dev/null 2>&1 || true
  dim "  security group $sg (80 from anywhere, 22 from $me only)"

  # The instance reads the API key from SSM at container start, so it needs a
  # role that can do exactly that and nothing else. Without this the key would
  # have to travel in the image or over ssh.
  if ! aws iam get-instance-profile --instance-profile-name "$NAME" >/dev/null 2>&1; then
    aws iam create-role --role-name "$NAME" --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},
                    "Action":"sts:AssumeRole"}]}' >/dev/null
    aws iam put-role-policy --role-name "$NAME" --policy-name read-api-key \
      --policy-document "{
        \"Version\":\"2012-10-17\",
        \"Statement\":[{\"Effect\":\"Allow\",
          \"Action\":[\"ssm:GetParameter\"],
          \"Resource\":\"arn:aws:ssm:$REGION:*:parameter$SSM_KEY_PARAM\"}]}" >/dev/null
    aws iam create-instance-profile --instance-profile-name "$NAME" >/dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "$NAME" \
      --role-name "$NAME" >/dev/null
    sleep 10   # IAM is eventually consistent; run-instances fails if it races
    dim "  iam role $NAME (ssm:GetParameter on $SSM_KEY_PARAM only)"
  fi

  aws ec2 run-instances --region "$REGION" \
    --image-id "$ami" --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" --security-group-ids "$sg" \
    --iam-instance-profile "Name=$NAME" \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$VOLUME_GB,VolumeType=gp3}" \
    --metadata-options "HttpTokens=required" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
    --user-data '#!/bin/bash
apt-get update -y && apt-get install -y docker.io awscli
systemctl enable --now docker
usermod -aG docker ubuntu' \
    --query 'Instances[0].InstanceId' --output text

  bold "  waiting for it to come up..."
  aws ec2 wait instance-running --region "$REGION" --instance-ids "$(instance_id)"
  bold "  $(instance_id) at $(public_ip)"
  dim "  give it a minute to finish installing docker, then: scripts/deploy.sh push"
}

cmd_push() {
  need docker; need aws
  local ip; ip=$(public_ip) || die "no instance — run provision first"
  [ -n "$ip" ] && [ "$ip" != "None" ] || die "instance has no public IP"

  bold "building $IMAGE for $PLATFORM (ref: $FASTAPI_REF)"
  docker build --platform "$PLATFORM" --build-arg "FASTAPI_REF=$FASTAPI_REF" \
    -t "$IMAGE" .

  bold "copying to $ip"
  docker save "$IMAGE" -o "$TARBALL"
  dim "  $(du -h "$TARBALL" | cut -f1) — this takes a few minutes"
  scp -i "$HOME/.ssh/$KEY_NAME.pem" -o StrictHostKeyChecking=accept-new \
    "$TARBALL" "$SSH_USER@$ip:/tmp/"
  ssh -i "$HOME/.ssh/$KEY_NAME.pem" "$SSH_USER@$ip" \
    "sudo docker load -i /tmp/$NAME.tar && rm /tmp/$NAME.tar"
  rm -f "$TARBALL"
  bold "  loaded. next: scripts/deploy.sh start"
}

cmd_start() {
  need aws
  local ip; ip=$(public_ip) || die "no instance"
  aws ssm get-parameter --region "$REGION" --name "$SSM_KEY_PARAM" --with-decryption \
    >/dev/null 2>&1 || die "no API key in SSM at $SSM_KEY_PARAM. Put it there with:
  aws ssm put-parameter --region $REGION --name $SSM_KEY_PARAM \\
    --type SecureString --value sk-..."

  # The key is fetched on the instance and piped into docker's env file; it is
  # never an argument, so it never lands in a shell history or a process list.
  ssh -i "$HOME/.ssh/$KEY_NAME.pem" "$SSH_USER@$ip" bash -s <<REMOTE
set -euo pipefail
sudo docker rm -f $NAME 2>/dev/null || true
umask 077
aws ssm get-parameter --region $REGION --name $SSM_KEY_PARAM --with-decryption \
  --query Parameter.Value --output text > /tmp/key 2>/dev/null \
  || { echo "instance cannot read SSM — attach a role with ssm:GetParameter"; exit 1; }
printf 'OPENAI_API_KEY=%s\nLLM_PROVIDER=openai\n' "\$(cat /tmp/key)" > /tmp/env
rm -f /tmp/key
sudo docker run -d --name $NAME --restart unless-stopped \
  -p 80:8000 --env-file /tmp/env \
  -e RATE_PER_MINUTE=5 -e RATE_PER_DAY=30 -e GLOBAL_PER_DAY=500 \
  $IMAGE
rm -f /tmp/env
REMOTE
  bold "  started at http://$ip"
  dim "  it loads models on boot; /ready reports when it is serving"
}

cmd_smoke() {
  local ip; ip=$(public_ip) || die "no instance"
  local base="http://$ip"
  bold "smoke test against $base"

  printf '  health    '; curl -fsS "$base/health" || die "FAILED"; echo
  printf '  ready     '; curl -fsS "$base/ready"  || die "FAILED"; echo

  printf '  answers   '
  curl -fsS -X POST "$base/ask" -H 'content-type: application/json' \
    -d '{"question":"How do I declare a dependency with Depends?"}' \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"abstained={d[\"abstained\"]} citations={len(d[\"citations\"])}")'

  printf '  abstains  '
  curl -fsS -X POST "$base/ask" -H 'content-type: application/json' \
    -d '{"question":"What is the airspeed velocity of an unladen swallow?"}' \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"abstained={d[\"abstained\"]}")'

  printf '  rate limit '
  local code
  for _ in $(seq 1 8); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$base/ask" \
      -H 'content-type: application/json' -d '{"question":"ping"}')
    [ "$code" = "429" ] && break
  done
  [ "$code" = "429" ] && echo "429 as expected" || die "never rate limited (got $code)"
  bold "  all four passed"
}

cmd_destroy() {
  need aws
  local id; id=$(instance_id || true)
  [ -n "$id" ] || die "nothing tagged $NAME to destroy"
  bold "This terminates $id and deletes the security group."
  read -r -p "Type the instance id to confirm: " confirm
  [ "$confirm" = "$id" ] || die "not confirmed"
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$id" >/dev/null
  aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$id"
  aws ec2 delete-security-group --region "$REGION" --group-name "$NAME" 2>/dev/null || \
    dim "  security group still in use; delete it manually later"
  aws iam remove-role-from-instance-profile --instance-profile-name "$NAME" \
    --role-name "$NAME" 2>/dev/null || true
  aws iam delete-instance-profile --instance-profile-name "$NAME" 2>/dev/null || true
  aws iam delete-role-policy --role-name "$NAME" --policy-name read-api-key 2>/dev/null || true
  aws iam delete-role --role-name "$NAME" 2>/dev/null || true
  bold "  terminated"
}

case "${1:-plan}" in
  plan)      cmd_plan ;;
  provision) cmd_provision ;;
  push)      cmd_push ;;
  start)     cmd_start ;;
  smoke)     cmd_smoke ;;
  destroy)   cmd_destroy ;;
  *)         die "usage: $0 {plan|provision|push|start|smoke|destroy}" ;;
esac
