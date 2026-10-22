# 6. Deploy to a single EC2 instance, not a serverless platform

Date: 2026-10-22

## Context

The demo needs somewhere to live. The obvious modern answer for a container
that nobody uses most of the time is a scale-to-zero platform — Cloud Run, or
Lambda with a container image — where an idle demo costs nothing.

Measuring the container first changed that answer.

| what | measured |
| --- | --- |
| image size | 3.6 GB, almost all PyTorch |
| resident memory, warm | **638 MB** (both models, index, after queries) |
| retrieval, warm | ~270 ms p50 |
| end-to-end p50 | ~13 s, almost entirely the LLM call |

Scale-to-zero means every cold request pays the startup cost: process start,
~180 MB of model weights loaded, index opened. For a demo whose entire point is
showing a fast, observable pipeline, a visitor's first question is the worst
possible moment to pay that. The one thing serverless is good at — being idle
cheaply — is precisely what hurts here.

Lambda is additionally impractical: a 3.6 GB image's cold start would very
likely exceed API Gateway's 29-second limit.

AWS has no container service that scales to zero *and* starts fast. App Runner
and Fargate both keep an instance warm, so they cost like a small VM while
adding a layer.

The 638 MB figure also matters: the image is big but the process is not, so a
2 GB instance is comfortable where I had assumed 4 GB.

## Decision

A single EC2 instance running one container.

- **`t4g.small`** — 2 vCPU, 2 GB. 638 MB resident leaves real headroom;
  `t4g.micro` at 1 GB would OOM on any spike.
- **Graviton (arm64)**, not x86. ~20% cheaper for identical specs, and the
  development machine is arm64, so images build natively instead of under
  emulation. Containers are not architecture-neutral — an arm64 image will not
  run on an x86_64 host — so this is a decision the build has to state, not
  discover.
- **`ap-south-1` (Mumbai)**, nearest to where this is operated. Network latency
  barely matters for the visitor — a 50 ms vs 250 ms round trip is noise next to
  a 13-second LLM call — so this is chosen for operational convenience.
- **One container, one worker.** Embedded Qdrant holds an exclusive lock on its
  storage directory; a second worker fails to start. This was not a theoretical
  concern — it surfaced during development when an ablation script and an eval
  run collided on the lock.
- **Everything baked into the image**: models, corpus and both indexes are built
  at image build time, so a started container needs nothing from the network
  except the LLM provider.

Roughly $14/month: ~$12 compute plus ~$1.60 for 20 GB of gp3 (estimates, not
quoted prices).

## Alternatives

- **Cloud Run.** The original plan. Genuinely good at idle cost, and the
  measurements turned its main advantage into its main drawback. Also a second
  cloud account to hold.
- **Lambda container.** The only AWS option with true scale-to-zero. A 3.6 GB
  cold start against a 29-second gateway timeout is not a demo.
- **App Runner / Fargate.** Managed, no scale-to-zero, and more expensive than
  the VM they are running on. Worth revisiting if this ever needs to scale.
- **Co-host on an existing box.** Cheapest option — marginal cost near zero.
  Rejected for a portfolio piece: a separate instance gets its own URL and its
  own failure domain, and cannot take anything else down.
- **Shrink the image with ONNX instead of PyTorch.** Plausibly under 1 GB and a
  few hundred MB of memory, which would put `t4g.micro` (~$8/month) in reach.
  That is a model change: it needs the eval re-run and the numbers to hold
  before adopting. Deliberately not bundled into a deployment decision.

## Consequences

- **No autoscaling, and no redundancy.** One instance, one container. If it dies
  the demo is down until it restarts. Acceptable for a demo; not a pattern to
  copy for anything that matters.
- **The instance bills whether or not anyone visits.** ~$14/month is the cost of
  not making visitors wait for a cold start.
- **Horizontal scaling is blocked by design**, not by accident: embedded Qdrant
  is single-process. Scaling out means running Qdrant as a service again, which
  the code already supports via `QDRANT_URL`.
- **A rebuild is needed to update the corpus**, since the index lives in the
  image. That is a feature as much as a cost: the running container's index is
  immutable and matches a known build.
- **Spend protection is not optional.** A public `/ask` spends money per request
  on a live API key, so rate limits and a global daily cap ship with it rather
  than after it.
