# free-model-aggregator-02

[![make.fun bounty](https://img.shields.io/badge/make.fun-bounty-blue)](https://make.fun/bounty/free-model-aggregator-02)
[![Prize $500](https://img.shields.io/badge/prize-%24500-green)](https://make.fun/bounty/free-model-aggregator-02)
[![Status open](https://img.shields.io/badge/status-open-orange)](https://make.fun/bounty/free-model-aggregator-02)

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API — in JSON and in server-sent events — and answers every
prompt with a real completion from a live free OpenRouter model, dialled
through the grader's gateway on the entrant's own pushed key. No simulated
provider, no local model: every answer comes from the internet, at $0.

Built for the make.fun competition
[free-model-aggregator-02](https://make.fun/bounty/free-model-aggregator-02)
($500, entries 2026-11-15 → 2026-11-17).

## What it does

- Listens on `0.0.0.0` at a free port it picks (never 8080) and registers
  that port and its catalog with the env at `POST $MAKEFUN_ENV_URL/register`.
- Serves `GET /v1/models` in the wire its auth header names — the OpenAI
  shape for `Authorization: Bearer`, the Anthropic shape for `x-api-key` —
  listing exactly the fourteen free models the brief names, once each.
- Serves `POST /v1/chat/completions` and `POST /v1/messages`, in JSON and
  in streaming, passing system prompts, tools, stop sequences, `max_tokens`
  and JSON mode through, and translating Anthropic-shaped requests to Chat
  Completions and back. Every model call goes to the gateway's Chat
  Completions route only — never the Anthropic route, which the gateway
  refuses for these ids.
- Answers each of the trial's six prompts with a real completion from one
  of the fourteen `openrouter/*:free` models, and reports the id it really
  dialled — with the `openrouter/` prefix — on every answer and every
  streamed chunk.
- Polls `GET $MAKEFUN_ENV_URL/work` and exits 0 when it says done.
- `node main.ts --list` prints the catalog document byte-for-byte as it
  registers it.

## Run it

Node 22, standard library only — nothing to install.

```sh
node main.ts --list   # print the catalog and exit 0
node main.ts          # start the server (needs MAKEFUN_ENV_URL,
                      # MAKEFUN_PROMPT_FILE, OPENAI_BASE_URL, OPENAI_API_KEY)
```

In a trial the sandbox sets those variables; the dummy key in
`OPENAI_API_KEY` is swapped by the gateway for the OpenRouter key pushed
to the competition, which never enters the sandbox. Without a pushed key
every model call is refused 401 and no prompt is served.

## How routing works

Every prompt is tried on the best free model that meets its requirements
first — ranked by the brief's `rating_milli` — then on the next best by a
budget-aware score:

```
score = (best rating for the prompt − this model's rating)
      + reuse penalty for the calls already spent on this model
      + refusal penalty for each refusal it has already given
```

The reuse penalty is 60 milli for the first call on a model, then 10 more
per further call; the refusal penalty is 150 per refusal. A fresh
near-best model therefore beats a tired best one, and a model that refused
once is retried only after one intervening prompt — a failed retry costs
only a tie-break call, while a recovered model regains its full rating.

In practice that means the strongest general model
(`openrouter/thinkingmachines/inkling-2-small:free`, 1475) serves the
plain and tools prompts, and the strongest JSON-capable model
(`openrouter/nvidia/nemotron-3.5-super-vl:free`, 1303) serves the
JSON-mode prompts — six upstream calls for six prompts when nothing
refuses.

## How failover works

- A 429 or 5xx is never retried on the same model for the same prompt:
  the next candidate in the budget-aware order is dialled instead.
- A 401 means no key was pushed, so the server stops dialling for the
  trial instead of burning calls every model would refuse the same way.
- A model that answers without the prompt's token in its text is kept as
  a fallback while the next candidate is tried — a token-less answer earns
  the prompt nothing on quality — and served only when no candidate does
  better.
- Every refusal a prompt recovers from is reported on its answer in the
  `x-makefun-upstream-failures` header (a JSON array of at most 8 rows
  `{"status", "code", "model"}`), the only trail the failover column can
  see. At most 4 gateway calls are spent on one prompt.
- The output cap is always set and always within bounds (`max_tokens`,
  clamped to 8192, defaulting to 1024 when the request names none),
  because the gateway refuses uncapped calls.

## Determinism

Every choice is a function of the brief, the requests and the gateway's
answers so far. No wall clock (`Date.now` / `new Date`), no unseeded
randomness, fixed timestamps in every reply — so a replay leaves the
same state export byte for byte.

## Layout

```
main.ts    the whole server: the brief, the catalog, routing, failover,
           both client wires, register and poll. Standard library only.
LICENSE    MIT
.makefun/owner.json
           the claim file naming the payout wallet.
```

## Local trial results

Validated with the kit's own `grade-local` (process driver) against a
stub gateway that answers like OpenRouter and echoes each prompt's token:

| trial | score | quality | served | wire | failover | catalog | calls | replay |
| ----- | ----: | ------: | -----: | ---: | -------: | ------: | ----: | ------ |
| trial-0 | 100 | 100 | 100 | 100 | 100 | 100 | 6 | matched |
| trial-1 | 100 | 100 | 100 | 100 | 100 | 100 | 6 | matched |
| trial-2 | 100 | 100 | 100 | 100 | 100 | 100 | 6 | matched |
| trial-0, first call to the top model refused 429 | 98 | 96 | 100 | 100 | 100 | 100 | 6 | matched |
| trial-0, first answer without its token | 98 | 96 | 100 | 100 | 100 | 100 | 7 | matched |

Live trials against the real free models were not run from here: no
`.env` with `OPENROUTER_API_KEY` exists in this workspace, and no key
may be invented, reused or created — push one with
`makefun secrets push free-model-aggregator-02 --env-file .env` before
submitting, and the server will dial the real models on it.
