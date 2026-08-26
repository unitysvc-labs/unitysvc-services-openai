# unitysvc-services-openai

UnitySVC service catalog for **OpenAI** models — GPT chat models published as
bring-your-own-key (BYOK) services on the UnitySVC gateway/marketplace.
Tracking issue: [unitysvc-labs/unitysvc-labs#41](https://github.com/unitysvc-labs/unitysvc-labs/issues/41).

Each service fronts one OpenAI chat model (`gpt-5.1`, `gpt-4o`, `o3`, …)
through the gateway. The gateway price is **free**: customers bring their own
OpenAI API key (as the `OPENAI_API_KEY` customer secret) and OpenAI bills
their usage directly. The gateway endpoint accepts both OpenAI Chat
Completions and Anthropic Messages requests — the `llm_translator`
auto-detects the dialect and translates for the upstream.

## Layout

Services are authored as **param files** rendered by the repo template
(`services/templates/`), one param file per model:

```
services/
├── templates/                 # offering.json.j2 + listing.json.j2 + provider.json
├── scripts/update_params.py   # populate: fetch /v1/models, filter to chat aliases
├── seller.secrets.txt         # committed secrets manifest (.env.example)
└── specs/openai/
    ├── <model>.json           # param file — the service definition
    └── <model>.service.json   # identity sidecar (auto-written on upload; commit it)
```

Scope: **chat models only**, alias ids only (`gpt-4o`, not
`gpt-4o-2024-08-06`). Audio, image, embedding, realtime, and Responses-only
models (`gpt-5-pro`, codex, deep-research) are filtered out by
`scripts/update_params.py` — see `SKIP_SUBSTRINGS` there.

## Workflow

```bash
# environment (staging seller credentials + OPENAI_API_KEY)
set -a; . ./services/seller.secrets.txt; set +a

usvc_seller specs validate
usvc_seller specs format
usvc_seller specs run-tests 'openai/%'             # upstream-side (needs OPENAI_API_KEY)
usvc_seller specs upload 'openai/%'                # push to staging
usvc_seller services run-tests 'openai/%' --force  # gateway-side
```

The nightly populate workflow re-runs `update_params.py` against OpenAI's
`/v1/models` (needs the `OPENAI_API_KEY` GitHub secret) and opens a PR with
catalog changes. Merges to `main` upload to staging automatically; production
uploads are manual (`workflow_dispatch`).

See `CLAUDE.md` for repo conventions and
[unitysvc-sellers/docs](https://github.com/unitysvc/unitysvc-sellers/tree/main/docs)
for the authoritative format documentation.
