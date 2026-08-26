#!/usr/bin/env python3
"""
Template-based update_services.py for OpenAI.

Yields model dictionaries that are rendered using Jinja2 templates.

Unlike single-purpose providers, OpenAI's /v1/models lists every model the
account can reach — audio, image, embedding, moderation, realtime, and dated
snapshots of each chat model. The catalog here is chat-over-Chat-Completions
only (the gateway's llm_translator speaks that dialect), so everything else is
filtered out, and dated snapshots are skipped in favour of their alias ids
(`gpt-4o`, not `gpt-4o-2024-08-06`) to keep the catalog stable and small.

Usage: python scripts/update_services.py
"""

import os
import re
import sys
from pathlib import Path
from typing import Iterator

import httpx

from unitysvc_sellers.model_data import ModelDataFetcher, ModelDataLookup
from unitysvc_sellers.params_render import write_params_from_iterator

# Provider Configuration
PROVIDER_NAME = "openai"
PROVIDER_DISPLAY_NAME = "OpenAI"
# No /v1 suffix: the example/connectivity presets (and gateway callers)
# carry the version prefix in the request path, so a /v1 here doubles up
# into /v1/v1/… and 404s every test.
API_BASE_URL = "https://api.openai.com"
ENV_API_KEY_NAME = "OPENAI_API_KEY"

SCRIPT_DIR = Path(__file__).parent

# Model ids containing any of these are not chat-completions services —
# different endpoint, different request shape, or retired families.
SKIP_SUBSTRINGS = (
    "embed",         # /v1/embeddings
    "whisper",       # /v1/audio/transcriptions
    "transcribe",    # /v1/audio/transcriptions
    "tts",           # /v1/audio/speech
    "audio",         # audio-native chat variants
    "realtime",      # websocket realtime API
    "dall-e",        # /v1/images
    "image",         # gpt-image-* → /v1/images
    "sora",          # video generation
    "moderation",    # /v1/moderations
    "search",        # search-preview variants
    "computer-use",  # responses-only agentic model
    "codex",         # responses-only coding models
    "deep-research", # responses-only research models
    "-pro",          # gpt-5-pro etc. are Responses API only
    "instruct",      # legacy /v1/completions
    "davinci",       # legacy completions family
    "babbage",       # legacy completions family
    "preview",       # dated/retired preview lines
)

# Dated snapshots: gpt-4o-2024-08-06, gpt-4-0613, o1-2024-12-17, …
DATED_SNAPSHOT_RE = re.compile(r"-\d{4}(-\d{2}-\d{2})?$")


class ModelSource:
    """Fetches models and yields template dictionaries."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.data_fetcher = ModelDataFetcher()
        self.litellm_data = None

    def iter_models(self) -> Iterator[dict]:
        """Yield model dictionaries for template rendering."""
        # Fetch LiteLLM data once
        self.litellm_data = self.data_fetcher.fetch_litellm_model_data()

        print(f"Fetching models from {PROVIDER_DISPLAY_NAME} API...")
        try:
            r = httpx.get(
                f"{API_BASE_URL}/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
            r.raise_for_status()
            models = r.json().get("data", [])
            print(f"Found {len(models)} models\n")
        except Exception as e:
            print(f"Error listing models: {e}")
            return

        for i, model_info in enumerate(models, 1):
            model_id = model_info.get("id", "")
            print(f"[{i}/{len(models)}] {model_id}")

            if self._skip_reason(model_id):
                print(f"  SKIP ({self._skip_reason(model_id)})")
                continue

            if not self._probe_chat(model_id):
                continue

            # Build template variables
            template_vars = self._build_template_vars(model_id, model_info)
            if template_vars:
                template_vars["supports_tools"] = self._probe_tools(model_id)
                yield template_vars
                print("  OK")

    def _probe_chat(self, model_id: str) -> bool:
        """One live chat-completions request per surviving candidate.

        /v1/models still lists models the API refuses to serve —
        gpt-5-chat-latest sits in the listing while every request returns
        400 model_not_found "has been deprecated". Publishing such an id
        ships a service whose every call fails, so each candidate must
        prove it answers. Deliberately parameter-free (no token cap):
        a cap param rejected by one model family would read as a dead
        model. Transient failures (429/5xx/network) keep the model —
        only an explicit invalid_request refusal drops it.
        """
        try:
            r = httpx.post(
                f"{API_BASE_URL}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": model_id,
                      "messages": [{"role": "user", "content": "ping"}]},
                timeout=60.0,
            )
        except Exception as e:
            print(f"  probe error ({e}) — keeping")
            return True
        if r.status_code == 200:
            return True
        if r.status_code in (400, 404):
            try:
                err = r.json().get("error", {})
            except Exception:
                err = {}
            print(f"  SKIP (probe {r.status_code}: "
                  f"{err.get('code')} — {err.get('message', '')[:80]})")
            return False
        print(f"  probe HTTP {r.status_code} — keeping")
        return True

    def _probe_tools(self, model_id: str) -> bool:
        """Does Chat Completions accept a function tool for this model?

        Some models refuse the combination — gpt-5.6-* returns 400
        "Function tools with reasoning_effort are not supported … in
        /v1/chat/completions" unless reasoning_effort is 'none'. The
        catalog's canonical tools example sends neither knob, so a model
        that refuses it must not declare the tools capability (the
        gateway test would fail exactly like a customer's first call).
        Only an explicit 400/404 refusal clears the flag.
        """
        tool = {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the current time",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        try:
            r = httpx.post(
                f"{API_BASE_URL}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": model_id,
                      "messages": [{"role": "user", "content": "What time is it?"}],
                      "tools": [tool]},
                timeout=60.0,
            )
        except Exception as e:
            print(f"  tools probe error ({e}) — assuming supported")
            return True
        if r.status_code in (400, 404):
            try:
                msg = r.json().get("error", {}).get("message", "")
            except Exception:
                msg = ""
            print(f"  tools NOT supported ({msg[:70]})")
            return False
        return True

    def _skip_reason(self, model_id: str) -> str | None:
        model_lower = model_id.lower()
        for kw in SKIP_SUBSTRINGS:
            if kw in model_lower:
                return f"non-chat family: {kw}"
        if DATED_SNAPSHOT_RE.search(model_lower):
            return "dated snapshot — alias id covers it"
        return None

    @staticmethod
    def _display_name(model_id: str) -> str:
        """OpenAI-style display names.

        `str.title()` mangles OpenAI's ids ("Gpt 5.1", "Chatgpt 4O Latest",
        "O3"), so follow the provider's own casing: o-series ids are
        canonically lowercase (o3, o4-mini); GPT ids keep the hyphenated
        version ("GPT-5.1", "GPT-4o") with any trailing words title-cased
        ("GPT-4o Mini", "ChatGPT-4o Latest").
        """
        if re.match(r"^o\d", model_id):
            return model_id
        m = re.match(r"^(chatgpt|gpt)-([\d.]+[a-z]*)(?:-(.+))?$", model_id)
        if not m:
            return model_id.replace("-", " ").replace("_", " ").title()
        prefix = "ChatGPT" if m.group(1) == "chatgpt" else "GPT"
        rest = ""
        if m.group(3):
            rest = " " + " ".join(w.title() for w in m.group(3).split("-"))
        return f"{prefix}-{m.group(2)}{rest}"

    def _build_template_vars(self, model_id: str, model_info: dict) -> dict | None:
        """Build template variables for a model."""
        display_name = self._display_name(model_id)

        # Build details from LiteLLM data and model info
        details = {}
        model_data = ModelDataLookup.lookup_model_details(
            model_id, self.litellm_data or {})

        # Second gate: whatever survives the id filter must still be a chat
        # model according to LiteLLM's registry (unknown models pass — the
        # registry lags new releases).
        if model_data and model_data.get("mode") not in (None, "chat"):
            print(f"  SKIP (litellm mode: {model_data.get('mode')})")
            return None

        if model_data:
            for field in [
                    "max_tokens", "max_input_tokens", "max_output_tokens",
                    "mode"
            ]:
                if field in model_data:
                    details[field] = model_data[field]
            if "litellm_provider" in model_data:
                details["litellm_provider"] = model_data["litellm_provider"]

        if "owned_by" in model_info:
            details["owned_by"] = model_info["owned_by"]
        if "object" in model_info:
            details["object"] = model_info["object"]

        # Canonical (snake_case) metadata required by the platform validator
        # for LLM offerings.  Both keys must be present; null asserts
        # "unknown".  OpenAI models are closed-source so parameter_count
        # is permanently null per the canonical helper.  metadata_sources
        # records provenance so reviewers can triage stale-value reports.
        canonical = ModelDataLookup.get_canonical_metadata(
            model_id,
            fetcher=self.data_fetcher,
        )
        details["context_length"] = canonical["context_length"]
        details["parameter_count"] = canonical["parameter_count"]
        if canonical["sources"]:
            details["metadata_sources"] = canonical["sources"]

        # BYOK: the customer supplies their own API key, so usage is billed by
        # the provider directly and UnitySVC meters nothing — the price is Free.
        # This plain description is what payout_price keeps (seller-facing). The
        # customer-facing listing cell is composed in listing.json.j2 from
        # pricing_note, into the "<amount> — <PILL> | <note>" grammar; do not
        # build it here, since this dict feeds payout_price too.
        pricing = {
            "type": "constant",
            "price": "0",
            "description": "Free (BYOK)",
        }
        pricing_note = None
        if model_data and "input_cost_per_token" in model_data and "output_cost_per_token" in model_data:
            input_price = round(float(
                model_data["input_cost_per_token"]) * 1_000_000, 4)
            output_price = round(float(
                model_data["output_cost_per_token"]) * 1_000_000, 4)
            if "cache_read_input_token_cost" in model_data:
                cached_price = round(float(
                    model_data["cache_read_input_token_cost"]) * 1_000_000, 4)
                pricing_note = (
                    f"${self._format_price(input_price)} / "
                    f"${self._format_price(output_price)} / "
                    f"${self._format_price(cached_price)} "
                    f"per 1M input/output/cached tokens"
                )
            else:
                pricing_note = (
                    f"${self._format_price(input_price)} / "
                    f"${self._format_price(output_price)} "
                    f"per 1M input/output tokens"
                )

        return {
            # Folder path under specs/ == listing.name == "<provider>/<model_id>"
            # (flat layout, #1263). populate_from_iterator preserves the slash.
            "name": f"{PROVIDER_NAME}/{model_id}",
            # Offering name is the bare upstream model_id
            "offering_name": model_id,
            # Offering fields
            "display_name": display_name,
            "description": f"{display_name} language model",
            "service_type": "llm",
            "status": "ready",
            "details": details,
            "payout_price": pricing,
            # Reference rates for the BYOK pricing paragraph (template-rendered)
            "pricing_note": pricing_note,
            # Listing fields
            "list_price": pricing,
            # Visibility the upload applies, from the DEFAULT_VISIBILITY GitHub
            # variable. `or` rather than a get() default: an unset variable
            # reaches the runner as an empty string, which would render an
            # invalid "" visibility.
            "default_visibility": os.environ.get("DEFAULT_VISIBILITY") or "unlisted",
            # Provider config (for templates)
            "provider_name": PROVIDER_NAME,
            "provider_display_name": PROVIDER_DISPLAY_NAME,
            "api_base_url": API_BASE_URL,
            "env_api_key_name": ENV_API_KEY_NAME,
        }

    def _format_price(self, price: float) -> str:
        """Format price without trailing .0 for whole numbers."""
        if price == int(price):
            return str(int(price))
        return str(price)


def main():
    api_key = os.environ.get(ENV_API_KEY_NAME)
    if not api_key:
        print(f"Error: {ENV_API_KEY_NAME} not set")
        sys.exit(1)

    source = ModelSource(api_key)
    write_params_from_iterator(
        iterator=source.iter_models(),
        output_dir=SCRIPT_DIR.parent / "specs",
    )


if __name__ == "__main__":
    main()
