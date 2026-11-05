"""Two-level cache keys: visual observations vs prompt selections (task #5).

This module deliberately owns NO hash implementation: every key is computed
with :func:`vibecut.core.cache.identity_digest` - the same single-pass,
canonical-JSON cache-identity mechanism behind the repo's normalized/render
caches. Candidate and observation content enters each key exactly once, as
canonical ``to_json()`` data (never pre-hashed digests stacked on digests).

Invalidation contract (what makes a key change):
  * cheap features    <- source content fingerprint, tool identity,
                         sampling identity, analysis schema version;
  * observation batch <- the above + canonical asset id, model id, the exact
                         batch spec (img ids / times / view ids) and the
                         observe prompt version. Never the focus text:
                         changing the focus reuses every observation;
  * selection         <- source fingerprint, asset id, model id, focus JSON
                         (priority/target duration/exclusions), the canonical
                         candidate windows, the canonical observations and
                         the select prompt + selection algorithm versions.
                         Any change recomputes the selection and never reuses
                         an old one.
Camera-path-only edits never reach this module at all. Old entries are never
mutated: a changed input yields a new key and the stale entry simply stops
being reachable (content-addressed garbage), so an algorithm upgrade can
never serve a selection computed by the previous algorithm.
"""

from __future__ import annotations

from typing import Any

from vibecut.analysis.model import OBSERVE_PROMPT_VERSION, SELECT_PROMPT_VERSION, SELECTION_ALGORITHM_VERSION
from vibecut.analysis.plan import ANALYSIS_SCHEMA_VERSION, VISUAL_CACHE_SCHEMA_VERSION
from vibecut.core.cache import identity_digest


def cheap_features_key(*, source_fingerprint: str, asset_id: str, tool_identity: str, sampling_identity: dict) -> str:
    return identity_digest(
        {"namespace": "analysis-cheap", "schema": VISUAL_CACHE_SCHEMA_VERSION},
        source_fingerprint, asset_id, tool_identity, sampling_identity,
    )


def observation_key(
    *,
    source_fingerprint: str,
    asset_id: str,
    tool_identity: str,
    sampling_identity: dict,
    model_id: str,
    batch: list[dict],
    inference_parameters: dict | None = None,
) -> str:
    """Prompt-independent observation-batch key; ``batch`` is canonical data."""
    return identity_digest(
        {"namespace": "analysis-observation", "schema": VISUAL_CACHE_SCHEMA_VERSION,
         "prompt_version": OBSERVE_PROMPT_VERSION},
        source_fingerprint, asset_id, tool_identity, sampling_identity, model_id, batch, inference_parameters,
    )


def selection_key(
    *,
    source_fingerprint: str,
    asset_id: str,
    model_id: str,
    focus_json: dict,
    candidates: list[dict],
    observations: list[dict],
    inference_parameters: dict | None = None,
) -> str:
    """Prompt/duration-dependent selection key over canonical candidate data."""
    return identity_digest(
        {"namespace": "analysis-selection", "schema": ANALYSIS_SCHEMA_VERSION,
         "prompt_version": SELECT_PROMPT_VERSION, "algorithm": SELECTION_ALGORITHM_VERSION},
        source_fingerprint, asset_id, model_id, focus_json, candidates, observations, inference_parameters,
    )
