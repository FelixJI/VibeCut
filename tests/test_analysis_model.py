"""Model client against the offline loopback fixture: probe, observe, select,
one-bounded-repair, structured failures, OOM/vision error mapping (AC5)."""

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis_offline_server import OfflineLM, observe_payload, select_payload

from vibecut.analysis import validate
from vibecut.analysis.errors import (
    AIUnavailableError,
    ModelAPIError,
    ModelOOMError,
    NoVisionError,
    StructuredOutputError,
)
from vibecut.analysis.model import ModelClient
from vibecut.analysis.plan import MeteringReport
from vibecut.analysis.transport import LoopbackTransport

MODEL = "qwen3.8-27b@q4_k_xl"


def _client(lm, **kw) -> ModelClient:
    metering = MeteringReport()
    transport = LoopbackTransport(lm.base_url, request_timeout_s=kw.pop("timeout", 10.0))
    return ModelClient(transport, MODEL, metering=metering, **kw), metering


def test_probe_reports_state_and_records():
    lm = OfflineLM().start()
    try:
        client, metering = _client(lm)
        info = client.probe()
        assert info["id"] == MODEL and info["state"] == "loaded" and info["vision"] is True
        assert metering.requests[0]["kind"] == "probe"
    finally:
        lm.stop()


def test_probe_not_loaded_is_ai_unavailable():
    lm = OfflineLM(model_state="not-loaded").start()
    try:
        client, _ = _client(lm)
        with pytest.raises(AIUnavailableError) as exc:
            client.probe()
        assert exc.value.code == "model-not-loaded"
    finally:
        lm.stop()


def test_probe_no_vision():
    lm = OfflineLM(vision=False).start()
    try:
        client, _ = _client(lm)
        with pytest.raises(NoVisionError):
            client.probe()
    finally:
        lm.stop()


def test_probe_unknown_model_lists_available():
    lm = OfflineLM().start()
    try:
        client, _ = _client(lm)
        client.model_id = "ghost-model"
        with pytest.raises(AIUnavailableError) as exc:
            client.probe()
        assert exc.value.code == "model-not-found"
    finally:
        lm.stop()


def _observe_images():
    # tiny valid JPEG header bytes are enough - fixture never decodes them
    return [("s00.eq+030", b"\xff\xd8\xff\xe0fake"), ("s00.eq-030", b"\xff\xd8\xff\xe0fake")]


def _observe_meta():
    from fractions import Fraction

    return {"s00.eq+030": (Fraction("1/2"), "eq+030"), "s00.eq-030": (Fraction("1/2"), "eq-030")}


def test_observe_happy_path_meters_images():
    lm = OfflineLM().start()
    try:
        client, metering = _client(lm)
        lm.chat_script.append({"content": observe_payload(["s00.eq+030", "s00.eq-030"])})
        obs, repairs = client.observe(_observe_images(), _observe_meta())
        assert repairs == [] and len(obs) == 2
        assert metering.requests[-1]["images"] == 2
        assert metering.requests[-1]["kind"] == "observe"
        # no prompt/image content ever lands in metering
        import json as _json

        assert "fake" not in _json.dumps(metering.to_json())
    finally:
        lm.stop()


def test_observe_repair_retry_once_then_fail():
    lm = OfflineLM().start()
    try:
        client, metering = _client(lm)
        lm.chat_script.append({"content": observe_payload(["ghost-id"])})  # invalid
        lm.chat_script.append({"content": observe_payload(["s00.eq+030", "s00.eq-030"])})  # repaired
        obs, repairs = client.observe(_observe_images(), _observe_meta())
        assert len(obs) == 2 and len(repairs) == 1
        assert metering.retry_count == 1
        # still invalid after repair -> stage fails
        lm.chat_script.append({"content": observe_payload(["ghost"])})
        lm.chat_script.append({"content": observe_payload(["ghost"])})
        with pytest.raises((StructuredOutputError, validate.ValidationFailureError)):
            client.observe(_observe_images(), _observe_meta())
        assert metering.retry_count == 2  # exactly one retry per call, never more
    finally:
        lm.stop()


def test_malformed_content_json_fails_after_one_retry():
    lm = OfflineLM().start()
    try:
        client, metering = _client(lm)
        # HTTP body itself is not JSON: transport-level API error, no retry consumed
        lm.chat_script.append({"raw": "not-json"})
        with pytest.raises(ModelAPIError) as exc:
            client.observe(_observe_images(), _observe_meta())
        assert exc.value.code == "bad-json"
        assert metering.retry_count == 0
        # body is JSON but message.content is not a JSON string: structured-output
        # failure with exactly one bounded repair retry, then stage failure
        bad_content = '{"choices": [{"message": {"content": "still not json"}}]}'
        lm.chat_script.append({"raw": bad_content})
        lm.chat_script.append({"raw": bad_content})
        with pytest.raises(StructuredOutputError):
            client.observe(_observe_images(), _observe_meta())
        assert metering.retry_count == 1
    finally:
        lm.stop()


def test_select_validates_and_maps_oom():
    from fractions import Fraction

    lm = OfflineLM().start()
    try:
        client, _ = _client(lm)
        lm.chat_script.append({"content": select_payload([
            {"candidate_id": "c0001", "img_ids": ["image"], "relevance": 0.9, "confidence": 0.8,
             "reason": "有人物跳舞"},
        ])})
        sels, repairs, dropped = client.select(
            {"priority": "person"}, [{"candidate_id": "c0001", "observed": [{"img_id": "image", "time_rel": "1/1"}]}], {"c0001": (Fraction(0), Fraction(2))},
        )
        assert len(sels) == 1 and not dropped and not repairs

        lm.chat_script.append({"status": 500, "message": "CUDA out of memory while loading KV cache"})
        with pytest.raises(ModelOOMError):
            client.select({"priority": "person"}, [{"candidate_id": "c0001", "observed": [{"img_id": "image", "time_rel": "1/1"}]}], {"c0001": (Fraction(0), Fraction(2))})

        lm.chat_script.append({"status": 400, "message": "image input is not supported by this model"})
        with pytest.raises((NoVisionError, ModelAPIError)):
            client.select({"priority": "person"}, [{"candidate_id": "c0001", "observed": [{"img_id": "image", "time_rel": "1/1"}]}], {"c0001": (Fraction(0), Fraction(2))})
    finally:
        lm.stop()


def test_select_repair_drops_bad_items_keeps_good_ones():
    from fractions import Fraction

    lm = OfflineLM().start()
    try:
        client, _ = _client(lm)
        windows = {"c0001": (Fraction(0), Fraction(2)), "c0002": (Fraction(5), Fraction(8))}
        lm.chat_script.append({"content": select_payload([
            {"candidate_id": "cXXXX", "img_ids": ["image"], "relevance": 1, "confidence": 1,
             "reason": "x"},
            {"candidate_id": "c0002", "img_ids": ["image"], "relevance": 0.7, "confidence": 0.6,
             "reason": "风景好"},
        ])})
        sels, repairs, dropped = client.select({"priority": "scenery"}, [{"candidate_id": "c0002", "observed": [{"img_id": "image", "time_rel": "6/1"}]}], windows)
        assert [s.candidate_id for s in sels] == ["c0002"]
        assert dropped  # bad id recorded, dropped item never enters a plan
    finally:
        lm.stop()


def test_cancel_between_requests():
    lm = OfflineLM().start()
    try:
        client, _ = _client(lm)
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(Exception):
            client.observe(_observe_images(), _observe_meta(), cancel_event=cancel)
        assert len(lm.chat_calls) == 0
    finally:
        lm.stop()


@pytest.mark.parametrize("limits, expected_calls", [
    ({"max_model_requests": 1}, 1),
    ({"max_repair_retries": 0}, 1),
    ({"max_request_input_bytes": 1}, 0),
])
def test_repair_and_input_budgets_are_hard_limits(limits, expected_calls):
    from vibecut.analysis.errors import BudgetExhaustedError, ValidationFailureError
    lm = OfflineLM().start()
    try:
        client, metering = _client(lm, **limits)
        lm.chat_script.extend([{"content": observe_payload(["ghost"])}] * 2)
        with pytest.raises((BudgetExhaustedError, ValidationFailureError)):
            client.observe(_observe_images(), _observe_meta())
        assert len(lm.chat_calls) == expected_calls
        assert metering.request_count == expected_calls
        assert metering.retry_count == 0
    finally:
        lm.stop()


def test_total_model_budget_bounds_the_active_request():
    import time
    from vibecut.analysis.errors import BudgetExhaustedError
    lm = OfflineLM().start()
    try:
        client, metering = _client(lm, request_timeout_s=5.0, total_model_budget_s=0.15)
        lm.chat_script.append({"delay": 1.0, "content": observe_payload(["s00.eq+030", "s00.eq-030"])})
        start = time.monotonic()
        with pytest.raises(BudgetExhaustedError):
            client.observe(_observe_images(), _observe_meta())
        assert time.monotonic() - start < 0.8
        assert metering.request_count == 1
    finally:
        lm.stop()



def test_truncated_output_is_metered_without_reasoning_text():
    import json
    lm = OfflineLM().start()
    try:
        client, meter = _client(lm, max_repair_retries=0)
        lm.chat_script.append({"raw": json.dumps({"choices": [{"finish_reason": "length", "message":
            {"content": "", "reasoning_content": "private thinking"}}]})})
        with pytest.raises(StructuredOutputError):
            client.observe(_observe_images(), _observe_meta())
        assert meter.requests[0]["finish_reason"] == "length"
        assert meter.requests[0]["output_token_budget"] == 2048
        assert "private thinking" not in json.dumps(meter.to_json())
    finally:
        lm.stop()



def test_one_observed_image_cannot_be_moved_to_another_time_or_bbox():
    from fractions import Fraction
    from vibecut.analysis.errors import ValidationFailureError
    lm = OfflineLM().start()
    try:
        client, _ = _client(lm, max_repair_retries=0)
        payload = [{"candidate_id": "c1", "observed": [{"img_id": "actual-image", "time_rel": "5/1"}]}]
        base = {"candidate_id": "c1", "img_ids": ["actual-image"], "relevance": 1, "confidence": 1, "reason": "red square"}
        for extra in ({"event_time_s": 9}, {"bbox": {"x": 0, "y": 0, "w": 1, "h": 1}}, {"relevance": 10 ** 400}):
            lm.chat_script.append({"content": {"selections": [{**base, **extra}]}})
            with pytest.raises(ValidationFailureError):
                client.select({}, payload, {"c1": (Fraction(0), Fraction(12))})
        lm.chat_script.append({"content": {"selections": [base]}})
        selections, _, _ = client.select({}, payload, {"c1": (Fraction(0), Fraction(12))})
        assert selections[0].times_rel == (Fraction(5),)
        assert selections[0].img_ids == ("actual-image",)
    finally:
        lm.stop()
