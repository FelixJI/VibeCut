"""Stage-specific diagnostics for the local-analysis module (task #5).

Every failure carries ``(stage, code)`` so callers can render concrete
diagnostics (AC5): model not started / not loaded / no vision capability /
structured-output failure / timeout / OOM / cancellation, at the exact stage
they occurred. None of these errors ever carries model free text that could
be mistaken for executable content.
"""

from __future__ import annotations


class AnalysisError(RuntimeError):
    """Base class for all analysis failures with a stage/code diagnostic."""

    stage = "analysis"
    code = "error"

    def __init__(self, message: str, *, stage: str | None = None, code: str | None = None) -> None:
        super().__init__(message)
        if stage is not None:
            self.stage = stage
        if code is not None:
            self.code = code

    @property
    def diagnostic(self) -> dict:
        return {"stage": self.stage, "code": self.code, "message": str(self)}


class EndpointError(AnalysisError):
    """Endpoint is not an explicit loopback http address (or redirected)."""

    stage = "endpoint"


class AIUnavailableError(AnalysisError):
    """Loopback model service not started / model not found / not loaded."""

    stage = "model-probe"
    code = "not-started"


class NoVisionError(AnalysisError):
    stage = "model-probe"
    code = "no-vision"


class ModelRequestError(AnalysisError):
    stage = "model-request"


class ModelTimeoutError(ModelRequestError):
    code = "timeout"


class ModelCancelledError(AnalysisError):
    stage = "model-request"
    code = "cancelled"


class ModelAPIError(ModelRequestError):
    code = "api-error"


class ModelOOMError(ModelAPIError):
    code = "oom"


class StructuredOutputError(ModelRequestError):
    code = "structured-output"


class ValidationFailureError(ModelRequestError):
    """Application-layer business validation failed (after bounded repair)."""

    code = "validation"


class BudgetExhaustedError(AnalysisError):
    stage = "budget"
    code = "budget-exceeded"


class ExtractionError(AnalysisError):
    stage = "extraction"


class SourceUnavailableError(AnalysisError):
    stage = "source"
