from __future__ import annotations

from dataclasses import replace
import time

from frugalrouter.evaluation.verifier import LocalVerifier
from frugalrouter.model_policy import ModelPolicy
from frugalrouter.prompting import REASONING_CATEGORIES, clean_answer, looks_like_reasoning_spill
from frugalrouter.providers.fireworks import FireworksError, FireworksProvider
from frugalrouter.providers.local import LocalProvider
from frugalrouter.providers.local_model import LocalModelProvider
from frugalrouter.task_classifier import classify_prompt
from frugalrouter.types import RouteResult, Task, Verification


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

FIREWORKS_MODEL_PREFIX = "accounts/fireworks/models/"


def _model_name_variants(model: str) -> list[str]:
    """Both naming forms for a model: as given, and prefix-toggled.

    The public API only accepts full "accounts/fireworks/models/x" paths
    while hackathon material uses short launch names; trying both makes the
    agent robust to whichever convention the harness passes.
    """
    if model.startswith(FIREWORKS_MODEL_PREFIX):
        return [model, model[len(FIREWORKS_MODEL_PREFIX) :]]
    return [model, f"{FIREWORKS_MODEL_PREFIX}{model}"]


class FrugalRouter:
    def __init__(self, config: dict, allow_remote: bool = True) -> None:
        self.config = config
        self.allow_remote = allow_remote
        self.local_confidence_threshold = float(config.get("local_confidence_threshold", 0.92))
        self.remote_max_attempts = int(config.get("remote_max_attempts", 3))
        self.remote_backoff_seconds = float(config.get("remote_backoff_seconds", 2.0))
        self.retry_truncated = bool(config.get("retry_truncated", True))
        self.rescue_max_tokens = int(config.get("rescue_max_tokens", 900))
        self.local = LocalProvider()
        self.local_model = LocalModelProvider(config)
        self.remote = FireworksProvider(config)
        self.model_policy = ModelPolicy(config)
        self.verifier = LocalVerifier()

    def run(self, task: Task) -> RouteResult:
        category = classify_prompt(task.input)
        local_candidate = self.local.answer(task)

        if local_candidate.confidence >= self.local_confidence_threshold:
            return RouteResult(
                task_id=task.id,
                answer=local_candidate.answer,
                verification=Verification(confidence=local_candidate.confidence, reasons=local_candidate.reasons),
                route="local",
                used_remote=False,
                category=category,
            )

        # Zero-token tier 2: a bundled small model answers gated easy
        # categories for free. Skipped entirely (no-op) when no model is
        # present, so the Fireworks-only baseline is unchanged.
        if self.local_model.available_for(category):
            try:
                lm_answer = self.local_model.answer(task, category=category)
            except Exception:  # noqa: BLE001 - never let a local-model bug kill the task
                lm_answer = None
            if lm_answer is not None and lm_answer.text.strip():
                lm_verification = self.verifier.score(task, lm_answer)
                return RouteResult(
                    task_id=task.id,
                    answer=lm_answer,
                    verification=lm_verification,
                    route="local_model",
                    used_remote=False,
                    category=category,
                )

        fallback_reason = (
            f"local_confidence_{local_candidate.confidence:.2f}_below_{self.local_confidence_threshold:.2f}"
        )
        if not self.allow_remote:
            return RouteResult(
                task_id=task.id,
                answer=local_candidate.answer,
                verification=Verification(confidence=local_candidate.confidence, reasons=local_candidate.reasons),
                route="local_remote_disabled",
                used_remote=False,
                fallback_reason=fallback_reason,
                category=category,
            )

        errors: list[str] = []
        remote_answer = None
        for model in self.model_policy.candidates(category, prompt=task.input):
            remote_answer, abort = self._call_with_retry(task, model, category, errors)
            if remote_answer is not None or abort:
                break

        if remote_answer is None:
            return RouteResult(
                task_id=task.id,
                answer=local_candidate.answer,
                verification=Verification(confidence=local_candidate.confidence, reasons=local_candidate.reasons),
                route="remote_error",
                used_remote=False,
                fallback_reason=f"{fallback_reason};remote_error:{' | '.join(errors)}",
                category=category,
            )

        # Keep a final normalization boundary in the router. Providers already
        # clean responses, but this is intentionally idempotent and protects
        # exact semantic answers when a fallback provider returns raw text.
        remote_answer = replace(
            remote_answer,
            text=clean_answer(remote_answer.text, category, prompt=task.input),
        )
        remote_verification = self.verifier.score(task, remote_answer)
        return RouteResult(
            task_id=task.id,
            answer=remote_answer,
            verification=remote_verification,
            route="remote",
            used_remote=True,
            fallback_reason=fallback_reason,
            category=category,
        )

    def _call_with_retry(self, task: Task, model: str, category: str, errors: list[str]):
        """Call one model, trying both name forms on 404.

        Returns (answer, abort). answer is None when this model failed;
        abort=True means the error is not worth trying other models for.
        """
        for variant in _model_name_variants(model):
            answer, abort, not_found = self._call_single_model(task, variant, category, errors)
            if answer is not None:
                return answer, False
            if abort:
                return None, True
            if not not_found:
                # Transient errors exhausted; the alternate name form hits
                # the same backend, so move on to the next model instead.
                return None, False
        return None, False

    def _call_single_model(self, task: Task, model: str, category: str, errors: list[str]):
        """Returns (answer, abort, not_found) for one exact model name."""
        max_tokens_override: int | None = None
        no_reasoning = False
        skip_overrides = False
        for attempt in range(self.remote_max_attempts):
            kwargs = {}
            if max_tokens_override:
                kwargs["max_tokens_override"] = max_tokens_override
            if no_reasoning:
                kwargs["no_reasoning_directive"] = True
            if skip_overrides:
                kwargs["skip_model_overrides"] = True
            try:
                answer = self.remote.answer(task, model=model, category=category, **kwargs)
            except FireworksError as error:
                errors.append(f"{model}:{error}")
                if error.status_code == 404:
                    return None, False, True
                if error.status_code in RETRYABLE_STATUS_CODES:
                    if attempt < self.remote_max_attempts - 1:
                        time.sleep(self.remote_backoff_seconds * (2**attempt))
                    continue
                # A 400 may mean the serving stack rejects our extra request
                # fields (e.g. reasoning_effort); retry once without them.
                if error.status_code == 400 and not skip_overrides:
                    skip_overrides = True
                    errors.append(f"{model}:retrying_without_model_overrides")
                    continue
                return None, True, False
            except Exception as error:  # noqa: BLE001 - any provider bug must not kill the batch
                errors.append(f"{model}:{error}")
                return None, True, False

            # Reasoning-heavy models can spend the whole cap before finishing
            # the real answer, leaving it empty or full of spilled reasoning;
            # a truncated submission is near-certain zero, so retry once with
            # a much larger cap.
            if self.retry_truncated and answer.finish_reason == "length" and max_tokens_override is None:
                max_tokens_override = self._rescue_cap(category)
                errors.append(f"{model}:truncated_at_cap_retrying_with_{max_tokens_override}")
                continue

            # Thinking-mode deliberation leaked into the answer text cannot be
            # parsed apart reliably; ask the model to answer directly instead.
            # Reasoning categories WANT step-by-step working, so skip this there.
            if (
                category not in REASONING_CATEGORIES
                and not no_reasoning
                and looks_like_reasoning_spill(answer.text)
            ):
                no_reasoning = True
                errors.append(f"{model}:reasoning_spill_retrying_with_directive")
                continue
            return answer, False, False
        return None, False, False

    def _rescue_cap(self, category: str) -> int:
        caps = self.config.get("fireworks", {}).get("max_tokens", {})
        if isinstance(caps, dict):
            base = int(caps.get(category, caps.get("general", 220)))
        else:
            base = int(caps)
        return max(self.rescue_max_tokens, base * 2)
