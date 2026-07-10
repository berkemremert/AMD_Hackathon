import json

from frugalrouter.config import load_config
from frugalrouter.io import read_tasks, write_results
from frugalrouter.model_policy import ModelPolicy
from frugalrouter.providers.fireworks import FireworksError
from frugalrouter.router import FrugalRouter
from frugalrouter.types import Answer, Task


def test_router_uses_local_for_direct_arithmetic():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(id="test", input="Calculate 42 * 17. Return only the number.")

    result = router.run(task)

    assert result.answer.text == "714"
    assert result.used_remote is False
    assert result.route == "local"
    assert result.category == "math"


def test_router_uses_local_for_direct_percentage():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(id="test", input="What is 25 percent of 200? Return only the number.")

    result = router.run(task)

    assert result.answer.text == "50"
    assert result.used_remote is False
    assert result.route == "local"


def test_router_computes_us_state_missing_letter_fact():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(
        id="test",
        input="What is the only letter of the English alphabet that does not appear in the name of any U.S. state?",
    )

    result = router.run(task)

    assert result.answer.text == "Q"
    assert result.used_remote is False
    assert result.route == "local"


def test_router_computes_chained_power_strip_empty_outlets():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(
        id="test",
        input=(
            "I have three identical power strips. Each strip has exactly 4 outlets. "
            "I plug the first power strip into the single wall outlet. I then plug the second "
            "strip into the first strip, and the third strip into the second strip. "
            "How many total empty outlets remain available?"
        ),
    )

    result = router.run(task)

    assert result.answer.text == "10"
    assert result.used_remote is False
    assert result.route == "local"


def test_router_computes_time_word_problem():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(
        id="test",
        input=(
            "I started baking a roast at 2:15 PM. It requires 45 minutes of preparation time "
            "before going into the oven, and then it needs to bake for 1 hour and 20 minutes. "
            "What time is it completely done? Use HH:MM PM format."
        ),
    )

    result = router.run(task)

    assert result.answer.text == "04:20 PM"
    assert result.used_remote is False
    assert result.route == "local"


def test_router_computes_weekday_from_reference_date():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(
        id="test",
        input=(
            "Assume February 29th, 2024 is a Thursday. "
            "What day of the week will February 29th, 2028 be? Output only the day."
        ),
    )

    result = router.run(task)

    assert result.answer.text == "Tuesday"
    assert result.used_remote is False
    assert result.route == "local"


def test_router_does_not_guess_vague_power_strip_prompt():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(id="test", input="Explain what a power strip is and how many outlets it usually has.")

    result = router.run(task)

    assert result.answer.text == ""
    assert result.route == "local_remote_disabled"


def test_router_uses_local_for_exact_response_instruction():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(id="test", input="Reply with exactly: OK")

    result = router.run(task)

    assert result.answer.text == "OK"
    assert result.used_remote is False
    assert result.route == "local"


def test_router_does_not_shortcut_negated_sentiment():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(id="test", input="Classify the sentiment as positive, neutral, or negative: The docs are not good.")

    result = router.run(task)

    assert result.route == "local_remote_disabled"
    assert result.used_remote is False


def test_router_does_not_shortcut_exact_response_with_alternatives():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(id="test", input="Answer with exactly 'yes' or 'no': Is the Atlantic larger than the Pacific?")

    result = router.run(task)

    assert result.route == "local_remote_disabled"
    assert result.answer.text == ""


def test_router_does_not_use_exact_response_for_length_constraint():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(id="test", input="Answer with exactly three words: Explain ROCm.")

    result = router.run(task)

    assert result.answer.text == ""
    assert result.route == "local_remote_disabled"


def test_router_uses_remote_for_general_tasks_without_local_shortcut():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=True)

    class FakeProvider:
        def answer(self, task, model=None, category="general"):
            return Answer(text=f"{category}:{model}:ok", provider="fireworks", model=model or "fake")

    router.remote = FakeProvider()
    task = Task(id="test", input="Explain in one sentence what ROCm is.")

    result = router.run(task)

    assert result.route == "remote"
    assert result.used_remote is True
    assert result.category == "factual"
    assert "gemma" in result.answer.text


def test_router_falls_back_to_empty_local_when_remote_fails():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=True)

    class FailingProvider:
        def answer(self, task, model=None, category="general"):
            raise RuntimeError("simulated remote outage")

    router.remote = FailingProvider()
    task = Task(id="test", input="Write a function that computes a rolling median from a stream.")

    result = router.run(task)

    assert result.route == "remote_error"
    assert result.used_remote is False
    assert "simulated remote outage" in result.fallback_reason


def test_router_tries_next_model_after_404():
    config = load_config("config/models.json")
    config["allowed_models"] = ["missing-model", "accounts/fireworks/models/minimax-m3"]
    config["model_policy"]["factual"] = ["missing-model", "minimax-m3"]
    router = FrugalRouter(config=config, allow_remote=True)

    class FlakyProvider:
        def answer(self, task, model=None, category="general"):
            if "missing-model" in model:
                raise FireworksError("not found", status_code=404)
            return Answer(text=f"{model}:ok", provider="fireworks", model=model or "fake")

    router.remote = FlakyProvider()

    result = router.run(Task(id="test", input="Explain what ROCm is."))

    assert result.route == "remote"
    assert result.used_remote is True
    assert result.answer.model == "accounts/fireworks/models/minimax-m3"


def test_router_retries_404_with_prefixed_model_name():
    config = load_config("config/models.json")
    config["allowed_models"] = ["kimi-k2p7-code"]
    config["model_policy"]["factual"] = ["kimi-k2p7-code"]
    router = FrugalRouter(config=config, allow_remote=True)

    class PrefixOnlyProvider:
        def answer(self, task, model=None, category="general"):
            if not model.startswith("accounts/fireworks/models/"):
                raise FireworksError("not found", status_code=404)
            return Answer(text="ok", provider="fireworks", model=model)

    router.remote = PrefixOnlyProvider()

    result = router.run(Task(id="test", input="Explain what ROCm is."))

    assert result.route == "remote"
    assert result.answer.model == "accounts/fireworks/models/kimi-k2p7-code"


def test_router_retries_rate_limited_model_with_backoff():
    config = load_config("config/models.json")
    config["remote_backoff_seconds"] = 0.001
    router = FrugalRouter(config=config, allow_remote=True)

    calls = {"count": 0}

    class RateLimitedOnceProvider:
        def answer(self, task, model=None, category="general"):
            calls["count"] += 1
            if calls["count"] == 1:
                raise FireworksError("rate limited", status_code=429)
            return Answer(text="recovered", provider="fireworks", model=model or "fake")

    router.remote = RateLimitedOnceProvider()

    result = router.run(Task(id="test", input="Explain what ROCm is."))

    assert result.route == "remote"
    assert result.answer.text == "recovered"
    assert calls["count"] == 2


def test_router_moves_to_next_model_after_exhausting_retries():
    config = load_config("config/models.json")
    config["remote_backoff_seconds"] = 0.001
    config["remote_max_attempts"] = 2
    config["allowed_models"] = ["always-limited", "healthy-model"]
    config["model_policy"]["factual"] = ["always-limited", "healthy-model"]
    router = FrugalRouter(config=config, allow_remote=True)

    class SplitProvider:
        def answer(self, task, model=None, category="general"):
            if model == "always-limited":
                raise FireworksError("rate limited", status_code=429)
            return Answer(text="ok", provider="fireworks", model=model)

    router.remote = SplitProvider()

    result = router.run(Task(id="test", input="Explain what ROCm is."))

    assert result.route == "remote"
    assert result.answer.model == "healthy-model"


def test_router_rescues_truncated_nonempty_answer():
    config = load_config("config/models.json")
    config["remote_backoff_seconds"] = 0.001
    config["retry_truncated"] = True
    router = FrugalRouter(config=config, allow_remote=True)

    class SpilledReasoningProvider:
        def answer(self, task, model=None, category="general", max_tokens_override=None):
            if max_tokens_override is None:
                return Answer(
                    text="We need answer user. They want",
                    provider="fireworks",
                    model=model or "fake",
                    finish_reason="length",
                )
            return Answer(text="def fixed(): pass", provider="fireworks", model=model or "fake", finish_reason="stop")

    router.remote = SpilledReasoningProvider()

    result = router.run(Task(id="test", input="Debug this code:\n\ndef broken(): pass"))

    assert result.route == "remote"
    assert result.answer.text == "def fixed(): pass"


def test_router_retries_reasoning_spill_with_directive():
    config = load_config("config/models.json")
    config["remote_backoff_seconds"] = 0.001
    router = FrugalRouter(config=config, allow_remote=True)

    class ThinkingProvider:
        def answer(self, task, model=None, category="general", no_reasoning_directive=False):
            if not no_reasoning_directive:
                return Answer(
                    text="thought\n*   The user wants the capital.\n\nCanberra.Canberra",
                    provider="fireworks",
                    model=model or "fake",
                    finish_reason="stop",
                )
            return Answer(text="Canberra", provider="fireworks", model=model or "fake", finish_reason="stop")

    router.remote = ThinkingProvider()

    result = router.run(Task(id="test", input="What is the capital of Australia? Answer with only the city name."))

    assert result.route == "remote"
    assert result.answer.text == "Canberra"


def test_router_retries_400_without_model_overrides():
    config = load_config("config/models.json")
    config["remote_backoff_seconds"] = 0.001
    router = FrugalRouter(config=config, allow_remote=True)

    class RejectsExtrasProvider:
        def answer(self, task, model=None, category="general", skip_model_overrides=False):
            if not skip_model_overrides:
                raise FireworksError("unknown field reasoning_effort", status_code=400)
            return Answer(text="ok", provider="fireworks", model=model or "fake", finish_reason="stop")

    router.remote = RejectsExtrasProvider()

    result = router.run(Task(id="test", input="Explain what ROCm is."))

    assert result.route == "remote"
    assert result.answer.text == "ok"


def test_router_rescues_empty_answer_at_token_cap():
    config = load_config("config/models.json")
    config["remote_backoff_seconds"] = 0.001
    config["retry_truncated"] = True
    router = FrugalRouter(config=config, allow_remote=True)

    seen_overrides = []

    class EmptyAtCapProvider:
        def answer(self, task, model=None, category="general", max_tokens_override=None):
            seen_overrides.append(max_tokens_override)
            if max_tokens_override is None:
                return Answer(text="", provider="fireworks", model=model or "fake", finish_reason="length")
            return Answer(text="rescued", provider="fireworks", model=model or "fake", finish_reason="stop")

    router.remote = EmptyAtCapProvider()

    result = router.run(Task(id="test", input="Explain what ROCm is."))

    assert result.route == "remote"
    assert result.answer.text == "rescued"
    assert seen_overrides == [None, 900]


def test_router_does_not_retry_truncation_in_compact_mode():
    config = load_config("config/models.json")
    config["retry_truncated"] = False
    router = FrugalRouter(config=config, allow_remote=True)
    seen_overrides = []

    class CappedProvider:
        def answer(self, task, model=None, category="general", max_tokens_override=None):
            seen_overrides.append(max_tokens_override)
            return Answer(text="42", provider="fireworks", model=model or "fake", finish_reason="length")

    router.remote = CappedProvider()
    result = router.run(Task(id="test", input="What is six times seven?"))

    assert result.answer.text == "42"
    assert seen_overrides == [None]


def test_router_applies_final_semantic_normalization():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=True)

    class RawFragmentProvider:
        def answer(self, task, model=None, category="general", **kwargs):
            return Answer(
                text="useEffect(() => { fetchUser(); }, []);",
                provider="fireworks",
                model=model or "fake",
            )

    router.remote = RawFragmentProvider()
    task = Task(
        id="test",
        input="Debug this React effect. Which exact two characters are missing from its dependency list?",
    )
    result = router.run(task)

    assert result.answer.text == "[]"


def test_router_solves_missing_loop_update_locally():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=False)
    task = Task(
        id="test",
        input=(
            "This JavaScript code causes the browser to freeze: "
            "`let i = 0; while(i < 10) { console.log(i); }`. "
            "What operation is missing inside the loop body?"
        ),
    )

    result = router.run(task)

    assert result.route == "local"
    assert result.answer.text == "i++"


def test_router_uses_local_model_for_gated_category():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=True)

    class FakeLocalModel:
        def available_for(self, category):
            return category == "factual"

        def answer(self, task, category="general"):
            return Answer(text="Canberra", provider="local_model", model="fake-2b")

    class ExplodingRemote:
        def answer(self, *a, **k):
            raise AssertionError("remote should not be called when local model answers")

    router.local_model = FakeLocalModel()
    router.remote = ExplodingRemote()

    result = router.run(Task(id="t", input="What is the capital of Australia?"))

    assert result.route == "local_model"
    assert result.used_remote is False
    assert result.answer.text == "Canberra"


def test_router_falls_back_to_remote_when_local_model_empty():
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=True)

    class EmptyLocalModel:
        def available_for(self, category):
            return True

        def answer(self, task, category="general"):
            return Answer(text="", provider="local_model", model="fake-2b")

    class FakeRemote:
        def answer(self, task, model=None, category="general", **kwargs):
            return Answer(text="remote-answer", provider="fireworks", model=model or "fake")

    router.local_model = EmptyLocalModel()
    router.remote = FakeRemote()

    result = router.run(Task(id="t", input="Explain what ROCm is."))

    assert result.route == "remote"
    assert result.used_remote is True
    assert result.answer.text == "remote-answer"


def test_local_model_inert_without_weights_matches_baseline():
    # Default config has a local_model section, but no weights exist locally,
    # so available_for must be False and routing must be pure Fireworks.
    config = load_config("config/models.json")
    router = FrugalRouter(config=config, allow_remote=True)

    assert router.local_model.available_for("factual") is False


def test_model_policy_uses_allowed_models_from_harness():
    config = load_config("config/models.json")
    config["include_all_allowed_fallbacks"] = True
    config["allowed_models"] = ["accounts/fireworks/models/kimi-k2p7-code", "accounts/fireworks/models/minimax-m3"]
    config["model_policy"]["factual"] = ["minimax-m3"]
    policy = ModelPolicy(config)

    assert policy.choose("code_generation") == "accounts/fireworks/models/kimi-k2p7-code"
    assert policy.choose("factual") == "accounts/fireworks/models/minimax-m3"


def test_model_policy_logic_routing_specialist_only_for_syllogisms():
    # V13: most logic goes to the primary with brief written reasoning (the
    # broad Kimi one-shot specialist produced 5 of the 9 blind failures), but
    # premise-based syllogisms stay on Kimi, which is measured to follow the
    # UNKNOWN-when-undetermined convention.
    config = load_config("config/models.json")
    config["allowed_models"] = [
        "accounts/fireworks/models/gemma-4-31b-it",
        "accounts/fireworks/models/kimi-k2p7-code",
    ]
    policy = ModelPolicy(config)

    assert policy.choose("logic", "Premise 1: All A are B. Are all A also C?").endswith("kimi-k2p7-code")
    assert policy.choose("logic", "Every box is incorrectly labeled. Which is Apples?").endswith("gemma-4-31b-it")
    assert policy.choose("logic", "The day before yesterday was Tuesday. What day is tomorrow?").endswith(
        "gemma-4-31b-it"
    )
    assert policy.choose("logic", "Server A writes false logs on odd days.").endswith("gemma-4-31b-it")


def test_model_policy_keeps_gemma_for_ordering_logic():
    config = load_config("config/models.json")
    config["allowed_models"] = [
        "accounts/fireworks/models/gemma-4-31b-it",
        "accounts/fireworks/models/kimi-k2p7-code",
    ]
    policy = ModelPolicy(config)

    prompt = "A finished before B and after C. Who came second?"
    assert policy.choose("logic", prompt).endswith("gemma-4-31b-it")


def test_v12_policy_uses_only_the_two_measured_models():
    config = load_config("config/models.json")
    config["allowed_models"] = [
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/gemma-4-31b-it",
        "accounts/fireworks/models/gemma-4-26b-a4b-it",
        "accounts/fireworks/models/gemma-4-31b-it-nvfp4",
    ]
    policy = ModelPolicy(config)

    assert policy.candidates("factual") == [
        "accounts/fireworks/models/gemma-4-31b-it",
        "accounts/fireworks/models/kimi-k2p7-code",
    ]
    assert policy.candidates("code_generation") == [
        "accounts/fireworks/models/kimi-k2p7-code",
        "accounts/fireworks/models/gemma-4-31b-it",
    ]


def test_v13_remote_output_caps_stay_lean_outside_reasoning():
    # V13: only math/logic get headroom for brief written reasoning; every
    # other category keeps V12's lean caps.
    config = load_config("config/models.json")
    caps = config["fireworks"]["max_tokens"]

    assert max(caps[cat] for cat in ("math", "logic")) <= 320
    assert max(v for k, v in caps.items() if k not in {"math", "logic"}) <= 128


def test_track1_json_io_contract(tmp_path):
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "results.json"
    input_path.write_text(
        json.dumps([{"task_id": "t1", "prompt": "Calculate 2 + 2. Return only the number."}]),
        encoding="utf-8",
    )

    tasks = read_tasks(input_path)
    write_results(output_path, [{"task_id": tasks[0].id, "answer": "4", "route": "local"}])

    assert tasks[0].id == "t1"
    assert tasks[0].input.startswith("Calculate")
    assert json.loads(output_path.read_text(encoding="utf-8")) == [{"task_id": "t1", "answer": "4"}]
