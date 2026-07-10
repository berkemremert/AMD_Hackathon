from frugalrouter.providers.fireworks import FireworksProvider


def test_model_overrides_can_set_category_specific_reasoning_budget():
    provider = FireworksProvider(
        {
            "model_overrides": {
                "gemma": {
                    "reasoning_effort": {
                        "default": "none",
                        "math": 64,
                        "logic": 64,
                    }
                }
            }
        }
    )

    assert provider._request_overrides("gemma-4-31b-it", "math") == {"reasoning_effort": 64}
    assert provider._request_overrides("gemma-4-31b-it", "factual") == {"reasoning_effort": "none"}
    assert provider._request_overrides("kimi-k2p7-code", "math") == {}
