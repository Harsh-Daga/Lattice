"""Tests for cost estimator.

Covers:
- Builtin pricing lookup for all major providers
- Cache-hit savings calculation
- Custom pricing override
- Unknown model fallback
- Aggregation
- Formatting
"""

from __future__ import annotations

import pytest

from lattice.core.cost_estimator import (
    CostEstimate,
    CostEstimator,
    extract_cached_tokens,
    format_cost_usd,
    normalize_usage,
)


@pytest.fixture
def estimator():
    return CostEstimator()


class TestBuiltinPricing:
    def test_openai_gpt4o(self, estimator) -> None:
        est = estimator.estimate_request(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=1_000_000,
            completion_tokens=500_000,
        )
        assert est.prompt_cost_usd == pytest.approx(2.50)
        assert est.completion_cost_usd == pytest.approx(5.00)
        assert est.total_cost_usd == pytest.approx(7.50)

    def test_openai_gpt4o_mini(self, estimator) -> None:
        est = estimator.estimate_request(
            provider="openai",
            model="gpt-4o-mini",
            prompt_tokens=1_000_000,
            completion_tokens=500_000,
        )
        assert est.prompt_cost_usd == pytest.approx(0.15)
        assert est.completion_cost_usd == pytest.approx(0.30)

    def test_anthropic_sonnet(self, estimator) -> None:
        est = estimator.estimate_request(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            prompt_tokens=1_000_000,
            completion_tokens=200_000,
        )
        assert est.prompt_cost_usd == pytest.approx(3.00)
        assert est.completion_cost_usd == pytest.approx(3.00)

    def test_anthropic_haiku(self, estimator) -> None:
        est = estimator.estimate_request(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            prompt_tokens=4_000_000,
            completion_tokens=1_000_000,
        )
        assert est.prompt_cost_usd == pytest.approx(1.00)
        assert est.completion_cost_usd == pytest.approx(1.25)

    def test_deepseek(self, estimator) -> None:
        est = estimator.estimate_request(
            provider="deepseek",
            model="deepseek-chat",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        assert est.prompt_cost_usd == pytest.approx(0.14)
        assert est.completion_cost_usd == pytest.approx(0.28)

    def test_ollama_zero_cost(self, estimator) -> None:
        est = estimator.estimate_request(
            provider="ollama",
            model="llama3.1",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        assert est.total_cost_usd == 0.0


class TestCacheSavings:
    def test_cached_tokens_discount(self, estimator) -> None:
        est = estimator.estimate_request(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=1_000_000,
            completion_tokens=0,
            cached_tokens=800_000,
        )
        # 200k uncached @ 2.50 + 800k cached @ 1.25
        expected = (200_000 / 1e6) * 2.50 + (800_000 / 1e6) * 1.25
        assert est.prompt_cost_usd == pytest.approx(expected)
        # Savings = 800k * (2.50 - 1.25) / 1M
        assert est.cached_savings_usd == pytest.approx(1.00)

    def test_no_cached_pricing_means_full_rate(self, estimator) -> None:
        est = estimator.estimate_request(
            provider="openai",
            model="gpt-4",
            prompt_tokens=1_000_000,
            completion_tokens=0,
            cached_tokens=800_000,
        )
        # gpt-4 has no cached_input_per_1m, so all prompt tokens remain full rate.
        assert est.prompt_cost_usd == pytest.approx(30.00)
        assert est.cached_savings_usd == 0.0

    def test_cached_tokens_are_clamped_to_prompt_tokens(self, estimator) -> None:
        est = estimator.estimate_request(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=100_000,
            completion_tokens=0,
            cached_tokens=200_000,
        )
        assert est.prompt_cost_usd == pytest.approx((100_000 / 1e6) * 1.25)
        assert est.cached_savings_usd == pytest.approx((100_000 / 1e6) * 1.25)


class TestReasoningTokens:
    def test_o1_reasoning(self, estimator) -> None:
        est = estimator.estimate_request(
            provider="openai",
            model="o1",
            prompt_tokens=100_000,
            completion_tokens=50_000,
            reasoning_tokens=20_000,
        )
        assert est.completion_cost_usd == pytest.approx(
            (50_000 / 1e6) * 60.00 + (20_000 / 1e6) * 60.00
        )


class TestUnknownModel:
    def test_unknown_returns_zero(self, estimator) -> None:
        est = estimator.estimate_request(
            provider="openai",
            model="unknown-model-v99",
            prompt_tokens=1_000_000,
            completion_tokens=500_000,
        )
        assert est.total_cost_usd == 0.0
        assert est.pricing_source == "unknown"


class TestActualUsage:
    def test_normalize_usage_openai_shape(self) -> None:
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 30},
            "completion_tokens_details": {"reasoning_tokens": 20},
        }
        assert normalize_usage(usage) == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cached_tokens": 30,
            "reasoning_tokens": 20,
        }

    def test_normalize_usage_bedrock_shape(self) -> None:
        usage = {
            "InputTokens": 100,
            "OutputTokens": 50,
            "CacheReadInputTokens": 25,
        }
        assert normalize_usage(usage) == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cached_tokens": 25,
            "reasoning_tokens": 0,
        }

    def test_normalize_usage_wrapped_gemini_shape(self) -> None:
        usage = {
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "cachedContentTokenCount": 25,
            }
        }
        assert normalize_usage(usage) == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cached_tokens": 25,
            "reasoning_tokens": 0,
        }

    def test_extract_cached_tokens_uses_same_normalization_surface(self) -> None:
        usage = {"usage_metadata": {"cached_content_token_count": 25}}
        assert extract_cached_tokens(usage) == normalize_usage(usage)["cached_tokens"]

    def test_extract_cached_tokens_invalid_usage(self) -> None:
        assert extract_cached_tokens(None) == 0
        assert extract_cached_tokens("bad") == 0

    def test_compute_actual(self, estimator) -> None:
        usage = {
            "prompt_tokens": 2_000_000,
            "completion_tokens": 500_000,
            "cached_tokens": 1_500_000,
        }
        est = estimator.compute_actual(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            usage=usage,
        )
        # 500k uncached @ 3.00 + 1.5M cached @ 0.30
        expected_prompt = (500_000 / 1e6) * 3.00 + (1_500_000 / 1e6) * 0.30
        assert est.prompt_cost_usd == pytest.approx(expected_prompt)

    def test_compute_actual_anthropic_usage_shape(self, estimator) -> None:
        usage = {
            "input_tokens": 2_000_000,
            "output_tokens": 500_000,
            "cache_read_input_tokens": 1_500_000,
        }
        est = estimator.compute_actual(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            usage=usage,
        )
        expected_prompt = (500_000 / 1e6) * 3.00 + (1_500_000 / 1e6) * 0.30
        assert est.prompt_cost_usd == pytest.approx(expected_prompt)
        assert est.completion_cost_usd == pytest.approx((500_000 / 1e6) * 15.00)

    def test_compute_actual_bedrock_claude_usage_shape(self, estimator) -> None:
        usage = {
            "inputTokens": 2_000_000,
            "outputTokens": 500_000,
            "cacheReadInputTokens": 1_500_000,
        }
        est = estimator.compute_actual(
            provider="bedrock",
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            usage=usage,
        )
        expected_prompt = (500_000 / 1e6) * 3.00 + (1_500_000 / 1e6) * 0.30
        assert est.prompt_cost_usd == pytest.approx(expected_prompt)
        assert est.completion_cost_usd == pytest.approx((500_000 / 1e6) * 15.00)
        assert est.pricing_source == "builtin"

    def test_compute_actual_gemini_usage_metadata(self, estimator) -> None:
        usage = {
            "usageMetadata": {
                "promptTokenCount": 1_000_000,
                "candidatesTokenCount": 500_000,
                "cachedContentTokenCount": 250_000,
            }
        }
        est = estimator.compute_actual(
            provider="gemini",
            model="gemini/gemini-1.5-flash",
            usage=usage,
        )
        assert est.prompt_cost_usd == pytest.approx((1_000_000 / 1e6) * 0.35)
        assert est.completion_cost_usd == pytest.approx((500_000 / 1e6) * 1.05)

    def test_compute_actual_openai_reasoning_tokens(self, estimator) -> None:
        usage = {
            "prompt_tokens": 100_000,
            "completion_tokens": 50_000,
            "completion_tokens_details": {"reasoning_tokens": 20_000},
        }
        est = estimator.compute_actual(
            provider="openai",
            model="o1",
            usage=usage,
        )
        assert est.completion_cost_usd == pytest.approx(
            (50_000 / 1e6) * 60.00 + (20_000 / 1e6) * 60.00
        )


class TestCustomPricing:
    def test_override_builtin(self) -> None:
        est = CostEstimator()
        est.add_custom_pricing(
            provider="openai",
            model="gpt-4o",
            input_per_1m=1.00,
            output_per_1m=2.00,
        )
        result = est.estimate_request(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=1_000_000,
            completion_tokens=500_000,
        )
        assert result.prompt_cost_usd == pytest.approx(1.00)
        assert result.completion_cost_usd == pytest.approx(1.00)

    def test_new_provider(self) -> None:
        est = CostEstimator()
        est.add_custom_pricing(
            provider="custom",
            model="custom-large",
            input_per_1m=5.00,
            output_per_1m=10.00,
        )
        result = est.estimate_request(
            provider="custom",
            model="custom-large",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        assert result.total_cost_usd == pytest.approx(15.00)


class TestAggregation:
    def test_aggregate_multiple(self) -> None:
        estimates = [
            CostEstimate(
                prompt_cost_usd=1.0,
                completion_cost_usd=2.0,
                cached_savings_usd=0.5,
                total_cost_usd=3.0,
            ),
            CostEstimate(
                prompt_cost_usd=3.0,
                completion_cost_usd=4.0,
                cached_savings_usd=1.0,
                total_cost_usd=7.0,
            ),
        ]
        agg = CostEstimator.aggregate(estimates)
        assert agg.prompt_cost_usd == 4.0
        assert agg.completion_cost_usd == 6.0
        assert agg.cached_savings_usd == 1.5
        assert agg.total_cost_usd == 10.0


class TestFormatting:
    def test_micro(self) -> None:
        assert "µ" in format_cost_usd(0.000_050)

    def test_cents(self) -> None:
        assert "¢" in format_cost_usd(0.50)

    def test_dollars(self) -> None:
        assert format_cost_usd(12.34).startswith("$")

    def test_extremely_small(self) -> None:
        assert "e" in format_cost_usd(0.000_000_000_1)


class TestProviderPrefixStrip:
    def test_openai_prefix(self, estimator) -> None:
        est = estimator.estimate_request(
            provider="openai",
            model="openai/gpt-4o",
            prompt_tokens=1_000_000,
            completion_tokens=0,
        )
        assert est.prompt_cost_usd == pytest.approx(2.50)

    def test_anthropic_prefix(self, estimator) -> None:
        est = estimator.estimate_request(
            provider="anthropic",
            model="anthropic/claude-3-5-sonnet-20241022",
            prompt_tokens=1_000_000,
            completion_tokens=0,
        )
        assert est.prompt_cost_usd == pytest.approx(3.00)

    def test_azure_uses_openai_pricing(self, estimator) -> None:
        est = estimator.estimate_request(
            provider="azure",
            model="azure/gpt-4o",
            prompt_tokens=1_000_000,
            completion_tokens=0,
        )
        assert est.prompt_cost_usd == pytest.approx(2.50)

    def test_vertex_uses_google_pricing(self, estimator) -> None:
        est = estimator.estimate_request(
            provider="vertex",
            model="vertex/gemini-1.5-flash",
            prompt_tokens=1_000_000,
            completion_tokens=0,
        )
        assert est.prompt_cost_usd == pytest.approx(0.35)

    def test_bedrock_claude_model_id_uses_anthropic_pricing(self, estimator) -> None:
        est = estimator.estimate_request(
            provider="bedrock",
            model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            prompt_tokens=1_000_000,
            completion_tokens=0,
        )
        assert est.prompt_cost_usd == pytest.approx(0.25)

    def test_bedrock_cohere_model_id_uses_cohere_pricing(self, estimator) -> None:
        est = estimator.estimate_request(
            provider="bedrock",
            model="bedrock/cohere.command-r-plus-v1:0",
            prompt_tokens=1_000_000,
            completion_tokens=0,
        )
        assert est.prompt_cost_usd == pytest.approx(3.00)


class TestPricingInfo:
    def test_get_pricing_info(self, estimator) -> None:
        info = estimator.get_pricing_info("openai", "gpt-4o")
        assert info is not None
        assert info["input_per_1m"] == 2.50
        assert info["cached_input_per_1m"] == 1.25

    def test_get_pricing_info_unknown(self, estimator) -> None:
        assert estimator.get_pricing_info("unknown", "model") is None
