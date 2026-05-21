# LATTICE Production Evals

- Runner: `production_evals`
- Total sections: `11`
- Passed sections: `11`

## feature_eval

- Kind: `local`
- Status: `ok`

| Metric | Value |
| --- | ---: |
| scenario_count | 21 |
| iterations | 1 |
| avg_reduction_ratio | 0.2868 |
| avg_pipeline_latency_ms | 100.3722 |
| avg_quality_score | 0.8844 |
| proof_passed | 16 |
| proof_failed | 5 |
| feature_failed | 5 |
| tier_failed | 0 |
| budget_failed | 0 |

- Benchmark provider: `ollama-cloud`
- Benchmark model: `ollama-cloud/kimi-k2.6:cloud`
- Scenarios: `21`
- Quality passed: `15`
- Total token savings: `8722`
- Avg reduction ratio: `0.2868`
- Avg pipeline latency: `100.3722 ms`
- Avg quality score: `0.8844`

| Scenario | Category | Saved | Ratio | Task Eq. |
| --- | --- | ---: | ---: | ---: |

| Proof Metric | Value |
| --- | ---: |
| proof_passed | 16 |
| proof_failed | 5 |
| uuid_deduplication | reference_substitution | 0 | 0.0000 | 0.9016 |
| tool_output_filtering | tool_filter | 4164 | 0.9438 | 0.7467 |
| table_compression | format_conversion | 1579 | 0.9552 | 0.7806 |
| prefix_optimization | prefix_opt | 0 | 0.0000 | 0.9583 |
| code_review_patterns | reference_sub | 0 | 0.0000 | 0.9875 |
| api_docs_summarization | reference_optimizer | 0 | 0.0000 | 0.9922 |
| tool_call_preservation | integration | 0 | 0.0000 | 0.8347 |
| json_response_format | format_conversion | 0 | 0.0000 | 0.9844 |
| cache_arbitrage_prefix | cache_arbitrage | 0 | 0.0000 | 1.0000 |
| dictionary_repetition | reference_optimizer | 61 | 0.4803 | 0.8150 |
| cleanup_noise | normalization | 0 | 0.0000 | 0.9250 |
| message_dedup_turns | message_dedup | 0 | 0.0000 | 0.9167 |
| rate_distortion_longform | rate_distortion | 511 | 0.9569 | 0.9688 |
| context_budget_pressure | context_selector | 669 | 0.7917 | 0.9808 |
| runtime_contract_pressure | runtime_contract | 0 | 0.0000 | 0.8542 |
| grammar_json_table | structure_optimizer | 925 | 0.9259 | 0.6873 |
| simple_baseline | baseline | 0 | 0.0000 | 0.8936 |
| mixed_realworld | mixed | 813 | 0.9679 | 0.6781 |
| debugging_log_analysis | debugging | 0 | 0.0000 | 1.0000 |
| reasoning_root_cause | reasoning | 0 | 0.0000 | 0.9875 |
| repeated_identifier_investigation | reference_substitution | 0 | 0.0000 | 0.9833 |

### Coverage Summary

| Metric | Value |
| --- | ---: |
| scenario_count | 21 |
| proof_passed | 16 |
| proof_failed | 5 |
| feature_failed | 5 |
| tier_failed | 0 |
| budget_failed | 0 |
| tier::simple | 8 |
| tier::medium | 9 |
| tier::complex | 1 |
| tier::reasoning | 3 |
| flaw::blocked by runtime contract: rate_distortion | 1 |
| flaw::features not reached by pipeline: context_selector | 1 |
| flaw::features not reached by pipeline: format_conversion | 2 |
| flaw::features not reached by pipeline: message_dedup | 1 |
| feature::cache_arbitrage | 1 |
| feature::context_selector | 0 |
| feature::format_conversion | 3 |
| feature::message_dedup | 0 |
| feature::prefix_opt | 1 |
| feature::rate_distortion | 2 |
| feature::reference_sub | 4 |
| feature::runtime_contract | 3 |
| feature::strategy_selector | 1 |
| feature::tool_filter | 3 |

### Transform Summary

| Transform | Avg Before | Avg After | Avg Saved | Count |
| --- | ---: | ---: | ---: | ---: |
| CandidateSearch | 731.25 | 358.6 | 372.65 | 20 |
| cache_arbitrage | 863.71 | 508.81 | 354.9 | 21 |
| content_profiler | 863.71 | 508.81 | 354.9 | 21 |
| format_conversion | 1593.75 | 234.5 | 1359.25 | 4 |
| message_dedup | 992.0 | 992.0 | 0.0 | 1 |
| pipeline_v2 | 863.71 | 508.81 | 354.9 | 21 |
| prefix_optimizer | 863.71 | 508.81 | 354.9 | 21 |
| rate_distortion | 1437.43 | 569.57 | 867.86 | 7 |
| reference_optimizer | 802.0 | 164.0 | 638.0 | 1 |
| reference_sub | 731.25 | 358.6 | 372.65 | 20 |
| runtime_contract | 863.71 | 508.81 | 354.9 | 21 |
| strategy_selector | 731.25 | 358.6 | 372.65 | 20 |
| structure_optimizer | 1272.0 | 57.0 | 1215.0 | 1 |
| tool_projection | 1261.33 | 502.67 | 758.67 | 9 |

### Scenario Proof

| Scenario | Complexity | Expected Tier | Observed Tier | Features | Status | Flaws |
| --- | --- | --- | --- | --- | --- | --- |
| uuid_deduplication | medium | SIMPLE | SIMPLE | reference_sub | pass | - |
| tool_output_filtering | complex | COMPLEX | COMPLEX | tool_filter | pass | - |
| table_compression | complex | MEDIUM | MEDIUM | format_conversion | pass | - |
| prefix_optimization | medium | MEDIUM | MEDIUM | prefix_opt | pass | - |
| code_review_patterns | complex | MEDIUM | MEDIUM | reference_sub | pass | - |
| api_docs_summarization | medium | MEDIUM | MEDIUM | rate_distortion | pass | - |
| tool_call_preservation | complex | SIMPLE | SIMPLE | tool_filter | pass | - |
| json_response_format | medium | MEDIUM | MEDIUM | format_conversion | pass | - |
| cache_arbitrage_prefix | medium | SIMPLE | SIMPLE | cache_arbitrage | pass | - |
| dictionary_repetition | medium | SIMPLE | SIMPLE | rate_distortion | fail | blocked by runtime contract: rate_distortion |
| cleanup_noise | simple | SIMPLE | SIMPLE | format_conversion | fail | features not reached by pipeline: format_conversion |
| message_dedup_turns | medium | SIMPLE | SIMPLE | message_dedup | fail | features not reached by pipeline: message_dedup |
| rate_distortion_longform | complex | MEDIUM | MEDIUM | rate_distortion | pass | - |
| context_budget_pressure | complex | MEDIUM | MEDIUM | context_selector, strategy_selector | fail | features not reached by pipeline: context_selector |
| runtime_contract_pressure | reasoning | REASONING | REASONING | runtime_contract | pass | - |
| grammar_json_table | complex | MEDIUM | MEDIUM | format_conversion | pass | - |
| simple_baseline | simple | SIMPLE | SIMPLE | - | pass | - |
| mixed_realworld | reasoning | SIMPLE | SIMPLE | reference_sub, format_conversion, tool_filter | fail | features not reached by pipeline: format_conversion |
| debugging_log_analysis | reasoning | REASONING | REASONING | runtime_contract | pass | - |
| reasoning_root_cause | reasoning | REASONING | REASONING | runtime_contract | pass | - |
| repeated_identifier_investigation | medium | MEDIUM | MEDIUM | reference_sub | pass | - |

### Usage

| Scenario | Baseline Prompt Tokens | Optimized Prompt Tokens | Baseline Cost USD | Optimized Cost USD | Delta USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| uuid_deduplication | 728 | 728 | 0.000000 | 0.000000 | 0.000000 |
| tool_output_filtering | 4412 | 248 | 0.000000 | 0.000000 | 0.000000 |
| table_compression | 1653 | 74 | 0.000000 | 0.000000 | 0.000000 |
| prefix_optimization | 202 | 202 | 0.000000 | 0.000000 | 0.000000 |
| code_review_patterns | 1037 | 1037 | 0.000000 | 0.000000 | 0.000000 |
| api_docs_summarization | 2183 | 2183 | 0.000000 | 0.000000 | 0.000000 |
| tool_call_preservation | 167 | 167 | 0.000000 | 0.000000 | 0.000000 |
| json_response_format | 1034 | 1034 | 0.000000 | 0.000000 | 0.000000 |
| cache_arbitrage_prefix | 213 | 213 | 0.000000 | 0.000000 | 0.000000 |
| dictionary_repetition | 127 | 66 | 0.000000 | 0.000000 | 0.000000 |
| cleanup_noise | 56 | 56 | 0.000000 | 0.000000 | 0.000000 |
| message_dedup_turns | 73 | 73 | 0.000000 | 0.000000 | 0.000000 |
| rate_distortion_longform | 534 | 23 | 0.000000 | 0.000000 | 0.000000 |
| context_budget_pressure | 845 | 176 | 0.000000 | 0.000000 | 0.000000 |
| runtime_contract_pressure | 3513 | 3513 | 0.000000 | 0.000000 | 0.000000 |
| grammar_json_table | 999 | 74 | 0.000000 | 0.000000 | 0.000000 |
| simple_baseline | 14 | 14 | 0.000000 | 0.000000 | 0.000000 |
| mixed_realworld | 840 | 27 | 0.000000 | 0.000000 | 0.000000 |
| debugging_log_analysis | 1732 | 1732 | 0.000000 | 0.000000 | 0.000000 |
| reasoning_root_cause | 104 | 104 | 0.000000 | 0.000000 | 0.000000 |
| repeated_identifier_investigation | 950 | 950 | 0.000000 | 0.000000 | 0.000000 |

### Response Samples

| Scenario | Baseline | Optimized |
| --- | --- | --- |
| uuid_deduplication | The user is asking why duplicate UUIDs failed in a list of transactions. Let me analyze the provided list. Looking at the list: - [0] 550e8400-e29b-41d4-a716-446655440000 - [1] 6b… | The user is asking why duplicate UUIDs failed in a list of transactions. Let me analyze the provided list. Looking at the list: - [0] 550e8400-e29b-41d4-a716-446655440000 - [1] 6b… |
| tool_output_filtering | The user wants to know why the build failed. I have the build errors from the function call. Let me analyze the data. There are 60 errors total. Let me categorize them by severity… | The user wants me to summarize the build errors. I already have the tool output provided in the system message, so I don't need to invoke the tool again. Let me analyze the data: … |
| table_compression | The user wants me to analyze employee data and summarize salary trends. Let me look at the data carefully. The data shows: - 100 employees (ID 0-99) - All in Engineering departmen… | The user wants me to analyze employee data and summarize salary trends. They've provided a snippet of data that looks like a table with 100 employees (Emp_0 to Emp_99), all in Eng… |
| prefix_optimization | The user is asking for guidance on when to use list comprehensions, dict comprehensions, set comprehensions, and generator expressions. This is a common Python question about choo… | The user is asking for guidance on when to use list comprehensions, dict comprehensions, set comprehensions, and generator expressions. This is a common Python question about choo… |
| code_review_patterns | The user wants me to review code changes across 10 files (module_0.py through module_9.py) and summarize the common issues. Looking at the code: - All 10 files have identical code… | The user wants me to review code changes across 10 files (module_0.py through module_9.py) and summarize the common issues. Looking at the code: - All 10 files have identical code… |
| api_docs_summarization | The user wants me to document these API endpoints concisely. Looking at the input, there are 20 endpoints (resource_0 through resource_19) that all have the exact same structure: … | The user wants me to document these API endpoints concisely. Looking at the input, there are 20 endpoints (resource_0 through resource_19) that all have the exact same structure: … |
| tool_call_preservation | The user is asking for the average temperature between NYC and SF. From the previous results, NYC is 72°F and SF is 65°F. I need to calculate the average: (72 + 65) / 2 = 137 / 2 … | The user is asking for the average temperature between NYC and SF based on the previous results. NYC was 72°F and SF was 65°F. I need to calculate the average: (72 + 65) / 2 = 137… |
| json_response_format | The user wants me to analyze the provided JSON data and return the top 5 users as JSON. Looking at the data: - There are 50 users with ids 0-49 - The scores are: 0, 10, 20, 30, ..… | The user wants me to analyze the provided JSON data and return the top 5 users as JSON. Looking at the data: - There are 50 users with ids 0-49 - User_0 has score 0 - User_1 has s… |
| cache_arbitrage_prefix | The user is asking what the next "delta" should contain given the repeated context. The context consists of 12 design notes (0-11), all saying the same thing: "keep manifest order… | The user is asking what the next "delta" should contain given the repeated context. The context consists of 12 design notes (0-11), all saying the same thing: "keep manifest order… |
| dictionary_repetition | The user wants me to compress repeated language losslessly. The input is a repetition of the sentence "The session manifest should remain stable and the session manifest should re… | The user wants me to compress repeated language losslessly. The input is: `<crossref_1>. <crossref_1>.<crossref_1>. <crossref_1>.<crossref_1>. <crossref_1>.<crossref_1>. <crossref… |
| cleanup_noise | The user wants me to summarize something, but they haven't actually provided the text to summarize. They said "Please summarize this" but there's no content after that. Wait, look… | The user wants me to summarize something, but they haven't actually provided the text to summarize. They said "Please summarize this" but there's no content after that. They also … |
| message_dedup_turns | The user wants me to explain the root cause once, based on the previous context. However, looking at the conversation history, the only "incident report" mentioned is the one I su… | The user wants me to explain the root cause once, presumably based on the same (unseen) incident report context. Since I don't have the actual incident report text in front of me,… |
| rate_distortion_longform | The user wants me to summarize an incident clearly while keeping the root cause. The input is highly repetitive (the same paragraph repeated many times). I need to extract the key… | The user wants me to summarize an incident clearly and keep the root cause. However, the user hasn't provided any specific incident details in their message. I need to ask them to… |
| context_budget_pressure | The user wants to know which documents mention the "error path" and "request ids". Looking at the provided documents: - Document 0: mentions "request id 0000" and "error path" - D… | The user wants to know which documents mention the "error path" and "request ids". Looking at the provided context, there are 18 documents (Document 0 through Document 17, though … |
| runtime_contract_pressure | The user is asking me to "Prove the optimization contract is correct" repeatedly. This seems to be a continuation of the previous context where I was analyzing stack traces and co… | The user is asking me to "prove the optimization contract is correct" and wants a formal theorem, lemma, deduction, proof obligations, and final mitigation. This seems to be a con… |
| grammar_json_table | The user wants me to summarize structured data (JSON and a table) without losing keys. This means I need to provide a condensed representation that preserves the schema/keys but c… | The user wants me to summarize structured data without losing keys. The data appears to be a table with columns, though the exact headers aren't explicitly stated. Let me analyze … |
| simple_baseline | Hello! I'm doing well, thank you. How can I help you today? | Hello! I'm doing well, thank you. As an AI, I don't have feelings, but I'm fully operational and ready to help. How can I assist you today? |
| mixed_realworld | The user wants me to investigate a production error from a transaction. Let me analyze the provided JSON trace. The trace shows 30 function calls (foo_0 through foo_29) with times… | The user has provided a very minimal error message: "Error from transaction <ref_29>:". This is clearly a fragment of a log or error message, and they likely want help debugging i… |
| debugging_log_analysis | The user wants me to analyze error logs from a production outage and identify the root cause and mitigation steps. Looking at the logs: - There are 5 services: service_0 through s… | The user wants me to analyze error logs from a production outage and identify the root cause and mitigation steps. Looking at the logs: - There are 5 services: service_0 through s… |
| reasoning_root_cause | The user wants me to explain a system crash based on provided data points: - Crash at 14:32 with error code E0503 - Memory usage trajectory: 94% (14:30), 97% (14:31), 99% (14:32) … | The user wants an explanation of a system crash based on provided data points. Let me analyze the information: 1. Crash at 14:32 with error code E0503 2. Memory usage trajectory: … |
| repeated_identifier_investigation | The user wants me to investigate transaction failures from the provided list and identify a pattern. First, let me list the failed transactions: - TX-0003: UUID 0x550e840000000003… | The user wants me to investigate transaction failures from the provided list and identify which ones failed plus any pattern. Let me list out the failed transactions: - TX-0003: U… |

### Prompt Safety

| Scenario | Structured | Code | Strict | Tool Calls | High Stakes | Long Form | Lossy OK |
| --- | --- | --- | --- | --- | --- | --- | --- |
| uuid_deduplication | False | False | False | False | True | True | True |
| tool_output_filtering | True | False | False | False | True | True | False |
| table_compression | True | False | False | False | True | True | False |
| prefix_optimization | False | False | True | False | False | True | False |
| code_review_patterns | False | True | True | False | True | True | False |
| api_docs_summarization | False | True | False | False | True | True | False |
| tool_call_preservation | True | False | False | True | False | False | False |
| json_response_format | False | False | True | False | True | True | False |
| cache_arbitrage_prefix | False | False | True | False | True | True | False |
| dictionary_repetition | False | False | False | False | False | False | True |
| cleanup_noise | False | False | True | False | False | False | False |
| message_dedup_turns | False | False | False | False | False | False | True |
| rate_distortion_longform | False | False | False | False | False | True | True |
| context_budget_pressure | False | False | True | False | True | True | False |
| runtime_contract_pressure | True | True | False | False | True | True | False |
| grammar_json_table | True | False | True | False | True | True | False |
| simple_baseline | False | False | False | False | False | False | False |
| mixed_realworld | False | True | False | False | True | True | False |
| debugging_log_analysis | False | False | False | False | True | True | True |
| reasoning_root_cause | False | False | True | False | True | False | False |
| repeated_identifier_investigation | False | False | False | False | True | True | True |

## provider_eval

- Kind: `live`
- Status: `ok`

| Metric | Value |
| --- | ---: |
| target_count | 1 |
| active_targets | 21 |
| skipped_targets | 0 |
| scenario_count | 21 |
| avg_quality_score | 0.8525 |

```json
{
  "targets": [
    {
      "adapter": "ollama-cloud",
      "benchmark": {
        "config": {
          "adapter": "ollama-cloud",
          "iterations": 1,
          "scenario_count": 21,
          "target": {
            "available": true,
            "base_url": "https://ollama.com",
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "skip_reason": ""
          },
          "warmup": 0
        },
        "model": "ollama-cloud/kimi-k2.6:cloud",
        "provider": "ollama-cloud",
        "runner": "provider_eval",
        "scenarios": [
          {
            "category": "reference_substitution",
            "cost": {
              "savings_per_1k_requests_usd": 0.0
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 4361.867165891454,
                "network_stdev": 0.0,
                "total_ms": 4361.867999890819,
                "total_stdev": 0.0
              },
              "delta": -62.49,
              "optimized": {
                "network_ms": 4245.264541823417,
                "network_stdev": 0.0,
                "pipeline_ms": 54.11299993284047,
                "pipeline_stdev": 0.0,
                "total_ms": 4299.37845794484,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.9375,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user is asking why duplicate UUIDs failed in a list of transactions. Let me analyze the provided list.\n\nLooking at the list:\n- [0] 550e8400-e29b-41d4-a716-446655440000\n- [1] 6ba7b810-9dad-11d1-80b4-00c04fd430c8\n- [2] a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11\n- [3] f47ac10b-58cc-4372-a567-0e02b2c3d479\n- [4] 6ba7b810-9dad-11d1-80b4-00c04fd430c8 (duplicate of [1])\n- [5] 550e8400-e29b-41d4-a716-446655440000 (duplicate of [0])\n\nThen [6] through [29] are sequential UUIDs with incrementing first segment",
              "optimized": "The user is asking why duplicate UUIDs failed in a list of transactions. Let me analyze the provided list.\n\nLooking at the list:\n- [0] 550e8400-e29b-41d4-a716-446655440000\n- [1] 6ba7b810-9dad-11d1-80b4-00c04fd430c8\n- [2] a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11\n- [3] f47ac10b-58cc-4372-a567-0e02b2c3d479\n- [4] 6ba7b810-9dad-11d1-80b4-00c04fd430c8 (duplicate of [1])\n- [5] 550e8400-e29b-41d4-a716-446655440000 (duplicate of [0])\n\nThen [6] through [29] are sequential UUIDs with incrementing hex values in"
            },
            "scenario": "uuid_deduplication",
            "telemetry": {
              "pipeline": {
                "latency_ms": 54.04019355773926,
                "tokens_in": 353,
                "tokens_out": 353,
                "transform_latency_ms": 51.795
              },
              "prompt": {
                "lossy_transform_allowed": true,
                "safety_profile": {
                  "has_code_blocks": false,
                  "has_high_stakes_entities": true,
                  "has_strict_instructions": false,
                  "has_structured_content": false,
                  "has_tool_calls": false,
                  "long_form": true
                },
                "structure_signature": {
                  "chars": 1393,
                  "has_code": false,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 34,
                  "number_count": 99,
                  "url_count": 0,
                  "uuid_count": 30
                }
              },
              "reachability": {
                "activated": [],
                "activated_count": 0,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [],
                "useful_count": 0
              },
              "runtime": {
                "confidence": 0.35,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 20.0,
                  "mode": "minimal",
                  "preferred_strategy": "full",
                  "score": 13,
                  "skipped_transforms": [
                    "information_theoretic_selector",
                    "rate_distortion"
                  ]
                },
                "features": {
                  "code": 0,
                  "debugging": 3,
                  "depth": 0,
                  "length": 10,
                  "reasoning": 0,
                  "root_cause": 0,
                  "tools": 0
                },
                "score": 13,
                "tier": "SIMPLE"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 20.0,
                "mode": "minimal",
                "preferred_strategy": "full",
                "score": 13,
                "skipped_transforms": [
                  "information_theoretic_selector",
                  "rate_distortion"
                ]
              },
              "scenario": {
                "category": "reference_substitution",
                "complexity": "medium",
                "expected_tier": "SIMPLE",
                "name": "uuid_deduplication",
                "proof": "UUID-heavy logs should collapse to aliases without changing the failure explanation.",
                "target_features": [
                  "reference_sub"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.088,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.0169971671388102,
                  "stable_tokens": 6,
                  "tokens_after": 353,
                  "tokens_before": 353
                },
                "content_profiler": {
                  "latency_ms": 18.591,
                  "profile": "narrative_long",
                  "risk_level": "LOW",
                  "risk_score": 12.0,
                  "sig_protected": 31,
                  "sig_total_spans": 62,
                  "task_class": "retrieval",
                  "tokens_after": 353,
                  "tokens_before": 353,
                  "total_tokens": 353
                },
                "pipeline_v2": {
                  "beam_candidates": 5,
                  "beam_search_latency_ms": 31.504,
                  "core_transforms": 7,
                  "ir_sections": 33,
                  "latency_ms": 32.48,
                  "optimizers": 5,
                  "tokens_after": 353,
                  "tokens_before": 353,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.033,
                  "prefix_tokens": 6,
                  "suffix_tokens": 347,
                  "tokens_after": 353,
                  "tokens_before": 353
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.35,
                  "latency_ms": 0.603,
                  "tier_score": 13,
                  "tokens_after": 353,
                  "tokens_before": 353,
                  "transform_budget_ms": 20.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.9562901832273594,
                  "rate_distortion": true,
                  "selected_strategy": "full",
                  "submodular": false,
                  "tool_filter": true,
                  "tool_optimizer": true
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "message_dedup",
                "strategy_selector",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 728,
                "before": 728,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 728,
                "before": 728,
                "ratio": 0.0,
                "saved": 0
              },
              "savings": {
                "after": 728,
                "before": 728,
                "ratio": 0.0,
                "saved": 0
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.088,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.0169971671388102,
                "stable_tokens": 6,
                "tokens_after": 353,
                "tokens_before": 353
              },
              "content_profiler": {
                "latency_ms": 18.591,
                "profile": "narrative_long",
                "risk_level": "LOW",
                "risk_score": 12.0,
                "sig_protected": 31,
                "sig_total_spans": 62,
                "task_class": "retrieval",
                "tokens_after": 353,
                "tokens_before": 353,
                "total_tokens": 353
              },
              "pipeline_v2": {
                "beam_candidates": 5,
                "beam_search_latency_ms": 31.504,
                "core_transforms": 7,
                "ir_sections": 33,
                "latency_ms": 32.48,
                "optimizers": 5,
                "tokens_after": 353,
                "tokens_before": 353,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.033,
                "prefix_tokens": 6,
                "suffix_tokens": 347,
                "tokens_after": 353,
                "tokens_before": 353
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.35,
                "latency_ms": 0.603,
                "tier_score": 13,
                "tokens_after": 353,
                "tokens_before": 353,
                "transform_budget_ms": 20.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.9562901832273594,
                "latency_ms": 0,
                "rate_distortion": true,
                "selected_strategy": "full",
                "submodular": false,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 728,
                "total_tokens": 984
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 256,
                "prompt_tokens": 728,
                "total_tokens": 984
              }
            }
          },
          {
            "category": "tool_filter",
            "cost": {
              "savings_per_1k_requests_usd": 6.246
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 4185.111792059615,
                "network_stdev": 0.0,
                "total_ms": 4185.112334089354,
                "total_stdev": 0.0
              },
              "delta": 2781.676,
              "optimized": {
                "network_ms": 6770.729917101562,
                "network_stdev": 0.0,
                "pipeline_ms": 196.05683395639062,
                "pipeline_stdev": 0.0,
                "total_ms": 6966.788833960891,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": false,
              "semantic_similarity": 0.7067,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user wants to know why the build failed. I have the build errors data. Let me analyze it.\n\nLooking at the errors array:\n- There are 60 total entries (module_0 through module_59)\n- The messages cycle through: \"Module not found\", \"Syntax error\", \"Type mismatch\"\n- The severities are: \"error\" for \"Module not found\" and \"Syntax error\", \"warning\" for \"Type mismatch\"\n- Let me count:\n  - \"Module not found\" (error): modules 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57 = ",
              "optimized": "The user wants me to summarize the build errors. I already have the tool output provided in the system message, so I don't need to invoke the tool again. Let me analyze the data:\n\n- Total errors: 60\n- Showing: 10 (sample)\n- Breakdown from the sample:\n  - \"Module not found\": modules 0, 3, 6, 9 (4 occurrences in sample) - severity: error\n  - \"Syntax error\": modules 1, 4, 7 (3 occurrences in sample) - severity: error\n  - \"Type mismatch\": modules 2, 5, 8 (3 occurrences in sample) - severity: warning"
            },
            "scenario": "tool_output_filtering",
            "telemetry": {
              "pipeline": {
                "latency_ms": 196.0318088531494,
                "tokens_in": 3770,
                "tokens_out": 242,
                "transform_latency_ms": 192.541
              },
              "prompt": {
                "lossy_transform_allowed": false,
                "safety_profile": {
                  "has_code_blocks": false,
                  "has_high_stakes_entities": true,
                  "has_strict_instructions": false,
                  "has_structured_content": true,
                  "has_tool_calls": false,
                  "long_form": true
                },
                "structure_signature": {
                  "chars": 14958,
                  "has_code": false,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 726,
                  "number_count": 240,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [
                  "pipeline_v2"
                ],
                "activated_count": 1,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [
                  "pipeline_v2"
                ],
                "useful_count": 1
              },
              "runtime": {
                "confidence": 0.5714,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 20.0,
                  "mode": "aggressive",
                  "preferred_strategy": "hybrid",
                  "score": 40,
                  "skipped_transforms": []
                },
                "features": {
                  "code": 0,
                  "debugging": 0,
                  "depth": 0,
                  "length": 30,
                  "reasoning": 0,
                  "root_cause": 0,
                  "tools": 10
                },
                "score": 40,
                "tier": "COMPLEX"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 20.0,
                "mode": "aggressive",
                "preferred_strategy": "hybrid",
                "score": 40,
                "skipped_transforms": []
              },
              "scenario": {
                "category": "tool_filter",
                "complexity": "complex",
                "expected_tier": "COMPLEX",
                "name": "tool_output_filtering",
                "proof": "Tool JSON should retain schema fields while removing internal-only payload.",
                "target_features": [
                  "tool_filter"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.063,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.9899204244031831,
                  "stable_tokens": 3732,
                  "tokens_after": 3770,
                  "tokens_before": 3770
                },
                "content_profiler": {
                  "latency_ms": 80.188,
                  "profile": "mixed",
                  "risk_level": "MEDIUM",
                  "risk_score": 32.0,
                  "sig_protected": 100,
                  "sig_total_spans": 122,
                  "task_class": "analysis",
                  "tokens_after": 3770,
                  "tokens_before": 3770,
                  "total_tokens": 3770
                },
                "format_conversion": {
                  "spans_converted": 1,
                  "tokens_saved_estimate": 2006
                },
                "pipeline_v2": {
                  "beam_candidates": 6,
                  "beam_search_latency_ms": 106.432,
                  "core_transforms": 11,
                  "ir_sections": 4,
                  "latency_ms": 111.885,
                  "optimizers": 7,
                  "tokens_after": 242,
                  "tokens_before": 3770,
                  "tokens_saved": 14120
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.015,
                  "prefix_tokens": 14,
                  "suffix_tokens": 3756,
                  "tokens_after": 3770,
                  "tokens_before": 3770
                },
                "rate_distortion": {
                  "guarded": 1
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.5714285714285714,
                  "latency_ms": 0.389,
                  "tier_score": 40,
                  "tokens_after": 3770,
                  "tokens_before": 3770,
                  "transform_budget_ms": 20.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.4221462653327892,
                  "rate_distortion": true,
                  "selected_strategy": "hybrid",
                  "submodular": true,
                  "tool_filter": true,
                  "tool_optimizer": true
                },
                "tool_projection": {
                  "chars_saved": 0
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "message_dedup",
                "strategy_selector",
                "rate_distortion",
                "path_prefix",
                "format_conversion",
                "tool_projection",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 4412,
                "before": 4412,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 248,
                "before": 4412,
                "ratio": 0.9437896645512239,
                "saved": 4164
              },
              "savings": {
                "after": 248,
                "before": 4412,
                "ratio": 0.9437896645512239,
                "saved": 4164
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.063,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.9899204244031831,
                "stable_tokens": 3732,
                "tokens_after": 3770,
                "tokens_before": 3770
              },
              "content_profiler": {
                "latency_ms": 80.188,
                "profile": "mixed",
                "risk_level": "MEDIUM",
                "risk_score": 32.0,
                "sig_protected": 100,
                "sig_total_spans": 122,
                "task_class": "analysis",
                "tokens_after": 3770,
                "tokens_before": 3770,
                "total_tokens": 3770
              },
              "format_conversion": {
                "latency_ms": 0,
                "spans_converted": 1,
                "tokens_after": 0,
                "tokens_saved_estimate": 2006
              },
              "pipeline_v2": {
                "beam_candidates": 6,
                "beam_search_latency_ms": 106.432,
                "core_transforms": 11,
                "ir_sections": 4,
                "latency_ms": 111.885,
                "optimizers": 7,
                "tokens_after": 242,
                "tokens_before": 3770,
                "tokens_saved": 14120
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.015,
                "prefix_tokens": 14,
                "suffix_tokens": 3756,
                "tokens_after": 3770,
                "tokens_before": 3770
              },
              "rate_distortion": {
                "guarded": 1,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.5714285714285714,
                "latency_ms": 0.389,
                "tier_score": 40,
                "tokens_after": 3770,
                "tokens_before": 3770,
                "transform_budget_ms": 20.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.4221462653327892,
                "latency_ms": 0,
                "rate_distortion": true,
                "selected_strategy": "hybrid",
                "submodular": true,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              },
              "tool_projection": {
                "chars_saved": 0,
                "latency_ms": 0,
                "tokens_after": 0
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 4412,
                "total_tokens": 4668
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 256,
                "prompt_tokens": 248,
                "total_tokens": 504
              }
            }
          },
          {
            "category": "format_conversion",
            "cost": {
              "savings_per_1k_requests_usd": 2.3685
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 4469.397667096928,
                "network_stdev": 0.0,
                "total_ms": 4469.398542074487,
                "total_stdev": 0.0
              },
              "delta": -153.009,
              "optimized": {
                "network_ms": 4166.385916993022,
                "network_stdev": 0.0,
                "pipeline_ms": 150.0025000423193,
                "pipeline_stdev": 0.0,
                "total_ms": 4316.389167215675,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": false,
              "semantic_similarity": 0.804,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user wants me to analyze employee data and summarize salary trends. Let me look at the data provided.\n\nThe data shows 100 employees (ID 0-99), all in Engineering department, all active, with salaries from $100,000 to $199,000.\n\nLet me identify the pattern:\n- ID 0: $100,000\n- ID 1: $101,000\n- ID 2: $102,000\n...\n- ID 99: $199,000\n\nSo the salary = $100,000 + ID \u00d7 $1,000\n\nKey observations:\n1. All employees are in the Engineering department\n2. All employees are active\n3. Salaries range from $100,",
              "optimized": "The user wants me to analyze employee data and summarize salary trends. They've provided a pattern of data that looks like:\n\n```\n100x | 0 | Emp_0 | Engineering | $100000 | active |\n- | 0 | Emp_0 | Engineering | $100000 | active |\n- ...\n- | 99 | Emp_99 | Engineering | $199000 | active |\n```\n\nWait, the first line says \"100x | 0 | Emp_0 | Engineering | $100000 | active |\" which might mean there are 100 employees, indexed 0 to 99.\n\nLooking at the pattern:\n- Employee IDs: Emp_0 to Emp_99\n- Department"
            },
            "scenario": "table_compression",
            "telemetry": {
              "pipeline": {
                "latency_ms": 149.94502067565918,
                "tokens_in": 1272,
                "tokens_out": 57,
                "transform_latency_ms": 147.888
              },
              "prompt": {
                "lossy_transform_allowed": false,
                "safety_profile": {
                  "has_code_blocks": false,
                  "has_high_stakes_entities": true,
                  "has_strict_instructions": false,
                  "has_structured_content": true,
                  "has_tool_calls": false,
                  "long_form": true
                },
                "structure_signature": {
                  "chars": 5069,
                  "has_code": false,
                  "has_table": true,
                  "json_keys": [],
                  "lines": 106,
                  "number_count": 200,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [
                  "pipeline_v2"
                ],
                "activated_count": 1,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [
                  "pipeline_v2"
                ],
                "useful_count": 1
              },
              "runtime": {
                "confidence": 0.75,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 20.0,
                  "mode": "balanced",
                  "preferred_strategy": "submodular",
                  "score": 30,
                  "skipped_transforms": []
                },
                "features": {
                  "code": 0,
                  "debugging": 0,
                  "depth": 0,
                  "length": 30,
                  "reasoning": 0,
                  "root_cause": 0,
                  "tools": 0
                },
                "score": 30,
                "tier": "MEDIUM"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 20.0,
                "mode": "balanced",
                "preferred_strategy": "submodular",
                "score": 30,
                "skipped_transforms": []
              },
              "scenario": {
                "category": "format_conversion",
                "complexity": "complex",
                "expected_tier": "MEDIUM",
                "name": "table_compression",
                "proof": "Wide markdown tables should convert to a more compact transport format.",
                "target_features": [
                  "format_conversion"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.051,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.009433962264150943,
                  "stable_tokens": 12,
                  "tokens_after": 1272,
                  "tokens_before": 1272
                },
                "content_profiler": {
                  "latency_ms": 34.779,
                  "profile": "table_heavy",
                  "risk_level": "LOW",
                  "risk_score": 15.0,
                  "sig_protected": 2,
                  "sig_total_spans": 2,
                  "task_class": "retrieval",
                  "tokens_after": 1272,
                  "tokens_before": 1272,
                  "total_tokens": 1272
                },
                "format_conversion": {
                  "spans_converted": 1,
                  "tokens_saved_estimate": 314
                },
                "pipeline_v2": {
                  "beam_candidates": 6,
                  "beam_search_latency_ms": 106.852,
                  "core_transforms": 11,
                  "ir_sections": 2,
                  "latency_ms": 111.234,
                  "optimizers": 7,
                  "tokens_after": 57,
                  "tokens_before": 1272,
                  "tokens_saved": 4861
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.02,
                  "prefix_tokens": 12,
                  "suffix_tokens": 1260,
                  "tokens_after": 1272,
                  "tokens_before": 1272
                },
                "rate_distortion": {
                  "guarded": 1
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.75,
                  "latency_ms": 1.804,
                  "tier_score": 30,
                  "tokens_after": 1272,
                  "tokens_before": 1272,
                  "transform_budget_ms": 20.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.948471772954384,
                  "rate_distortion": false,
                  "selected_strategy": "submodular",
                  "submodular": true,
                  "tool_filter": true,
                  "tool_optimizer": true
                },
                "structure_optimizer": {
                  "rejected": "tokens_after (3762) > tokens_before (207) with no gain"
                },
                "tool_projection": {
                  "chars_saved": 0
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "message_dedup",
                "strategy_selector",
                "rate_distortion",
                "path_prefix",
                "format_conversion",
                "tool_projection",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 1653,
                "before": 1653,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 74,
                "before": 1653,
                "ratio": 0.9552329098608591,
                "saved": 1579
              },
              "savings": {
                "after": 74,
                "before": 1653,
                "ratio": 0.9552329098608591,
                "saved": 1579
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.051,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.009433962264150943,
                "stable_tokens": 12,
                "tokens_after": 1272,
                "tokens_before": 1272
              },
              "content_profiler": {
                "latency_ms": 34.779,
                "profile": "table_heavy",
                "risk_level": "LOW",
                "risk_score": 15.0,
                "sig_protected": 2,
                "sig_total_spans": 2,
                "task_class": "retrieval",
                "tokens_after": 1272,
                "tokens_before": 1272,
                "total_tokens": 1272
              },
              "format_conversion": {
                "latency_ms": 0,
                "spans_converted": 1,
                "tokens_after": 0,
                "tokens_saved_estimate": 314
              },
              "pipeline_v2": {
                "beam_candidates": 6,
                "beam_search_latency_ms": 106.852,
                "core_transforms": 11,
                "ir_sections": 2,
                "latency_ms": 111.234,
                "optimizers": 7,
                "tokens_after": 57,
                "tokens_before": 1272,
                "tokens_saved": 4861
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.02,
                "prefix_tokens": 12,
                "suffix_tokens": 1260,
                "tokens_after": 1272,
                "tokens_before": 1272
              },
              "rate_distortion": {
                "guarded": 1,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.75,
                "latency_ms": 1.804,
                "tier_score": 30,
                "tokens_after": 1272,
                "tokens_before": 1272,
                "transform_budget_ms": 20.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.948471772954384,
                "latency_ms": 0,
                "rate_distortion": false,
                "selected_strategy": "submodular",
                "submodular": true,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              },
              "structure_optimizer": {
                "latency_ms": 0,
                "rejected": "tokens_after (3762) > tokens_before (207) with no gain",
                "tokens_after": 0
              },
              "tool_projection": {
                "chars_saved": 0,
                "latency_ms": 0,
                "tokens_after": 0
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 1653,
                "total_tokens": 1909
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 256,
                "prompt_tokens": 74,
                "total_tokens": 330
              }
            }
          },
          {
            "category": "prefix_opt",
            "cost": {
              "savings_per_1k_requests_usd": 0.0
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 4906.404708046466,
                "network_stdev": 0.0,
                "total_ms": 4906.405167188495,
                "total_stdev": 0.0
              },
              "delta": 110.429,
              "optimized": {
                "network_ms": 4972.296041902155,
                "network_stdev": 0.0,
                "pipeline_ms": 44.53754215501249,
                "pipeline_stdev": 0.0,
                "total_ms": 5016.834167065099,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": false,
              "semantic_similarity": 0.5875,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user is asking for guidance on when to use list comprehensions, dict comprehensions, set comprehensions, and generator expressions. This is a natural follow-up to the previous questions where I explained each one individually.\n\nI need to provide a concise, practical comparison that helps them choose the right tool. Key factors to consider:\n1. Memory usage (lists/sets/dicts are eager, generators are lazy)\n2. Use case (building a collection vs iterating once)\n3. Performance implications (overh",
              "optimized": "**Use the data structure you actually need:**\n\n| | Use when... | Memory |\n|---|---|---|\n| **List** | You need indexing, `len()`, or multiple passes. Result order"
            },
            "scenario": "prefix_optimization",
            "telemetry": {
              "pipeline": {
                "latency_ms": 44.45791244506836,
                "tokens_in": 207,
                "tokens_out": 207,
                "transform_latency_ms": 42.13
              },
              "prompt": {
                "lossy_transform_allowed": false,
                "safety_profile": {
                  "has_code_blocks": false,
                  "has_high_stakes_entities": false,
                  "has_strict_instructions": true,
                  "has_structured_content": false,
                  "has_tool_calls": false,
                  "long_form": true
                },
                "structure_signature": {
                  "chars": 734,
                  "has_code": false,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 10,
                  "number_count": 1,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [],
                "activated_count": 0,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [],
                "useful_count": 0
              },
              "runtime": {
                "confidence": 0.5,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 20.0,
                  "mode": "balanced",
                  "preferred_strategy": "submodular",
                  "score": 20,
                  "skipped_transforms": []
                },
                "features": {
                  "code": 0,
                  "debugging": 0,
                  "depth": 10,
                  "length": 10,
                  "reasoning": 0,
                  "root_cause": 0,
                  "tools": 0
                },
                "score": 20,
                "tier": "MEDIUM"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 20.0,
                "mode": "balanced",
                "preferred_strategy": "submodular",
                "score": 20,
                "skipped_transforms": []
              },
              "scenario": {
                "category": "prefix_opt",
                "complexity": "medium",
                "expected_tier": "MEDIUM",
                "name": "prefix_optimization",
                "proof": "Stable prefix material should not be re-sent on every turn.",
                "target_features": [
                  "prefix_opt"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.136,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.42028985507246375,
                  "stable_tokens": 87,
                  "tokens_after": 207,
                  "tokens_before": 207
                },
                "content_profiler": {
                  "latency_ms": 9.791,
                  "profile": "narrative_long",
                  "risk_level": "LOW",
                  "risk_score": 3.5,
                  "sig_protected": 3,
                  "sig_total_spans": 17,
                  "task_class": "retrieval",
                  "tokens_after": 207,
                  "tokens_before": 207,
                  "total_tokens": 207
                },
                "pipeline_v2": {
                  "beam_candidates": 5,
                  "beam_search_latency_ms": 30.437,
                  "core_transforms": 7,
                  "ir_sections": 11,
                  "latency_ms": 31.665,
                  "optimizers": 5,
                  "tokens_after": 207,
                  "tokens_before": 207,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.029,
                  "prefix_tokens": 87,
                  "suffix_tokens": 120,
                  "tokens_after": 207,
                  "tokens_before": 207
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.5,
                  "latency_ms": 0.509,
                  "tier_score": 20,
                  "tokens_after": 207,
                  "tokens_before": 207,
                  "transform_budget_ms": 20.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.8004556499732676,
                  "rate_distortion": false,
                  "selected_strategy": "submodular",
                  "submodular": true,
                  "tool_filter": true,
                  "tool_optimizer": true
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "message_dedup",
                "strategy_selector",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 202,
                "before": 202,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 202,
                "before": 202,
                "ratio": 0.0,
                "saved": 0
              },
              "savings": {
                "after": 202,
                "before": 202,
                "ratio": 0.0,
                "saved": 0
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.136,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.42028985507246375,
                "stable_tokens": 87,
                "tokens_after": 207,
                "tokens_before": 207
              },
              "content_profiler": {
                "latency_ms": 9.791,
                "profile": "narrative_long",
                "risk_level": "LOW",
                "risk_score": 3.5,
                "sig_protected": 3,
                "sig_total_spans": 17,
                "task_class": "retrieval",
                "tokens_after": 207,
                "tokens_before": 207,
                "total_tokens": 207
              },
              "pipeline_v2": {
                "beam_candidates": 5,
                "beam_search_latency_ms": 30.437,
                "core_transforms": 7,
                "ir_sections": 11,
                "latency_ms": 31.665,
                "optimizers": 5,
                "tokens_after": 207,
                "tokens_before": 207,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.029,
                "prefix_tokens": 87,
                "suffix_tokens": 120,
                "tokens_after": 207,
                "tokens_before": 207
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.5,
                "latency_ms": 0.509,
                "tier_score": 20,
                "tokens_after": 207,
                "tokens_before": 207,
                "transform_budget_ms": 20.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.8004556499732676,
                "latency_ms": 0,
                "rate_distortion": false,
                "selected_strategy": "submodular",
                "submodular": true,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 202,
                "total_tokens": 458
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 256,
                "prompt_tokens": 202,
                "total_tokens": 458
              }
            }
          },
          {
            "category": "reference_sub",
            "cost": {
              "savings_per_1k_requests_usd": 0.0
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 4807.6808750629425,
                "network_stdev": 0.0,
                "total_ms": 4807.681417092681,
                "total_stdev": 0.0
              },
              "delta": 4.159,
              "optimized": {
                "network_ms": 4664.48266687803,
                "network_stdev": 0.0,
                "pipeline_ms": 147.35654182732105,
                "pipeline_stdev": 0.0,
                "total_ms": 4811.839999631047,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 1.0,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user wants me to review code changes across 10 files (module_0.py through module_9.py) and summarize the common issues.\n\nLooking at the code:\n- All 10 files have identical code structure, just different function names (process_data_0 through process_data_9)\n- Each has a for loop that filters items where status is \"active\" and creates a new dictionary\n- The issue noted for each is: \"Line 4-6 has repeated dictionary access pattern. Consider list comprehension.\"\n\nCommon issues I can identify:\n1",
              "optimized": "The user wants me to review code changes across 10 files (module_0.py through module_9.py) and summarize the common issues.\n\nLooking at the code:\n- All 10 files have identical code structure, just different function names (process_data_0 through process_data_9)\n- Each has a for loop that filters items where status is \"active\" and creates a new dictionary\n- The issue noted for each is: \"Line 4-6 has repeated dictionary access pattern. Consider list comprehension.\"\n\nCommon issues I can identify:\n1"
            },
            "scenario": "code_review_patterns",
            "telemetry": {
              "pipeline": {
                "latency_ms": 147.27425575256348,
                "tokens_in": 992,
                "tokens_out": 992,
                "transform_latency_ms": 145.431
              },
              "prompt": {
                "lossy_transform_allowed": false,
                "safety_profile": {
                  "has_code_blocks": true,
                  "has_high_stakes_entities": true,
                  "has_strict_instructions": true,
                  "has_structured_content": false,
                  "has_tool_calls": false,
                  "long_form": true
                },
                "structure_signature": {
                  "chars": 3949,
                  "has_code": true,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 134,
                  "number_count": 30,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [],
                "activated_count": 0,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [],
                "useful_count": 0
              },
              "runtime": {
                "confidence": 0.575,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 20.0,
                  "mode": "balanced",
                  "preferred_strategy": "submodular",
                  "score": 23,
                  "skipped_transforms": []
                },
                "features": {
                  "code": 3,
                  "debugging": 0,
                  "depth": 0,
                  "length": 20,
                  "reasoning": 0,
                  "root_cause": 0,
                  "tools": 0
                },
                "score": 23,
                "tier": "MEDIUM"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 20.0,
                "mode": "balanced",
                "preferred_strategy": "submodular",
                "score": 23,
                "skipped_transforms": []
              },
              "scenario": {
                "category": "reference_sub",
                "complexity": "complex",
                "expected_tier": "MEDIUM",
                "name": "code_review_patterns",
                "proof": "Repeated code review structure should be compressed while preserving code blocks.",
                "target_features": [
                  "reference_sub"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.06,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.008064516129032258,
                  "stable_tokens": 8,
                  "tokens_after": 992,
                  "tokens_before": 992
                },
                "content_profiler": {
                  "latency_ms": 34.513,
                  "profile": "code_heavy",
                  "risk_level": "LOW",
                  "risk_score": 20.0,
                  "sig_protected": 12,
                  "sig_total_spans": 42,
                  "task_class": "retrieval",
                  "tokens_after": 992,
                  "tokens_before": 992,
                  "total_tokens": 992
                },
                "message_dedup": {
                  "original_count": 22,
                  "removed_count": 8,
                  "tokens_saved_estimate": 295
                },
                "pipeline_v2": {
                  "beam_candidates": 6,
                  "beam_search_latency_ms": 103.518,
                  "core_transforms": 11,
                  "ir_sections": 14,
                  "latency_ms": 109.19,
                  "optimizers": 7,
                  "tokens_after": 992,
                  "tokens_before": 992,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.021,
                  "prefix_tokens": 8,
                  "suffix_tokens": 984,
                  "tokens_after": 992,
                  "tokens_before": 992
                },
                "rate_distortion": {
                  "guarded": 1
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.575,
                  "latency_ms": 1.646,
                  "tier_score": 23,
                  "tokens_after": 992,
                  "tokens_before": 992,
                  "transform_budget_ms": 20.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.9402894756153457,
                  "rate_distortion": false,
                  "selected_strategy": "submodular",
                  "submodular": true,
                  "tool_filter": true,
                  "tool_optimizer": true
                },
                "tool_projection": {
                  "chars_saved": 0
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "message_dedup",
                "strategy_selector",
                "rate_distortion",
                "path_prefix",
                "format_conversion",
                "tool_projection",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 1037,
                "before": 1037,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 1037,
                "before": 1037,
                "ratio": 0.0,
                "saved": 0
              },
              "savings": {
                "after": 1037,
                "before": 1037,
                "ratio": 0.0,
                "saved": 0
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.06,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.008064516129032258,
                "stable_tokens": 8,
                "tokens_after": 992,
                "tokens_before": 992
              },
              "content_profiler": {
                "latency_ms": 34.513,
                "profile": "code_heavy",
                "risk_level": "LOW",
                "risk_score": 20.0,
                "sig_protected": 12,
                "sig_total_spans": 42,
                "task_class": "retrieval",
                "tokens_after": 992,
                "tokens_before": 992,
                "total_tokens": 992
              },
              "message_dedup": {
                "latency_ms": 0,
                "original_count": 22,
                "removed_count": 8,
                "tokens_after": 0,
                "tokens_saved_estimate": 295
              },
              "pipeline_v2": {
                "beam_candidates": 6,
                "beam_search_latency_ms": 103.518,
                "core_transforms": 11,
                "ir_sections": 14,
                "latency_ms": 109.19,
                "optimizers": 7,
                "tokens_after": 992,
                "tokens_before": 992,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.021,
                "prefix_tokens": 8,
                "suffix_tokens": 984,
                "tokens_after": 992,
                "tokens_before": 992
              },
              "rate_distortion": {
                "guarded": 1,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.575,
                "latency_ms": 1.646,
                "tier_score": 23,
                "tokens_after": 992,
                "tokens_before": 992,
                "transform_budget_ms": 20.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.9402894756153457,
                "latency_ms": 0,
                "rate_distortion": false,
                "selected_strategy": "submodular",
                "submodular": true,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              },
              "tool_projection": {
                "chars_saved": 0,
                "latency_ms": 0,
                "tokens_after": 0
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 1037,
                "total_tokens": 1293
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 256,
                "prompt_tokens": 1037,
                "total_tokens": 1293
              }
            }
          },
          {
            "category": "reference_optimizer",
            "cost": {
              "savings_per_1k_requests_usd": 0.0
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 4190.5970829539,
                "network_stdev": 0.0,
                "total_ms": 4190.597750013694,
                "total_stdev": 0.0
              },
              "delta": 110.865,
              "optimized": {
                "network_ms": 4137.432625051588,
                "network_stdev": 0.0,
                "pipeline_ms": 164.0278750564903,
                "pipeline_stdev": 0.0,
                "total_ms": 4301.462750183418,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 1.0,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user wants me to document 20 API endpoints concisely. Looking at the input, all endpoints follow the exact same pattern:\n\n- GET /api/v1/resource_{N} where N ranges from 0 to 19\n- Same parameters: id (required string), limit (optional int, default 50), offset (optional int)\n- Same response structure: {\"id\": \"N\", \"name\": \"Resource N\", \"status\": \"active\"}\n- Same errors: 404, 429, 500\n\nSince they are all identical except for the resource number/id, I should group them together rather than listin",
              "optimized": "The user wants me to document 20 API endpoints concisely. Looking at the input, all endpoints follow the exact same pattern:\n\n- GET /api/v1/resource_{N} where N ranges from 0 to 19\n- Same parameters: id (required string), limit (optional int, default 50), offset (optional int)\n- Same response structure: {\"id\": \"N\", \"name\": \"Resource N\", \"status\": \"active\"}\n- Same errors: 404, 429, 500\n\nSince they are all identical except for the resource number/id, I should group them together rather than listin"
            },
            "scenario": "api_docs_summarization",
            "telemetry": {
              "pipeline": {
                "latency_ms": 163.94710540771484,
                "tokens_in": 1893,
                "tokens_out": 1893,
                "transform_latency_ms": 162.463
              },
              "prompt": {
                "lossy_transform_allowed": false,
                "safety_profile": {
                  "has_code_blocks": true,
                  "has_high_stakes_entities": true,
                  "has_strict_instructions": false,
                  "has_structured_content": false,
                  "has_tool_calls": false,
                  "long_form": true
                },
                "structure_signature": {
                  "chars": 7550,
                  "has_code": true,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 362,
                  "number_count": 120,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [],
                "activated_count": 0,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [],
                "useful_count": 0
              },
              "runtime": {
                "confidence": 0.825,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 20.0,
                  "mode": "balanced",
                  "preferred_strategy": "submodular",
                  "score": 33,
                  "skipped_transforms": []
                },
                "features": {
                  "code": 0,
                  "debugging": 3,
                  "depth": 0,
                  "length": 30,
                  "reasoning": 0,
                  "root_cause": 0,
                  "tools": 0
                },
                "score": 33,
                "tier": "MEDIUM"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 20.0,
                "mode": "balanced",
                "preferred_strategy": "submodular",
                "score": 33,
                "skipped_transforms": []
              },
              "scenario": {
                "category": "reference_optimizer",
                "complexity": "medium",
                "expected_tier": "MEDIUM",
                "name": "api_docs_summarization",
                "proof": "Boilerplate docs should collapse through lossless phrase compression.",
                "target_features": [
                  "rate_distortion"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.096,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.003169572107765452,
                  "stable_tokens": 6,
                  "tokens_after": 1893,
                  "tokens_before": 1893
                },
                "content_profiler": {
                  "latency_ms": 47.347,
                  "profile": "code_heavy",
                  "risk_level": "LOW",
                  "risk_score": 20.0,
                  "sig_protected": 40,
                  "sig_total_spans": 42,
                  "task_class": "retrieval",
                  "tokens_after": 1893,
                  "tokens_before": 1893,
                  "total_tokens": 1893
                },
                "pipeline_v2": {
                  "beam_candidates": 6,
                  "beam_search_latency_ms": 105.742,
                  "core_transforms": 11,
                  "ir_sections": 43,
                  "latency_ms": 112.848,
                  "optimizers": 7,
                  "tokens_after": 1893,
                  "tokens_before": 1893,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.02,
                  "prefix_tokens": 6,
                  "suffix_tokens": 1887,
                  "tokens_after": 1893,
                  "tokens_before": 1893
                },
                "rate_distortion": {
                  "guarded": 1
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.825,
                  "latency_ms": 2.152,
                  "tier_score": 33,
                  "tokens_after": 1893,
                  "tokens_before": 1893,
                  "transform_budget_ms": 20.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.6662081376826976,
                  "rate_distortion": false,
                  "selected_strategy": "submodular",
                  "submodular": true,
                  "tool_filter": true,
                  "tool_optimizer": true
                },
                "tool_projection": {
                  "chars_saved": 100
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "message_dedup",
                "strategy_selector",
                "rate_distortion",
                "path_prefix",
                "format_conversion",
                "tool_projection",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 2183,
                "before": 2183,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 2183,
                "before": 2183,
                "ratio": 0.0,
                "saved": 0
              },
              "savings": {
                "after": 2183,
                "before": 2183,
                "ratio": 0.0,
                "saved": 0
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.096,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.003169572107765452,
                "stable_tokens": 6,
                "tokens_after": 1893,
                "tokens_before": 1893
              },
              "content_profiler": {
                "latency_ms": 47.347,
                "profile": "code_heavy",
                "risk_level": "LOW",
                "risk_score": 20.0,
                "sig_protected": 40,
                "sig_total_spans": 42,
                "task_class": "retrieval",
                "tokens_after": 1893,
                "tokens_before": 1893,
                "total_tokens": 1893
              },
              "pipeline_v2": {
                "beam_candidates": 6,
                "beam_search_latency_ms": 105.742,
                "core_transforms": 11,
                "ir_sections": 43,
                "latency_ms": 112.848,
                "optimizers": 7,
                "tokens_after": 1893,
                "tokens_before": 1893,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.02,
                "prefix_tokens": 6,
                "suffix_tokens": 1887,
                "tokens_after": 1893,
                "tokens_before": 1893
              },
              "rate_distortion": {
                "guarded": 1,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.825,
                "latency_ms": 2.152,
                "tier_score": 33,
                "tokens_after": 1893,
                "tokens_before": 1893,
                "transform_budget_ms": 20.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.6662081376826976,
                "latency_ms": 0,
                "rate_distortion": false,
                "selected_strategy": "submodular",
                "submodular": true,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              },
              "tool_projection": {
                "chars_saved": 100,
                "latency_ms": 0,
                "tokens_after": 0
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 2183,
                "total_tokens": 2439
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 256,
                "prompt_tokens": 2183,
                "total_tokens": 2439
              }
            }
          },
          {
            "category": "integration",
            "cost": {
              "savings_per_1k_requests_usd": 0.0
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 1962.135958019644,
                "network_stdev": 0.0,
                "total_ms": 1962.1367501094937,
                "total_stdev": 0.0
              },
              "delta": 257.807,
              "optimized": {
                "network_ms": 2176.291999872774,
                "network_stdev": 0.0,
                "pipeline_ms": 43.6509158462286,
                "pipeline_stdev": 0.0,
                "total_ms": 2219.9440407566726,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.9167,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user is asking for the average temperature between NYC and SF based on the previous results. NYC was 72\u00b0F and SF was 65\u00b0F. I should use the calculate tool to compute the average: (72 + 65) / 2.",
              "optimized": "The user is asking for the average temperature between NYC and SF based on the previous results. NYC was 72\u00b0F and SF was 65\u00b0F. I should use the calculate tool to compute the average: (72 + 65) / 2."
            },
            "scenario": "tool_call_preservation",
            "telemetry": {
              "pipeline": {
                "latency_ms": 43.52402687072754,
                "tokens_in": 133,
                "tokens_out": 133,
                "transform_latency_ms": 40.328
              },
              "prompt": {
                "lossy_transform_allowed": false,
                "safety_profile": {
                  "has_code_blocks": false,
                  "has_high_stakes_entities": false,
                  "has_strict_instructions": false,
                  "has_structured_content": true,
                  "has_tool_calls": true,
                  "long_form": false
                },
                "structure_signature": {
                  "chars": 139,
                  "has_code": false,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 6,
                  "number_count": 2,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [],
                "activated_count": 0,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [],
                "useful_count": 0
              },
              "runtime": {
                "confidence": 0.2,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 20.0,
                  "mode": "minimal",
                  "preferred_strategy": "full",
                  "score": 16,
                  "skipped_transforms": [
                    "information_theoretic_selector",
                    "rate_distortion"
                  ]
                },
                "features": {
                  "code": 0,
                  "debugging": 0,
                  "depth": 5,
                  "length": 1,
                  "reasoning": 0,
                  "root_cause": 0,
                  "tools": 10
                },
                "score": 16,
                "tier": "SIMPLE"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 20.0,
                "mode": "minimal",
                "preferred_strategy": "full",
                "score": 16,
                "skipped_transforms": [
                  "information_theoretic_selector",
                  "rate_distortion"
                ]
              },
              "scenario": {
                "category": "integration",
                "complexity": "complex",
                "expected_tier": "SIMPLE",
                "name": "tool_call_preservation",
                "proof": "Tool-call conversations must preserve call IDs and arguments.",
                "target_features": [
                  "tool_filter"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "latency_ms": 0.024,
                  "tokens_after": 133,
                  "tokens_before": 133
                },
                "content_profiler": {
                  "latency_ms": 5.167,
                  "profile": "tool_output",
                  "risk_level": "LOW",
                  "risk_score": 10.5,
                  "sig_protected": 2,
                  "sig_total_spans": 3,
                  "task_class": "simple",
                  "tokens_after": 133,
                  "tokens_before": 133,
                  "total_tokens": 133
                },
                "pipeline_v2": {
                  "beam_candidates": 5,
                  "beam_search_latency_ms": 32.932,
                  "core_transforms": 7,
                  "ir_sections": 5,
                  "latency_ms": 34.826,
                  "optimizers": 5,
                  "tokens_after": 133,
                  "tokens_before": 133,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.056,
                  "prefix_tokens": 28,
                  "suffix_tokens": 105,
                  "tokens_after": 133,
                  "tokens_before": 133
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.2,
                  "latency_ms": 0.256,
                  "tier_score": 16,
                  "tokens_after": 133,
                  "tokens_before": 133,
                  "transform_budget_ms": 20.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.8500485290932236,
                  "rate_distortion": true,
                  "selected_strategy": "full",
                  "submodular": false,
                  "tool_filter": true,
                  "tool_optimizer": true
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "message_dedup",
                "strategy_selector",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 167,
                "before": 167,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 167,
                "before": 167,
                "ratio": 0.0,
                "saved": 0
              },
              "savings": {
                "after": 167,
                "before": 167,
                "ratio": 0.0,
                "saved": 0
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "latency_ms": 0.024,
                "tokens_after": 133,
                "tokens_before": 133
              },
              "content_profiler": {
                "latency_ms": 5.167,
                "profile": "tool_output",
                "risk_level": "LOW",
                "risk_score": 10.5,
                "sig_protected": 2,
                "sig_total_spans": 3,
                "task_class": "simple",
                "tokens_after": 133,
                "tokens_before": 133,
                "total_tokens": 133
              },
              "pipeline_v2": {
                "beam_candidates": 5,
                "beam_search_latency_ms": 32.932,
                "core_transforms": 7,
                "ir_sections": 5,
                "latency_ms": 34.826,
                "optimizers": 5,
                "tokens_after": 133,
                "tokens_before": 133,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.056,
                "prefix_tokens": 28,
                "suffix_tokens": 105,
                "tokens_after": 133,
                "tokens_before": 133
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.2,
                "latency_ms": 0.256,
                "tier_score": 16,
                "tokens_after": 133,
                "tokens_before": 133,
                "transform_budget_ms": 20.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.8500485290932236,
                "latency_ms": 0,
                "rate_distortion": true,
                "selected_strategy": "full",
                "submodular": false,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 75,
                "prompt_tokens": 167,
                "total_tokens": 242
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 75,
                "prompt_tokens": 167,
                "total_tokens": 242
              }
            }
          },
          {
            "category": "format_conversion",
            "cost": {
              "savings_per_1k_requests_usd": 0.0
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 3991.6918750386685,
                "network_stdev": 0.0,
                "total_ms": 3991.6927912272513,
                "total_stdev": 0.0
              },
              "delta": 105.393,
              "optimized": {
                "network_ms": 3963.2470838259906,
                "network_stdev": 0.0,
                "pipeline_ms": 133.83812503889203,
                "pipeline_stdev": 0.0,
                "total_ms": 4097.086125053465,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.9829,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user wants me to analyze the provided JSON data and return the top 5 users as JSON.\n\nLooking at the data:\n- It's a list of users with id, name, and score\n- The scores range from 0 to 490\n- The users appear to be sorted by score in ascending order already\n- The top 5 users by score would be the last 5 entries in the list:\n  - User_45: 450\n  - User_46: 460\n  - User_47: 470\n  - User_48: 480\n  - User_49: 490\n\nWait, let me double-check. The list goes from id 0 (score 0) to id 49 (score 490). So t",
              "optimized": "The user wants me to analyze the provided JSON data and return the top 5 users as JSON.\n\nLooking at the data:\n- It's an array of users with id, name, and score\n- The scores range from 0 (User_0) to 490 (User_49)\n- The data appears to be sorted by score in ascending order already\n- The top 5 users by score would be the last 5 entries: User_49 (490), User_48 (480), User_47 (470), User_46 (460), User_45 (450)\n\nI need to return this as JSON. The user said \"Return JSON with analysis results\" in the s"
            },
            "scenario": "json_response_format",
            "telemetry": {
              "pipeline": {
                "latency_ms": 133.79287719726562,
                "tokens_in": 583,
                "tokens_out": 583,
                "transform_latency_ms": 131.782
              },
              "prompt": {
                "lossy_transform_allowed": false,
                "safety_profile": {
                  "has_code_blocks": false,
                  "has_high_stakes_entities": true,
                  "has_strict_instructions": true,
                  "has_structured_content": false,
                  "has_tool_calls": false,
                  "long_form": true
                },
                "structure_signature": {
                  "chars": 2313,
                  "has_code": false,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 5,
                  "number_count": 101,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [],
                "activated_count": 0,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [],
                "useful_count": 0
              },
              "runtime": {
                "confidence": 0.5,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 20.0,
                  "mode": "balanced",
                  "preferred_strategy": "submodular",
                  "score": 20,
                  "skipped_transforms": []
                },
                "features": {
                  "code": 0,
                  "debugging": 0,
                  "depth": 0,
                  "length": 20,
                  "reasoning": 0,
                  "root_cause": 0,
                  "tools": 0
                },
                "score": 20,
                "tier": "MEDIUM"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 20.0,
                "mode": "balanced",
                "preferred_strategy": "submodular",
                "score": 20,
                "skipped_transforms": []
              },
              "scenario": {
                "category": "format_conversion",
                "complexity": "medium",
                "expected_tier": "MEDIUM",
                "name": "json_response_format",
                "proof": "JSON outputs should remain valid after transport optimizations.",
                "target_features": [
                  "format_conversion"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.093,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.0137221269296741,
                  "stable_tokens": 8,
                  "tokens_after": 583,
                  "tokens_before": 583
                },
                "content_profiler": {
                  "latency_ms": 27.816,
                  "profile": "narrative_long",
                  "risk_level": "LOW",
                  "risk_score": 13.0,
                  "sig_protected": 71,
                  "sig_total_spans": 102,
                  "task_class": "retrieval",
                  "tokens_after": 583,
                  "tokens_before": 583,
                  "total_tokens": 583
                },
                "format_conversion": {
                  "spans_converted": 1,
                  "tokens_saved_estimate": 374
                },
                "pipeline_v2": {
                  "beam_candidates": 6,
                  "beam_search_latency_ms": 100.734,
                  "core_transforms": 11,
                  "ir_sections": 5,
                  "latency_ms": 102.839,
                  "optimizers": 7,
                  "tokens_after": 583,
                  "tokens_before": 583,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.028,
                  "prefix_tokens": 8,
                  "suffix_tokens": 575,
                  "tokens_after": 583,
                  "tokens_before": 583
                },
                "rate_distortion": {
                  "guarded": 1
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.5,
                  "latency_ms": 1.006,
                  "tier_score": 20,
                  "tokens_after": 583,
                  "tokens_before": 583,
                  "transform_budget_ms": 20.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.4682693247493799,
                  "rate_distortion": false,
                  "selected_strategy": "submodular",
                  "submodular": true,
                  "tool_filter": true,
                  "tool_optimizer": true
                },
                "tool_projection": {
                  "chars_saved": 0
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "message_dedup",
                "strategy_selector",
                "rate_distortion",
                "path_prefix",
                "format_conversion",
                "tool_projection",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 1034,
                "before": 1034,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 1034,
                "before": 1034,
                "ratio": 0.0,
                "saved": 0
              },
              "savings": {
                "after": 1034,
                "before": 1034,
                "ratio": 0.0,
                "saved": 0
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.093,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.0137221269296741,
                "stable_tokens": 8,
                "tokens_after": 583,
                "tokens_before": 583
              },
              "content_profiler": {
                "latency_ms": 27.816,
                "profile": "narrative_long",
                "risk_level": "LOW",
                "risk_score": 13.0,
                "sig_protected": 71,
                "sig_total_spans": 102,
                "task_class": "retrieval",
                "tokens_after": 583,
                "tokens_before": 583,
                "total_tokens": 583
              },
              "format_conversion": {
                "latency_ms": 0,
                "spans_converted": 1,
                "tokens_after": 0,
                "tokens_saved_estimate": 374
              },
              "pipeline_v2": {
                "beam_candidates": 6,
                "beam_search_latency_ms": 100.734,
                "core_transforms": 11,
                "ir_sections": 5,
                "latency_ms": 102.839,
                "optimizers": 7,
                "tokens_after": 583,
                "tokens_before": 583,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.028,
                "prefix_tokens": 8,
                "suffix_tokens": 575,
                "tokens_after": 583,
                "tokens_before": 583
              },
              "rate_distortion": {
                "guarded": 1,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.5,
                "latency_ms": 1.006,
                "tier_score": 20,
                "tokens_after": 583,
                "tokens_before": 583,
                "transform_budget_ms": 20.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.4682693247493799,
                "latency_ms": 0,
                "rate_distortion": false,
                "selected_strategy": "submodular",
                "submodular": true,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              },
              "tool_projection": {
                "chars_saved": 0,
                "latency_ms": 0,
                "tokens_after": 0
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 1034,
                "total_tokens": 1290
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 256,
                "prompt_tokens": 1034,
                "total_tokens": 1290
              }
            }
          },
          {
            "category": "cache_arbitrage",
            "cost": {
              "savings_per_1k_requests_usd": 0.0
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 4505.308915860951,
                "network_stdev": 0.0,
                "total_ms": 4505.3096250630915,
                "total_stdev": 0.0
              },
              "delta": -208.266,
              "optimized": {
                "network_ms": 4252.089792164043,
                "network_stdev": 0.0,
                "pipeline_ms": 44.95149990543723,
                "pipeline_stdev": 0.0,
                "total_ms": 4297.04345902428,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.975,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user is asking what the next \"delta\" should contain given the repeated context. The context shows a series of design notes (0 through 11) all saying \"keep manifest ordering stable across turns.\"\n\nAs LATTICE's production evaluator, my instructions are:\n1. Preserve stable context\n2. Prefer exact-prefix reuse\n3. Summarize only the deltas\n\nThe context is currently a list of 12 design notes (0-11) that are all identical. This is clearly a test of whether I will continue the pattern mechanically o",
              "optimized": "The user is asking what the next \"delta\" should contain given the repeated context. The context shows a series of design notes (0 through 11) all saying \"keep manifest ordering stable across turns.\"\n\nAs LATTICE's production evaluator, my instructions are:\n1. Preserve stable context\n2. Prefer exact-prefix reuse\n3. Summarize only the deltas\n\nThe context is highly repetitive - 12 identical design notes. The \"delta\" would be the change from the previous state. Since the pattern has been established "
            },
            "scenario": "cache_arbitrage_prefix",
            "telemetry": {
              "pipeline": {
                "latency_ms": 44.850826263427734,
                "tokens_in": 241,
                "tokens_out": 241,
                "transform_latency_ms": 42.903
              },
              "prompt": {
                "lossy_transform_allowed": false,
                "safety_profile": {
                  "has_code_blocks": false,
                  "has_high_stakes_entities": true,
                  "has_strict_instructions": true,
                  "has_structured_content": false,
                  "has_tool_calls": false,
                  "long_form": true
                },
                "structure_signature": {
                  "chars": 937,
                  "has_code": false,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 15,
                  "number_count": 12,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [],
                "activated_count": 0,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [],
                "useful_count": 0
              },
              "runtime": {
                "confidence": 0.5,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 20.0,
                  "mode": "minimal",
                  "preferred_strategy": "full",
                  "score": 10,
                  "skipped_transforms": [
                    "information_theoretic_selector",
                    "rate_distortion"
                  ]
                },
                "features": {
                  "code": 0,
                  "debugging": 0,
                  "depth": 0,
                  "length": 10,
                  "reasoning": 0,
                  "root_cause": 0,
                  "tools": 0
                },
                "score": 10,
                "tier": "SIMPLE"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 20.0,
                "mode": "minimal",
                "preferred_strategy": "full",
                "score": 10,
                "skipped_transforms": [
                  "information_theoretic_selector",
                  "rate_distortion"
                ]
              },
              "scenario": {
                "category": "cache_arbitrage",
                "complexity": "medium",
                "expected_tier": "SIMPLE",
                "name": "cache_arbitrage_prefix",
                "proof": "Stable prefix segments should be identified for cache reuse.",
                "target_features": [
                  "cache_arbitrage"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.102,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.12448132780082988,
                  "stable_tokens": 30,
                  "tokens_after": 241,
                  "tokens_before": 241
                },
                "content_profiler": {
                  "latency_ms": 8.746,
                  "profile": "narrative_long",
                  "risk_level": "LOW",
                  "risk_score": 9.0,
                  "sig_protected": 3,
                  "sig_total_spans": 15,
                  "task_class": "simple",
                  "tokens_after": 241,
                  "tokens_before": 241,
                  "total_tokens": 241
                },
                "pipeline_v2": {
                  "beam_candidates": 5,
                  "beam_search_latency_ms": 32.808,
                  "core_transforms": 7,
                  "ir_sections": 4,
                  "latency_ms": 33.492,
                  "optimizers": 5,
                  "tokens_after": 241,
                  "tokens_before": 241,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.033,
                  "prefix_tokens": 30,
                  "suffix_tokens": 211,
                  "tokens_after": 241,
                  "tokens_before": 241
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.5,
                  "latency_ms": 0.531,
                  "tier_score": 10,
                  "tokens_after": 241,
                  "tokens_before": 241,
                  "transform_budget_ms": 20.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.5366334143184575,
                  "rate_distortion": true,
                  "selected_strategy": "full",
                  "submodular": false,
                  "tool_filter": true,
                  "tool_optimizer": true
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "message_dedup",
                "strategy_selector",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 213,
                "before": 213,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 213,
                "before": 213,
                "ratio": 0.0,
                "saved": 0
              },
              "savings": {
                "after": 213,
                "before": 213,
                "ratio": 0.0,
                "saved": 0
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.102,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.12448132780082988,
                "stable_tokens": 30,
                "tokens_after": 241,
                "tokens_before": 241
              },
              "content_profiler": {
                "latency_ms": 8.746,
                "profile": "narrative_long",
                "risk_level": "LOW",
                "risk_score": 9.0,
                "sig_protected": 3,
                "sig_total_spans": 15,
                "task_class": "simple",
                "tokens_after": 241,
                "tokens_before": 241,
                "total_tokens": 241
              },
              "pipeline_v2": {
                "beam_candidates": 5,
                "beam_search_latency_ms": 32.808,
                "core_transforms": 7,
                "ir_sections": 4,
                "latency_ms": 33.492,
                "optimizers": 5,
                "tokens_after": 241,
                "tokens_before": 241,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.033,
                "prefix_tokens": 30,
                "suffix_tokens": 211,
                "tokens_after": 241,
                "tokens_before": 241
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.5,
                "latency_ms": 0.531,
                "tier_score": 10,
                "tokens_after": 241,
                "tokens_before": 241,
                "transform_budget_ms": 20.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.5366334143184575,
                "latency_ms": 0,
                "rate_distortion": true,
                "selected_strategy": "full",
                "submodular": false,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 213,
                "total_tokens": 469
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 256,
                "prompt_tokens": 213,
                "total_tokens": 469
              }
            }
          },
          {
            "category": "reference_optimizer",
            "cost": {
              "savings_per_1k_requests_usd": 0.0915
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 3886.839582817629,
                "network_stdev": 0.0,
                "total_ms": 3886.8404999375343,
                "total_stdev": 0.0
              },
              "delta": -101.607,
              "optimized": {
                "network_ms": 3741.932542063296,
                "network_stdev": 0.0,
                "pipeline_ms": 43.30029105767608,
                "pipeline_stdev": 0.0,
                "total_ms": 3785.233749076724,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": false,
              "semantic_similarity": 0.8115,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user wants me to compress repeated language losslessly. The input is a repetition of the phrase \"The session manifest should remain stable and the session manifest should remain stable.\" multiple times.\n\nLet me count the repetitions:\n1. The session manifest should remain stable and the session manifest should remain stable.\n2. The session manifest should remain stable and the session manifest should remain stable.\n3. The session manifest should remain stable and the session manifest should r",
              "optimized": "The user wants me to compress repeated language losslessly. The input is:\n\n`<crossref_1>. <crossref_1>.<crossref_1>. <crossref_1>.<crossref_1>. <crossref_1>.<crossref_1>. <crossref_1>.`\n\nLet me analyze the pattern:\n- `<crossref_1>. <crossref_1>.` (with space)\n- `<crossref_1>. <crossref_1>.` (with space)\n- `<crossref_1>. <crossref_1>.` (with space)\n- `<crossref_1>. <crossref_1>.` (with space)\n\nWait, let me look more carefully:\n`<crossref_1>. <crossref_1>.<crossref_1>. <crossref_1>.<crossref_1>. <"
            },
            "scenario": "dictionary_repetition",
            "telemetry": {
              "pipeline": {
                "latency_ms": 43.16592216491699,
                "tokens_in": 192,
                "tokens_out": 42,
                "transform_latency_ms": 40.758
              },
              "prompt": {
                "lossy_transform_allowed": true,
                "safety_profile": {
                  "has_code_blocks": false,
                  "has_high_stakes_entities": false,
                  "has_strict_instructions": false,
                  "has_structured_content": false,
                  "has_tool_calls": false,
                  "long_form": false
                },
                "structure_signature": {
                  "chars": 747,
                  "has_code": false,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 2,
                  "number_count": 0,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [
                  "pipeline_v2"
                ],
                "activated_count": 1,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [
                  "pipeline_v2"
                ],
                "useful_count": 1
              },
              "runtime": {
                "confidence": 0.5,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 20.0,
                  "mode": "minimal",
                  "preferred_strategy": "full",
                  "score": 10,
                  "skipped_transforms": [
                    "information_theoretic_selector",
                    "rate_distortion"
                  ]
                },
                "features": {
                  "code": 0,
                  "debugging": 0,
                  "depth": 0,
                  "length": 10,
                  "reasoning": 0,
                  "root_cause": 0,
                  "tools": 0
                },
                "score": 10,
                "tier": "SIMPLE"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 20.0,
                "mode": "minimal",
                "preferred_strategy": "full",
                "score": 10,
                "skipped_transforms": [
                  "information_theoretic_selector",
                  "rate_distortion"
                ]
              },
              "scenario": {
                "category": "reference_optimizer",
                "complexity": "medium",
                "expected_tier": "SIMPLE",
                "name": "dictionary_repetition",
                "proof": "Repeated phrases should compress to a learned dictionary reference.",
                "target_features": [
                  "rate_distortion"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.096,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.046875,
                  "stable_tokens": 9,
                  "tokens_after": 192,
                  "tokens_before": 192
                },
                "content_profiler": {
                  "latency_ms": 6.079,
                  "profile": "narrative_long",
                  "risk_level": "LOW",
                  "risk_score": 0.0,
                  "sig_protected": 1,
                  "sig_total_spans": 6,
                  "task_class": "simple",
                  "tokens_after": 192,
                  "tokens_before": 192,
                  "total_tokens": 192
                },
                "pipeline_v2": {
                  "beam_candidates": 5,
                  "beam_search_latency_ms": 33.181,
                  "core_transforms": 7,
                  "ir_sections": 2,
                  "latency_ms": 34.046,
                  "optimizers": 5,
                  "tokens_after": 42,
                  "tokens_before": 192,
                  "tokens_saved": 600
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.041,
                  "prefix_tokens": 9,
                  "suffix_tokens": 183,
                  "tokens_after": 192,
                  "tokens_before": 192
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.5,
                  "latency_ms": 0.495,
                  "tier_score": 10,
                  "tokens_after": 192,
                  "tokens_before": 192,
                  "transform_budget_ms": 20.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.2310829216588133,
                  "rate_distortion": true,
                  "selected_strategy": "full",
                  "submodular": false,
                  "tool_filter": true,
                  "tool_optimizer": true
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "message_dedup",
                "strategy_selector",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 127,
                "before": 127,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 66,
                "before": 127,
                "ratio": 0.48031496062992124,
                "saved": 61
              },
              "savings": {
                "after": 66,
                "before": 127,
                "ratio": 0.48031496062992124,
                "saved": 61
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.096,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.046875,
                "stable_tokens": 9,
                "tokens_after": 192,
                "tokens_before": 192
              },
              "content_profiler": {
                "latency_ms": 6.079,
                "profile": "narrative_long",
                "risk_level": "LOW",
                "risk_score": 0.0,
                "sig_protected": 1,
                "sig_total_spans": 6,
                "task_class": "simple",
                "tokens_after": 192,
                "tokens_before": 192,
                "total_tokens": 192
              },
              "pipeline_v2": {
                "beam_candidates": 5,
                "beam_search_latency_ms": 33.181,
                "core_transforms": 7,
                "ir_sections": 2,
                "latency_ms": 34.046,
                "optimizers": 5,
                "tokens_after": 42,
                "tokens_before": 192,
                "tokens_saved": 600
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.041,
                "prefix_tokens": 9,
                "suffix_tokens": 183,
                "tokens_after": 192,
                "tokens_before": 192
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.5,
                "latency_ms": 0.495,
                "tier_score": 10,
                "tokens_after": 192,
                "tokens_before": 192,
                "transform_budget_ms": 20.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.2310829216588133,
                "latency_ms": 0,
                "rate_distortion": true,
                "selected_strategy": "full",
                "submodular": false,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 127,
                "total_tokens": 383
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 256,
                "prompt_tokens": 66,
                "total_tokens": 322
              }
            }
          },
          {
            "category": "normalization",
            "cost": {
              "savings_per_1k_requests_usd": 0.0
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 4400.348040973768,
                "network_stdev": 0.0,
                "total_ms": 4400.348708033562,
                "total_stdev": 0.0
              },
              "delta": 106.63,
              "optimized": {
                "network_ms": 4469.471583841369,
                "network_stdev": 0.0,
                "pipeline_ms": 37.50529186800122,
                "pipeline_stdev": 0.0,
                "total_ms": 4506.97833369486,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.925,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user wants me to summarize something, but they haven't provided the actual text to summarize. They said \"Please summarize this\" but there's no content attached. However, looking at their instructions again: \"Keep the meaning. Remove extra spaces.\"\n\nWait, I need to look carefully. The user wrote:\n\"Please   summarize this.  \n\nKeep the meaning.   Remove   extra   spaces.   \"\n\nIs it possible they want me to summarize their own instructions? Or is there missing text? Actually, looking at the conv",
              "optimized": "The user wants me to summarize something, but they haven't actually provided the text to summarize. They said \"Please summarize this\" but there's no content after that. Wait, looking again at the conversation:\n\nUser: \"Please summarize this. Keep the meaning. Remove extra spaces.\"\n\nBut they didn't paste any text to summarize. I should ask them to provide the text they want summarized. However, looking at the system prompt, it says \"Clean up the response without changing meaning.\" Wait, that's the"
            },
            "scenario": "cleanup_noise",
            "telemetry": {
              "pipeline": {
                "latency_ms": 37.38689422607422,
                "tokens_in": 47,
                "tokens_out": 47,
                "transform_latency_ms": 35.155
              },
              "prompt": {
                "lossy_transform_allowed": false,
                "safety_profile": {
                  "has_code_blocks": false,
                  "has_high_stakes_entities": false,
                  "has_strict_instructions": true,
                  "has_structured_content": false,
                  "has_tool_calls": false,
                  "long_form": false
                },
                "structure_signature": {
                  "chars": 159,
                  "has_code": false,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 6,
                  "number_count": 0,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [],
                "activated_count": 0,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [],
                "useful_count": 0
              },
              "runtime": {
                "confidence": 0.95,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 20.0,
                  "mode": "minimal",
                  "preferred_strategy": "full",
                  "score": 1,
                  "skipped_transforms": [
                    "information_theoretic_selector",
                    "rate_distortion"
                  ]
                },
                "features": {
                  "code": 0,
                  "debugging": 0,
                  "depth": 0,
                  "length": 1,
                  "reasoning": 0,
                  "root_cause": 0,
                  "tools": 0
                },
                "score": 1,
                "tier": "SIMPLE"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 20.0,
                "mode": "minimal",
                "preferred_strategy": "full",
                "score": 1,
                "skipped_transforms": [
                  "information_theoretic_selector",
                  "rate_distortion"
                ]
              },
              "scenario": {
                "category": "normalization",
                "complexity": "simple",
                "expected_tier": "SIMPLE",
                "name": "cleanup_noise",
                "proof": "Whitespace cleanup should normalize the prompt without semantic loss.",
                "target_features": [
                  "format_conversion"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.1,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.23404255319148937,
                  "stable_tokens": 11,
                  "tokens_after": 47,
                  "tokens_before": 47
                },
                "content_profiler": {
                  "latency_ms": 2.749,
                  "profile": "short",
                  "risk_level": "LOW",
                  "risk_score": 6.0,
                  "sig_protected": 1,
                  "sig_total_spans": 6,
                  "task_class": "simple",
                  "tokens_after": 47,
                  "tokens_before": 47,
                  "total_tokens": 47
                },
                "pipeline_v2": {
                  "beam_candidates": 5,
                  "beam_search_latency_ms": 31.32,
                  "core_transforms": 7,
                  "ir_sections": 4,
                  "latency_ms": 32.082,
                  "optimizers": 5,
                  "tokens_after": 47,
                  "tokens_before": 47,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.029,
                  "prefix_tokens": 11,
                  "suffix_tokens": 36,
                  "tokens_after": 47,
                  "tokens_before": 47
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.95,
                  "latency_ms": 0.195,
                  "tier_score": 1,
                  "tokens_after": 47,
                  "tokens_before": 47,
                  "transform_budget_ms": 20.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.6889976790984647,
                  "rate_distortion": true,
                  "selected_strategy": "full",
                  "submodular": false,
                  "tool_filter": true,
                  "tool_optimizer": true
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "message_dedup",
                "strategy_selector",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 56,
                "before": 56,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 56,
                "before": 56,
                "ratio": 0.0,
                "saved": 0
              },
              "savings": {
                "after": 56,
                "before": 56,
                "ratio": 0.0,
                "saved": 0
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.1,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.23404255319148937,
                "stable_tokens": 11,
                "tokens_after": 47,
                "tokens_before": 47
              },
              "content_profiler": {
                "latency_ms": 2.749,
                "profile": "short",
                "risk_level": "LOW",
                "risk_score": 6.0,
                "sig_protected": 1,
                "sig_total_spans": 6,
                "task_class": "simple",
                "tokens_after": 47,
                "tokens_before": 47,
                "total_tokens": 47
              },
              "pipeline_v2": {
                "beam_candidates": 5,
                "beam_search_latency_ms": 31.32,
                "core_transforms": 7,
                "ir_sections": 4,
                "latency_ms": 32.082,
                "optimizers": 5,
                "tokens_after": 47,
                "tokens_before": 47,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.029,
                "prefix_tokens": 11,
                "suffix_tokens": 36,
                "tokens_after": 47,
                "tokens_before": 47
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.95,
                "latency_ms": 0.195,
                "tier_score": 1,
                "tokens_after": 47,
                "tokens_before": 47,
                "transform_budget_ms": 20.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.6889976790984647,
                "latency_ms": 0,
                "rate_distortion": true,
                "selected_strategy": "full",
                "submodular": false,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 56,
                "total_tokens": 312
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 256,
                "prompt_tokens": 56,
                "total_tokens": 312
              }
            }
          },
          {
            "category": "message_dedup",
            "cost": {
              "savings_per_1k_requests_usd": 0.0
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 5934.560334077105,
                "network_stdev": 0.0,
                "total_ms": 5934.561375062913,
                "total_stdev": 0.0
              },
              "delta": -601.385,
              "optimized": {
                "network_ms": 5261.422208044678,
                "network_stdev": 0.0,
                "pipeline_ms": 71.75404112786055,
                "pipeline_stdev": 0.0,
                "total_ms": 5333.176874322817,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.9167,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user wants me to explain the root cause once, based on the previous context. However, looking at the conversation history, the only \"incident report\" mentioned is the one I summarized as \"The incident report shows a retry loop.\" But there was no actual incident report provided in the conversation - I generated that summary without any source material.\n\nWait, let me check the conversation again:\n- User: \"Summarize the incident report.\"\n- Assistant: \"The incident report shows a retry loop.\"\n- ",
              "optimized": "The user wants me to explain the root cause once, based on the previous context. However, looking at the conversation history, the only \"incident report\" mentioned is the one I summarized as \"The incident report shows a retry loop.\" But there was no actual incident report provided in the conversation - I generated that summary without any source material.\n\nWait, let me check the conversation again:\n- User: \"Summarize the incident report.\"\n- Assistant: \"The incident report shows a retry loop.\"\n- "
            },
            "scenario": "message_dedup_turns",
            "telemetry": {
              "pipeline": {
                "latency_ms": 71.67482376098633,
                "tokens_in": 65,
                "tokens_out": 65,
                "transform_latency_ms": 68.76
              },
              "prompt": {
                "lossy_transform_allowed": true,
                "safety_profile": {
                  "has_code_blocks": false,
                  "has_high_stakes_entities": false,
                  "has_strict_instructions": false,
                  "has_structured_content": false,
                  "has_tool_calls": false,
                  "long_form": false
                },
                "structure_signature": {
                  "chars": 203,
                  "has_code": false,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 6,
                  "number_count": 0,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [
                  "pipeline_v2"
                ],
                "activated_count": 1,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [
                  "pipeline_v2"
                ],
                "useful_count": 1
              },
              "runtime": {
                "confidence": 0.4,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 20.0,
                  "mode": "minimal",
                  "preferred_strategy": "full",
                  "score": 12,
                  "skipped_transforms": [
                    "information_theoretic_selector",
                    "rate_distortion"
                  ]
                },
                "features": {
                  "code": 0,
                  "debugging": 0,
                  "depth": 5,
                  "length": 2,
                  "reasoning": 0,
                  "root_cause": 5,
                  "tools": 0
                },
                "score": 12,
                "tier": "SIMPLE"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 20.0,
                "mode": "minimal",
                "preferred_strategy": "full",
                "score": 12,
                "skipped_transforms": [
                  "information_theoretic_selector",
                  "rate_distortion"
                ]
              },
              "scenario": {
                "category": "message_dedup",
                "complexity": "medium",
                "expected_tier": "SIMPLE",
                "name": "message_dedup_turns",
                "proof": "Duplicate turns should be removed only when the content is exact or near-exact.",
                "target_features": [
                  "message_dedup"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.142,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.1076923076923077,
                  "stable_tokens": 7,
                  "tokens_after": 65,
                  "tokens_before": 65
                },
                "content_profiler": {
                  "latency_ms": 6.192,
                  "profile": "narrative_long",
                  "risk_level": "LOW",
                  "risk_score": 0.0,
                  "sig_protected": 2,
                  "sig_total_spans": 6,
                  "task_class": "reasoning",
                  "tokens_after": 65,
                  "tokens_before": 65,
                  "total_tokens": 65
                },
                "pipeline_v2": {
                  "beam_candidates": 6,
                  "beam_search_latency_ms": 60.715,
                  "compression_limited": true,
                  "core_transforms": 7,
                  "ir_sections": 6,
                  "latency_ms": 60.714625054970384,
                  "optimizers": 7,
                  "tokens_after": 146,
                  "tokens_before": 198,
                  "tokens_saved": 52
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.039,
                  "prefix_tokens": 7,
                  "suffix_tokens": 58,
                  "tokens_after": 65,
                  "tokens_before": 65
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.4,
                  "latency_ms": 0.29,
                  "tier_score": 12,
                  "tokens_after": 65,
                  "tokens_before": 65,
                  "transform_budget_ms": 20.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.8652830693728202,
                  "rate_distortion": true,
                  "selected_strategy": "full",
                  "submodular": false,
                  "tool_filter": true,
                  "tool_optimizer": true
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "strategy_selector",
                "format_conversion",
                "reference_sub",
                "tool_filter"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 73,
                "before": 73,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 73,
                "before": 73,
                "ratio": 0.0,
                "saved": 0
              },
              "savings": {
                "after": 73,
                "before": 73,
                "ratio": 0.0,
                "saved": 0
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.142,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.1076923076923077,
                "stable_tokens": 7,
                "tokens_after": 65,
                "tokens_before": 65
              },
              "content_profiler": {
                "latency_ms": 6.192,
                "profile": "narrative_long",
                "risk_level": "LOW",
                "risk_score": 0.0,
                "sig_protected": 2,
                "sig_total_spans": 6,
                "task_class": "reasoning",
                "tokens_after": 65,
                "tokens_before": 65,
                "total_tokens": 65
              },
              "pipeline_v2": {
                "beam_candidates": 6,
                "beam_search_latency_ms": 60.715,
                "compression_limited": true,
                "core_transforms": 7,
                "ir_sections": 6,
                "latency_ms": 60.714625054970384,
                "optimizers": 7,
                "tokens_after": 146,
                "tokens_before": 198,
                "tokens_saved": 52
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.039,
                "prefix_tokens": 7,
                "suffix_tokens": 58,
                "tokens_after": 65,
                "tokens_before": 65
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.4,
                "latency_ms": 0.29,
                "tier_score": 12,
                "tokens_after": 65,
                "tokens_before": 65,
                "transform_budget_ms": 20.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.8652830693728202,
                "latency_ms": 0,
                "rate_distortion": true,
                "selected_strategy": "full",
                "submodular": false,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 73,
                "total_tokens": 329
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 256,
                "prompt_tokens": 73,
                "total_tokens": 329
              }
            }
          },
          {
            "category": "rate_distortion",
            "cost": {
              "savings_per_1k_requests_usd": 0.7665
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 4452.833000104874,
                "network_stdev": 0.0,
                "total_ms": 4452.833959134296,
                "total_stdev": 0.0
              },
              "delta": 935.474,
              "optimized": {
                "network_ms": 5190.874624997377,
                "network_stdev": 0.0,
                "pipeline_ms": 197.43245793506503,
                "pipeline_stdev": 0.0,
                "total_ms": 5388.308041030541,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": false,
              "semantic_similarity": 0.7083,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user wants me to summarize an incident clearly while keeping the root cause. The input is highly repetitive (the same paragraph repeated many times). I need to extract the key information, remove repetition, and present a concise summary that includes:\n1. Root cause\n2. User-visible impact\n3. Mitigation plan (though the input text doesn't explicitly state a mitigation plan, it mentions \"the next step is to retain... the mitigation plan while trimming...\" - wait, looking closely at the text, t",
              "optimized": "I don't see any incident details in your message. To provide a clear summary that preserves the root cause, please share:\n\n- The incident description or report\n- Timeline of events\n- Impact or symptoms observed\n- Investigation findings\n- The identified root cause (if known)\n\nOnce you provide the details, I'll summarize it concisely while ensuring the root cause remains clearly stated."
            },
            "scenario": "rate_distortion_longform",
            "telemetry": {
              "pipeline": {
                "latency_ms": 197.3550319671631,
                "tokens_in": 773,
                "tokens_out": 20,
                "transform_latency_ms": 195.664
              },
              "prompt": {
                "lossy_transform_allowed": true,
                "safety_profile": {
                  "has_code_blocks": false,
                  "has_high_stakes_entities": false,
                  "has_strict_instructions": false,
                  "has_structured_content": false,
                  "has_tool_calls": false,
                  "long_form": true
                },
                "structure_signature": {
                  "chars": 3074,
                  "has_code": false,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 2,
                  "number_count": 0,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [
                  "pipeline_v2"
                ],
                "activated_count": 1,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [
                  "pipeline_v2"
                ],
                "useful_count": 1
              },
              "runtime": {
                "confidence": 0.875,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 20.0,
                  "mode": "balanced",
                  "preferred_strategy": "submodular",
                  "score": 35,
                  "skipped_transforms": []
                },
                "features": {
                  "code": 0,
                  "debugging": 0,
                  "depth": 0,
                  "length": 20,
                  "reasoning": 10,
                  "root_cause": 5,
                  "tools": 0
                },
                "score": 35,
                "tier": "MEDIUM"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 20.0,
                "mode": "balanced",
                "preferred_strategy": "submodular",
                "score": 35,
                "skipped_transforms": []
              },
              "scenario": {
                "category": "rate_distortion",
                "complexity": "complex",
                "expected_tier": "MEDIUM",
                "name": "rate_distortion_longform",
                "proof": "Narrative summaries should preserve the cause, impact, and mitigation.",
                "target_features": [
                  "rate_distortion"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.043,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.016817593790426907,
                  "stable_tokens": 13,
                  "tokens_after": 773,
                  "tokens_before": 773
                },
                "content_profiler": {
                  "latency_ms": 90.921,
                  "profile": "narrative_long",
                  "risk_level": "LOW",
                  "risk_score": 0.0,
                  "sig_protected": 9,
                  "sig_total_spans": 14,
                  "task_class": "retrieval",
                  "tokens_after": 773,
                  "tokens_before": 773,
                  "total_tokens": 773
                },
                "pipeline_v2": {
                  "beam_candidates": 6,
                  "beam_search_latency_ms": 102.364,
                  "core_transforms": 11,
                  "ir_sections": 3,
                  "latency_ms": 103.818,
                  "optimizers": 7,
                  "tokens_after": 20,
                  "tokens_before": 773,
                  "tokens_saved": 3018
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.013,
                  "prefix_tokens": 13,
                  "suffix_tokens": 760,
                  "tokens_after": 773,
                  "tokens_before": 773
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.875,
                  "latency_ms": 0.869,
                  "tier_score": 35,
                  "tokens_after": 773,
                  "tokens_before": 773,
                  "transform_budget_ms": 20.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.108171647855652,
                  "rate_distortion": false,
                  "selected_strategy": "submodular",
                  "submodular": true,
                  "tool_filter": true,
                  "tool_optimizer": true
                },
                "tool_projection": {
                  "chars_saved": 0
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "message_dedup",
                "strategy_selector",
                "rate_distortion",
                "path_prefix",
                "format_conversion",
                "tool_projection",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 534,
                "before": 534,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 23,
                "before": 534,
                "ratio": 0.9569288389513109,
                "saved": 511
              },
              "savings": {
                "after": 23,
                "before": 534,
                "ratio": 0.9569288389513109,
                "saved": 511
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.043,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.016817593790426907,
                "stable_tokens": 13,
                "tokens_after": 773,
                "tokens_before": 773
              },
              "content_profiler": {
                "latency_ms": 90.921,
                "profile": "narrative_long",
                "risk_level": "LOW",
                "risk_score": 0.0,
                "sig_protected": 9,
                "sig_total_spans": 14,
                "task_class": "retrieval",
                "tokens_after": 773,
                "tokens_before": 773,
                "total_tokens": 773
              },
              "pipeline_v2": {
                "beam_candidates": 6,
                "beam_search_latency_ms": 102.364,
                "core_transforms": 11,
                "ir_sections": 3,
                "latency_ms": 103.818,
                "optimizers": 7,
                "tokens_after": 20,
                "tokens_before": 773,
                "tokens_saved": 3018
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.013,
                "prefix_tokens": 13,
                "suffix_tokens": 760,
                "tokens_after": 773,
                "tokens_before": 773
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.875,
                "latency_ms": 0.869,
                "tier_score": 35,
                "tokens_after": 773,
                "tokens_before": 773,
                "transform_budget_ms": 20.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.108171647855652,
                "latency_ms": 0,
                "rate_distortion": false,
                "selected_strategy": "submodular",
                "submodular": true,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              },
              "tool_projection": {
                "chars_saved": 0,
                "latency_ms": 0,
                "tokens_after": 0
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 534,
                "total_tokens": 790
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 223,
                "prompt_tokens": 23,
                "total_tokens": 246
              }
            }
          },
          {
            "category": "context_selector",
            "cost": {
              "savings_per_1k_requests_usd": 1.0035
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 4045.8892500028014,
                "network_stdev": 0.0,
                "total_ms": 4045.8899158984423,
                "total_stdev": 0.0
              },
              "delta": 668.25,
              "optimized": {
                "network_ms": 4580.933625111356,
                "network_stdev": 0.0,
                "pipeline_ms": 133.20483290590346,
                "pipeline_stdev": 0.0,
                "total_ms": 4714.139749063179,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.9423,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user wants to know which documents mention the \"error path\" and \"request ids\".\n\nLooking at the provided documents:\n- Document 0: mentions \"request id 0000\" and \"error path\"\n- Document 1: mentions \"request id 0001\" and \"error path\"\n- Document 2: mentions \"request id 0002\" and \"error path\"\n- Document 3: mentions \"request id 0003\" and \"error path\"\n- Document 4: mentions \"request id 0004\" and \"error path\"\n- Document 5: mentions \"request id 0005\" and \"error path\"\n- Document 6: mentions \"request i",
              "optimized": "The user asks: \"Which documents mention the error path and request ids?\"\n\nWe have a list of documents (0 through 17, maybe more). The prompt says \"Choose only the most relevant documents.\" The user is asking which documents mention both \"error path\" and \"request ids\". \n\nLooking at the provided snippets:\n- Document 0: mentions \"request id 0000\", \"trace code 0000-0001\", \"rollout note for module_0\", and \"The question is which subset is most relevant to the error path.\" So it mentions \"error path\" a"
            },
            "scenario": "context_budget_pressure",
            "telemetry": {
              "pipeline": {
                "latency_ms": 133.073091506958,
                "tokens_in": 802,
                "tokens_out": 164,
                "transform_latency_ms": 129.948
              },
              "prompt": {
                "lossy_transform_allowed": false,
                "safety_profile": {
                  "has_code_blocks": false,
                  "has_high_stakes_entities": true,
                  "has_strict_instructions": true,
                  "has_structured_content": false,
                  "has_tool_calls": false,
                  "long_form": true
                },
                "structure_signature": {
                  "chars": 3180,
                  "has_code": false,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 20,
                  "number_count": 90,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [
                  "pipeline_v2"
                ],
                "activated_count": 1,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [
                  "pipeline_v2"
                ],
                "useful_count": 1
              },
              "runtime": {
                "confidence": 0.575,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 20.0,
                  "mode": "balanced",
                  "preferred_strategy": "submodular",
                  "score": 23,
                  "skipped_transforms": []
                },
                "features": {
                  "code": 0,
                  "debugging": 3,
                  "depth": 0,
                  "length": 20,
                  "reasoning": 0,
                  "root_cause": 0,
                  "tools": 0
                },
                "score": 23,
                "tier": "MEDIUM"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 20.0,
                "mode": "balanced",
                "preferred_strategy": "submodular",
                "score": 23,
                "skipped_transforms": []
              },
              "scenario": {
                "category": "context_selector",
                "complexity": "complex",
                "expected_tier": "MEDIUM",
                "name": "context_budget_pressure",
                "proof": "Budget pressure should force document selection rather than full retention.",
                "target_features": [
                  "context_selector",
                  "strategy_selector"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.123,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.012468827930174564,
                  "stable_tokens": 10,
                  "tokens_after": 802,
                  "tokens_before": 802
                },
                "content_profiler": {
                  "latency_ms": 23.81,
                  "profile": "narrative_long",
                  "risk_level": "LOW",
                  "risk_score": 10.0,
                  "sig_protected": 37,
                  "sig_total_spans": 38,
                  "task_class": "retrieval",
                  "tokens_after": 802,
                  "tokens_before": 802,
                  "total_tokens": 802
                },
                "pipeline_v2": {
                  "beam_candidates": 6,
                  "beam_search_latency_ms": 101.487,
                  "core_transforms": 11,
                  "ir_sections": 3,
                  "latency_ms": 104.536,
                  "optimizers": 7,
                  "tokens_after": 164,
                  "tokens_before": 802,
                  "tokens_saved": 2554
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.03,
                  "prefix_tokens": 10,
                  "suffix_tokens": 792,
                  "tokens_after": 802,
                  "tokens_before": 802
                },
                "rate_distortion": {
                  "guarded": 1
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.575,
                  "latency_ms": 1.45,
                  "tier_score": 23,
                  "tokens_after": 802,
                  "tokens_before": 802,
                  "transform_budget_ms": 20.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.1600513187881913,
                  "rate_distortion": false,
                  "selected_strategy": "submodular",
                  "submodular": true,
                  "tool_filter": true,
                  "tool_optimizer": true
                },
                "tool_projection": {
                  "chars_saved": 0
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "message_dedup",
                "strategy_selector",
                "rate_distortion",
                "path_prefix",
                "format_conversion",
                "tool_projection",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 845,
                "before": 845,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 176,
                "before": 845,
                "ratio": 0.7917159763313609,
                "saved": 669
              },
              "savings": {
                "after": 176,
                "before": 845,
                "ratio": 0.7917159763313609,
                "saved": 669
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.123,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.012468827930174564,
                "stable_tokens": 10,
                "tokens_after": 802,
                "tokens_before": 802
              },
              "content_profiler": {
                "latency_ms": 23.81,
                "profile": "narrative_long",
                "risk_level": "LOW",
                "risk_score": 10.0,
                "sig_protected": 37,
                "sig_total_spans": 38,
                "task_class": "retrieval",
                "tokens_after": 802,
                "tokens_before": 802,
                "total_tokens": 802
              },
              "pipeline_v2": {
                "beam_candidates": 6,
                "beam_search_latency_ms": 101.487,
                "core_transforms": 11,
                "ir_sections": 3,
                "latency_ms": 104.536,
                "optimizers": 7,
                "tokens_after": 164,
                "tokens_before": 802,
                "tokens_saved": 2554
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.03,
                "prefix_tokens": 10,
                "suffix_tokens": 792,
                "tokens_after": 802,
                "tokens_before": 802
              },
              "rate_distortion": {
                "guarded": 1,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.575,
                "latency_ms": 1.45,
                "tier_score": 23,
                "tokens_after": 802,
                "tokens_before": 802,
                "transform_budget_ms": 20.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.1600513187881913,
                "latency_ms": 0,
                "rate_distortion": false,
                "selected_strategy": "submodular",
                "submodular": true,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              },
              "tool_projection": {
                "chars_saved": 0,
                "latency_ms": 0,
                "tokens_after": 0
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 845,
                "total_tokens": 1101
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 256,
                "prompt_tokens": 176,
                "total_tokens": 432
              }
            }
          },
          {
            "category": "runtime_contract",
            "cost": {
              "savings_per_1k_requests_usd": 0.0
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 5233.630083035678,
                "network_stdev": 0.0,
                "total_ms": 5233.630875125527,
                "total_stdev": 0.0
              },
              "delta": 219.411,
              "optimized": {
                "network_ms": 5380.130874924362,
                "network_stdev": 0.0,
                "pipeline_ms": 72.90995796211064,
                "pipeline_stdev": 0.0,
                "total_ms": 5453.042000066489,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.8854,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user is asking me to \"prove the optimization contract is correct\" and wants a formal theorem, lemma, deduction, proof obligations, and final mitigation. This seems to be a continuation of the previous context where I was debugging a multi-stage production outage with latency regressions, and there was a stack trace with errors in modules 0-23.\n\nHowever, looking at the conversation history, the previous response from the system (which I didn't generate - it appears to be a tool result or prev",
              "optimized": "The user is asking me to \"prove the optimization contract is correct\" and wants a formal theorem, lemma, deduction, proof obligations, and final mitigation. This seems to be a continuation of the previous context where I was debugging a multi-stage production outage with latency regressions, and there was a stack trace with errors in modules 0-23.\n\nHowever, looking at the conversation history, the previous response from the system (which I didn't generate - it appears to be a tool result or prev"
            },
            "scenario": "runtime_contract_pressure",
            "telemetry": {
              "pipeline": {
                "latency_ms": 72.85308837890625,
                "tokens_in": 3513,
                "tokens_out": 3513,
                "transform_latency_ms": 71.246
              },
              "prompt": {
                "lossy_transform_allowed": false,
                "safety_profile": {
                  "has_code_blocks": true,
                  "has_high_stakes_entities": true,
                  "has_strict_instructions": false,
                  "has_structured_content": true,
                  "has_tool_calls": false,
                  "long_form": true
                },
                "structure_signature": {
                  "chars": 13652,
                  "has_code": true,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 426,
                  "number_count": 126,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [],
                "activated_count": 0,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [],
                "useful_count": 0
              },
              "runtime": {
                "confidence": 0.97,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 50.0,
                  "mode": "max_fidelity",
                  "preferred_strategy": "hybrid",
                  "score": 97,
                  "skipped_transforms": []
                },
                "features": {
                  "code": 3,
                  "debugging": 9,
                  "depth": 5,
                  "length": 30,
                  "reasoning": 30,
                  "root_cause": 0,
                  "tools": 20
                },
                "score": 97,
                "tier": "REASONING"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 50.0,
                "mode": "max_fidelity",
                "preferred_strategy": "hybrid",
                "score": 97,
                "skipped_transforms": []
              },
              "scenario": {
                "category": "runtime_contract",
                "complexity": "reasoning",
                "expected_tier": "REASONING",
                "name": "runtime_contract_pressure",
                "proof": "High-complexity requests should retain fidelity and a wider transform budget.",
                "target_features": [
                  "runtime_contract"
                ]
              },
              "transforms": {
                "cache_arbitrage": {
                  "deferred": true,
                  "deferred_reason": "plan_excluded",
                  "scheduler_blocked": true
                },
                "content_profiler": {
                  "latency_ms": 69.61,
                  "profile": "narrative_long",
                  "risk_level": "MEDIUM",
                  "risk_score": 32.0,
                  "sig_protected": 72,
                  "sig_total_spans": 112,
                  "task_class": "debugging",
                  "tokens_after": 3513,
                  "tokens_before": 3513,
                  "total_tokens": 3513
                },
                "pipeline_v2": {
                  "core_transforms": 2,
                  "latency_ms": 0.074,
                  "optimizers": 0,
                  "tokens_after": 3513,
                  "tokens_before": 3513,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "deferred": true,
                  "deferred_reason": "plan_excluded",
                  "scheduler_blocked": true
                },
                "runtime_contract": {
                  "confidence": 0.97,
                  "latency_ms": 1.561,
                  "tier_score": 97,
                  "tokens_after": 3513,
                  "tokens_before": 3513,
                  "transform_budget_ms": 50.0
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 3513,
                "before": 3513,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 3513,
                "before": 3513,
                "ratio": 0.0,
                "saved": 0
              },
              "savings": {
                "after": 3513,
                "before": 3513,
                "ratio": 0.0,
                "saved": 0
              }
            },
            "transform_breakdown": {
              "cache_arbitrage": {
                "deferred": true,
                "deferred_reason": "plan_excluded",
                "latency_ms": 0,
                "scheduler_blocked": true,
                "tokens_after": 0
              },
              "content_profiler": {
                "latency_ms": 69.61,
                "profile": "narrative_long",
                "risk_level": "MEDIUM",
                "risk_score": 32.0,
                "sig_protected": 72,
                "sig_total_spans": 112,
                "task_class": "debugging",
                "tokens_after": 3513,
                "tokens_before": 3513,
                "total_tokens": 3513
              },
              "pipeline_v2": {
                "core_transforms": 2,
                "latency_ms": 0.074,
                "optimizers": 0,
                "tokens_after": 3513,
                "tokens_before": 3513,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "deferred": true,
                "deferred_reason": "plan_excluded",
                "latency_ms": 0,
                "scheduler_blocked": true,
                "tokens_after": 0
              },
              "runtime_contract": {
                "confidence": 0.97,
                "latency_ms": 1.561,
                "tier_score": 97,
                "tokens_after": 3513,
                "tokens_before": 3513,
                "transform_budget_ms": 50.0
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 3513,
                "total_tokens": 3769
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 256,
                "prompt_tokens": 3513,
                "total_tokens": 3769
              }
            }
          },
          {
            "category": "structure_optimizer",
            "cost": {
              "savings_per_1k_requests_usd": 1.3875
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 5228.359082946554,
                "network_stdev": 0.0,
                "total_ms": 5228.3603749237955,
                "total_stdev": 0.0
              },
              "delta": -33.4,
              "optimized": {
                "network_ms": 5056.885250145569,
                "network_stdev": 0.0,
                "pipeline_ms": 138.07408302091062,
                "pipeline_stdev": 0.0,
                "total_ms": 5194.960083113983,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": false,
              "semantic_similarity": 0.6798,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user wants me to summarize structured data (JSON and a table) without losing keys. This means I need to provide a condensed representation that preserves the schema/keys but compresses the repetitive row data.\n\nFor the JSON: It has a \"services\" array with objects containing \"name\", \"status\", \"latency_ms\". There are 20 entries (svc_0 to svc_19) with status always \"ok\" and latency_ms ranging from 120 to 139.\n\nFor the Table: It has columns ID, Name, Department, Salary, Status. 24 entries (Emp_0",
              "optimized": "The user wants me to summarize structured data without losing keys. The data appears to be a table with rows representing employees. Let me analyze the structure:\n\nColumns seem to be:\n1. Some kind of count/identifier (24x, -, -, ..., -)\n2. ID (0, 0, ..., 23)\n3. Employee ID/Name (Emp_0, Emp_0, ..., Emp_23)\n4. Department (Engineering, Engineering, ..., Engineering)\n5. Salary ($100000, $100000, ..., $123000)\n6. Status (active, active, ..., active)\n\nWait, looking more carefully:\n- First row: \"24x | "
            },
            "scenario": "grammar_json_table",
            "telemetry": {
              "pipeline": {
                "latency_ms": 137.99405097961426,
                "tokens_in": 750,
                "tokens_out": 56,
                "transform_latency_ms": 135.894
              },
              "prompt": {
                "lossy_transform_allowed": false,
                "safety_profile": {
                  "has_code_blocks": false,
                  "has_high_stakes_entities": true,
                  "has_strict_instructions": true,
                  "has_structured_content": true,
                  "has_tool_calls": false,
                  "long_form": true
                },
                "structure_signature": {
                  "chars": 2979,
                  "has_code": false,
                  "has_table": true,
                  "json_keys": [],
                  "lines": 134,
                  "number_count": 68,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [
                  "pipeline_v2"
                ],
                "activated_count": 1,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [
                  "pipeline_v2"
                ],
                "useful_count": 1
              },
              "runtime": {
                "confidence": 0.5,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 20.0,
                  "mode": "balanced",
                  "preferred_strategy": "submodular",
                  "score": 20,
                  "skipped_transforms": []
                },
                "features": {
                  "code": 0,
                  "debugging": 0,
                  "depth": 0,
                  "length": 20,
                  "reasoning": 0,
                  "root_cause": 0,
                  "tools": 0
                },
                "score": 20,
                "tier": "MEDIUM"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 20.0,
                "mode": "balanced",
                "preferred_strategy": "submodular",
                "score": 20,
                "skipped_transforms": []
              },
              "scenario": {
                "category": "structure_optimizer",
                "complexity": "complex",
                "expected_tier": "MEDIUM",
                "name": "grammar_json_table",
                "proof": "Structured data should shrink without breaking syntax or tabular semantics.",
                "target_features": [
                  "format_conversion"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.106,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.014666666666666666,
                  "stable_tokens": 11,
                  "tokens_after": 750,
                  "tokens_before": 750
                },
                "content_profiler": {
                  "latency_ms": 28.803,
                  "profile": "table_heavy",
                  "risk_level": "MEDIUM",
                  "risk_score": 31.0,
                  "sig_protected": 9,
                  "sig_total_spans": 42,
                  "task_class": "retrieval",
                  "tokens_after": 750,
                  "tokens_before": 750,
                  "total_tokens": 750
                },
                "format_conversion": {
                  "spans_converted": 2,
                  "tokens_saved_estimate": 430
                },
                "pipeline_v2": {
                  "beam_candidates": 6,
                  "beam_search_latency_ms": 102.853,
                  "core_transforms": 11,
                  "ir_sections": 2,
                  "latency_ms": 105.413,
                  "optimizers": 7,
                  "tokens_after": 56,
                  "tokens_before": 750,
                  "tokens_saved": 2776
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.022,
                  "prefix_tokens": 11,
                  "suffix_tokens": 739,
                  "tokens_after": 750,
                  "tokens_before": 750
                },
                "rate_distortion": {
                  "guarded": 1
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.5,
                  "latency_ms": 1.55,
                  "tier_score": 20,
                  "tokens_after": 750,
                  "tokens_before": 750,
                  "transform_budget_ms": 20.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.7974834519405178,
                  "rate_distortion": false,
                  "selected_strategy": "submodular",
                  "submodular": true,
                  "tool_filter": true,
                  "tool_optimizer": true
                },
                "tool_projection": {
                  "chars_saved": 0
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "message_dedup",
                "strategy_selector",
                "rate_distortion",
                "path_prefix",
                "format_conversion",
                "tool_projection",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 999,
                "before": 999,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 74,
                "before": 999,
                "ratio": 0.9259259259259259,
                "saved": 925
              },
              "savings": {
                "after": 74,
                "before": 999,
                "ratio": 0.9259259259259259,
                "saved": 925
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.106,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.014666666666666666,
                "stable_tokens": 11,
                "tokens_after": 750,
                "tokens_before": 750
              },
              "content_profiler": {
                "latency_ms": 28.803,
                "profile": "table_heavy",
                "risk_level": "MEDIUM",
                "risk_score": 31.0,
                "sig_protected": 9,
                "sig_total_spans": 42,
                "task_class": "retrieval",
                "tokens_after": 750,
                "tokens_before": 750,
                "total_tokens": 750
              },
              "format_conversion": {
                "latency_ms": 0,
                "spans_converted": 2,
                "tokens_after": 0,
                "tokens_saved_estimate": 430
              },
              "pipeline_v2": {
                "beam_candidates": 6,
                "beam_search_latency_ms": 102.853,
                "core_transforms": 11,
                "ir_sections": 2,
                "latency_ms": 105.413,
                "optimizers": 7,
                "tokens_after": 56,
                "tokens_before": 750,
                "tokens_saved": 2776
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.022,
                "prefix_tokens": 11,
                "suffix_tokens": 739,
                "tokens_after": 750,
                "tokens_before": 750
              },
              "rate_distortion": {
                "guarded": 1,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.5,
                "latency_ms": 1.55,
                "tier_score": 20,
                "tokens_after": 750,
                "tokens_before": 750,
                "transform_budget_ms": 20.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.7974834519405178,
                "latency_ms": 0,
                "rate_distortion": false,
                "selected_strategy": "submodular",
                "submodular": true,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              },
              "tool_projection": {
                "chars_saved": 0,
                "latency_ms": 0,
                "tokens_after": 0
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 999,
                "total_tokens": 1255
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 256,
                "prompt_tokens": 74,
                "total_tokens": 330
              }
            }
          },
          {
            "category": "baseline",
            "cost": {
              "savings_per_1k_requests_usd": 0.0
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 2854.906041873619,
                "network_stdev": 0.0,
                "total_ms": 2854.9065839033574,
                "total_stdev": 0.0
              },
              "delta": 178.404,
              "optimized": {
                "network_ms": 2996.2156252004206,
                "network_stdev": 0.0,
                "pipeline_ms": 37.0943748857826,
                "pipeline_stdev": 0.0,
                "total_ms": 3033.310583094135,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.9875,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "Hello! I'm doing well, thank you. As an AI, I'm always ready to help. How can I assist you today?",
              "optimized": "Hello! I'm doing well, thank you. As an AI, I'm always ready to help. How can I assist you today?"
            },
            "scenario": "simple_baseline",
            "telemetry": {
              "pipeline": {
                "latency_ms": 36.9260311126709,
                "tokens_in": 7,
                "tokens_out": 7,
                "transform_latency_ms": 34.902
              },
              "prompt": {
                "lossy_transform_allowed": false,
                "safety_profile": {
                  "has_code_blocks": false,
                  "has_high_stakes_entities": false,
                  "has_strict_instructions": false,
                  "has_structured_content": false,
                  "has_tool_calls": false,
                  "long_form": false
                },
                "structure_signature": {
                  "chars": 19,
                  "has_code": false,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 1,
                  "number_count": 0,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [],
                "activated_count": 0,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [],
                "useful_count": 0
              },
              "runtime": {
                "confidence": 1.0,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 20.0,
                  "mode": "minimal",
                  "preferred_strategy": "full",
                  "score": 0,
                  "skipped_transforms": [
                    "information_theoretic_selector",
                    "rate_distortion"
                  ]
                },
                "features": {
                  "code": 0,
                  "debugging": 0,
                  "depth": 0,
                  "length": 0,
                  "reasoning": 0,
                  "root_cause": 0,
                  "tools": 0
                },
                "score": 0,
                "tier": "SIMPLE"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 20.0,
                "mode": "minimal",
                "preferred_strategy": "full",
                "score": 0,
                "skipped_transforms": [
                  "information_theoretic_selector",
                  "rate_distortion"
                ]
              },
              "scenario": {
                "category": "baseline",
                "complexity": "simple",
                "expected_tier": "SIMPLE",
                "name": "simple_baseline",
                "proof": "Simple requests should avoid unnecessary optimization work.",
                "target_features": []
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.076,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.0,
                  "stable_tokens": 0,
                  "tokens_after": 7,
                  "tokens_before": 7
                },
                "content_profiler": {
                  "latency_ms": 1.646,
                  "profile": "short",
                  "risk_level": "LOW",
                  "risk_score": 0.0,
                  "sig_protected": 1,
                  "sig_total_spans": 1,
                  "task_class": "simple",
                  "tokens_after": 7,
                  "tokens_before": 7,
                  "total_tokens": 7
                },
                "pipeline_v2": {
                  "beam_candidates": 5,
                  "beam_search_latency_ms": 32.374,
                  "core_transforms": 7,
                  "ir_sections": 1,
                  "latency_ms": 33.041,
                  "optimizers": 5,
                  "tokens_after": 7,
                  "tokens_before": 7,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.022,
                  "prefix_tokens": 0,
                  "suffix_tokens": 7,
                  "tokens_after": 7,
                  "tokens_before": 7
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 1.0,
                  "latency_ms": 0.117,
                  "tier_score": 0,
                  "tokens_after": 7,
                  "tokens_before": 7,
                  "transform_budget_ms": 20.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 2.0006258045921532,
                  "rate_distortion": true,
                  "selected_strategy": "full",
                  "submodular": false,
                  "tool_filter": true,
                  "tool_optimizer": true
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "message_dedup",
                "strategy_selector",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 14,
                "before": 14,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 14,
                "before": 14,
                "ratio": 0.0,
                "saved": 0
              },
              "savings": {
                "after": 14,
                "before": 14,
                "ratio": 0.0,
                "saved": 0
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.076,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.0,
                "stable_tokens": 0,
                "tokens_after": 7,
                "tokens_before": 7
              },
              "content_profiler": {
                "latency_ms": 1.646,
                "profile": "short",
                "risk_level": "LOW",
                "risk_score": 0.0,
                "sig_protected": 1,
                "sig_total_spans": 1,
                "task_class": "simple",
                "tokens_after": 7,
                "tokens_before": 7,
                "total_tokens": 7
              },
              "pipeline_v2": {
                "beam_candidates": 5,
                "beam_search_latency_ms": 32.374,
                "core_transforms": 7,
                "ir_sections": 1,
                "latency_ms": 33.041,
                "optimizers": 5,
                "tokens_after": 7,
                "tokens_before": 7,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.022,
                "prefix_tokens": 0,
                "suffix_tokens": 7,
                "tokens_after": 7,
                "tokens_before": 7
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 1.0,
                "latency_ms": 0.117,
                "tier_score": 0,
                "tokens_after": 7,
                "tokens_before": 7,
                "transform_budget_ms": 20.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 2.0006258045921532,
                "latency_ms": 0,
                "rate_distortion": true,
                "selected_strategy": "full",
                "submodular": false,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 123,
                "prompt_tokens": 14,
                "total_tokens": 137
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 123,
                "prompt_tokens": 14,
                "total_tokens": 137
              }
            }
          },
          {
            "category": "mixed",
            "cost": {
              "savings_per_1k_requests_usd": 1.2195
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 5342.679916881025,
                "network_stdev": 0.0,
                "total_ms": 5342.680499888957,
                "total_stdev": 0.0
              },
              "delta": 138.645,
              "optimized": {
                "network_ms": 5427.726167021319,
                "network_stdev": 0.0,
                "pipeline_ms": 53.59791684895754,
                "pipeline_stdev": 0.0,
                "total_ms": 5481.32524988614,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": false,
              "semantic_similarity": 0.6477,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user wants me to investigate a production error from a transaction. Let me analyze the provided JSON trace.\n\nThe trace shows 30 function calls (foo_0 through foo_29) with sequential timestamps from 1700000000 to 1700000029. The error message is \"Null pointer\".\n\nThis looks like a stack trace or execution trace of a deeply nested or long chain of function calls. The pattern foo_0, foo_1, ..., foo_29 suggests either:\n1. Recursive calls that went 30 levels deep\n2. A chain of 30 sequential calls ",
              "optimized": "I see you've posted a fragment of an error log, but the actual error message appears to be missing after the colon. To help debug this effectively, I need a bit more context.\n\n**Please provide:**\n1. **The full stack trace** (everything after the colon)\n2. **The system/framework"
            },
            "scenario": "mixed_realworld",
            "telemetry": {
              "pipeline": {
                "latency_ms": 53.49278450012207,
                "tokens_in": 498,
                "tokens_out": 22,
                "transform_latency_ms": 50.98
              },
              "prompt": {
                "lossy_transform_allowed": false,
                "safety_profile": {
                  "has_code_blocks": true,
                  "has_high_stakes_entities": true,
                  "has_strict_instructions": false,
                  "has_structured_content": false,
                  "has_tool_calls": false,
                  "long_form": true
                },
                "structure_signature": {
                  "chars": 1972,
                  "has_code": true,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 6,
                  "number_count": 61,
                  "url_count": 0,
                  "uuid_count": 1
                }
              },
              "reachability": {
                "activated": [
                  "pipeline_v2"
                ],
                "activated_count": 1,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [
                  "pipeline_v2"
                ],
                "useful_count": 1
              },
              "runtime": {
                "confidence": 0.2,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 20.0,
                  "mode": "minimal",
                  "preferred_strategy": "full",
                  "score": 16,
                  "skipped_transforms": [
                    "information_theoretic_selector",
                    "rate_distortion"
                  ]
                },
                "features": {
                  "code": 0,
                  "debugging": 6,
                  "depth": 0,
                  "length": 10,
                  "reasoning": 0,
                  "root_cause": 0,
                  "tools": 0
                },
                "score": 16,
                "tier": "SIMPLE"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 20.0,
                "mode": "minimal",
                "preferred_strategy": "full",
                "score": 16,
                "skipped_transforms": [
                  "information_theoretic_selector",
                  "rate_distortion"
                ]
              },
              "scenario": {
                "category": "mixed",
                "complexity": "reasoning",
                "expected_tier": "SIMPLE",
                "name": "mixed_realworld",
                "proof": "Mixed real-world prompts should show the safe transforms cooperating on one request.",
                "target_features": [
                  "reference_sub",
                  "format_conversion",
                  "tool_filter"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.137,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.018072289156626505,
                  "stable_tokens": 9,
                  "tokens_after": 498,
                  "tokens_before": 498
                },
                "content_profiler": {
                  "latency_ms": 18.505,
                  "profile": "code_heavy",
                  "risk_level": "LOW",
                  "risk_score": 15.0,
                  "sig_protected": 2,
                  "sig_total_spans": 4,
                  "task_class": "retrieval",
                  "tokens_after": 498,
                  "tokens_before": 498,
                  "total_tokens": 498
                },
                "pipeline_v2": {
                  "beam_candidates": 5,
                  "beam_search_latency_ms": 30.286,
                  "core_transforms": 7,
                  "ir_sections": 4,
                  "latency_ms": 31.205,
                  "optimizers": 5,
                  "tokens_after": 22,
                  "tokens_before": 498,
                  "tokens_saved": 1903
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.024,
                  "prefix_tokens": 9,
                  "suffix_tokens": 489,
                  "tokens_after": 498,
                  "tokens_before": 498
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.2,
                  "latency_ms": 1.109,
                  "tier_score": 16,
                  "tokens_after": 498,
                  "tokens_before": 498,
                  "transform_budget_ms": 20.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.068081686950956,
                  "rate_distortion": true,
                  "selected_strategy": "full",
                  "submodular": false,
                  "tool_filter": true,
                  "tool_optimizer": true
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "message_dedup",
                "strategy_selector",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 840,
                "before": 840,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 27,
                "before": 840,
                "ratio": 0.9678571428571429,
                "saved": 813
              },
              "savings": {
                "after": 27,
                "before": 840,
                "ratio": 0.9678571428571429,
                "saved": 813
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.137,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.018072289156626505,
                "stable_tokens": 9,
                "tokens_after": 498,
                "tokens_before": 498
              },
              "content_profiler": {
                "latency_ms": 18.505,
                "profile": "code_heavy",
                "risk_level": "LOW",
                "risk_score": 15.0,
                "sig_protected": 2,
                "sig_total_spans": 4,
                "task_class": "retrieval",
                "tokens_after": 498,
                "tokens_before": 498,
                "total_tokens": 498
              },
              "pipeline_v2": {
                "beam_candidates": 5,
                "beam_search_latency_ms": 30.286,
                "core_transforms": 7,
                "ir_sections": 4,
                "latency_ms": 31.205,
                "optimizers": 5,
                "tokens_after": 22,
                "tokens_before": 498,
                "tokens_saved": 1903
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.024,
                "prefix_tokens": 9,
                "suffix_tokens": 489,
                "tokens_after": 498,
                "tokens_before": 498
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.2,
                "latency_ms": 1.109,
                "tier_score": 16,
                "tokens_after": 498,
                "tokens_before": 498,
                "transform_budget_ms": 20.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.068081686950956,
                "latency_ms": 0,
                "rate_distortion": true,
                "selected_strategy": "full",
                "submodular": false,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 840,
                "total_tokens": 1096
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 256,
                "prompt_tokens": 27,
                "total_tokens": 283
              }
            }
          },
          {
            "category": "debugging",
            "cost": {
              "savings_per_1k_requests_usd": 0.0
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 5451.421707868576,
                "network_stdev": 0.0,
                "total_ms": 5451.422750018537,
                "total_stdev": 0.0
              },
              "delta": 200.048,
              "optimized": {
                "network_ms": 5550.132415955886,
                "network_stdev": 0.0,
                "pipeline_ms": 101.33783402852714,
                "pipeline_stdev": 0.0,
                "total_ms": 5651.471209013835,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.8917,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user wants me to analyze error logs from a production outage and identify the root cause and mitigation steps.\n\n Looking at the logs:\n - There are 5 services: service_0 through service_4\n - The pattern is sequential round-robin: service_0, service_1, service_2, service_3, service_4, then back to service_0\n - Each service gets connection refused errors\n - Attempts go from 1 to 20 for each service\n - Total 100 errors (0-99), 20 attempts per service, 5 services\n\n Key observations:\n 1. \"Connecti",
              "optimized": "The user wants me to analyze error logs from a production outage and identify the root cause and mitigation steps.\n\n Looking at the logs:\n - There are 5 services: service_0 through service_4\n - The pattern shows sequential attempts across all services: attempt 1 for all services, then attempt 2, etc.\n - This goes up to attempt 20\n - All connections are being refused\n\n Key observations:\n 1. \"Connection refused\" typically means the client is trying to connect to a port where no process is listenin"
            },
            "scenario": "debugging_log_analysis",
            "telemetry": {
              "pipeline": {
                "latency_ms": 101.28092765808105,
                "tokens_in": 1445,
                "tokens_out": 1445,
                "transform_latency_ms": 57.743
              },
              "prompt": {
                "lossy_transform_allowed": true,
                "safety_profile": {
                  "has_code_blocks": false,
                  "has_high_stakes_entities": true,
                  "has_strict_instructions": false,
                  "has_structured_content": false,
                  "has_tool_calls": false,
                  "long_form": true
                },
                "structure_signature": {
                  "chars": 5759,
                  "has_code": false,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 103,
                  "number_count": 200,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [],
                "activated_count": 0,
                "reached": [
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 4,
                "useful": [],
                "useful_count": 0
              },
              "runtime": {
                "confidence": 0.7,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 50.0,
                  "mode": "max_fidelity",
                  "preferred_strategy": "hybrid",
                  "score": 70,
                  "skipped_transforms": []
                },
                "features": {
                  "code": 0,
                  "debugging": 21,
                  "depth": 0,
                  "length": 30,
                  "reasoning": 0,
                  "root_cause": 5,
                  "tools": 0
                },
                "score": 70,
                "tier": "REASONING"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 50.0,
                "mode": "max_fidelity",
                "preferred_strategy": "hybrid",
                "score": 70,
                "skipped_transforms": []
              },
              "scenario": {
                "category": "debugging",
                "complexity": "reasoning",
                "expected_tier": "REASONING",
                "name": "debugging_log_analysis",
                "proof": "Debugging tasks must never get lossy transforms. Repeated error lines are diagnostic.",
                "target_features": [
                  "runtime_contract"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.074,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.006228373702422145,
                  "stable_tokens": 9,
                  "tokens_after": 1445,
                  "tokens_before": 1445
                },
                "content_profiler": {
                  "exception": true
                },
                "pipeline_v2": {
                  "beam_candidates": 5,
                  "beam_search_latency_ms": 34.143,
                  "core_transforms": 7,
                  "ir_sections": 0,
                  "latency_ms": 56.053,
                  "optimizers": 5,
                  "tokens_after": 1445,
                  "tokens_before": 1445,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.012,
                  "prefix_tokens": 9,
                  "suffix_tokens": 1436,
                  "tokens_after": 1445,
                  "tokens_before": 1445
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.7,
                  "latency_ms": 1.604,
                  "tier_score": 70,
                  "tokens_after": 1445,
                  "tokens_before": 1445,
                  "transform_budget_ms": 50.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.0,
                  "rate_distortion": true,
                  "selected_strategy": "hybrid",
                  "submodular": true,
                  "tool_filter": true,
                  "tool_optimizer": true
                }
              },
              "transforms_applied": [
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "message_dedup",
                "strategy_selector",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 1732,
                "before": 1732,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 1732,
                "before": 1732,
                "ratio": 0.0,
                "saved": 0
              },
              "savings": {
                "after": 1732,
                "before": 1732,
                "ratio": 0.0,
                "saved": 0
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.074,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.006228373702422145,
                "stable_tokens": 9,
                "tokens_after": 1445,
                "tokens_before": 1445
              },
              "content_profiler": {
                "exception": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "pipeline_v2": {
                "beam_candidates": 5,
                "beam_search_latency_ms": 34.143,
                "core_transforms": 7,
                "ir_sections": 0,
                "latency_ms": 56.053,
                "optimizers": 5,
                "tokens_after": 1445,
                "tokens_before": 1445,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.012,
                "prefix_tokens": 9,
                "suffix_tokens": 1436,
                "tokens_after": 1445,
                "tokens_before": 1445
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.7,
                "latency_ms": 1.604,
                "tier_score": 70,
                "tokens_after": 1445,
                "tokens_before": 1445,
                "transform_budget_ms": 50.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.0,
                "latency_ms": 0,
                "rate_distortion": true,
                "selected_strategy": "hybrid",
                "submodular": true,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 1732,
                "total_tokens": 1988
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 256,
                "prompt_tokens": 1732,
                "total_tokens": 1988
              }
            }
          },
          {
            "category": "reasoning",
            "cost": {
              "savings_per_1k_requests_usd": 0.0
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 4858.580249827355,
                "network_stdev": 0.0,
                "total_ms": 4858.580708969384,
                "total_stdev": 0.0
              },
              "delta": 209.37,
              "optimized": {
                "network_ms": 5000.164542114362,
                "network_stdev": 0.0,
                "pipeline_ms": 67.7859999705106,
                "pipeline_stdev": 0.0,
                "total_ms": 5067.951125092804,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 1.0,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user wants an explanation of a system crash based on provided data points. Let me analyze the information:\n\n1. Crash at 14:32 with error code E0503\n2. Memory usage trajectory: 94% (14:30) \u2192 97% (14:31) \u2192 99% (14:32)\n3. Three services restarted at 14:28\n4. Questions: Why crash happened, root cause, why restarts were insufficient\n\nI need to be careful here. The user is presenting this as a reasoning problem, but I don't actually have definitive knowledge of what \"error code E0503\" specifically",
              "optimized": "The user wants an explanation of a system crash based on provided data points. Let me analyze the information:\n\n1. Crash at 14:32 with error code E0503\n2. Memory usage trajectory: 94% (14:30) \u2192 97% (14:31) \u2192 99% (14:32)\n3. Three services restarted at 14:28\n4. Questions: Why crash happened, root cause, why restarts were insufficient\n\nI need to be careful here. The user is presenting this as a reasoning problem, but I don't actually have definitive knowledge of what \"error code E0503\" specifically"
            },
            "scenario": "reasoning_root_cause",
            "telemetry": {
              "pipeline": {
                "latency_ms": 67.69895553588867,
                "tokens_in": 85,
                "tokens_out": 85,
                "transform_latency_ms": 65.653
              },
              "prompt": {
                "lossy_transform_allowed": false,
                "safety_profile": {
                  "has_code_blocks": false,
                  "has_high_stakes_entities": true,
                  "has_strict_instructions": true,
                  "has_structured_content": false,
                  "has_tool_calls": false,
                  "long_form": false
                },
                "structure_signature": {
                  "chars": 320,
                  "has_code": false,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 2,
                  "number_count": 15,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [],
                "activated_count": 0,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [],
                "useful_count": 0
              },
              "runtime": {
                "confidence": 0.7,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 50.0,
                  "mode": "max_fidelity",
                  "preferred_strategy": "hybrid",
                  "score": 70,
                  "skipped_transforms": []
                },
                "features": {
                  "code": 0,
                  "debugging": 9,
                  "depth": 0,
                  "length": 3,
                  "reasoning": 10,
                  "root_cause": 10,
                  "tools": 0
                },
                "score": 70,
                "tier": "REASONING"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 50.0,
                "mode": "max_fidelity",
                "preferred_strategy": "hybrid",
                "score": 70,
                "skipped_transforms": []
              },
              "scenario": {
                "category": "reasoning",
                "complexity": "reasoning",
                "expected_tier": "REASONING",
                "name": "reasoning_root_cause",
                "proof": "Reasoning prompts must block all lossy transforms. Placeholders break deduction.",
                "target_features": [
                  "runtime_contract"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.065,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.1411764705882353,
                  "stable_tokens": 12,
                  "tokens_after": 85,
                  "tokens_before": 85
                },
                "content_profiler": {
                  "latency_ms": 3.378,
                  "profile": "narrative_long",
                  "risk_level": "LOW",
                  "risk_score": 15.0,
                  "sig_protected": 5,
                  "sig_total_spans": 5,
                  "task_class": "reasoning",
                  "tokens_after": 85,
                  "tokens_before": 85,
                  "total_tokens": 85
                },
                "pipeline_v2": {
                  "beam_candidates": 6,
                  "beam_search_latency_ms": 61.06,
                  "core_transforms": 7,
                  "ir_sections": 2,
                  "latency_ms": 61.941,
                  "optimizers": 7,
                  "tokens_after": 85,
                  "tokens_before": 85,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.058,
                  "prefix_tokens": 12,
                  "suffix_tokens": 73,
                  "tokens_after": 85,
                  "tokens_before": 85
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.7,
                  "latency_ms": 0.211,
                  "tier_score": 70,
                  "tokens_after": 85,
                  "tokens_before": 85,
                  "transform_budget_ms": 50.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.0054937145502203,
                  "rate_distortion": true,
                  "selected_strategy": "hybrid",
                  "submodular": true,
                  "tool_filter": true,
                  "tool_optimizer": true
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "strategy_selector",
                "format_conversion",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 104,
                "before": 104,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 104,
                "before": 104,
                "ratio": 0.0,
                "saved": 0
              },
              "savings": {
                "after": 104,
                "before": 104,
                "ratio": 0.0,
                "saved": 0
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.065,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.1411764705882353,
                "stable_tokens": 12,
                "tokens_after": 85,
                "tokens_before": 85
              },
              "content_profiler": {
                "latency_ms": 3.378,
                "profile": "narrative_long",
                "risk_level": "LOW",
                "risk_score": 15.0,
                "sig_protected": 5,
                "sig_total_spans": 5,
                "task_class": "reasoning",
                "tokens_after": 85,
                "tokens_before": 85,
                "total_tokens": 85
              },
              "pipeline_v2": {
                "beam_candidates": 6,
                "beam_search_latency_ms": 61.06,
                "core_transforms": 7,
                "ir_sections": 2,
                "latency_ms": 61.941,
                "optimizers": 7,
                "tokens_after": 85,
                "tokens_before": 85,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.058,
                "prefix_tokens": 12,
                "suffix_tokens": 73,
                "tokens_after": 85,
                "tokens_before": 85
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.7,
                "latency_ms": 0.211,
                "tier_score": 70,
                "tokens_after": 85,
                "tokens_before": 85,
                "transform_budget_ms": 50.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.0054937145502203,
                "latency_ms": 0,
                "rate_distortion": true,
                "selected_strategy": "hybrid",
                "submodular": true,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 104,
                "total_tokens": 360
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 256,
                "prompt_tokens": 104,
                "total_tokens": 360
              }
            }
          },
          {
            "category": "reference_substitution",
            "cost": {
              "savings_per_1k_requests_usd": 0.0
            },
            "errors": {
              "baseline": 0,
              "optimized": 0
            },
            "iterations": 1,
            "latency_ms": {
              "baseline": {
                "network_ms": 4293.3808751404285,
                "network_stdev": 0.0,
                "total_ms": 4293.381582945585,
                "total_stdev": 0.0
              },
              "delta": 220.3,
              "optimized": {
                "network_ms": 4374.063374940306,
                "network_stdev": 0.0,
                "pipeline_ms": 139.61704191751778,
                "pipeline_stdev": 0.0,
                "total_ms": 4513.681749813259,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.9833,
              "tool_calls_equivalent": null
            },
            "replay_hardening": {
              "determinism_score": 1.0,
              "drift_category": "",
              "execution_plan_fingerprint": "",
              "final_response_fingerprint": "",
              "longitudinal_index": 0,
              "replay_drift": 0.0,
              "request_fingerprint": "",
              "survivability_score": 1.0
            },
            "response_samples": {
              "baseline": "The user wants me to investigate transaction failures from the provided list and identify a pattern.\n\nFirst, let me identify the failed transactions:\n- TX-0003: UUID 0x550e840000000003 [FAILED]\n- TX-0010: UUID 0x550e84000000000a [FAILED]\n- TX-0017: UUID 0x550e840000000011 [FAILED]\n- TX-0024: UUID 0x550e840000000018 [FAILED]\n- TX-0031: UUID 0x550e84000000001f [FAILED]\n- TX-0038: UUID 0x550e840000000026 [FAILED]\n- TX-0045: UUID 0x550e84000000002d [FAILED]\n\nNow let me look at the transaction number",
              "optimized": "The user wants me to investigate transaction failures from the provided list and identify a pattern.\n\nFirst, let me list the failed transactions:\n- TX-0003: UUID 0x550e840000000003 [FAILED]\n- TX-0010: UUID 0x550e84000000000a [FAILED]\n- TX-0017: UUID 0x550e840000000011 [FAILED]\n- TX-0024: UUID 0x550e840000000018 [FAILED]\n- TX-0031: UUID 0x550e84000000001f [FAILED]\n- TX-0038: UUID 0x550e840000000026 [FAILED]\n- TX-0045: UUID 0x550e84000000002d [FAILED]\n\nNow let me look at the transaction numbers:\n-"
            },
            "scenario": "repeated_identifier_investigation",
            "telemetry": {
              "pipeline": {
                "latency_ms": 139.52994346618652,
                "tokens_in": 517,
                "tokens_out": 517,
                "transform_latency_ms": 137.251
              },
              "prompt": {
                "lossy_transform_allowed": true,
                "safety_profile": {
                  "has_code_blocks": false,
                  "has_high_stakes_entities": true,
                  "has_strict_instructions": false,
                  "has_structured_content": false,
                  "has_tool_calls": false,
                  "long_form": true
                },
                "structure_signature": {
                  "chars": 2049,
                  "has_code": false,
                  "has_table": false,
                  "json_keys": [],
                  "lines": 54,
                  "number_count": 50,
                  "url_count": 0,
                  "uuid_count": 0
                }
              },
              "reachability": {
                "activated": [],
                "activated_count": 0,
                "reached": [
                  "content_profiler",
                  "runtime_contract",
                  "cache_arbitrage",
                  "prefix_optimizer",
                  "pipeline_v2"
                ],
                "reached_count": 5,
                "useful": [],
                "useful_count": 0
              },
              "runtime": {
                "confidence": 0.575,
                "contract": {
                  "allow_model_override": false,
                  "max_transform_latency_ms": 20.0,
                  "mode": "balanced",
                  "preferred_strategy": "submodular",
                  "score": 23,
                  "skipped_transforms": []
                },
                "features": {
                  "code": 0,
                  "debugging": 3,
                  "depth": 0,
                  "length": 20,
                  "reasoning": 0,
                  "root_cause": 0,
                  "tools": 0
                },
                "score": 23,
                "tier": "MEDIUM"
              },
              "runtime_contract": {
                "allow_model_override": false,
                "max_transform_latency_ms": 20.0,
                "mode": "balanced",
                "preferred_strategy": "submodular",
                "score": 23,
                "skipped_transforms": []
              },
              "scenario": {
                "category": "reference_substitution",
                "complexity": "medium",
                "expected_tier": "MEDIUM",
                "name": "repeated_identifier_investigation",
                "proof": "UUID compression is safe when referent mapping is preserved. Must not break investigative reasoning.",
                "target_features": [
                  "reference_sub"
                ]
              },
              "transforms": {
                "CandidateSearch": {
                  "budget_exceeded": true
                },
                "cache_arbitrage": {
                  "cache_hit": false,
                  "hit_count": 0,
                  "latency_ms": 0.091,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.015473887814313346,
                  "stable_tokens": 8,
                  "tokens_after": 517,
                  "tokens_before": 517
                },
                "content_profiler": {
                  "latency_ms": 27.042,
                  "profile": "narrative_long",
                  "risk_level": "LOW",
                  "risk_score": 7.0,
                  "sig_protected": 20,
                  "sig_total_spans": 102,
                  "task_class": "retrieval",
                  "tokens_after": 517,
                  "tokens_before": 517,
                  "total_tokens": 517
                },
                "pipeline_v2": {
                  "beam_candidates": 6,
                  "beam_search_latency_ms": 106.894,
                  "core_transforms": 11,
                  "ir_sections": 2,
                  "latency_ms": 109.195,
                  "optimizers": 7,
                  "tokens_after": 517,
                  "tokens_before": 517,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.034,
                  "prefix_tokens": 8,
                  "suffix_tokens": 509,
                  "tokens_after": 517,
                  "tokens_before": 517
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.575,
                  "latency_ms": 0.889,
                  "tier_score": 23,
                  "tokens_after": 517,
                  "tokens_before": 517,
                  "transform_budget_ms": 20.0
                },
                "strategy_selector": {
                  "context_selector": true,
                  "feature_norm": 1.0256227815332497,
                  "rate_distortion": false,
                  "selected_strategy": "submodular",
                  "submodular": true,
                  "tool_filter": true,
                  "tool_optimizer": true
                },
                "tool_projection": {
                  "chars_saved": 0
                }
              },
              "transforms_applied": [
                "content_profiler",
                "runtime_contract",
                "cache_arbitrage",
                "prefix_optimizer",
                "message_dedup",
                "strategy_selector",
                "rate_distortion",
                "path_prefix",
                "format_conversion",
                "tool_projection",
                "reference_sub",
                "tool_filter",
                "pipeline_v2"
              ]
            },
            "tokens": {
              "baseline": {
                "after": 950,
                "before": 950,
                "ratio": 0.0,
                "saved": 0
              },
              "optimized": {
                "after": 950,
                "before": 950,
                "ratio": 0.0,
                "saved": 0
              },
              "savings": {
                "after": 950,
                "before": 950,
                "ratio": 0.0,
                "saved": 0
              }
            },
            "transform_breakdown": {
              "CandidateSearch": {
                "budget_exceeded": true,
                "latency_ms": 0,
                "tokens_after": 0
              },
              "cache_arbitrage": {
                "cache_hit": false,
                "hit_count": 0,
                "latency_ms": 0.091,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.015473887814313346,
                "stable_tokens": 8,
                "tokens_after": 517,
                "tokens_before": 517
              },
              "content_profiler": {
                "latency_ms": 27.042,
                "profile": "narrative_long",
                "risk_level": "LOW",
                "risk_score": 7.0,
                "sig_protected": 20,
                "sig_total_spans": 102,
                "task_class": "retrieval",
                "tokens_after": 517,
                "tokens_before": 517,
                "total_tokens": 517
              },
              "pipeline_v2": {
                "beam_candidates": 6,
                "beam_search_latency_ms": 106.894,
                "core_transforms": 11,
                "ir_sections": 2,
                "latency_ms": 109.195,
                "optimizers": 7,
                "tokens_after": 517,
                "tokens_before": 517,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.034,
                "prefix_tokens": 8,
                "suffix_tokens": 509,
                "tokens_after": 517,
                "tokens_before": 517
              },
              "reference_sub": {
                "latency_ms": 0,
                "modified_count": 0,
                "tokens_after": 0,
                "tokens_saved_estimate": 0,
                "unique_refs": 0
              },
              "runtime_contract": {
                "confidence": 0.575,
                "latency_ms": 0.889,
                "tier_score": 23,
                "tokens_after": 517,
                "tokens_before": 517,
                "transform_budget_ms": 20.0
              },
              "strategy_selector": {
                "context_selector": true,
                "feature_norm": 1.0256227815332497,
                "latency_ms": 0,
                "rate_distortion": false,
                "selected_strategy": "submodular",
                "submodular": true,
                "tokens_after": 0,
                "tool_filter": true,
                "tool_optimizer": true
              },
              "tool_projection": {
                "chars_saved": 0,
                "latency_ms": 0,
                "tokens_after": 0
              }
            },
            "usage": {
              "baseline": {
                "completion_tokens": 256,
                "prompt_tokens": 950,
                "total_tokens": 1206
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 256,
                "prompt_tokens": 950,
                "total_tokens": 1206
              }
            }
          }
        ],
        "summary": {
          "avg_pipeline_latency_ms": 98.6738,
          "avg_quality_score": 0.8525,
          "avg_reduction_ratio": 0.2868,
          "passed_scenarios": 14,
          "total_scenarios": 21,
          "total_token_savings": 8722
        },
        "timestamp": 1779371330.7317061
      },
      "target": {
        "available": true,
        "base_url": "https://ollama.com",
        "model": "ollama-cloud/kimi-k2.6:cloud",
        "provider": "ollama-cloud",
        "skip_reason": ""
      }
    }
  ]
}
```

## provider_validation

- Kind: `live`
- Status: `ok`

| Metric | Value |
| --- | ---: |
| total_evaluated | 18 |
| total_passed | 4 |
| pass_rate | 0.2222 |

### Provider Validation Results

| Scenario | Provider | Task Eq. | Token Pass | Overall | Tokens Before | Tokens After | Risk | Forbidden Applied |
| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |
| uuid_deduplication | ollama-cloud | 0.892 | pass | pass | 353 | 353 | unknown | none |
| tool_output_filtering | ollama-cloud | 0.709 | pass | FAIL | 3770 | 242 | unknown | rate_distortion |
| table_compression | ollama-cloud | 0.721 | pass | FAIL | 1272 | 57 | unknown | rate_distortion |
| prefix_optimization | ollama-cloud | 1.000 | pass | pass | 207 | 207 | unknown | none |
| code_review_patterns | ollama-cloud | 1.000 | pass | FAIL | 992 | 992 | unknown | rate_distortion |
| api_docs_summarization | ollama-cloud | 0.969 | pass | FAIL | 1893 | 1893 | unknown | rate_distortion |
| tool_call_preservation | ollama-cloud | 0.917 | pass | FAIL | 133 | 133 | unknown | reference_sub |
| json_response_format | ollama-cloud | 0.990 | pass | FAIL | 583 | 583 | unknown | rate_distortion |
| cache_arbitrage_prefix | ollama-cloud | 0.966 | pass | pass | 241 | 241 | unknown | none |
| dictionary_repetition | ollama-cloud | 0.767 | pass | FAIL | 192 | 42 | unknown | none |
| cleanup_noise | ollama-cloud | 0.925 | pass | FAIL | 47 | 47 | unknown | reference_sub |
| message_dedup_turns | ollama-cloud | 0.792 | pass | FAIL | 65 | 65 | unknown | none |
| rate_distortion_longform | ollama-cloud | 0.714 | pass | FAIL | 773 | 20 | unknown | none |
| context_budget_pressure | ollama-cloud | 0.966 | pass | FAIL | 802 | 164 | unknown | rate_distortion |
| runtime_contract_pressure | ollama-cloud | 0.917 | pass | pass | 3513 | 3513 | unknown | none |
| grammar_json_table | ollama-cloud | 0.674 | pass | FAIL | 750 | 56 | unknown | rate_distortion |
| simple_baseline | ollama-cloud | 0.747 | pass | FAIL | 7 | 7 | unknown | message_dedup, reference_sub |
| mixed_realworld | ollama-cloud | 0.633 | pass | FAIL | 498 | 22 | unknown | none |

### PSG Safety Decisions

| Scenario | Judge Verdict | Safety Rollbacks | Expansion Aborts | Scheduler Blocks |
| --- | --- | --- | --- | --- |
| uuid_deduplication | WARN: minor meaning drift detected | none | none | none |
| tool_output_filtering | FAIL: significant meaning drift | none | none | none |
| table_compression | FAIL: significant meaning drift | none | none | none |
| prefix_optimization | PASS: optimized output preserves meaning well | none | none | none |
| code_review_patterns | PASS: optimized output preserves meaning well | none | none | none |
| api_docs_summarization | PASS: optimized output preserves meaning well | none | none | none |
| tool_call_preservation | PASS: optimized output preserves meaning well | none | none | none |
| json_response_format | PASS: optimized output preserves meaning well | none | none | none |
| cache_arbitrage_prefix | PASS: optimized output preserves meaning well | none | none | none |
| dictionary_repetition | FAIL: significant meaning drift; placeholder leakage detected (drift=0.20) | none | none | none |
| cleanup_noise | PASS: optimized output preserves meaning well | none | none | none |
| message_dedup_turns | FAIL: significant meaning drift | none | none | none |
| rate_distortion_longform | FAIL: significant meaning drift | none | none | none |
| context_budget_pressure | PASS: optimized output preserves meaning well | none | none | none |
| runtime_contract_pressure | PASS: optimized output preserves meaning well | none | none | none |
| grammar_json_table | FAIL: significant meaning drift | none | none | none |
| simple_baseline | FAIL: significant meaning drift | none | none | none |
| mixed_realworld | FAIL: significant meaning drift; placeholder leakage detected (drift=0.30) | none | none | none |

### Task Equivalence Breakdown

| Scenario | Constraint | Entity | Format | Reasoning | Refusal | Complete | Drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uuid_deduplication | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| tool_output_filtering | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| table_compression | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| prefix_optimization | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| code_review_patterns | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| api_docs_summarization | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| tool_call_preservation | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| json_response_format | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| cache_arbitrage_prefix | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| dictionary_repetition | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.200 |
| cleanup_noise | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| message_dedup_turns | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| rate_distortion_longform | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| context_budget_pressure | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| runtime_contract_pressure | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| grammar_json_table | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| simple_baseline | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| mixed_realworld | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.300 |

### Safety Gate Decisions

| Scenario | Risk Level | Risk Score | Risky Applied | Expansion Detected | Judge Verdict |
| --- | --- | ---: | --- | --- | --- |
| uuid_deduplication | unknown | 0.0 | none | none | WARN: minor meaning drift detected |
| tool_output_filtering | unknown | 0.0 | reference_sub | none | FAIL: significant meaning drift |
| table_compression | unknown | 0.0 | reference_sub | none | FAIL: significant meaning drift |
| prefix_optimization | unknown | 0.0 | reference_sub | none | PASS: optimized output preserves meaning well |
| code_review_patterns | unknown | 0.0 | none | none | PASS: optimized output preserves meaning well |
| api_docs_summarization | unknown | 0.0 | none | none | PASS: optimized output preserves meaning well |
| tool_call_preservation | unknown | 0.0 | none | none | PASS: optimized output preserves meaning well |
| json_response_format | unknown | 0.0 | reference_sub | none | PASS: optimized output preserves meaning well |
| cache_arbitrage_prefix | unknown | 0.0 | none | none | PASS: optimized output preserves meaning well |
| dictionary_repetition | unknown | 0.0 | none | none | FAIL: significant meaning drift; placeholder leakage detected (drift=0.20) |
| cleanup_noise | unknown | 0.0 | none | none | PASS: optimized output preserves meaning well |
| message_dedup_turns | unknown | 0.0 | reference_sub | none | FAIL: significant meaning drift |
| rate_distortion_longform | unknown | 0.0 | rate_distortion | none | FAIL: significant meaning drift |
| context_budget_pressure | unknown | 0.0 | reference_sub | none | PASS: optimized output preserves meaning well |
| runtime_contract_pressure | unknown | 0.0 | none | none | PASS: optimized output preserves meaning well |
| grammar_json_table | unknown | 0.0 | reference_sub | none | FAIL: significant meaning drift |
| simple_baseline | unknown | 0.0 | none | none | FAIL: significant meaning drift |
| mixed_realworld | unknown | 0.0 | none | none | FAIL: significant meaning drift; placeholder leakage detected (drift=0.30) |

## feature_matrix_eval

- Kind: `replay`
- Status: `ok`

| Metric | Value |
| --- | ---: |
| trace_count | 8 |
| config_count | 6 |
| feature_count | 9 |
| replay_feature_count | 4 |
| n_a_feature_count | 6 |
| off_reduction_ratio | 0.2024 |
| on_reduction_ratio | 0.2024 |
| off_quality_score | 1.0 |
| on_quality_score | 1.0 |
| off_signal_score | 0.0 |
| on_signal_score | 0.0415 |
| improving_features | 1 |
| flat_features | 2 |
| regressing_features | 0 |

### Feature Matrix

| Feature | Surface | Evidence | Supporting Traces | Signal Metric | Off Signal | On Signal | Delta Signal | Off Savings | On Savings | Off Ratio | On Ratio | Delta Savings | Verdict |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| semantic_cache | execution | provider_eval | - |  | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0 | n/a |
| cache_arbitrage | replay | feature_matrix | trace_cache_arbitrage_001 | stability_score | 0.0000 | 0.1245 | 0.1245 | 0 | 0 | 0.0000 | 0.0000 | 0 | improves |
| stall_detection | execution | provider_eval | - |  | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0 | n/a |
| batching | execution | provider_eval | - |  | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0 | n/a |
| speculation | execution | provider_eval | - |  | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0 | n/a |
| tacc | simulation | tacc_eval | - |  | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0 | n/a |
| rate_distortion | replay | feature_matrix | - |  | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0 | n/a |
| message_dedup | replay | feature_matrix | trace_message_dedup_001 | tokens_saved_estimate | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0 | flat |
| reference_sub | replay | feature_matrix | trace_uuid_dedup_001 | tokens_saved_estimate | 0.0000 | 0.0000 | 0.0000 | 119 | 119 | 0.6071 | 0.6071 | 0 | flat |

## protocol_eval

- Kind: `local`
- Status: `ok`

| Metric | Value |
| --- | ---: |
| passed | 11 |
| total | 11 |

```json
{
  "cache_plans": {
    "anthropic": {
      "annotation_keys": [
        "cache",
        "cache_breakpoints",
        "max_breakpoints",
        "provider"
      ],
      "breakpoints": 2,
      "cached_tokens": 49
    },
    "bedrock": {
      "annotation_keys": [
        "cache",
        "cache_breakpoints",
        "max_breakpoints",
        "provider"
      ],
      "breakpoints": 2,
      "cached_tokens": 49
    },
    "gemini": {
      "annotation_keys": [
        "cache",
        "explicit_context_cache",
        "provider"
      ],
      "breakpoints": 2,
      "cached_tokens": 49
    },
    "openai": {
      "annotation_keys": [
        "auto_caching",
        "cache",
        "provider"
      ],
      "breakpoints": 1,
      "cached_tokens": 49
    },
    "vertex": {
      "annotation_keys": [
        "cache",
        "explicit_context_cache",
        "provider"
      ],
      "breakpoints": 2,
      "cached_tokens": 49
    }
  },
  "checks": {
    "boundary_flags": true,
    "cache_plans": true,
    "chunking": true,
    "dictionary_roundtrip": true,
    "frame_roundtrip": true,
    "manifest_segments": true,
    "mux_migration": true,
    "mux_primary_active": true,
    "reliability": true,
    "replay_window": true,
    "resume_valid": true
  },
  "manifest": {
    "anchor_hash": "8e2b20fc0fd0671e817ba0d1aaf9081a947dca728992b8221f064095c78966c4",
    "anchor_version": 0,
    "manifest_id": "mf-e7aeec84922dc320",
    "metadata": {
      "model": "openai/gpt-4o-mini",
      "provider": "openai"
    },
    "segment_count": 3,
    "segment_counts": {
      "messages": 1,
      "system": 1,
      "tools": 1
    },
    "segment_types": [
      "tools",
      "system",
      "messages"
    ],
    "session_id": "eval-protocol",
    "token_estimate": 61
  }
}
```

## transport_eval

- Kind: `local`
- Status: `ok`

| Metric | Value |
| --- | ---: |
| passed | 10 |
| total | 10 |

```json
{
  "checks": {
    "connection_pool_reuse": true,
    "framing_roundtrip": true,
    "local_socket": true,
    "negotiation_roundtrip": true,
    "proxy_roundtrip": true,
    "proxy_streaming": true,
    "replay_buffer": true,
    "resume_window": true,
    "sidecar_connect_url": true,
    "websocket_tunnel": true
  },
  "http_version": "http/1.1",
  "proxy": {
    "request_count": 2,
    "response": "{\"ok\": true, \"path\": \"/v1/chat/completions\", \"host\": \"127.0.0.1:62871\"}",
    "stream_response": "data: stream-one\n\ndata: stream-two\n\n"
  },
  "websocket": {
    "connected": true,
    "frames": 1
  }
}
```

## integration_eval

- Kind: `local`
- Status: `ok`

| Metric | Value |
| --- | ---: |
| passed | 7 |
| total | 7 |

```json
{
  "checks": {
    "claude_env": true,
    "codex_env": true,
    "copilot_env": true,
    "cursor_env": true,
    "opencode_env": true,
    "reverts": true,
    "writes": true
  },
  "envs": {
    "claude": {
      "ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"
    },
    "codex": {
      "OPENAI_BASE_URL": "http://127.0.0.1:8787/v1"
    },
    "copilot": {
      "AI_AGENT": "claude-code_2-1-146_agent",
      "BUN_INSTALL": "/Users/harshdaga/.bun",
      "CLAUDECODE": "1",
      "CLAUDE_CODE_ENTRYPOINT": "cli",
      "CLAUDE_CODE_EXECPATH": "/Users/harshdaga/.local/share/claude/versions/2.1.146",
      "CLAUDE_CODE_SESSION_ID": "<redacted>",
      "CLAUDE_EFFORT": "medium",
      "COPILOT_PROVIDER_BASE_URL": "http://127.0.0.1:8787/v1",
      "COPILOT_PROVIDER_TYPE": "openai",
      "COPILOT_PROVIDER_WIRE_API": "completions",
      "COREPACK_ENABLE_AUTO_PIN": "0",
      "FPATH": "/opt/homebrew/share/zsh/site-functions:/usr/local/share/zsh/site-functions:/usr/share/zsh/site-functions:/usr/share/zsh/5.9/functions",
      "GIT_EDITOR": "true",
      "HOME": "/Users/harshdaga",
      "HOMEBREW_CELLAR": "/opt/homebrew/Cellar",
      "HOMEBREW_PREFIX": "/opt/homebrew",
      "HOMEBREW_REPOSITORY": "/opt/homebrew",
      "INFOPATH": "/opt/homebrew/share/info:",
      "LC_CTYPE": "UTF-8",
      "LOGNAME": "harshdaga",
      "LaunchInstanceID": "6B4F1A8D-55F1-4831-86D2-349EB8426DF0",
      "NVM_BIN": "/Users/harshdaga/.nvm/versions/node/v16.20.2/bin",
      "NVM_CD_FLAGS": "-q",
      "NVM_DIR": "/Users/harshdaga/.nvm",
      "NVM_INC": "/Users/harshdaga/.nvm/versions/node/v16.20.2/include/node",
      "NoDefaultCurrentDirectoryInExePath": "1",
      "OLLAMA_CLOUD_API_KEY": "<redacted>",
      "PATH": "/Users/harshdaga/lattice/.venv/bin:/Users/harshdaga/.bun/bin:/Users/harshdaga/.pyenv/shims:/Users/harshdaga/.pyenv/bin:/Users/harshdaga/.local/bin:/Users/harshdaga/.nvm/versions/node/v16.20.2/bin:/Library/Frameworks/Python.framework/Versions/2.7/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/Frameworks/Python.framework/Versions/2.7/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/local/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/appleinternal/bin:/usr/local/go/bin:/Users/harshdaga/.claude/plugins/cache/claude-code-plugins/frontend-design/1.0.0/bin:/Users/harshdaga/.claude/plugins/cache/claude-code-plugins/security-guidance/1.0.0/bin:/Users/harshdaga/.claude/plugins/cache/insforge-official-marketplace/insforge/1.0.0/bin:/Users/harshdaga/.claude/plugins/cache/claude-plugins-official/superpowers/5.1.0/bin:/Users/harshdaga/.claude/plugins/cache/claude-plugins-official/code-simplifier/1.0.0/bin:/Users/harshdaga/.claude/plugins/cache/claude-plugins-official/playwright/unknown/bin:/Users/harshdaga/.claude/plugins/cache/claude-plugins-official/huggingface-skills/1.0.2/bin:/Users/harshdaga/.claude/plugins/cache/thedotmack/claude-mem/12.1.0/bin",
      "PWD": "/Users/harshdaga/lattice",
      "PYENV_ROOT": "/Users/harshdaga/.pyenv",
      "PYENV_SHELL": "zsh",
      "SECURITYSESSIONID": "<redacted>",
      "SHELL": "/bin/zsh",
      "SHLVL": "3",
      "SSH_AUTH_SOCK": "/private/tmp/com.apple.launchd.nPAzIWKoWU/Listeners",
      "TERM": "xterm-256color",
      "TERM_PROGRAM": "Apple_Terminal",
      "TERM_PROGRAM_VERSION": "453",
      "TERM_SESSION_ID": "<redacted>",
      "TMPDIR": "/var/folders/xf/1x6zsq6x0r5cmdqwxnr1_3600000gn/T/",
      "USER": "harshdaga",
      "UV": "/Users/harshdaga/.local/bin/uv",
      "UV_RUN_RECURSION_DEPTH": "1",
      "VIRTUAL_ENV": "/Users/harshdaga/lattice/.venv",
      "XPC_FLAGS": "0x0",
      "XPC_SERVICE_NAME": "0",
      "_": "/Users/harshdaga/.local/bin/uv",
      "__CFBundleIdentifier": "com.apple.Terminal",
      "__CF_USER_TEXT_ENCODING": "0x1F5:0x0:0x0"
    },
    "cursor": {
      "ANTHROPIC_BASE_URL": "http://127.0.0.1:8787",
      "OPENAI_BASE_URL": "http://127.0.0.1:8787/v1"
    },
    "opencode": {
      "OPENAI_BASE_URL": "http://127.0.0.1:8787/v1"
    }
  },
  "reverted": {
    "claude": true,
    "codex": true,
    "copilot": true,
    "opencode": true
  },
  "wrote": {
    "claude": true,
    "codex": true,
    "copilot": true,
    "cursor": true,
    "opencode": true
  }
}
```

## capability_eval

- Kind: `local`
- Status: `ok`

| Metric | Value |
| --- | ---: |
| passed | 5 |
| total | 5 |

```json
{
  "cache_modes": {
    "adapter": 1,
    "auto_prefix": 2,
    "explicit": 2,
    "explicit_context": 2,
    "none": 9
  },
  "checks": {
    "anthropic_cache": true,
    "cache_modes_present": true,
    "explicit_context": true,
    "matrix_size": true,
    "openai_cache": true
  },
  "providers": [
    {
      "cache_mode": "auto_prefix",
      "default_base_url": "https://api.openai.com",
      "provider": "openai",
      "supports_http2": true,
      "supports_prompt_caching": true,
      "supports_reasoning": true,
      "supports_streaming": true,
      "supports_structured_output": true,
      "supports_tool_calls": true
    },
    {
      "cache_mode": "explicit",
      "default_base_url": "https://api.anthropic.com",
      "provider": "anthropic",
      "supports_http2": true,
      "supports_prompt_caching": true,
      "supports_reasoning": true,
      "supports_streaming": true,
      "supports_structured_output": true,
      "supports_tool_calls": true
    },
    {
      "cache_mode": "explicit_context",
      "default_base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
      "provider": "gemini",
      "supports_http2": true,
      "supports_prompt_caching": true,
      "supports_reasoning": false,
      "supports_streaming": true,
      "supports_structured_output": true,
      "supports_tool_calls": true
    },
    {
      "cache_mode": "explicit_context",
      "default_base_url": "",
      "provider": "vertex",
      "supports_http2": true,
      "supports_prompt_caching": true,
      "supports_reasoning": false,
      "supports_streaming": true,
      "supports_structured_output": true,
      "supports_tool_calls": true
    },
    {
      "cache_mode": "explicit",
      "default_base_url": "",
      "provider": "bedrock",
      "supports_http2": true,
      "supports_prompt_caching": true,
      "supports_reasoning": true,
      "supports_streaming": true,
      "supports_structured_output": false,
      "supports_tool_calls": true
    },
    {
      "cache_mode": "none",
      "default_base_url": "http://127.0.0.1:11434",
      "provider": "ollama",
      "supports_http2": true,
      "supports_prompt_caching": false,
      "supports_reasoning": true,
      "supports_streaming": true,
      "supports_structured_output": false,
      "supports_tool_calls": true
    },
    {
      "cache_mode": "auto_prefix",
      "default_base_url": "",
      "provider": "azure",
      "supports_http2": true,
      "supports_prompt_caching": true,
      "supports_reasoning": true,
      "supports_streaming": true,
      "supports_structured_output": true,
      "supports_tool_calls": true
    },
    {
      "cache_mode": "none",
      "default_base_url": "https://api.groq.com/openai",
      "provider": "groq",
      "supports_http2": true,
      "supports_prompt_caching": false,
      "supports_reasoning": false,
      "supports_streaming": true,
      "supports_structured_output": true,
      "supports_tool_calls": true
    },
    {
      "cache_mode": "none",
      "default_base_url": "https://api.deepseek.com",
      "provider": "deepseek",
      "supports_http2": true,
      "supports_prompt_caching": false,
      "supports_reasoning": true,
      "supports_streaming": true,
      "supports_structured_output": false,
      "supports_tool_calls": true
    },
    {
      "cache_mode": "none",
      "default_base_url": "https://api.mistral.ai/v1",
      "provider": "mistral",
      "supports_http2": true,
      "supports_prompt_caching": false,
      "supports_reasoning": false,
      "supports_streaming": true,
      "supports_structured_output": true,
      "supports_tool_calls": true
    },
    {
      "cache_mode": "none",
      "default_base_url": "https://api.cohere.com/compatibility/v1",
      "provider": "cohere",
      "supports_http2": true,
      "supports_prompt_caching": false,
      "supports_reasoning": false,
      "supports_streaming": true,
      "supports_structured_output": true,
      "supports_tool_calls": true
    },
    {
      "cache_mode": "adapter",
      "default_base_url": "https://openrouter.ai/api/v1",
      "provider": "openrouter",
      "supports_http2": true,
      "supports_prompt_caching": false,
      "supports_reasoning": false,
      "supports_streaming": true,
      "supports_structured_output": false,
      "supports_tool_calls": true
    },
    {
      "cache_mode": "none",
      "default_base_url": "https://api.fireworks.ai/inference/v1",
      "provider": "fireworks",
      "supports_http2": true,
      "supports_prompt_caching": false,
      "supports_reasoning": false,
      "supports_streaming": true,
      "supports_structured_output": false,
      "supports_tool_calls": true
    },
    {
      "cache_mode": "none",
      "default_base_url": "https://api.together.xyz/v1",
      "provider": "together",
      "supports_http2": true,
      "supports_prompt_caching": false,
      "supports_reasoning": false,
      "supports_streaming": true,
      "supports_structured_output": false,
      "supports_tool_calls": true
    },
    {
      "cache_mode": "none",
      "default_base_url": "https://api.perplexity.ai",
      "provider": "perplexity",
      "supports_http2": true,
      "supports_prompt_caching": false,
      "supports_reasoning": false,
      "supports_streaming": true,
      "supports_structured_output": false,
      "supports_tool_calls": false
    },
    {
      "cache_mode": "none",
      "default_base_url": "https://api.ai21.com/studio/v1",
      "provider": "ai21",
      "supports_http2": true,
      "supports_prompt_caching": false,
      "supports_reasoning": false,
      "supports_streaming": true,
      "supports_structured_output": false,
      "supports_tool_calls": false
    }
  ]
}
```

## control_plane_eval

- Kind: `local`
- Status: `ok`

| Metric | Value |
| --- | ---: |
| passed | 7 |
| total | 7 |

```json
{
  "checks": {
    "delta_detected": true,
    "dictionary_roundtrip": true,
    "frame_roundtrip": true,
    "manifest_summary": true,
    "manifest_version": true,
    "session_created": true,
    "session_updated": true
  },
  "session": {
    "client_info": {},
    "connection_id": null,
    "created_at": 1779371329.496078,
    "last_accessed_at": 1779371329.4961112,
    "manifest": {
      "anchor_hash": "d5cb231de5479c8ce897b513f33edf50c0475228dd036e31d303b32153229c04",
      "anchor_version": 0,
      "created_at": 1779371329.495853,
      "manifest_id": "mf-350085defc8366f8",
      "metadata": {
        "model": "openai/gpt-4o-mini",
        "provider": "openai"
      },
      "segments": [
        {
          "created_at": 1779371329.49581,
          "hash": "c9643d11d5df6054",
          "metadata": {},
          "parts": [
            {
              "text": "[{\"function\": {\"description\": \"Lookup data\", \"name\": \"lookup\", \"parameters\": {\"properties\": {\"key\": {\"type\": \"string\"}}, \"type\": \"object\"}}, \"type\": \"function\"}]",
              "type": "text"
            }
          ],
          "type": "tools",
          "version": 0
        },
        {
          "created_at": 1779371329.495784,
          "hash": "cd278b5da992db10",
          "metadata": {},
          "parts": [
            {
              "text": "You are a production evaluator.",
              "type": "text"
            }
          ],
          "type": "system",
          "version": 0
        },
        {
          "created_at": 1779371329.495821,
          "hash": "8e6fda792cb344c3",
          "metadata": {},
          "parts": [
            {
              "text": "[USER] Summarize the manifest and wire state.",
              "type": "text"
            }
          ],
          "type": "messages",
          "version": 0
        }
      ],
      "session_id": "lattice-TDWR2vUXNK0BSrzNXW8J-Q"
    },
    "messages": [
      {
        "content": "You are a production evaluator.",
        "role": "Role.SYSTEM"
      },
      {
        "content": "Summarize the manifest and wire state.",
        "role": "Role.USER"
      },
      {
        "content": "Add one more turn.",
        "role": "Role.USER"
      }
    ],
    "metadata": {},
    "model": "openai/gpt-4o-mini",
    "provider": "openai",
    "session_id": "lattice-TDWR2vUXNK0BSrzNXW8J-Q",
    "system_prompt": "You are a production evaluator.",
    "tool_schemas": [
      {
        "function": {
          "description": "Lookup data",
          "name": "lookup",
          "parameters": {
            "properties": {
              "key": {
                "type": "string"
              }
            },
            "type": "object"
          }
        },
        "type": "function"
      }
    ],
    "version": 1
  }
}
```

## tacc_eval

- Kind: `simulation`
- Status: `ok`

| Metric | Value |
| --- | ---: |
| scenario_count | 4 |
| avg_static_p95_ms | 259.68 |
| avg_tacc_p95_ms | 257.99 |

```json
{
  "scenarios": [
    {
      "scenario": "light_load",
      "static": {
        "error_rate": 0.0,
        "p95_latency_ms": 131.21,
        "p99_latency_ms": 131.91,
        "throughput_qps": 20.0
      },
      "tacc": {
        "error_rate": 0.0,
        "p95_latency_ms": 131.21,
        "p99_latency_ms": 131.91,
        "throughput_qps": 20.0
      }
    },
    {
      "scenario": "moderate_load",
      "static": {
        "error_rate": 0.0803,
        "p95_latency_ms": 302.51,
        "p99_latency_ms": 305.17,
        "throughput_qps": 41.25
      },
      "tacc": {
        "error_rate": 0.0506,
        "p95_latency_ms": 300.25,
        "p99_latency_ms": 359.68,
        "throughput_qps": 42.25
      }
    },
    {
      "scenario": "heavy_load",
      "static": {
        "error_rate": 0.0866,
        "p95_latency_ms": 302.27,
        "p99_latency_ms": 305.33,
        "throughput_qps": 41.15
      },
      "tacc": {
        "error_rate": 0.0506,
        "p95_latency_ms": 300.25,
        "p99_latency_ms": 359.68,
        "throughput_qps": 42.25
      }
    },
    {
      "scenario": "burst_load",
      "static": {
        "error_rate": 0.0871,
        "p95_latency_ms": 302.71,
        "p99_latency_ms": 305.53,
        "throughput_qps": 41.4
      },
      "tacc": {
        "error_rate": 0.0506,
        "p95_latency_ms": 300.25,
        "p99_latency_ms": 359.68,
        "throughput_qps": 42.25
      }
    }
  ]
}
```

## replay_eval

- Kind: `replay`
- Status: `ok`

| Metric | Value |
| --- | ---: |
| trace_count | 8 |
| iterations | 1 |
| avg_reduction_ratio | 0.1704 |
| avg_pipeline_latency_ms | 46.9517 |
| avg_quality_score | 0.9737 |

- Benchmark provider: `openai`
- Benchmark model: `gpt-4`
- Scenarios: `8`
- Quality passed: `7`
- Total token savings: `369`
- Avg reduction ratio: `0.1704`
- Avg pipeline latency: `46.9517 ms`
- Avg quality score: `0.9737`

| Scenario | Category | Saved | Ratio | Task Eq. |
| --- | --- | ---: | ---: | ---: |
| trace_uuid_dedup_001 | reference_substitution | 119 | 0.6071 | 1.0000 |
| trace_tool_filter_001 | tool_output_heavy | 176 | 0.2489 | 1.0000 |
| trace_json_integrity_001 | json_integrity | 74 | 0.5068 | 1.0000 |
| trace_cache_arbitrage_001 | cache_arbitrage | 0 | 0.0000 | 1.0000 |
| trace_message_dedup_001 | message_dedup | 0 | 0.0000 | 1.0000 |
| trace_semantic_compress_001 | semantic_compress | 0 | 0.0000 | 1.0000 |
| trace_debugging_logs_001 | debugging | 0 | 0.0000 | 0.7059 |
| trace_reasoning_root_cause_001 | reasoning | 0 | 0.0000 | 1.0000 |

### Response Samples

| Scenario | Baseline | Optimized |
| --- | --- | --- |
| trace_uuid_dedup_001 | Duplicate UUIDs and repeated transaction IDs indicate the same transaction appeared multiple times, likely causing a unique constraint or deduplication conflict. | Duplicate UUIDs and repeated transaction IDs indicate the same transaction appeared multiple times, likely causing a unique constraint or deduplication conflict. |
| trace_tool_filter_001 | The build failed because it encountered Module not found, Syntax error, Type mismatch, and Dependency timeout issues across repeated errors. The internal stack fields are noise. | The build failed because it encountered Module not found, Syntax error, Type mismatch, and Dependency timeout issues across repeated errors. The internal stack fields are noise. |
| trace_json_integrity_001 | There are 2 users, both with dark theme and notifications enabled. | There are 2 users, both with dark theme and notifications enabled. |
| trace_cache_arbitrage_001 | The next delta should contain only the new request-specific change, because the stable context is already shared. | The next delta should contain only the new request-specific change, because the stable context is already shared. |
| trace_message_dedup_001 | The incident report shows a retry loop caused by repeated failures in the request path. | The incident report shows a retry loop caused by repeated failures in the request path. |
| trace_semantic_compress_001 | The slowdown was caused by a cache invalidation storm that triggered repeated recomputation and increased tail latency. The mitigation is to preserve the root cause, impact, and r… | The slowdown was caused by a cache invalidation storm that triggered repeated recomputation and increased tail latency. The mitigation is to preserve the root cause, impact, and r… |
| trace_debugging_logs_001 | Service connectivity failure across all 5 services. Repeated at attempt 2 for services 0 and 1. Root cause: connection pool exhaustion. | Service connectivity failure across all 5 services with repeated attempts. Root cause: connection pool exhaustion. |
| trace_reasoning_root_cause_001 | Memory leak caused OOM crash. Restarts at 14:28 freed memory but the leak continued, reaching 99% in 4 minutes. Root cause: unreleased memory in the service. | Memory leak caused OOM crash. Restarts at 14:28 freed memory but the leak continued, reaching 99% in 4 minutes. Root cause: unreleased memory in the service. |
