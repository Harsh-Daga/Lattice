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
| avg_pipeline_latency_ms | 96.7744 |
| avg_quality_score | 0.8302 |
| proof_passed | 16 |
| proof_failed | 5 |
| feature_failed | 5 |
| tier_failed | 0 |
| budget_failed | 0 |

- Benchmark provider: `ollama-cloud`
- Benchmark model: `ollama-cloud/kimi-k2.6:cloud`
- Scenarios: `21`
- Quality passed: `12`
- Total token savings: `8722`
- Avg reduction ratio: `0.2868`
- Avg pipeline latency: `96.7744 ms`
- Avg quality score: `0.8302`

| Scenario | Category | Saved | Ratio | Task Eq. |
| --- | --- | ---: | ---: | ---: |

| Proof Metric | Value |
| --- | ---: |
| proof_passed | 16 |
| proof_failed | 5 |
| uuid_deduplication | reference_substitution | 0 | 0.0000 | 0.9250 |
| tool_output_filtering | tool_filter | 4164 | 0.9438 | 0.7133 |
| table_compression | format_conversion | 1579 | 0.9552 | 0.7644 |
| prefix_optimization | prefix_opt | 0 | 0.0000 | 0.7591 |
| code_review_patterns | reference_sub | 0 | 0.0000 | 0.9583 |
| api_docs_summarization | reference_optimizer | 0 | 0.0000 | 0.8125 |
| tool_call_preservation | integration | 0 | 0.0000 | 0.9271 |
| json_response_format | format_conversion | 0 | 0.0000 | 1.0000 |
| cache_arbitrage_prefix | cache_arbitrage | 0 | 0.0000 | 0.9204 |
| dictionary_repetition | reference_optimizer | 61 | 0.4803 | 0.8356 |
| cleanup_noise | normalization | 0 | 0.0000 | 0.8000 |
| message_dedup_turns | message_dedup | 0 | 0.0000 | 0.9583 |
| rate_distortion_longform | rate_distortion | 511 | 0.9569 | 0.6667 |
| context_budget_pressure | context_selector | 669 | 0.7917 | 0.9615 |
| runtime_contract_pressure | runtime_contract | 0 | 0.0000 | 0.8750 |
| grammar_json_table | structure_optimizer | 925 | 0.9259 | 0.6844 |
| simple_baseline | baseline | 0 | 0.0000 | 0.9875 |
| mixed_realworld | mixed | 813 | 0.9679 | 0.5000 |
| debugging_log_analysis | debugging | 0 | 0.0000 | 0.8558 |
| reasoning_root_cause | reasoning | 0 | 0.0000 | 0.9785 |
| repeated_identifier_investigation | reference_substitution | 0 | 0.0000 | 0.9955 |

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
| structure_optimizer | 1011.0 | 56.5 | 954.5 | 2 |
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
| tool_output_filtering | The user wants to know why the build failed. I have the build errors from the function call. Let me analyze the errors. Looking at the errors array: - There are 60 total entries (… | The user wants me to summarize the build errors based on the tool output provided. Let me analyze the data: - Total errors: 60 - Showing: 10 (sample) - Error types in sample: - "M… |
| table_compression | The user wants me to analyze employee data and summarize salary trends. Let me look at the data carefully. The data has 100 employees (ID 0-99), all in Engineering department, all… | The user wants me to analyze employee data and summarize salary trends. They've provided a snippet of data that looks like it contains 100 employees (Emp_0 through Emp_99), all in… |
| prefix_optimization | **Use this decision tree:** \| Construct \| Use When \| Memory \| Access \| \|-----------\|----------\|--------\|--------\| \| **List comp** \| You need all results, multiple pass… | The user is asking for a comparison of when to use list comprehensions, dict comprehensions, set comprehensions, and generator expressions. This is a natural follow-up to the prev… |
| code_review_patterns | The user wants me to review code changes across 10 files (module_0.py through module_9.py) and summarize the common issues. Looking at the code: - All 10 files have identical code… | The user wants me to review code changes across 10 files (module_0.py through module_9.py) and summarize the common issues. Looking at all the files, they all contain identical co… |
| api_docs_summarization | The user wants me to summarize/document 20 API endpoints concisely. Looking at the input, all endpoints follow the exact same pattern: - Method: GET - Path: /api/v1/resource_{N} w… | The user wants me to summarize/document 20 API endpoints concisely. They all follow the exact same pattern: - GET /api/v1/resource_{N} where N ranges from 0 to 19 - Same parameter… |
| tool_call_preservation | The user is asking for the average temperature between NYC and SF based on the previous results. NYC was 72°F and SF was 65°F. I need to calculate the average: (72 + 65) / 2 = 137… | The user is asking for the average temperature between NYC and SF based on the previous weather results. NYC was 72°F and SF was 65°F. I need to calculate the average: (72 + 65) /… |
| json_response_format | The user wants me to analyze the provided JSON data and return the top 5 users as JSON. Looking at the data: - There are 50 users (id 0 through 49) - Each user has a score that in… | The user wants me to analyze the provided JSON data and return the top 5 users as JSON. Looking at the data: - There are 50 users (id 0 through 49) - Each user has a score that in… |
| cache_arbitrage_prefix | The user is asking what the next delta should contain given the repeated context. The context shows a series of design notes (0 through 11) all saying "keep manifest ordering stab… | The user is asking: "Given the repeated context, what should the next delta contain?" The context provided is a list of design notes (0 through 11), all identical: "keep manifest … |
| dictionary_repetition | The user wants me to compress repeated language losslessly. The input is a repetition of the sentence "The session manifest should remain stable and the session manifest should re… | The user wants me to compress repeated language losslessly. The input is: `<crossref_1>. <crossref_1>.<crossref_1>. <crossref_1>.<crossref_1>. <crossref_1>.<crossref_1>. <crossref… |
| cleanup_noise | The user wants me to summarize something, but they haven't actually provided the text to summarize. They said "Please summarize this." but there's no content attached. Wait, looki… | The user wants me to summarize something, but they haven't actually provided the text to summarize. They said "Please summarize this" but there's no content attached. They also me… |
| message_dedup_turns | The user is asking me to explain the root cause once, following up on the previous interaction where I summarized the incident report as showing a "retry loop." However, I don't a… | The user is asking me to explain the root cause of the incident described in the previous turns. However, looking at the conversation history, I don't actually see the content of … |
| rate_distortion_longform | The user wants a summary of the incident described in the prompt. The key requirements are: 1. Summarize the incident clearly 2. Keep the root cause 3. Retain: root cause, user-vi… | I don't see any incident details in your message. Could you please provide: - The incident report, description, or context - Any relevant timeline, logs, or events - Background on… |
| context_budget_pressure | The user wants to know which documents mention the "error path" and "request ids". Looking at the provided documents: - Document 0: mentions "request id 0000" and "error path" - D… | The user is asking: "Which documents mention the error path and request ids?" Looking at the provided context, there are multiple documents (Document 0 through Document 17, and po… |
| runtime_contract_pressure | The user is asking me to prove an optimization contract is correct, but they haven't actually provided the optimization contract or the specific code to prove. However, looking at… | The user is asking me to prove an optimization contract is correct, but they haven't actually provided the optimization contract or the specific code to prove. However, looking at… |
| grammar_json_table | The user wants me to summarize structured data without losing keys. There are two datasets: 1. JSON with 20 services (svc_0 to svc_19), all status "ok", latencies from 120ms to 13… | The user wants me to summarize structured data without losing keys. The data appears to be a table with rows representing employees. Let me analyze the structure: Columns seem to … |
| simple_baseline | Hello! I'm doing well, thank you for asking. I'm here and ready to help you with whatever you need. How can I assist you today? | Hello! I'm doing well, thank you for asking. I'm here and ready to help you with whatever you need. How can I assist you today? |
| mixed_realworld | The user wants me to investigate a production error from a specific transaction. They've provided a JSON trace with 30 function calls (foo_0 through foo_29) with timestamps, and a… | I see you've pasted what appears to be the |
| debugging_log_analysis | The user wants me to analyze error logs from a production outage and identify the root cause and mitigation steps. Looking at the logs: - There are 5 services: service_0, service_… | The user wants me to analyze error logs from a production outage and identify the root cause and mitigation steps. Looking at the logs: - There are 5 services: service_0, service_… |
| reasoning_root_cause | The user wants me to explain a system crash based on the provided timeline and data points. Let me analyze what we know: 1. Crash at 14:32 with error code E0503 2. Memory usage pr… | The user is asking me to explain a system crash based on provided telemetry data. Let me analyze the information given: 1. Crash time: 14:32 with error code E0503 2. Memory usage … |
| repeated_identifier_investigation | The user wants me to investigate transaction failures from the provided list and identify which transactions failed and what pattern they follow. Let me list out the failed transa… | The user wants me to investigate transaction failures from the provided list and identify which transactions failed and what pattern they follow. Let me list out the failed transa… |

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
| avg_quality_score | 0.8764 |

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
                "network_ms": 3105.5650420021266,
                "network_stdev": 0.0,
                "total_ms": 3105.56570882909,
                "total_stdev": 0.0
              },
              "delta": 917.071,
              "optimized": {
                "network_ms": 3960.1158329751343,
                "network_stdev": 0.0,
                "pipeline_ms": 62.520166859030724,
                "pipeline_stdev": 0.0,
                "total_ms": 4022.636374924332,
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
              "baseline": "The user is asking why duplicate UUIDs failed in a list of transactions. Let me analyze the provided list.\n\nLooking at the list:\n- [0] 550e8400-e29b-41d4-a716-446655440000\n- [1] 6ba7b810-9dad-11d1-80b4-00c04fd430c8\n- [2] a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11\n- [4] 6ba7b810-9dad-11d1-80b4-00c04fd430c8 (duplicate of [1])\n- [5] 550e8400-e29b-41d4-a716-446655440000 (duplicate of [0])\n\nThe user asks \"Why did the duplicate UUIDs fail?\"\n\nWait, I need to be careful here. The user is presenting a list of ",
              "optimized": "The user is asking why duplicate UUIDs failed in a list of transactions. Let me analyze the provided list.\n\nLooking at the list:\n- [0] 550e8400-e29b-41d4-a716-446655440000\n- [1] 6ba7b810-9dad-11d1-80b4-00c04fd430c8\n- [2] a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11\n- [3] f47ac10b-58cc-4372-a567-0e02b2c3d479\n- [4] 6ba7b810-9dad-11d1-80b4-00c04fd430c8  <- Duplicate of [1]\n- [5] 550e8400-e29b-41d4-a716-446655440000  <- Duplicate of [0]\n- [6] through [29] are sequential UUIDs with incrementing first segment"
            },
            "scenario": "uuid_deduplication",
            "telemetry": {
              "pipeline": {
                "latency_ms": 62.44993209838867,
                "tokens_in": 353,
                "tokens_out": 353,
                "transform_latency_ms": 61.255
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
                  "latency_ms": 0.084,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.0169971671388102,
                  "stable_tokens": 6,
                  "tokens_after": 353,
                  "tokens_before": 353
                },
                "content_profiler": {
                  "latency_ms": 10.302,
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
                  "beam_search_latency_ms": 50.01,
                  "core_transforms": 7,
                  "ir_sections": 33,
                  "latency_ms": 50.494,
                  "optimizers": 5,
                  "tokens_after": 353,
                  "tokens_before": 353,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.014,
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
                  "latency_ms": 0.361,
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
                "latency_ms": 0.084,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.0169971671388102,
                "stable_tokens": 6,
                "tokens_after": 353,
                "tokens_before": 353
              },
              "content_profiler": {
                "latency_ms": 10.302,
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
                "beam_search_latency_ms": 50.01,
                "core_transforms": 7,
                "ir_sections": 33,
                "latency_ms": 50.494,
                "optimizers": 5,
                "tokens_after": 353,
                "tokens_before": 353,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.014,
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
                "latency_ms": 0.361,
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
                "network_ms": 4772.672082995996,
                "network_stdev": 0.0,
                "total_ms": 4772.672791965306,
                "total_stdev": 0.0
              },
              "delta": -1383.653,
              "optimized": {
                "network_ms": 3216.4613329805434,
                "network_stdev": 0.0,
                "pipeline_ms": 172.55775001831353,
                "pipeline_stdev": 0.0,
                "total_ms": 3389.020249946043,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": false,
              "semantic_similarity": 0.6957,
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
              "baseline": "The user wants to know why the build failed. I have the build errors data. Let me analyze it.\n\nThere are 60 errors total. Let me categorize them by severity and message:\n\nSeverity: error\n- \"Module not found\": modules 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57\n  That's 20 errors.\n- \"Syntax error\": modules 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58\n  That's 20 errors.\n\nSeverity: warning\n- \"Type mismatch\": modules 2, 5, 8, 11, 14, 17,",
              "optimized": "The user wants me to summarize the build errors based on the tool output provided. Let me analyze the data:\n\n- Total errors: 60\n- Showing: 10 (sample)\n- Error types in sample:\n  - \"Module not found\": modules 0, 3, 6, 9 (4 occurrences in sample)\n  - \"Syntax error\": modules 1, 4, 7 (3 occurrences in sample)\n  - \"Type mismatch\": modules 2, 5, 8 (3 occurrences in sample)\n\nSeverity breakdown in sample:\n- error: 7 (Module not found \u00d74 + Syntax error \u00d73)\n- warning: 3 (Type mismatch \u00d73)\n\nSince there are"
            },
            "scenario": "tool_output_filtering",
            "telemetry": {
              "pipeline": {
                "latency_ms": 172.45173454284668,
                "tokens_in": 3770,
                "tokens_out": 242,
                "transform_latency_ms": 168.474
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
                  "latency_ms": 0.102,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.9899204244031831,
                  "stable_tokens": 3732,
                  "tokens_after": 3770,
                  "tokens_before": 3770
                },
                "content_profiler": {
                  "latency_ms": 54.513,
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
                  "beam_search_latency_ms": 107.733,
                  "core_transforms": 11,
                  "ir_sections": 4,
                  "latency_ms": 113.427,
                  "optimizers": 7,
                  "tokens_after": 242,
                  "tokens_before": 3770,
                  "tokens_saved": 14120
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.019,
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
                  "latency_ms": 0.413,
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
                "latency_ms": 0.102,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.9899204244031831,
                "stable_tokens": 3732,
                "tokens_after": 3770,
                "tokens_before": 3770
              },
              "content_profiler": {
                "latency_ms": 54.513,
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
                "beam_search_latency_ms": 107.733,
                "core_transforms": 11,
                "ir_sections": 4,
                "latency_ms": 113.427,
                "optimizers": 7,
                "tokens_after": 242,
                "tokens_before": 3770,
                "tokens_saved": 14120
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.019,
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
                "latency_ms": 0.413,
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
                "network_ms": 3064.769540913403,
                "network_stdev": 0.0,
                "total_ms": 3064.770583063364,
                "total_stdev": 0.0
              },
              "delta": -53.049,
              "optimized": {
                "network_ms": 2874.6329999994487,
                "network_stdev": 0.0,
                "pipeline_ms": 137.0864158961922,
                "pipeline_stdev": 0.0,
                "total_ms": 3011.7217067163438,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": false,
              "semantic_similarity": 0.7775,
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
              "baseline": "The user wants me to analyze employee data and summarize salary trends. Let me look at the data carefully.\n\nThe data has 100 employees (ID 0-99), all in Engineering department, all active, with salaries from $100,000 to $199,000.\n\nLet me identify the pattern:\n- ID 0: $100,000\n- ID 1: $101,000\n- ID 2: $102,000\n- ...\n- ID 99: $199,000\n\nSo the salary = $100,000 + ID \u00d7 $1,000.\n\nKey observations:\n1. All employees are in the Engineering department\n2. All are active\n3. Salaries increase linearly by exa",
              "optimized": "The user wants me to analyze employee data and summarize salary trends. They've provided a pattern showing 100 employees (Emp_0 through Emp_99), all in Engineering, with salaries ranging from $100,000 to $199,000, all active.\n\nLet me parse the data:\n- Employee IDs: 0 to 99 (100 employees total)\n- Department: Engineering (all same department)\n- Salaries: $100,000 to $199,000 (incrementing by $1,000 each)\n- Status: all active\n\nSo the salary data is: 100000, 101000, 102000, ..., 199000.\n\nThis is an"
            },
            "scenario": "table_compression",
            "telemetry": {
              "pipeline": {
                "latency_ms": 137.00079917907715,
                "tokens_in": 1272,
                "tokens_out": 57,
                "transform_latency_ms": 134.761
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
                  "latency_ms": 0.076,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.009433962264150943,
                  "stable_tokens": 12,
                  "tokens_after": 1272,
                  "tokens_before": 1272
                },
                "content_profiler": {
                  "latency_ms": 28.69,
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
                  "beam_search_latency_ms": 102.051,
                  "core_transforms": 11,
                  "ir_sections": 2,
                  "latency_ms": 104.67,
                  "optimizers": 7,
                  "tokens_after": 57,
                  "tokens_before": 1272,
                  "tokens_saved": 4861
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.014,
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
                  "latency_ms": 1.312,
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
                "latency_ms": 0.076,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.009433962264150943,
                "stable_tokens": 12,
                "tokens_after": 1272,
                "tokens_before": 1272
              },
              "content_profiler": {
                "latency_ms": 28.69,
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
                "beam_search_latency_ms": 102.051,
                "core_transforms": 11,
                "ir_sections": 2,
                "latency_ms": 104.67,
                "optimizers": 7,
                "tokens_after": 57,
                "tokens_before": 1272,
                "tokens_saved": 4861
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.014,
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
                "latency_ms": 1.312,
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
                "network_ms": 6407.589374808595,
                "network_stdev": 0.0,
                "total_ms": 6407.646582927555,
                "total_stdev": 0.0
              },
              "delta": -1711.9,
              "optimized": {
                "network_ms": 4648.608000017703,
                "network_stdev": 0.0,
                "pipeline_ms": 47.13791608810425,
                "pipeline_stdev": 0.0,
                "total_ms": 4695.746748941019,
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
              "baseline": "The user is asking for a comparison of when to use list comprehensions, dict comprehensions, set comprehensions, and generator expressions. This is a natural follow-up to the previous questions. I should provide a concise, practical guide that covers:\n\n1. List comprehensions - when you need all results in memory as a list, multiple times\n2. Dict comprehensions - when mapping keys to values\n3. Set comprehensions - when you need unique elements and set operations\n4. Generator expressions - when me",
              "optimized": "The user is asking for a comparison of when to use list comprehensions, dict comprehensions, set comprehensions, and generator expressions. This is a natural follow-up to the previous questions. I should provide a concise, practical guide that helps them choose the right tool.\n\nKey points to cover:\n1. List comprehensions: When you need all results in memory, need indexing/length, or need to iterate multiple times\n2. Dict comprehensions: When mapping keys to values, transforming dicts\n3. Set comp"
            },
            "scenario": "prefix_optimization",
            "telemetry": {
              "pipeline": {
                "latency_ms": 46.96798324584961,
                "tokens_in": 207,
                "tokens_out": 207,
                "transform_latency_ms": 43.812
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
                  "latency_ms": 0.14,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.42028985507246375,
                  "stable_tokens": 87,
                  "tokens_after": 207,
                  "tokens_before": 207
                },
                "content_profiler": {
                  "latency_ms": 9.794,
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
                  "beam_search_latency_ms": 32.66,
                  "core_transforms": 7,
                  "ir_sections": 11,
                  "latency_ms": 33.469,
                  "optimizers": 5,
                  "tokens_after": 207,
                  "tokens_before": 207,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.024,
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
                  "latency_ms": 0.384,
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
                "latency_ms": 0.14,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.42028985507246375,
                "stable_tokens": 87,
                "tokens_after": 207,
                "tokens_before": 207
              },
              "content_profiler": {
                "latency_ms": 9.794,
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
                "beam_search_latency_ms": 32.66,
                "core_transforms": 7,
                "ir_sections": 11,
                "latency_ms": 33.469,
                "optimizers": 5,
                "tokens_after": 207,
                "tokens_before": 207,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.024,
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
                "latency_ms": 0.384,
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
                "network_ms": 3769.130792003125,
                "network_stdev": 0.0,
                "total_ms": 3769.131625071168,
                "total_stdev": 0.0
              },
              "delta": -411.156,
              "optimized": {
                "network_ms": 3219.9558748397976,
                "network_stdev": 0.0,
                "pipeline_ms": 138.01729190163314,
                "pipeline_stdev": 0.0,
                "total_ms": 3357.975207734853,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.9711,
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
              "baseline": "The user wants me to review code changes across 10 files (module_0.py through module_9.py) and summarize the common issues.\n\nLooking at the code:\n- All 10 files have identical code structure: a function `process_data_N` that takes a list of dicts and returns a list of dicts.\n- Each has a TODO comment about optimizing a loop.\n- Each uses an explicit for loop with `.append()` to build a result list.\n- Each filters items where `status` is \"active\" and transforms them into new dicts with `id` and `v",
              "optimized": "The user wants me to review code changes across 10 files (module_0.py through module_9.py) and summarize the common issues.\n\nLooking at all the files, they all contain identical code:\n```python\ndef process_data_N(data: list[dict]) -> list[dict]:\n    # TODO: optimize this loop\n    result = []\n    for item in data:\n        if item.get(\"status\") == \"active\":\n            result.append({\"id\": item[\"id\"], \"value\": item[\"value\"] * 2})\n    return result\n```\n\nAnd all have the same issue noted: \"Line 4-6 "
            },
            "scenario": "code_review_patterns",
            "telemetry": {
              "pipeline": {
                "latency_ms": 137.94898986816406,
                "tokens_in": 992,
                "tokens_out": 992,
                "transform_latency_ms": 136.123
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
                  "latency_ms": 0.066,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.008064516129032258,
                  "stable_tokens": 8,
                  "tokens_after": 992,
                  "tokens_before": 992
                },
                "content_profiler": {
                  "latency_ms": 29.66,
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
                  "beam_search_latency_ms": 101.154,
                  "core_transforms": 11,
                  "ir_sections": 14,
                  "latency_ms": 105.156,
                  "optimizers": 7,
                  "tokens_after": 992,
                  "tokens_before": 992,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.022,
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
                  "latency_ms": 1.219,
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
                "latency_ms": 0.066,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.008064516129032258,
                "stable_tokens": 8,
                "tokens_after": 992,
                "tokens_before": 992
              },
              "content_profiler": {
                "latency_ms": 29.66,
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
                "beam_search_latency_ms": 101.154,
                "core_transforms": 11,
                "ir_sections": 14,
                "latency_ms": 105.156,
                "optimizers": 7,
                "tokens_after": 992,
                "tokens_before": 992,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.022,
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
                "latency_ms": 1.219,
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
                "network_ms": 3796.135041862726,
                "network_stdev": 0.0,
                "total_ms": 3796.136707998812,
                "total_stdev": 0.0
              },
              "delta": 2979.538,
              "optimized": {
                "network_ms": 6627.755832858384,
                "network_stdev": 0.0,
                "pipeline_ms": 147.91779103688896,
                "pipeline_stdev": 0.0,
                "total_ms": 6775.6748320534825,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": false,
              "semantic_similarity": 0.7958,
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
              "baseline": "The user wants me to summarize/document 20 API endpoints concisely. They all follow the exact same pattern:\n\n- GET /api/v1/resource_{N} where N ranges from 0 to 19\n- Same parameters: id (required string), limit (optional int, default 50), offset (optional int)\n- Same response structure: {\"id\": \"N\", \"name\": \"Resource N\", \"status\": \"active\"}\n- Same errors: 404, 429, 500\n\nInstead of listing all 20 individually, I should group them since they're identical in structure except for the resource number ",
              "optimized": "The user wants me to document 20 API endpoints concisely. Looking at the input, all endpoints follow the exact same pattern:\n\n- GET /api/v1/resource_{N} where N ranges from 0 to 19\n- Same parameters: id (required string), limit (optional int, default 50), offset (optional int)\n- Same response structure: {\"id\": \"N\", \"name\": \"Resource N\", \"status\": \"active\"}\n- Same errors: 404, 429, 500\n\nSince the user asked for concise documentation, I should not list all 20 endpoints individually with repetitive"
            },
            "scenario": "api_docs_summarization",
            "telemetry": {
              "pipeline": {
                "latency_ms": 147.85194396972656,
                "tokens_in": 1893,
                "tokens_out": 1893,
                "transform_latency_ms": 146.604
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
                  "latency_ms": 0.041,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.003169572107765452,
                  "stable_tokens": 6,
                  "tokens_after": 1893,
                  "tokens_before": 1893
                },
                "content_profiler": {
                  "latency_ms": 33.41,
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
                  "beam_search_latency_ms": 105.221,
                  "core_transforms": 11,
                  "ir_sections": 43,
                  "latency_ms": 111.497,
                  "optimizers": 7,
                  "tokens_after": 1893,
                  "tokens_before": 1893,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.019,
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
                  "latency_ms": 1.638,
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
                "latency_ms": 0.041,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.003169572107765452,
                "stable_tokens": 6,
                "tokens_after": 1893,
                "tokens_before": 1893
              },
              "content_profiler": {
                "latency_ms": 33.41,
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
                "beam_search_latency_ms": 105.221,
                "core_transforms": 11,
                "ir_sections": 43,
                "latency_ms": 111.497,
                "optimizers": 7,
                "tokens_after": 1893,
                "tokens_before": 1893,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.019,
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
                "latency_ms": 1.638,
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
                "network_ms": 2087.039708159864,
                "network_stdev": 0.0,
                "total_ms": 2087.0401251595467,
                "total_stdev": 0.0
              },
              "delta": 100.152,
              "optimized": {
                "network_ms": 2149.3552499450743,
                "network_stdev": 0.0,
                "pipeline_ms": 37.835124880075455,
                "pipeline_stdev": 0.0,
                "total_ms": 2187.1917909011245,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.9583,
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
              "baseline": "The user is asking for the average temperature between NYC and SF based on the previous weather results. NYC was 72\u00b0F and SF was 65\u00b0F. I need to calculate the average: (72 + 65) / 2 = 137 / 2 = 68.5\u00b0F.\n\nI should use the calculate tool to do this calculation.",
              "optimized": "The user is asking for the average temperature between NYC and SF based on the previous weather results. NYC was 72\u00b0F and SF was 65\u00b0F. I need to calculate the average: (72 + 65) / 2 = 137 / 2 = 68.5\u00b0F. I should use the calculator tool to do this calculation."
            },
            "scenario": "tool_call_preservation",
            "telemetry": {
              "pipeline": {
                "latency_ms": 37.75477409362793,
                "tokens_in": 133,
                "tokens_out": 133,
                "transform_latency_ms": 35.861
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
                  "latency_ms": 0.013,
                  "tokens_after": 133,
                  "tokens_before": 133
                },
                "content_profiler": {
                  "latency_ms": 3.24,
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
                  "beam_search_latency_ms": 31.583,
                  "core_transforms": 7,
                  "ir_sections": 5,
                  "latency_ms": 32.387,
                  "optimizers": 5,
                  "tokens_after": 133,
                  "tokens_before": 133,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.051,
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
                  "latency_ms": 0.171,
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
                "latency_ms": 0.013,
                "tokens_after": 133,
                "tokens_before": 133
              },
              "content_profiler": {
                "latency_ms": 3.24,
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
                "beam_search_latency_ms": 31.583,
                "core_transforms": 7,
                "ir_sections": 5,
                "latency_ms": 32.387,
                "optimizers": 5,
                "tokens_after": 133,
                "tokens_before": 133,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.051,
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
                "latency_ms": 0.171,
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
                "completion_tokens": 94,
                "prompt_tokens": 167,
                "total_tokens": 261
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 94,
                "prompt_tokens": 167,
                "total_tokens": 261
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
                "network_ms": 3199.1683749947697,
                "network_stdev": 0.0,
                "total_ms": 3199.1700418293476,
                "total_stdev": 0.0
              },
              "delta": -126.007,
              "optimized": {
                "network_ms": 2952.3901669308543,
                "network_stdev": 0.0,
                "pipeline_ms": 120.77204207889736,
                "pipeline_stdev": 0.0,
                "total_ms": 3073.16283416003,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.9952,
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
              "baseline": "The user wants me to analyze the provided JSON data and return the top 5 users as JSON.\n\nLooking at the data:\n- There are 50 users (id 0 through 49)\n- Each user has an id, name, and score\n- The scores range from 0 to 490, increasing by 10 for each subsequent user\n- So the top 5 users by score would be the ones with the highest scores:\n  1. User_49: 490\n  2. User_48: 480\n  3. User_47: 470\n  4. User_46: 460\n  5. User_45: 450\n\nI need to return this as JSON. The user said \"Return top 5 users as JSON",
              "optimized": "The user wants me to analyze the provided JSON data and return the top 5 users as JSON.\n\nLooking at the data:\n- There are 50 users (id 0 through 49)\n- Each user has a score that increases by 10 for each subsequent user\n- User_0: score 0\n- User_1: score 10\n- ...\n- User_49: score 490\n\nSo the top 5 users by score would be the ones with the highest scores:\n1. User_49: 490\n2. User_48: 480\n3. User_47: 470\n4. User_46: 460\n5. User_45: 450\n\nI need to return this as JSON. The user said \"Return top 5 users"
            },
            "scenario": "json_response_format",
            "telemetry": {
              "pipeline": {
                "latency_ms": 120.7113265991211,
                "tokens_in": 583,
                "tokens_out": 583,
                "transform_latency_ms": 119.508
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
                  "latency_ms": 0.07,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.0137221269296741,
                  "stable_tokens": 8,
                  "tokens_after": 583,
                  "tokens_before": 583
                },
                "content_profiler": {
                  "latency_ms": 14.062,
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
                  "beam_search_latency_ms": 103.424,
                  "core_transforms": 11,
                  "ir_sections": 5,
                  "latency_ms": 104.714,
                  "optimizers": 7,
                  "tokens_after": 583,
                  "tokens_before": 583,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.015,
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
                  "latency_ms": 0.647,
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
                "latency_ms": 0.07,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.0137221269296741,
                "stable_tokens": 8,
                "tokens_after": 583,
                "tokens_before": 583
              },
              "content_profiler": {
                "latency_ms": 14.062,
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
                "beam_search_latency_ms": 103.424,
                "core_transforms": 11,
                "ir_sections": 5,
                "latency_ms": 104.714,
                "optimizers": 7,
                "tokens_after": 583,
                "tokens_before": 583,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.015,
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
                "latency_ms": 0.647,
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
                "network_ms": 2897.1640840172768,
                "network_stdev": 0.0,
                "total_ms": 2897.16483396478,
                "total_stdev": 0.0
              },
              "delta": 2366.567,
              "optimized": {
                "network_ms": 5223.435749998316,
                "network_stdev": 0.0,
                "pipeline_ms": 40.28950002975762,
                "pipeline_stdev": 0.0,
                "total_ms": 5263.731499901041,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.9886,
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
              "baseline": "The user is asking what the next delta should contain given the repeated context. The context shows a series of design notes (0 through 11) all saying \"keep manifest ordering stable across turns.\"\n\nAs LATTICE's production evaluator, my instructions are to:\n1. Preserve stable context\n2. Prefer exact-prefix reuse\n3. Summarize only the deltas\n\nThe context is currently a list of design notes 0-11, all identical. The \"next delta\" in this pattern would logically be design note 12, continuing the seque",
              "optimized": "The user is asking what the next \"delta\" should contain given the repeated context. The context consists of design notes 0 through 11, all saying \"keep manifest ordering stable across turns.\"\n\nThe system prompt says: \"Preserve stable context, prefer exact-prefix reuse, and summarize only the deltas.\"\n\nSo I need to:\n1. Preserve stable context - the manifest ordering stable across turns is the stable context\n2. Prefer exact-prefix reuse - reuse the prefix of previous turns\n3. Summarize only the de"
            },
            "scenario": "cache_arbitrage_prefix",
            "telemetry": {
              "pipeline": {
                "latency_ms": 40.21787643432617,
                "tokens_in": 241,
                "tokens_out": 241,
                "transform_latency_ms": 38.727
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
                  "latency_ms": 0.037,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.12448132780082988,
                  "stable_tokens": 30,
                  "tokens_after": 241,
                  "tokens_before": 241
                },
                "content_profiler": {
                  "latency_ms": 6.992,
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
                  "beam_search_latency_ms": 30.883,
                  "core_transforms": 7,
                  "ir_sections": 4,
                  "latency_ms": 31.363,
                  "optimizers": 5,
                  "tokens_after": 241,
                  "tokens_before": 241,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.012,
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
                  "latency_ms": 0.322,
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
                "latency_ms": 0.037,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.12448132780082988,
                "stable_tokens": 30,
                "tokens_after": 241,
                "tokens_before": 241
              },
              "content_profiler": {
                "latency_ms": 6.992,
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
                "beam_search_latency_ms": 30.883,
                "core_transforms": 7,
                "ir_sections": 4,
                "latency_ms": 31.363,
                "optimizers": 5,
                "tokens_after": 241,
                "tokens_before": 241,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.012,
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
                "latency_ms": 0.322,
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
                "network_ms": 3171.6742920689285,
                "network_stdev": 0.0,
                "total_ms": 3171.6782920993865,
                "total_stdev": 0.0
              },
              "delta": -101.101,
              "optimized": {
                "network_ms": 3031.095332931727,
                "network_stdev": 0.0,
                "pipeline_ms": 39.47862493805587,
                "pipeline_stdev": 0.0,
                "total_ms": 3070.577416801825,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": false,
              "semantic_similarity": 0.8121,
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
              "baseline": "The user wants me to compress repeated language losslessly. The input is a repetition of the sentence \"The session manifest should remain stable and the session manifest should remain stable.\" many times.\n\nLet me count the repetitions:\n1. \"The session manifest should remain stable and the session manifest should remain stable.\"\n2. \" The session manifest should remain stable and the session manifest should remain stable.\"\n3. \"The session manifest should remain stable and the session manifest shou",
              "optimized": "The user wants me to compress repeated language losslessly. The input is:\n\n`<crossref_1>. <crossref_1>.<crossref_1>. <crossref_1>.<crossref_1>. <crossref_1>.<crossref_1>. <crossref_1>.`\n\nThis is a string that repeats the pattern `<crossref_1>.` with some variations in spacing. Let me look closely:\n\nOriginal: `<crossref_1>. <crossref_1>.<crossref_1>. <crossref_1>.<crossref_1>. <crossref_1>.<crossref_1>. <crossref_1>.`\n\nLet me parse this:\n1. `<crossref_1>.`\n2. ` `\n3. `<crossref_1>.`\n4. `<crossref_"
            },
            "scenario": "dictionary_repetition",
            "telemetry": {
              "pipeline": {
                "latency_ms": 39.24989700317383,
                "tokens_in": 192,
                "tokens_out": 42,
                "transform_latency_ms": 37.312
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
                  "latency_ms": 0.081,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.046875,
                  "stable_tokens": 9,
                  "tokens_after": 192,
                  "tokens_before": 192
                },
                "content_profiler": {
                  "latency_ms": 4.161,
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
                  "beam_search_latency_ms": 32.195,
                  "core_transforms": 7,
                  "ir_sections": 2,
                  "latency_ms": 32.75,
                  "optimizers": 5,
                  "tokens_after": 42,
                  "tokens_before": 192,
                  "tokens_saved": 600
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.019,
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
                  "latency_ms": 0.3,
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
                "latency_ms": 0.081,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.046875,
                "stable_tokens": 9,
                "tokens_after": 192,
                "tokens_before": 192
              },
              "content_profiler": {
                "latency_ms": 4.161,
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
                "beam_search_latency_ms": 32.195,
                "core_transforms": 7,
                "ir_sections": 2,
                "latency_ms": 32.75,
                "optimizers": 5,
                "tokens_after": 42,
                "tokens_before": 192,
                "tokens_saved": 600
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.019,
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
                "latency_ms": 0.3,
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
                "network_ms": 4195.0421668589115,
                "network_stdev": 0.0,
                "total_ms": 4195.045958971605,
                "total_stdev": 0.0
              },
              "delta": 225.604,
              "optimized": {
                "network_ms": 4379.650208866224,
                "network_stdev": 0.0,
                "pipeline_ms": 40.99754197522998,
                "pipeline_stdev": 0.0,
                "total_ms": 4420.650208834559,
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
              "baseline": "The user wants me to summarize something, but they haven't actually provided the text to summarize. They said \"Please summarize this\" but there's no content attached. They also mentioned \"Keep the meaning. Remove extra spaces.\"\n\nWait, looking at the conversation:\n- User message: \"Please   summarize this.  \\n\\n\\nKeep the meaning.   Remove   extra   spaces.   \"\n- There is no actual text/content provided to summarize.\n\nSo I need to ask them to provide the text they want me to summarize. Or perhaps ",
              "optimized": "The user wants me to summarize something, but they haven't actually provided the text to summarize. They said \"Please summarize this.\" but there's no content attached. However, looking at the second part: \"Keep the meaning. Remove extra spaces.\" - this might be instructions for how to summarize, OR they might be asking me to clean up the text \"Please summarize this.\" itself?\n\nWait, let me re-read the user's message:\n\n\"Please   summarize this.  \n\n\nKeep the meaning.   Remove   extra   spaces.   \"\n"
            },
            "scenario": "cleanup_noise",
            "telemetry": {
              "pipeline": {
                "latency_ms": 40.927886962890625,
                "tokens_in": 47,
                "tokens_out": 47,
                "transform_latency_ms": 38.297
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
                  "latency_ms": 0.15,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.23404255319148937,
                  "stable_tokens": 11,
                  "tokens_after": 47,
                  "tokens_before": 47
                },
                "content_profiler": {
                  "latency_ms": 5.555,
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
                  "beam_search_latency_ms": 30.816,
                  "core_transforms": 7,
                  "ir_sections": 4,
                  "latency_ms": 32.217,
                  "optimizers": 5,
                  "tokens_after": 47,
                  "tokens_before": 47,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.043,
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
                  "latency_ms": 0.333,
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
                "latency_ms": 0.15,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.23404255319148937,
                "stable_tokens": 11,
                "tokens_after": 47,
                "tokens_before": 47
              },
              "content_profiler": {
                "latency_ms": 5.555,
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
                "beam_search_latency_ms": 30.816,
                "core_transforms": 7,
                "ir_sections": 4,
                "latency_ms": 32.217,
                "optimizers": 5,
                "tokens_after": 47,
                "tokens_before": 47,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.043,
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
                "latency_ms": 0.333,
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
                "network_ms": 7658.7659157812595,
                "network_stdev": 0.0,
                "total_ms": 7658.766915788874,
                "total_stdev": 0.0
              },
              "delta": -3656.34,
              "optimized": {
                "network_ms": 3930.2560419309884,
                "network_stdev": 0.0,
                "pipeline_ms": 72.16029218398035,
                "pipeline_stdev": 0.0,
                "total_ms": 4002.427084138617,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.9583,
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
              "baseline": "The user is asking me to explain the root cause once, following up on the previous interaction where I summarized an incident report as \"The incident report shows a retry loop.\" However, I don't actually have the content of the incident report. The user hasn't provided any specific incident report text, details, or context beyond the initial request to summarize it.\n\nWait, looking back at the conversation:\n- User: \"Summarize the incident report.\"\n- Assistant: \"The incident report shows a retry l",
              "optimized": "The user is asking me to explain the root cause of the incident described in the previous turns. However, looking at the conversation history, the user said \"Summarize the incident report\" twice, but never actually provided an incident report. There is no content to summarize or analyze for a root cause.\n\nI need to point out that no incident report was provided, so I cannot explain a root cause. I should ask them to share the report or provide the details.\n\nWait, let me check if there's any hidd"
            },
            "scenario": "message_dedup_turns",
            "telemetry": {
              "pipeline": {
                "latency_ms": 72.06511497497559,
                "tokens_in": 65,
                "tokens_out": 65,
                "transform_latency_ms": 69.316
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
                  "latency_ms": 0.177,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.1076923076923077,
                  "stable_tokens": 7,
                  "tokens_after": 65,
                  "tokens_before": 65
                },
                "content_profiler": {
                  "latency_ms": 7.199,
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
                  "beam_search_latency_ms": 60.632,
                  "compression_limited": true,
                  "core_transforms": 7,
                  "ir_sections": 6,
                  "latency_ms": 60.63158297911286,
                  "optimizers": 7,
                  "tokens_after": 146,
                  "tokens_before": 198,
                  "tokens_saved": 52
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.071,
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
                  "latency_ms": 0.311,
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
                "latency_ms": 0.177,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.1076923076923077,
                "stable_tokens": 7,
                "tokens_after": 65,
                "tokens_before": 65
              },
              "content_profiler": {
                "latency_ms": 7.199,
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
                "beam_search_latency_ms": 60.632,
                "compression_limited": true,
                "core_transforms": 7,
                "ir_sections": 6,
                "latency_ms": 60.63158297911286,
                "optimizers": 7,
                "tokens_after": 146,
                "tokens_before": 198,
                "tokens_saved": 52
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.071,
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
                "latency_ms": 0.311,
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
                "network_ms": 3770.789625123143,
                "network_stdev": 0.0,
                "total_ms": 3770.790125010535,
                "total_stdev": 0.0
              },
              "delta": 429.333,
              "optimized": {
                "network_ms": 4011.3334578927606,
                "network_stdev": 0.0,
                "pipeline_ms": 188.78883309662342,
                "pipeline_stdev": 0.0,
                "total_ms": 4200.123041169718,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.8958,
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
              "baseline": "The user wants me to summarize an incident clearly while keeping the root cause. They've provided a repetitive text that describes:\n\n1. Root cause: Cache invalidation storm leading to cascading slowdown\n2. Mechanism: Repeated recomputation of expensive work\n3. Impact: Increased queueing delay and higher tail latency\n4. Observation: Request path remained logically correct, but runtime degraded\n5. Requirement: Retain root cause, user-visible impact, and mitigation plan (though no specific mitigati",
              "optimized": "The user wants me to summarize an incident clearly and keep the root cause. However, they haven't provided any specific incident details in their message. \n\nI need to:\n1. Acknowledge that no incident details were provided\n2. Ask them to provide the incident information so I can summarize it properly\n3. Or provide a template/framework for how I would summarize an incident once they provide the details\n\nSince the instruction is to summarize \"the incident\" but there's no incident described, I shoul"
            },
            "scenario": "rate_distortion_longform",
            "telemetry": {
              "pipeline": {
                "latency_ms": 188.69280815124512,
                "tokens_in": 773,
                "tokens_out": 20,
                "transform_latency_ms": 187.004
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
                  "latency_ms": 0.083,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.016817593790426907,
                  "stable_tokens": 13,
                  "tokens_after": 773,
                  "tokens_before": 773
                },
                "content_profiler": {
                  "latency_ms": 83.126,
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
                  "beam_search_latency_ms": 100.773,
                  "core_transforms": 11,
                  "ir_sections": 3,
                  "latency_ms": 102.961,
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
                  "latency_ms": 0.821,
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
                "latency_ms": 0.083,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.016817593790426907,
                "stable_tokens": 13,
                "tokens_after": 773,
                "tokens_before": 773
              },
              "content_profiler": {
                "latency_ms": 83.126,
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
                "beam_search_latency_ms": 100.773,
                "core_transforms": 11,
                "ir_sections": 3,
                "latency_ms": 102.961,
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
                "latency_ms": 0.821,
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
                "completion_tokens": 256,
                "prompt_tokens": 23,
                "total_tokens": 279
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
                "network_ms": 3378.3120838925242,
                "network_stdev": 0.0,
                "total_ms": 3378.3165419008583,
                "total_stdev": 0.0
              },
              "delta": 1124.237,
              "optimized": {
                "network_ms": 4381.230250000954,
                "network_stdev": 0.0,
                "pipeline_ms": 121.32208305411041,
                "pipeline_stdev": 0.0,
                "total_ms": 4502.553083002567,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.9615,
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
              "optimized": "The user is asking: \"Which documents mention the error path and request ids?\"\n\nLooking at the provided context, there are multiple documents (Document 0 through Document 17, and potentially Document 18x which seems to be a typo for Document 18 or just a label). Each document follows a pattern:\n\n\"The subsystem X stores request id XXXX, trace code XXXX-YYYY, and rollout note for module_X. The question is which subset is most relevant to the error path.\"\n\nWait, let me look carefully at the provided"
            },
            "scenario": "context_budget_pressure",
            "telemetry": {
              "pipeline": {
                "latency_ms": 121.14500999450684,
                "tokens_in": 802,
                "tokens_out": 164,
                "transform_latency_ms": 119.296
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
                  "latency_ms": 0.082,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.012468827930174564,
                  "stable_tokens": 10,
                  "tokens_after": 802,
                  "tokens_before": 802
                },
                "content_profiler": {
                  "latency_ms": 13.212,
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
                  "beam_search_latency_ms": 103.239,
                  "core_transforms": 11,
                  "ir_sections": 3,
                  "latency_ms": 105.137,
                  "optimizers": 7,
                  "tokens_after": 164,
                  "tokens_before": 802,
                  "tokens_saved": 2554
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.019,
                  "prefix_tokens": 10,
                  "suffix_tokens": 792,
                  "tokens_after": 802,
                  "tokens_before": 802
                },
                "rate_distortion": {
                  "guarded": 1
                },
                "reference_optimizer": {
                  "rejected": "tokens_after (2260) > tokens_before (624) with no gain"
                },
                "reference_sub": {
                  "modified_count": 0,
                  "tokens_saved_estimate": 0,
                  "unique_refs": 0
                },
                "runtime_contract": {
                  "confidence": 0.575,
                  "latency_ms": 0.845,
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
                "latency_ms": 0.082,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.012468827930174564,
                "stable_tokens": 10,
                "tokens_after": 802,
                "tokens_before": 802
              },
              "content_profiler": {
                "latency_ms": 13.212,
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
                "beam_search_latency_ms": 103.239,
                "core_transforms": 11,
                "ir_sections": 3,
                "latency_ms": 105.137,
                "optimizers": 7,
                "tokens_after": 164,
                "tokens_before": 802,
                "tokens_saved": 2554
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.019,
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
              "reference_optimizer": {
                "latency_ms": 0,
                "rejected": "tokens_after (2260) > tokens_before (624) with no gain",
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
                "latency_ms": 0.845,
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
                "network_ms": 4499.289874918759,
                "network_stdev": 0.0,
                "total_ms": 4499.3004580028355,
                "total_stdev": 0.0
              },
              "delta": 288.875,
              "optimized": {
                "network_ms": 4728.091333061457,
                "network_stdev": 0.0,
                "pipeline_ms": 60.08245795965195,
                "pipeline_stdev": 0.0,
                "total_ms": 4788.17508299835,
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
              "baseline": "The user is asking me to prove an optimization contract is correct, but they haven't actually provided the optimization contract or the specific code transformation they want verified. However, looking at the previous context, there was a code review snippet showing multiple modules with a pattern like:\n\n```python\ndef process_data_X(data: list[dict]) -> list[dict]:\n    # TODO: optimize this loop\n    result = []\n    for item in data:\n        if item.get(\"status\") == \"active\":\n            result.a",
              "optimized": "The user is asking me to prove an optimization contract is correct, but they haven't actually provided the optimization contract or the specific code transformation they want verified. However, looking at the previous context, there was a code review showing multiple modules with a pattern that could be optimized using list comprehensions.\n\nThe user repeated the request many times (likely a copy-paste error or emphasis), but the core request is:\n1. State the theorem\n2. Derive the lemma\n3. Explai"
            },
            "scenario": "runtime_contract_pressure",
            "telemetry": {
              "pipeline": {
                "latency_ms": 60.00804901123047,
                "tokens_in": 3513,
                "tokens_out": 3513,
                "transform_latency_ms": 58.037
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
                  "latency_ms": 56.275,
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
                  "latency_ms": 0.102,
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
                  "latency_ms": 1.66,
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
                "latency_ms": 56.275,
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
                "latency_ms": 0.102,
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
                "latency_ms": 1.66,
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
                "network_ms": 5225.517374929041,
                "network_stdev": 0.0,
                "total_ms": 5225.524500012398,
                "total_stdev": 0.0
              },
              "delta": -381.195,
              "optimized": {
                "network_ms": 4703.016333049163,
                "network_stdev": 0.0,
                "pipeline_ms": 141.31204085424542,
                "pipeline_stdev": 0.0,
                "total_ms": 4844.329123850912,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": false,
              "semantic_similarity": 0.6963,
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
              "baseline": "The user wants me to summarize structured data (JSON and a table) without losing keys. This means I need to compress the information while preserving the structure/schema and key names, not just converting to prose.\n\nLooking at the JSON:\n- It's an array of 20 services (svc_0 through svc_19)\n- All have status \"ok\"\n- latency_ms ranges from 120 to 139 (incrementing by 1)\n\nLooking at the Table:\n- 24 employees (ID 0-23, Name Emp_0 through Emp_23)\n- All Department: Engineering\n- Salary: $100000 to $12",
              "optimized": "The user wants me to summarize structured data without losing keys. The data appears to be a table with rows representing employees. Let me analyze the structure:\n\nRow 1: 24x | 0 | Emp_0 | Engineering | $100000 | active |\nRow 2: - | 0 | Emp_0 | Engineering | $100000 | active |\n...\nRow 24: - | 23 | Emp_23 | Engineering | $123000 | active |\n\nIt looks like there are 24 rows total (indices 0-23). The columns seem to be:\n1. Count/Index (24x, -, -, ...)\n2. ID (0-23)\n3. Employee Name (Emp_0 to Emp_23)\n"
            },
            "scenario": "grammar_json_table",
            "telemetry": {
              "pipeline": {
                "latency_ms": 141.251802444458,
                "tokens_in": 750,
                "tokens_out": 56,
                "transform_latency_ms": 139.209
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
                  "latency_ms": 0.098,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.014666666666666666,
                  "stable_tokens": 11,
                  "tokens_after": 750,
                  "tokens_before": 750
                },
                "content_profiler": {
                  "latency_ms": 32.363,
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
                  "beam_search_latency_ms": 103.808,
                  "core_transforms": 11,
                  "ir_sections": 2,
                  "latency_ms": 105.792,
                  "optimizers": 7,
                  "tokens_after": 56,
                  "tokens_before": 750,
                  "tokens_saved": 2776
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.021,
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
                  "latency_ms": 0.935,
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
                "latency_ms": 0.098,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.014666666666666666,
                "stable_tokens": 11,
                "tokens_after": 750,
                "tokens_before": 750
              },
              "content_profiler": {
                "latency_ms": 32.363,
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
                "beam_search_latency_ms": 103.808,
                "core_transforms": 11,
                "ir_sections": 2,
                "latency_ms": 105.792,
                "optimizers": 7,
                "tokens_after": 56,
                "tokens_before": 750,
                "tokens_saved": 2776
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.021,
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
                "latency_ms": 0.935,
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
                "network_ms": 3426.6376250889152,
                "network_stdev": 0.0,
                "total_ms": 3426.644208142534,
                "total_stdev": 0.0
              },
              "delta": -750.961,
              "optimized": {
                "network_ms": 2637.4379999469966,
                "network_stdev": 0.0,
                "pipeline_ms": 38.24387490749359,
                "pipeline_stdev": 0.0,
                "total_ms": 2675.6827500648797,
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
              "baseline": "Hello! I'm doing well, thank you. As an AI, I don't experience feelings, but I'm functioning properly and ready to help you. How can I assist you today?",
              "optimized": "Hello! I'm doing well, thank you for asking. As an AI, I don't have feelings in the traditional sense, but I'm fully operational and ready to help you.\n\nHow can I assist you today?"
            },
            "scenario": "simple_baseline",
            "telemetry": {
              "pipeline": {
                "latency_ms": 38.1617546081543,
                "tokens_in": 7,
                "tokens_out": 7,
                "transform_latency_ms": 35.99
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
                  "latency_ms": 0.127,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.0,
                  "stable_tokens": 0,
                  "tokens_after": 7,
                  "tokens_before": 7
                },
                "content_profiler": {
                  "latency_ms": 4.42,
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
                  "beam_search_latency_ms": 30.461,
                  "core_transforms": 7,
                  "ir_sections": 1,
                  "latency_ms": 31.076,
                  "optimizers": 5,
                  "tokens_after": 7,
                  "tokens_before": 7,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.024,
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
                  "latency_ms": 0.342,
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
                "latency_ms": 0.127,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.0,
                "stable_tokens": 0,
                "tokens_after": 7,
                "tokens_before": 7
              },
              "content_profiler": {
                "latency_ms": 4.42,
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
                "beam_search_latency_ms": 30.461,
                "core_transforms": 7,
                "ir_sections": 1,
                "latency_ms": 31.076,
                "optimizers": 5,
                "tokens_after": 7,
                "tokens_before": 7,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.024,
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
                "latency_ms": 0.342,
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
                "completion_tokens": 214,
                "prompt_tokens": 14,
                "total_tokens": 228
              },
              "costs": {
                "baseline_total_usd": 0.0,
                "delta_usd": 0.0,
                "optimized_total_usd": 0.0
              },
              "optimized": {
                "completion_tokens": 152,
                "prompt_tokens": 14,
                "total_tokens": 166
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
                "network_ms": 8481.708584120497,
                "network_stdev": 0.0,
                "total_ms": 8481.711500091478,
                "total_stdev": 0.0
              },
              "delta": -4492.482,
              "optimized": {
                "network_ms": 3937.1932079084218,
                "network_stdev": 0.0,
                "pipeline_ms": 52.035291912034154,
                "pipeline_stdev": 0.0,
                "total_ms": 3989.22916688025,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": false,
              "semantic_similarity": 0.5,
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
              "baseline": "The user wants me to investigate a production error from a specific transaction. The error trace shows a sequence of 30 function calls (foo_0 through foo_29) with timestamps from 1700000000 to 1700000029, and the error message is \"Null pointer\".\n\nLet me analyze what I can see:\n1. Transaction ID: 550e8400-e29b-41d4-a716-446655440000\n2. Trace: 30 frames, foo_0 to foo_29, timestamps incrementing by 1 second each\n3. Error message: \"Null pointer\" (NullPointerException or similar)\n\nThis looks like a s",
              "optimized": "I see you've pasted what appears to be"
            },
            "scenario": "mixed_realworld",
            "telemetry": {
              "pipeline": {
                "latency_ms": 51.95498466491699,
                "tokens_in": 498,
                "tokens_out": 23,
                "transform_latency_ms": 49.906
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
                  "latency_ms": 0.072,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.018072289156626505,
                  "stable_tokens": 9,
                  "tokens_after": 498,
                  "tokens_before": 498
                },
                "content_profiler": {
                  "latency_ms": 18.291,
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
                  "beam_search_latency_ms": 30.397,
                  "core_transforms": 7,
                  "ir_sections": 4,
                  "latency_ms": 30.806,
                  "optimizers": 5,
                  "tokens_after": 23,
                  "tokens_before": 498,
                  "tokens_saved": 1902
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.016,
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
                  "latency_ms": 0.722,
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
                "latency_ms": 0.072,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.018072289156626505,
                "stable_tokens": 9,
                "tokens_after": 498,
                "tokens_before": 498
              },
              "content_profiler": {
                "latency_ms": 18.291,
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
                "beam_search_latency_ms": 30.397,
                "core_transforms": 7,
                "ir_sections": 4,
                "latency_ms": 30.806,
                "optimizers": 5,
                "tokens_after": 23,
                "tokens_before": 498,
                "tokens_saved": 1902
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.016,
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
                "latency_ms": 0.722,
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
                "network_ms": 4507.933957967907,
                "network_stdev": 0.0,
                "total_ms": 4507.940375013277,
                "total_stdev": 0.0
              },
              "delta": -616.58,
              "optimized": {
                "network_ms": 3813.488374929875,
                "network_stdev": 0.0,
                "pipeline_ms": 77.86974986083806,
                "pipeline_stdev": 0.0,
                "total_ms": 3891.360124805942,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.9904,
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
              "baseline": "The user wants me to analyze error logs from a production outage and identify the root cause and mitigation steps.\n\n Looking at the logs:\n - There are 5 services: service_0, service_1, service_2, service_3, service_4\n - Each service is getting \"Connection refused\" errors\n - The attempts go from 1 to 20 for each service\n - The pattern is sequential: service_0 attempt 1, service_1 attempt 1, ..., service_4 attempt 1, then service_0 attempt 2, etc.\n - This suggests a client/service mesh/load balanc",
              "optimized": "The user wants me to analyze error logs from a production outage and identify the root cause and mitigation steps.\n\n Looking at the logs:\n - There are 5 services: service_0, service_1, service_2, service_3, service_4\n - Each service is getting \"Connection refused\" errors\n - The attempts go from 1 to 20 for each service\n - The pattern is round-robin: service_0 attempt 1, service_1 attempt 1, ..., service_4 attempt 1, then service_0 attempt 2, etc.\n - All services are consistently refusing connect"
            },
            "scenario": "debugging_log_analysis",
            "telemetry": {
              "pipeline": {
                "latency_ms": 77.79407501220703,
                "tokens_in": 1445,
                "tokens_out": 1445,
                "transform_latency_ms": 55.366
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
                  "latency_ms": 0.134,
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
                  "beam_search_latency_ms": 34.203,
                  "core_transforms": 7,
                  "ir_sections": 0,
                  "latency_ms": 54.056,
                  "optimizers": 5,
                  "tokens_after": 1445,
                  "tokens_before": 1445,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.014,
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
                  "latency_ms": 1.162,
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
                "latency_ms": 0.134,
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
                "beam_search_latency_ms": 34.203,
                "core_transforms": 7,
                "ir_sections": 0,
                "latency_ms": 54.056,
                "optimizers": 5,
                "tokens_after": 1445,
                "tokens_before": 1445,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.014,
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
                "latency_ms": 1.162,
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
                "network_ms": 3674.2732918355614,
                "network_stdev": 0.0,
                "total_ms": 3674.273416865617,
                "total_stdev": 0.0
              },
              "delta": 630.044,
              "optimized": {
                "network_ms": 4240.5379170086235,
                "network_stdev": 0.0,
                "pipeline_ms": 63.777209026739,
                "pipeline_stdev": 0.0,
                "total_ms": 4304.317543050274,
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
              "baseline": "The user wants an explanation of a system crash based on provided data points:\n- Crash at 14:32 with error code E0503\n- Memory usage trajectory: 94% (14:30), 97% (14:31), 99% (14:32)\n- Three services restarted at 14:28\n\nI need to:\n1. Explain why the crash happened\n2. Identify the root cause\n3. Explain why the service restarts at 14:28 were insufficient\n\nFirst, let's analyze the error code E0503. In many systems (especially Linux, systemd, or application-specific codes), E0503 could indicate vari",
              "optimized": "The user wants an explanation of a system crash based on provided data points:\n- Crash at 14:32 with error code E0503\n- Memory usage trajectory: 94% (14:30), 97% (14:31), 99% (14:32)\n- Three services restarted at 14:28\n\nI need to:\n1. Explain why the crash happened\n2. Identify the root cause\n3. Explain why the service restarts at 14:28 were insufficient\n\nFirst, let's analyze the data:\n- Error code E0503: I should interpret what this likely means. E0503 isn't a universally standard error code, but"
            },
            "scenario": "reasoning_root_cause",
            "telemetry": {
              "pipeline": {
                "latency_ms": 63.70878219604492,
                "tokens_in": 85,
                "tokens_out": 85,
                "transform_latency_ms": 62.743
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
                  "latency_ms": 0.033,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.1411764705882353,
                  "stable_tokens": 12,
                  "tokens_after": 85,
                  "tokens_before": 85
                },
                "content_profiler": {
                  "latency_ms": 1.966,
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
                  "beam_search_latency_ms": 60.249,
                  "core_transforms": 7,
                  "ir_sections": 2,
                  "latency_ms": 60.615,
                  "optimizers": 7,
                  "tokens_after": 85,
                  "tokens_before": 85,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.021,
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
                  "latency_ms": 0.107,
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
                "latency_ms": 0.033,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.1411764705882353,
                "stable_tokens": 12,
                "tokens_after": 85,
                "tokens_before": 85
              },
              "content_profiler": {
                "latency_ms": 1.966,
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
                "beam_search_latency_ms": 60.249,
                "core_transforms": 7,
                "ir_sections": 2,
                "latency_ms": 60.615,
                "optimizers": 7,
                "tokens_after": 85,
                "tokens_before": 85,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.021,
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
                "latency_ms": 0.107,
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
                "network_ms": 4089.3257500138134,
                "network_stdev": 0.0,
                "total_ms": 4089.327207999304,
                "total_stdev": 0.0
              },
              "delta": 111.344,
              "optimized": {
                "network_ms": 4076.715749921277,
                "network_stdev": 0.0,
                "pipeline_ms": 123.95483395084739,
                "pipeline_stdev": 0.0,
                "total_ms": 4200.671667000279,
                "total_stdev": 0.0
              }
            },
            "model": "ollama-cloud/kimi-k2.6:cloud",
            "provider": "ollama-cloud",
            "quality": {
              "exact_match": false,
              "json_valid": null,
              "passed": true,
              "semantic_similarity": 0.9911,
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
              "baseline": "The user wants me to investigate transaction failures from the provided list and identify which transactions failed and what pattern they follow.\n\nLet me list out the failed transactions:\n- TX-0003: UUID 0x550e840000000003 [FAILED]\n- TX-0010: UUID 0x550e84000000000a [FAILED]\n- TX-0017: UUID 0x550e840000000011 [FAILED]\n- TX-0024: UUID 0x550e840000000018 [FAILED]\n- TX-0031: UUID 0x550e84000000001f [FAILED]\n- TX-0038: UUID 0x550e840000000026 [FAILED]\n- TX-0045: UUID 0x550e84000000002d [FAILED]\n\nNow",
              "optimized": "The user wants me to investigate transaction failures from the provided list and identify which transactions failed and what pattern they follow.\n\nLet me list out the failed transactions:\n- TX-0003: UUID 0x550e840000000003 [FAILED]\n- TX-0010: UUID 0x550e84000000000a [FAILED]\n- TX-0017: UUID 0x550e840000000011 [FAILED]\n- TX-0024: UUID 0x550e840000000018 [FAILED]\n- TX-0031: UUID 0x550e84000000001f [FAILED]\n- TX-0038: UUID 0x550e840000000026 [FAILED]\n- TX-0045: UUID 0x550e84000000002d [FAILED]\n\nNow"
            },
            "scenario": "repeated_identifier_investigation",
            "telemetry": {
              "pipeline": {
                "latency_ms": 123.87919425964355,
                "tokens_in": 517,
                "tokens_out": 517,
                "transform_latency_ms": 122.622
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
                  "latency_ms": 0.037,
                  "miss_count": 1,
                  "plan_skipped": "provider_no_cache_support",
                  "stability_score": 0.015473887814313346,
                  "stable_tokens": 8,
                  "tokens_after": 517,
                  "tokens_before": 517
                },
                "content_profiler": {
                  "latency_ms": 14.111,
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
                  "beam_search_latency_ms": 106.85,
                  "core_transforms": 11,
                  "ir_sections": 2,
                  "latency_ms": 108.023,
                  "optimizers": 7,
                  "tokens_after": 517,
                  "tokens_before": 517,
                  "tokens_saved": 0
                },
                "prefix_optimizer": {
                  "cache_hit": false,
                  "latency_ms": 0.013,
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
                  "latency_ms": 0.438,
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
                "latency_ms": 0.037,
                "miss_count": 1,
                "plan_skipped": "provider_no_cache_support",
                "stability_score": 0.015473887814313346,
                "stable_tokens": 8,
                "tokens_after": 517,
                "tokens_before": 517
              },
              "content_profiler": {
                "latency_ms": 14.111,
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
                "beam_search_latency_ms": 106.85,
                "core_transforms": 11,
                "ir_sections": 2,
                "latency_ms": 108.023,
                "optimizers": 7,
                "tokens_after": 517,
                "tokens_before": 517,
                "tokens_saved": 0
              },
              "prefix_optimizer": {
                "cache_hit": false,
                "latency_ms": 0.013,
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
                "latency_ms": 0.438,
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
          "avg_pipeline_latency_ms": 91.6265,
          "avg_quality_score": 0.8764,
          "avg_reduction_ratio": 0.2868,
          "passed_scenarios": 15,
          "total_scenarios": 21,
          "total_token_savings": 8722
        },
        "timestamp": 1779341638.277458
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
| total_passed | 6 |
| pass_rate | 0.3333 |

### Provider Validation Results

| Scenario | Provider | Task Eq. | Token Pass | Overall | Tokens Before | Tokens After | Risk | Forbidden Applied |
| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |
| uuid_deduplication | ollama-cloud | 0.898 | pass | pass | 353 | 353 | unknown | none |
| tool_output_filtering | ollama-cloud | 0.723 | pass | FAIL | 3770 | 3770 | unknown | rate_distortion |
| table_compression | ollama-cloud | 0.764 | pass | FAIL | 1272 | 57 | unknown | rate_distortion |
| prefix_optimization | ollama-cloud | 0.975 | pass | pass | 207 | 207 | unknown | none |
| code_review_patterns | ollama-cloud | 0.945 | pass | FAIL | 992 | 992 | unknown | rate_distortion |
| api_docs_summarization | ollama-cloud | 0.965 | pass | FAIL | 1893 | 1893 | unknown | rate_distortion |
| tool_call_preservation | ollama-cloud | 0.958 | pass | FAIL | 133 | 133 | unknown | reference_sub |
| json_response_format | ollama-cloud | 1.000 | pass | FAIL | 583 | 583 | unknown | rate_distortion |
| cache_arbitrage_prefix | ollama-cloud | 0.963 | pass | pass | 241 | 241 | unknown | none |
| dictionary_repetition | ollama-cloud | 0.790 | pass | FAIL | 192 | 42 | unknown | none |
| cleanup_noise | ollama-cloud | 0.925 | pass | FAIL | 47 | 47 | unknown | reference_sub |
| message_dedup_turns | ollama-cloud | 0.958 | pass | pass | 65 | 65 | unknown | none |
| rate_distortion_longform | ollama-cloud | 0.917 | pass | pass | 773 | 20 | unknown | none |
| context_budget_pressure | ollama-cloud | 0.995 | pass | FAIL | 802 | 164 | unknown | rate_distortion |
| runtime_contract_pressure | ollama-cloud | 0.917 | pass | pass | 3513 | 3513 | unknown | none |
| grammar_json_table | ollama-cloud | 0.694 | pass | FAIL | 750 | 56 | unknown | rate_distortion |
| simple_baseline | ollama-cloud | 0.988 | pass | FAIL | 7 | 7 | unknown | message_dedup, reference_sub |
| mixed_realworld | ollama-cloud | 0.477 | pass | FAIL | 498 | 22 | unknown | none |

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
| message_dedup_turns | PASS: optimized output preserves meaning well | none | none | none |
| rate_distortion_longform | PASS: optimized output preserves meaning well | none | none | none |
| context_budget_pressure | PASS: optimized output preserves meaning well | none | none | none |
| runtime_contract_pressure | PASS: optimized output preserves meaning well | none | none | none |
| grammar_json_table | FAIL: significant meaning drift | none | none | none |
| simple_baseline | PASS: optimized output preserves meaning well | none | none | none |
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
| message_dedup_turns | unknown | 0.0 | reference_sub | none | PASS: optimized output preserves meaning well |
| rate_distortion_longform | unknown | 0.0 | rate_distortion | none | PASS: optimized output preserves meaning well |
| context_budget_pressure | unknown | 0.0 | reference_sub | none | PASS: optimized output preserves meaning well |
| runtime_contract_pressure | unknown | 0.0 | none | none | PASS: optimized output preserves meaning well |
| grammar_json_table | unknown | 0.0 | reference_sub | none | FAIL: significant meaning drift |
| simple_baseline | unknown | 0.0 | none | none | PASS: optimized output preserves meaning well |
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
    "manifest_id": "mf-5a632a2c72d9f138",
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
    "response": "{\"ok\": true, \"path\": \"/v1/chat/completions\", \"host\": \"127.0.0.1:61832\"}",
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
    "created_at": 1779341637.056675,
    "last_accessed_at": 1779341637.056703,
    "manifest": {
      "anchor_hash": "d5cb231de5479c8ce897b513f33edf50c0475228dd036e31d303b32153229c04",
      "anchor_version": 0,
      "created_at": 1779341637.056672,
      "manifest_id": "mf-bef036c0762ad67d",
      "metadata": {
        "model": "openai/gpt-4o-mini",
        "provider": "openai"
      },
      "segments": [
        {
          "created_at": 1779341637.056638,
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
          "created_at": 1779341637.056618,
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
          "created_at": 1779341637.0566459,
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
      "session_id": "lattice-AaIxkhJdeqUi7xoG0GnhrQ"
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
    "session_id": "lattice-AaIxkhJdeqUi7xoG0GnhrQ",
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
| avg_pipeline_latency_ms | 46.7111 |
| avg_quality_score | 0.9737 |

- Benchmark provider: `openai`
- Benchmark model: `gpt-4`
- Scenarios: `8`
- Quality passed: `7`
- Total token savings: `369`
- Avg reduction ratio: `0.1704`
- Avg pipeline latency: `46.7111 ms`
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
