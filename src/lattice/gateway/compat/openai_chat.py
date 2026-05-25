"""OpenAI chat compat."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import status
from fastapi.responses import JSONResponse

from lattice.cache.semantic import assemble_cached_response
from lattice.core.context import TransformContext
from lattice.core.result import is_err, unwrap
from lattice.gateway.compat.headers import _extract_cached_tokens, _runtime_header_values
from lattice.gateway.compat.openai_chat_deps import ChatCompatDeps
from lattice.gateway.compat.openai_chat_ws import chat_completions_websocket_passthrough
from lattice.planner.runtime_state import (
    get_canonical_request_value,
    persist_execution_plan_state,
    persist_session_plan_state,
    sum_expected_cached_tokens,
)
from lattice.proxy.middleware import attach_routing_headers
from lattice.telemetry.agent_stats import identify_agent
from lattice.telemetry.cost_estimator import normalize_usage
from lattice.telemetry.downgrade import TransportOutcome

Handler = Callable[..., Awaitable[Any]]

__all__ = ["ChatCompatDeps", "chat_completions_websocket_passthrough", "make_chat_completion_handler"]


def make_chat_completion_handler(deps: ChatCompatDeps) -> Handler:
    """Create OpenAI-compatible chat completions handler."""

    async def _handle_chat_completion(
        fastapi_request: Any,
        body: dict[str, Any],
        x_lattice_session_id: str | None = None,
        x_lattice_disable_transforms: str | None = None,
        x_lattice_client_profile: str | None = None,
        x_lattice_provider: str | None = None,
        authorization: str | None = None,
        x_api_key: str | None = None,
    ) -> Any:
        request = deps.deserialize_openai_request(body)
        request.extra_headers["x-lattice-session-id"] = x_lattice_session_id or ""

        client_api_key = None
        if authorization and authorization.startswith("Bearer "):
            client_api_key = authorization[7:]

        # Preserve additional auth headers so upstream provider adapters can use them
        if x_api_key:
            request.extra_headers["x-api-key"] = x_api_key

        # ------------------------------------------------------------------
        # Provider detection — zero defaults, zero fallbacks
        # ------------------------------------------------------------------
        from lattice.gateway.routing import (
            ProviderAmbiguityError,
            ProviderNotDetectedError,
            ProviderRouter,
            RequestSignals,
        )

        router = ProviderRouter(deps.provider.registry)
        signals = RequestSignals.from_request(
            method="POST",
            path="/v1/chat/completions",
            headers={
                **request.extra_headers,
                "authorization": authorization or "",
                "x-api-key": x_api_key or "",
            },
            body=body,
            model=request.model or body.get("model", ""),
        )

        try:
            result = router.resolve(signals)
            provider_name = result.provider
        except (ProviderNotDetectedError, ProviderAmbiguityError) as exc:
            deps.logger.warning("chat_completion_provider_detection_failed", error=str(exc))
            return JSONResponse(
                {"error": "provider_detection_failed", "message": str(exc)},
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        # ------------------------------------------------------------------
        # Base URL, delta mode, and client profile
        # ------------------------------------------------------------------
        delta_mode = "delta" if request.metadata.get("_delta_wire") else ""
        base_url = deps.provider.provider_base_urls.get(provider_name)
        if not base_url:
            return JSONResponse(
                {
                    "error": "provider_not_configured",
                    "message": (
                        f"No base URL configured for provider '{provider_name}'. "
                        f"Set provider_base_urls['{provider_name}'] or "
                        f"LATTICE_PROVIDER_BASE_URLS env var."
                    ),
                },
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        http_version = deps.provider.pool.get_http_version(provider_name, base_url)

        # Throttled maintenance: clean up abandoned stream state and stale cache entries
        if deps.maintenance is not None:
            try:
                maintenance_results = await deps.maintenance.tick()
                for name, result in maintenance_results.items():
                    if result.did_work:
                        deps.logger.debug(
                            "maintenance_tick_did_work",
                            name=name,
                            stale_streams_removed=result.stale_streams_removed,
                            stale_cache_entries_removed=result.stale_cache_entries_removed,
                        )
            except (TypeError, AttributeError, KeyError, RuntimeError):
                # Never block the request path for maintenance failures.
                deps.logger.warning("maintenance_tick_failed", exc_info=True)
                pass

        # ---- Sampled post-transform guard production validation ----
        # High-risk requests are flagged for model-in-the-loop validation.
        # Low-risk requests are sampled at 1% for continuous quality monitoring.
        risk_data = get_canonical_request_value(request, None, "_lattice_risk_score", {})
        risk_level = risk_data.get("level", "unknown") if isinstance(risk_data, dict) else "unknown"
        task_data = get_canonical_request_value(request, None, "_lattice_task_classification", {})
        task_class = (
            task_data.get("task_class", "unknown") if isinstance(task_data, dict) else "unknown"
        )

        if risk_level in ("HIGH", "CRITICAL") or task_class in ("debugging", "reasoning"):
            deps.logger.info(
                "post_transform_guard_validation_candidate",
                risk_level=risk_level,
                task_class=task_class,
                session_id=x_lattice_session_id or "",
            )
            deps.metrics.increment("lattice_post_transform_guard_high_risk_flagged")
            # Record validation intent in metadata — production hook for post-transform guard
            request.metadata["_lattice_validation"] = {
                "flagged": True,
                "reason": f"high_risk_{risk_level}"
                if risk_level in ("HIGH", "CRITICAL")
                else f"conservative_task_{task_class}",
                "timestamp": time.time(),
            }
        elif hasattr(__import__("random"), "random") and __import__("random").random() < 0.01:
            request.metadata["_lattice_validation"] = {
                "flagged": True,
                "reason": "sampled_1pct",
                "timestamp": time.time(),
            }
        # ---- End post-transform guard production hook ----

        # Auto-generate session ID if absent
        if not x_lattice_session_id:
            import secrets

            x_lattice_session_id = f"sess_{secrets.token_hex(8)}"

        # ------------------------------------------------------------------
        # Build ExecutionPlan — unified plan for the entire request lifecycle
        # ------------------------------------------------------------------
        from lattice.planner.execution_builder import build_execution_plan

        execution_plan = build_execution_plan(
            request=request,
            provider_name=provider_name,
            model=request.model or body.get("model", "gpt-4"),
            session_id=x_lattice_session_id,
            config=deps.config,
            is_streaming=body.get("stream", False),
        )
        ctx = TransformContext(
            request_id=str(time.time()),
            session_id=x_lattice_session_id,
            provider=provider_name,
            model=request.model or body.get("model", "gpt-4"),
        )
        fastapi_request.state.transform_context = ctx
        ctx.session_state["client_profile"] = x_lattice_client_profile or "default"
        persist_execution_plan_state(
            request,
            ctx,
            execution_plan,
            cache_plan=execution_plan.cache_plan,
            cache_simulation=get_canonical_request_value(
                request, None, "_lattice_cache_simulation"
            ),
        )

        session, was_created = await deps.session_manager.get_or_create_session(
            session_id=x_lattice_session_id,
            provider=provider_name,
            model=request.model or body.get("model", "gpt-4"),
            messages=request.messages,
            tools=request.tools,
        )

        # Persist ExecutionPlan into session metadata for multi-turn consistency.
        # On first turn: store the plan. On subsequent turns: refresh from
        # existing session metadata (provider/model/risk are sticky across turns).
        if not was_created:
            prev = session.metadata.get("_lattice_execution_plan")
            if prev is not None:
                from lattice.planner.session_plan import SessionExecutionPlan as _ExecPlan

                restored = _ExecPlan.from_dict(prev)
                # Preserve sticky fields from previous turn in the new plan
                execution_plan.provider = restored.provider
                execution_plan.model = restored.model
                execution_plan.session_id = restored.session_id
                # Merge allowed optimizers (union of previous + new)
                prev_allowed = set(restored.allowed_optimizers)
                new_allowed = set(execution_plan.allowed_optimizers)
                execution_plan.allowed_optimizers = list(prev_allowed | new_allowed)
                # Use stricter quality_floor and budget across turns
                execution_plan.quality_floor = max(
                    execution_plan.quality_floor, restored.quality_floor
                )
                execution_plan.latency_budget_ms = max(
                    execution_plan.latency_budget_ms, restored.latency_budget_ms
                )
        # Persist merged plan back
        persist_session_plan_state(
            session.metadata,
            execution_plan,
            cache_plan=execution_plan.cache_plan,
            cache_simulation=get_canonical_request_value(
                request, None, "_lattice_cache_simulation"
            ),
        )

        # Compute delta savings against prior session state
        delta_savings_bytes = 0
        if not was_created:
            existing = session.messages
            new_msgs = deps.detect_new_messages(existing, request.messages)
            if new_msgs:
                full_messages = existing + new_msgs
                request.messages = full_messages
                await deps.session_manager.update_session(session.session_id, full_messages)
            else:
                await deps.session_manager.update_session(session.session_id, request.messages)
            # Calculate wire savings if client had used delta encoding
            from lattice.transport.delta_wire import delta_wire_bytes

            full_msgs = deps.serialize_messages(request)
            new_raw = [msg.to_dict() if hasattr(msg, "to_dict") else msg for msg in new_msgs]
            if new_raw:
                try:
                    full_bytes, delta_bytes = delta_wire_bytes(
                        full_msgs, new_raw, session.session_id, len(existing)
                    )
                    delta_savings_bytes = max(0, full_bytes - delta_bytes)
                except (TypeError, ValueError) as exc:
                    deps.logger.warning(
                        "delta_wire_bytes_failed", error=str(exc), session_id=session.session_id
                    )

        # Use ExecutionPlan cache_plan if available; fall back to manifest-based planner
        if execution_plan.cache_plan:
            ctx.session_state["cache_plan_entries"] = execution_plan.cache_plan
            total_expected = sum_expected_cached_tokens(execution_plan.cache_plan)
            ctx.record_metric("cache_planner", "expected_cached_tokens", total_expected)
            ctx.record_metric("cache_planner", "breakpoints", len(execution_plan.cache_plan))
        elif session.manifest:
            cache_planner = deps.get_cache_planner(provider_name)
            cache_plan = cache_planner.plan(session.manifest)
            ctx.session_state["cache_plan"] = cache_plan
            ctx.record_metric(
                "cache_planner",
                "expected_cached_tokens",
                cache_plan.expected_cached_tokens,
            )
            ctx.record_metric("cache_planner", "breakpoints", len(cache_plan.breakpoints))

        # Check ExecutionPlan fallback settings
        disable_optimizers = getattr(execution_plan.fallback_plan, "disable_optimizers", False)
        if disable_optimizers:
            x_lattice_disable_transforms = True  # type: ignore[assignment]

        if x_lattice_disable_transforms:
            compressed_request = request
        else:
            result = deps.pipeline.compress(request, ctx)
            if is_err(result):
                if deps.config.graceful_degradation:
                    compressed_request = request
                    deps.logger.warning("pipeline_degraded", error=str(result))
                else:
                    return JSONResponse(
                        {
                            "error": "pipeline_failed",
                            "message": "Transform error — set graceful_degradation=true to continue",
                        },
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    )
            else:
                compressed_request = unwrap(result)

        if client_api_key:
            compressed_request.metadata["_lattice_client_api_key"] = client_api_key

        # ------------------------------------------------------------------
        # Semantic cache check (after transforms, before provider call)
        # ------------------------------------------------------------------
        from lattice.cache.semantic import compute_cache_key
        from lattice.gateway.compat.openai_chat_cache import try_semantic_cache_hit

        cache_key: str | None = None
        disable_cache = request.extra_headers.get("x-lattice-disable-cache", "")
        if deps.semantic_cache and deps.semantic_cache.enabled and not disable_cache:
            cache_key = compute_cache_key(compressed_request)

        cache_hit_response = await try_semantic_cache_hit(
            deps,
            fastapi_request=fastapi_request,
            request=request,
            compressed_request=compressed_request,
            ctx=ctx,
            session=session,
            provider_name=provider_name,
            delta_mode=delta_mode,
            http_version=http_version,
            delta_savings_bytes=delta_savings_bytes,
            x_lattice_client_profile=x_lattice_client_profile,
            x_lattice_disable_transforms=x_lattice_disable_transforms,
        )
        if cache_hit_response is not None:
            return cache_hit_response


        start_llm = time.perf_counter()
        try:
            messages = deps.serialize_messages(compressed_request)
            stream = compressed_request.stream
            requested_model = compressed_request.model
            if not requested_model:
                return JSONResponse(
                    {"error": "model field is required", "message": "model field is required"},
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

            if stream:
                from lattice.gateway.compat.openai_chat_stream import handle_chat_stream

                return await handle_chat_stream(
                    deps,
                    fastapi_request=fastapi_request,
                    request=request,
                    compressed_request=compressed_request,
                    ctx=ctx,
                    session=session,
                    execution_plan=execution_plan,
                    messages=messages,
                    provider_name=provider_name,
                    requested_model=requested_model,
                    client_api_key=client_api_key,
                    cache_key=cache_key,
                    delta_mode=delta_mode,
                    http_version=http_version,
                    delta_savings_bytes=delta_savings_bytes,
                    start_llm=start_llm,
                    x_lattice_client_profile=x_lattice_client_profile,
                )

            used_speculative = False
            prediction_hit = False
            batching_eligible = (
                ctx.metrics.get("transforms", {}).get("batching", {}).get("eligible", False)
            )
            if batching_eligible and not x_lattice_disable_transforms:
                try:
                    compressed_request.metadata["_lattice_is_batch"] = True
                    internal_response = await deps.batching_engine.submit(compressed_request, ctx)
                    model_used = requested_model
                    deps.metrics.increment("lattice_batching_dispatched")
                    if hasattr(deps.provider, "tacc"):
                        await deps.provider.tacc.record_batch_pressure(provider_name, 1)
                except Exception as batch_exc:
                    deps.logger.warning("batching_failed", error=str(batch_exc))
                    batching_eligible = False

            if not batching_eligible:
                prediction = ctx.session_state.get("speculative_prediction")
                if prediction and not x_lattice_disable_transforms:
                    compressed_request.metadata["_lattice_is_speculative"] = True
                    speculative_task = asyncio.create_task(
                        deps.speculative_executor.run_speculative(compressed_request, prediction)
                    )
                    real_start = time.perf_counter()
                    from lattice.planner.fallback_executor import execute_with_fallback

                    internal_response = await execute_with_fallback(
                        deps.provider.completion,
                        execution_plan=execution_plan,
                        provider_name=provider_name,
                        model=requested_model,
                        logger=deps.logger,
                        metrics=deps.metrics,
                        messages=messages,
                        temperature=compressed_request.temperature,
                        max_tokens=compressed_request.max_tokens,
                        top_p=compressed_request.top_p,
                        tools=compressed_request.tools,
                        tool_choice=compressed_request.tool_choice,
                        stream=False,
                        stop=compressed_request.stop,
                        api_key=client_api_key if provider_name == "openai" else None,
                        metadata=compressed_request.metadata,
                        extra_headers=compressed_request.extra_headers,
                        extra_body=compressed_request.extra_body,
                    )
                    model_used = requested_model
                    real_latency_ms = (time.perf_counter() - real_start) * 1000

                    speculative_response = None
                    if not speculative_task.done():
                        with contextlib.suppress(asyncio.TimeoutError):
                            speculative_response = await asyncio.wait_for(
                                speculative_task, timeout=0.5
                            )
                    else:
                        speculative_response = speculative_task.result()

                    if speculative_response is not None:
                        actual = deps.speculative_executor.extract_actual_step(internal_response)
                        hit = deps.speculative_executor.is_hit(prediction, actual)
                        deps.speculative_executor.record_result(
                            hit=hit,
                            _predicted=prediction,
                            _actual=actual,
                            latency_ms=real_latency_ms,
                        )
                        if hit:
                            internal_response = speculative_response
                            used_speculative = True
                            prediction_hit = True
                            deps.metrics.increment("lattice_speculative_hit")
                        else:
                            deps.metrics.increment("lattice_speculative_miss")
                else:
                    from lattice.planner.fallback_executor import execute_with_fallback

                    internal_response = await execute_with_fallback(
                        deps.provider.completion,
                        execution_plan=execution_plan,
                        provider_name=provider_name,
                        model=requested_model,
                        logger=deps.logger,
                        metrics=deps.metrics,
                        messages=messages,
                        temperature=compressed_request.temperature,
                        max_tokens=compressed_request.max_tokens,
                        top_p=compressed_request.top_p,
                        tools=compressed_request.tools,
                        tool_choice=compressed_request.tool_choice,
                        stream=False,
                        stop=compressed_request.stop,
                        api_key=client_api_key if provider_name == "openai" else None,
                        metadata=compressed_request.metadata,
                        extra_headers=compressed_request.extra_headers,
                        extra_body=compressed_request.extra_body,
                    )
                    model_used = requested_model

            # Auto-continuation for truncated responses (non-streaming only)
            cont_result = None
            if (
                deps.auto_continuation
                and not stream
                and internal_response.finish_reason == "length"
            ):
                deps.logger.info(
                    "auto_continuation_triggered",
                    session_id=session.session_id,
                    turns=deps.auto_continuation.max_turns,
                )

                async def _continuation_provider_call(**kw: Any) -> Any:
                    from lattice.planner.fallback_executor import execute_with_fallback

                    return await execute_with_fallback(
                        deps.provider.completion,
                        execution_plan=execution_plan,
                        provider_name=kw.pop("provider_name", provider_name),
                        model=kw.pop("model", requested_model),
                        logger=deps.logger,
                        metrics=deps.metrics,
                        **kw,
                    )

                cont_result = await deps.auto_continuation.continue_if_needed(
                    request=compressed_request,
                    initial_response=internal_response,
                    provider_caller=_continuation_provider_call,
                    session_manager=deps.session_manager,
                    message_cls=deps.message_cls,
                    provider_name=provider_name,
                )
                if cont_result.was_continued:
                    internal_response = cont_result.response
                    deps.metrics.increment("lattice_auto_continuation_turns", cont_result.turns)
                    deps.logger.info(
                        "auto_continuation_complete",
                        session_id=session.session_id,
                        turns=cont_result.turns,
                        final_length=len(internal_response.content or ""),
                    )

            if internal_response.content or internal_response.tool_calls:
                msg = deps.message_cls(
                    role="assistant",
                    content=internal_response.content or "",
                    tool_calls=internal_response.tool_calls,
                )
                session.messages.append(msg)
                await deps.session_manager.update_session(session.session_id, session.messages)

            # Record cache telemetry from provider usage
            cached_tokens = _extract_cached_tokens(internal_response.usage)
            if cached_tokens > 0:
                session.record_cache_hit(cached_tokens)

            # Feed actual cache result back into request metadata for observability
            cache_plan = ctx.session_state.get("cache_plan")
            if cache_plan is None:
                cache_plan = ctx.session_state.get("cache_plan_entries")
            if cache_plan is not None:
                if isinstance(cache_plan, list):
                    total_expected = sum_expected_cached_tokens(cache_plan)
                    breakpoints = len(cache_plan)
                else:
                    total_expected = getattr(cache_plan, "expected_cached_tokens", 0)
                    breakpoints_int2: int = getattr(cache_plan, "breakpoints", [])  # type: ignore[assignment]
                    breakpoints = breakpoints_int2
                internal_response.metadata["_cache_arbitrage_actual"] = {
                    "expected_cached_tokens": total_expected,
                    "actual_cached_tokens": cached_tokens,
                    "breakpoints": breakpoints,
                    "provider": provider_name,
                }

            # Store non-streaming response in semantic cache
            if cache_key and deps.semantic_cache:
                await deps.semantic_cache.set(
                    cache_key,
                    assemble_cached_response(
                        model=model_used,
                        content=internal_response.content or "",
                        tool_calls=internal_response.tool_calls,
                        usage=internal_response.usage or {},
                        finish_reason=internal_response.finish_reason or "stop",
                    ),
                    compressed_request,
                )

        except deps.provider_timeout_error as exc:
            deps.logger.error(
                "provider_timeout",
                provider=provider_name,
                model=requested_model,
                error=str(exc),
                fallback_plan=execution_plan.fallback_plan.to_dict()
                if hasattr(execution_plan.fallback_plan, "to_dict")
                else {},
                retry_count=execution_plan.fallback_plan.retry_count if execution_plan else 0,
            )
            return JSONResponse(
                {"error": "provider_timeout", "message": str(exc)},
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            )
        except deps.provider_error as exc:
            deps.logger.error(
                "provider_error",
                provider=provider_name,
                model=requested_model,
                error=str(exc),
                status_code=getattr(exc, "status_code", None),
                fallback_plan=execution_plan.fallback_plan.to_dict()
                if hasattr(execution_plan.fallback_plan, "to_dict")
                else {},
            )
            return JSONResponse(
                {"error": "provider_error", "message": str(exc)},
                status_code=getattr(exc, "status_code", None) or status.HTTP_502_BAD_GATEWAY,
            )
        except Exception as exc:
            deps.logger.error(
                "provider_unexpected_error",
                provider=provider_name,
                model=requested_model,
                error=str(exc),
                error_type=type(exc).__name__,
                fallback_plan=execution_plan.fallback_plan.to_dict()
                if hasattr(execution_plan.fallback_plan, "to_dict")
                else {},
            )
            return JSONResponse(
                {"error": "provider_error", "message": str(exc)},
                status_code=status.HTTP_502_BAD_GATEWAY,
            )

        if not x_lattice_disable_transforms:
            internal_response = deps.pipeline.reverse(internal_response, ctx)

        # ---- Production post-transform guard enforcement ----
        # Check blank-output post-response for flagged high-risk requests.
        validation_flag = get_canonical_request_value(request, None, "_lattice_validation", {})
        if isinstance(validation_flag, dict) and validation_flag.get("flagged"):
            resp_text = internal_response.content if internal_response else ""
            if not resp_text or not resp_text.strip():
                deps.logger.warning(
                    "post_transform_guard_blank_output_detected",
                    session_id=x_lattice_session_id or "",
                    reason=validation_flag.get("reason", "unknown"),
                )
                deps.metrics.increment("lattice_post_transform_guard_blank_output_rollback")
                # Record the actual validation outcome
                request.metadata["_lattice_validation"] = {
                    **validation_flag,
                    "blank_output": True,
                    "rollback_reason": "blank_output_detected",
                    "task_equivalence_composite": 0.0,
                }
        # ---- End post-transform guard enforcement ----

        elapsed_ms = (time.perf_counter() - start_llm) * 1000
        deps.metrics.record_latency("lattice_llm_latency_ms", elapsed_ms)
        response_body = deps.serialize_openai_response(internal_response, compressed_request)
        response = JSONResponse(content=response_body)
        cached_tokens = session.metadata.get("last_cache_hit_tokens", 0)
        # Compute actual cost
        cost_usd = 0.0
        cache_savings_usd = 0.0
        if deps.cost_estimator:
            cost_estimate = deps.cost_estimator.compute_actual(
                provider=provider_name,
                model=model_used,
                usage=internal_response.usage or {},
            )
            cost_usd = cost_estimate.total_cost_usd
            cache_savings_usd = cost_estimate.cached_savings_usd
            deps.metrics.record_metric("cost_estimator", "request_cost_usd", cost_usd)
            deps.metrics.record_metric(
                "cost_estimator",
                "cache_savings_usd",
                cache_savings_usd,
            )

        # Record per-agent stats
        if deps.agent_stats:
            agent_name = identify_agent(
                request.extra_headers.get("user-agent"),
                x_lattice_client_profile,
            )
            normalized_usage = normalize_usage(internal_response.usage or {})
            await deps.agent_stats.record_request(
                agent=agent_name,
                provider=provider_name,
                model=model_used,
                prompt_tokens=normalized_usage["prompt_tokens"],
                completion_tokens=normalized_usage["completion_tokens"],
                compressed_tokens=compressed_request.token_estimate or 0,
                original_tokens=request.token_estimate or 0,
                cached_tokens=cached_tokens,
                cache_hit=cached_tokens > 0,
                cost_usd=cost_usd,
                speculative_hit=used_speculative and prediction_hit,
                batched=batching_eligible,
                auto_continuation_turns=cont_result.turns if cont_result else 0,
            )

        transport_outcome = TransportOutcome(
            semantic_cache_status="miss" if cache_key is not None else "",
            batching_status="batched" if batching_eligible else "",
            speculative_status="hit"
            if (used_speculative and prediction_hit)
            else ("miss" if used_speculative else ""),
            delta_mode=delta_mode,
            http_version=http_version,
        )
        attach_routing_headers(
            fastapi_request,
            ctx,
            deps.build_routing_headers(
                model_used,
                session_id=session.session_id,
                compressed_tokens=compressed_request.token_estimate,
                original_tokens=request.token_estimate,
                used_speculative=used_speculative,
                prediction_hit=prediction_hit,
                batched=batching_eligible,
                delta_savings_bytes=delta_savings_bytes,
                cache_hit=cached_tokens > 0,
                cached_tokens=cached_tokens,
                cost_usd=cost_usd,
                cache_savings_usd=cache_savings_usd,
                transport_outcome=transport_outcome,
                **_runtime_header_values(compressed_request),
            ),
        )
        return response

    return _handle_chat_completion

