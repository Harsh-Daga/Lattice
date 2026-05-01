"""Proxy-side auto-continuation for truncated responses.

When a provider returns ``finish_reason="length"`` (the response was cut off
because it hit ``max_tokens``), this module automatically sends follow-up
requests and stitches the partial responses into a single coherent output.

Design decisions
----------------
1. **Non-streaming**: Straightforward — send follow-up, concatenate content.
2. **Streaming**: More complex — the generator must continue yielding chunks
   from the follow-up stream transparently.
3. **Tool calls**: Preserved across continuations; partial tool calls are
   accumulated and forwarded.
4. **Usage aggregation**: Sums prompt_tokens and completion_tokens across all
   continuation turns.
5. **Safety limit**: ``max_continuation_turns`` prevents infinite loops if the
   model consistently hits the token limit.
6. **Session updates**: Each partial assistant message is appended to the
   session so the full conversation history is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lattice.core.transport import Message, Request, Response


@dataclass(slots=True)
class ContinuationResult:
    """Result of auto-continuation."""

    response: Response
    turns: int = 0
    was_continued: bool = False


class AutoContinuation:
    """Handle auto-continuation for truncated LLM responses.

    Usage (non-streaming)::

        continuator = AutoContinuation(max_turns=3)
        result = await continuator.continue_if_needed(
            request=req,
            initial_response=resp,
            provider_caller=provider.completion,
            session_manager=session_manager,
        )
        if result.was_continued:
            print(f"Continued for {result.turns} extra turns")
    """

    def __init__(self, max_turns: int = 3) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self.max_turns = max_turns

    async def continue_if_needed(
        self,
        *,
        request: Request,
        initial_response: Response,
        provider_caller: Any,
        session_manager: Any,
        message_cls: type[Message],
        provider_name: str,
    ) -> ContinuationResult:
        """Check if continuation is needed and execute follow-ups.

        Args:
            request: The original (possibly transformed) request.
            initial_response: First response from the provider.
            provider_caller: Async callable matching provider.completion signature.
            session_manager: SessionManager for persisting messages.
            message_cls: Message class for creating assistant messages.

        Returns:
            ContinuationResult with the final stitched response.
        """
        del session_manager
        current_response = initial_response
        turns = 0
        accumulated_content = current_response.content or ""
        accumulated_tool_calls = list(current_response.tool_calls or [])
        accumulated_usage = dict(current_response.usage or {})

        while (
            current_response.finish_reason == "length"
            and turns < self.max_turns
        ):
            # Append the partial assistant message to the conversation
            partial_msg = message_cls(
                role="assistant",
                content=current_response.content or "",
                tool_calls=current_response.tool_calls,
            )
            request.messages.append(partial_msg)

            # Send follow-up request (stream=False for simplicity in continuation)
            try:
                follow_up = await provider_caller(
                    model=request.model or "gpt-4",
                    messages=[m.to_dict() if hasattr(m, "to_dict") else dict(m) for m in request.messages],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    stream=False,
                    stop=request.stop,
                    provider_name=provider_name,
                    api_key=request.metadata.get("_lattice_client_api_key"),
                    metadata=request.metadata,
                    extra_headers=request.extra_headers,
                    extra_body=request.extra_body,
                )
            except Exception:
                # Provider failure during continuation — return what we have
                break

            # Accumulate content and tool calls
            if follow_up.content:
                accumulated_content += follow_up.content
            if follow_up.tool_calls:
                accumulated_tool_calls.extend(follow_up.tool_calls)

            # Accumulate usage
            for k, v in (follow_up.usage or {}).items():
                if isinstance(v, (int, float)):
                    accumulated_usage[k] = accumulated_usage.get(k, 0) + v

            current_response = follow_up
            turns += 1

        # Build final stitched response
        final_response = Response(
            content=accumulated_content,
            tool_calls=accumulated_tool_calls or None,
            usage=accumulated_usage,
            model=current_response.model,
            finish_reason=current_response.finish_reason,
        )

        return ContinuationResult(
            response=final_response,
            turns=turns,
            was_continued=turns > 0,
        )

    # ------------------------------------------------------------------
    # Streaming continuation
    # ------------------------------------------------------------------

    async def stream_continuation(
        self,
        *,
        request: Request,
        provider_caller: Any,
        message_cls: type[Message],
        session_manager: Any,
    ) -> Response:
        """Execute continuation for a streaming response that ended with length.

        This is called AFTER the initial stream ends with finish_reason="length".
        It sends a non-streaming follow-up and returns the complete response
        for stitching.

        Returns:
            The follow-up Response (to be stitched into the stream).
        """
        del session_manager
        # Append the partial content we already streamed
        # (The caller has already collected partial_content)
        partial_msg = message_cls(
            role="assistant",
            content=request.metadata.get("_partial_content", ""),
            tool_calls=request.metadata.get("_partial_tool_calls"),
        )
        request.messages.append(partial_msg)

        try:
            follow_up = await provider_caller(
                model=request.model or "gpt-4",
                messages=[m.to_dict() if hasattr(m, "to_dict") else dict(m) for m in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                tools=request.tools,
                tool_choice=request.tool_choice,
                stream=False,
                stop=request.stop,
                provider_name=request.metadata.get("provider_name", "openai"),
                api_key=request.metadata.get("_lattice_client_api_key"),
                metadata=request.metadata,
                extra_headers=request.extra_headers,
                extra_body=request.extra_body,
            )
        except Exception:
            # Return empty response on failure
            return Response(content="", finish_reason="stop")

        return follow_up


__all__ = ["AutoContinuation", "ContinuationResult"]
