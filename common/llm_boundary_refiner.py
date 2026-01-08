"""Optional LLM-based boundary refinement (guarded, non-blocking).

This is a soft post-check that can adjust sentence/unit boundaries before
final integrity checks. It only runs when USE_LLM_BOUNDARY_VERIFY is set and
OPENAI_API_KEY is available. Falls back silently on any error.
"""
from __future__ import annotations
import os
import json
from typing import List


def _dbg(enabled: bool, msg: str):
    if enabled:
        try:
            print(f"[LLM-BOUNDARY] {msg}")
        except Exception:
            pass


def refine_boundaries_with_llm(
    text: str,
    segments: List[str],
    task: str = "pa",
    max_segments: int = 12,
    model: str | None = None,
    reference_text: str | None = None,
) -> List[str]:
    """Let an LLM suggest boundary tweaks while preserving content.

    Rules enforced in prompt:
    - Do not alter characters; only move boundary cut points.
    - Keep overall order.
    - Return JSON {"segments": [...]}.
    Safety:
    - Skip when env USE_LLM_BOUNDARY_VERIFY is not set.
    - Skip if segment count outside [2, max_segments].
    - Skip on any exception.
    - Validate that concatenated text (ignoring whitespace) matches original.
    """
    debug = bool(os.getenv("LLM_BOUNDARY_DEBUG"))
    _dbg(debug, f"invoked: segs={len(segments)} max={max_segments} task={task} ref={bool(reference_text)}")

    if not os.getenv("USE_LLM_BOUNDARY_VERIFY"):
        _dbg(debug, "skip: USE_LLM_BOUNDARY_VERIFY not set")
        return segments
    if not segments or len(segments) < 2:
        _dbg(debug, "skip: not enough segments")
        return segments
    if len(segments) > max_segments:
        _dbg(debug, f"skip: segment count {len(segments)} > max {max_segments}")
        return segments

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        _dbg(debug, "skip: OPENAI_API_KEY missing")
        return segments

    model_name = model or os.getenv("LLM_BOUNDARY_MODEL", "gpt-4o-mini")

    try:
        from openai import OpenAI
    except Exception as e:
        _dbg(debug, f"skip: openai import/create failed ({e})")
        return segments

    prompt = _build_prompt(text, segments, task, reference_text)

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise boundary refiner. Move sentence/unit boundaries only. "
                        "Never add or delete characters. Preserve order. Respond with JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        content = resp.choices[0].message.content or ""
        parsed = json.loads(content)
        new_segments = parsed.get("segments") if isinstance(parsed, dict) else None
        if not (isinstance(new_segments, list) and all(isinstance(s, str) for s in new_segments)):
            return segments

        # Validate integrity: concatenation (whitespace-insensitive) must match
        def normalize(seq: List[str]) -> str:
            return "".join("".join(s.split()) for s in seq)

        if normalize(new_segments) != normalize(segments):
            _dbg(debug, "reject: integrity mismatch after LLM")
            return segments

        cleaned = [s.strip() for s in new_segments if s.strip()]
        _dbg(debug, f"applied: {len(segments)} -> {len(cleaned)} segments")
        return cleaned if cleaned else segments
    except Exception as e:
        _dbg(debug, f"skip: exception during LLM call ({e})")
        return segments


def _build_prompt(text: str, segments: List[str], task: str, reference_text: str | None) -> str:
    bullet = "\n - "
    prompt = (
        f"Task: adjust boundary cuts for '{task}' without changing characters.\n"
        f"Original text:\n{text}\n\n"
        f"Current segments ({len(segments)}):\n"
    )
    for i, seg in enumerate(segments, 1):
        prompt += f"[{i}] {seg}\n"
    
    # SA 특화: 원문 어절 단위를 명시하여 1:1 대응 강제
    if task == "sa" and reference_text:
        ref_units = reference_text.split()
        prompt += f"\nReference source units ({len(ref_units)}) - MUST have exactly {len(ref_units)} segments in output:\n"
        for i, unit in enumerate(ref_units, 1):
            prompt += f"[{i}] {unit}\n"
        prompt += (
            f"\n⚠️ CRITICAL CONSTRAINT for SA task:\n"
            f"{bullet}Output MUST contain EXACTLY {len(ref_units)} segments (same as reference units)."
            f"{bullet}Each segment should semantically correspond to one reference unit."
            f"{bullet}Do NOT merge or reduce segment count."
            f"{bullet}Only adjust WHERE to cut boundaries, not HOW MANY segments.\n"
        )
    elif reference_text:
        prompt += f"\nCounterpart text (meaning must align):\n{reference_text}\n"
    
    prompt += (
        "\nReturn JSON only: {\"segments\": [" "..." "]}."
        f"{bullet}Keep same characters and order."
        f"{bullet}You may adjust boundary positions to fix obvious mistakes."
        f"{bullet}Do not paraphrase or introduce punctuation."
        f"{bullet}Prefer boundaries at clause/punctuation/quotation ends."
        f"{bullet}Use counterpart meaning to decide better boundary placement.\n"
    )
    return prompt
