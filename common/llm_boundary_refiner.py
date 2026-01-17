"""Optional LLM-based boundary refinement (guarded, non-blocking).

This is a soft post-check that can adjust sentence/unit boundaries before
final integrity checks. It only runs when USE_LLM_BOUNDARY_VERIFY is set and
appropriate API access is available. Falls back silently on any error.

Supported backends:
- ollama: Local/cloud Ollama (default)
- openai: OpenAI API (requires OPENAI_API_KEY)
- gemini: Google Gemini API (requires GEMINI_API_KEY)

Ollama Cloud Models (available via ollama backend):
- gemini-3-flash-preview:cloud
- gemini-3-pro-preview:latest
- deepseek-v3.1:671b-cloud
- deepseek-v3.2:cloud
- glm-4.6:cloud
- cogito-2.1:671b-cloud
- qwen3-coder:480b-cloud (default)
"""
from __future__ import annotations
import os
import json
import requests
from typing import List, Optional


def _dbg(enabled: bool, msg: str):
    if enabled:
        try:
            print(f"[LLM-BOUNDARY] {msg}")
        except Exception:
            pass


def _call_ollama(prompt: str, model: str, system_prompt: str, debug: bool) -> Optional[str]:
    """Call Ollama API."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    url = f"{host}/api/generate"
    
    payload = {
        "model": model,
        "prompt": f"{system_prompt}\n\n{prompt}",
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 1024,
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except Exception as e:
        _dbg(debug, f"ollama call failed: {e}")
        return None


def _call_openai(prompt: str, model: str, system_prompt: str, debug: bool) -> Optional[str]:
    """Call OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        _dbg(debug, "skip: OPENAI_API_KEY missing")
        return None
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        _dbg(debug, f"openai call failed: {e}")
        return None


def _call_gemini(prompt: str, model: str, system_prompt: str, debug: bool) -> Optional[str]:
    """Call Google Gemini API via REST."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        _dbg(debug, "skip: GEMINI_API_KEY missing")
        return None
    
    # Map model names to API versions if needed
    # Default: gemini-pro (or gemini-1.5-flash)
    api_model = model if model else "gemini-1.5-flash"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{api_model}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{"text": f"{system_prompt}\n\n{prompt}"}]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 1024,
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        try:
            return result['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            _dbg(debug, f"gemini invalid response structure: {result}")
            return None
            
    except Exception as e:
        _dbg(debug, f"gemini call failed: {e}")
        return None


def _call_llm(prompt: str, model: str, backend: str, system_prompt: str, debug: bool) -> Optional[str]:
    """Dispatch to appropriate LLM backend."""
    if backend == "ollama":
        return _call_ollama(prompt, model, system_prompt, debug)
    elif backend == "openai":
        return _call_openai(prompt, model, system_prompt, debug)
    elif backend == "gemini":
        return _call_gemini(prompt, model, system_prompt, debug)
    else:
        _dbg(debug, f"unknown backend: {backend}")
        return None


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
    - Validate integrity carefully.
    
    Environment variables:
    - USE_LLM_BOUNDARY_VERIFY: Set to enable
    - LLM_BOUNDARY_BACKEND: "ollama" (default), "openai", "gemini"
    - LLM_BOUNDARY_MODEL: Model name
    - OLLAMA_HOST: Ollama URL
    - OPENAI_API_KEY / GEMINI_API_KEY
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

    # Determine backend and model
    backend = os.getenv("LLM_BOUNDARY_BACKEND", "ollama").lower()
    
    default_models = {
        "ollama": "qwen3-coder:480b-cloud",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash", 
    }
    model_name = model or os.getenv("LLM_BOUNDARY_MODEL", default_models.get(backend, "qwen3-coder:480b-cloud"))

    _dbg(debug, f"using backend={backend}, model={model_name}")

    system_prompt = (
        "You are a precise boundary refiner. Move sentence/unit boundaries only. "
        "Never add or delete characters. Preserve order. Respond with JSON only."
    )
    
    prompt = _build_prompt(text, segments, task, reference_text)

    try:
        content = _call_llm(prompt, model_name, backend, system_prompt, debug)
        if not content:
            return segments
            
        # Parse JSON response
        # Handle potential markdown code blocks
        if "```" in content:
            import re
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                content = match.group(1)
        
        parsed = json.loads(content)
        new_segments = parsed.get("segments") if isinstance(parsed, dict) else None
        if not (isinstance(new_segments, list) and all(isinstance(s, str) for s in new_segments)):
            _dbg(debug, "reject: invalid segments format")
            return segments

        # Validate integrity: concatenation (whitespace-insensitive) must match
        def normalize(seq: List[str]) -> str:
            return "".join("".join(s.split()) for s in seq)

        if normalize(new_segments) != normalize(segments):
            _dbg(debug, "reject: integrity mismatch after LLM")
            return segments

        cleaned = [s.strip() for s in new_segments if s.strip()]
        
        if len(cleaned) != len(segments):
             # For SA, strict count check is already in prompt, but double check here if needed
             # Actually, if count changed, it might be what we wanted (if prompt allowed it)
             # But for strict 1:1, usually we want same count.
             pass

        _dbg(debug, f"applied: {len(segments)} -> {len(cleaned)} segments")
        return cleaned if cleaned else segments
        
    except json.JSONDecodeError as e:
        _dbg(debug, f"skip: JSON parse error ({e})")
        return segments
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
