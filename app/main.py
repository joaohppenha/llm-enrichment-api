import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, status
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, APIError
from pydantic import ValidationError

from app.schemas.llm import NoticeInput, EnrichmentOutput, CategoryEnum, UrgencyEnum

load_dotenv()

app = FastAPI(
    title="LLM Support Message Enrichment API",
    version="1.0.0"
)

PROMPT_PATH = Path("prompts/enrich-v1.md")
QUARANTINE_PATH = Path("logs/quarantine.jsonl")

def load_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise RuntimeError(f"Prompt file not found at {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8")

def clean_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

def log_quarantine(raw_input: str, raw_output: str, error_msg: str):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_version": "enrich-v1.md",
        "input": raw_input,
        "raw_output": raw_output,
        "error": error_msg
    }
    QUARANTINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QUARANTINE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def log_cost(prompt_version: str, model: str, prompt_tokens: int, completion_tokens: int, duration_ms: float, repairs: int):
    log_entry = {
        "event": "llm_call_stats",
        "prompt_version": prompt_version,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "duration_ms": round(duration_ms, 2),
        "repairs": repairs
    }
    print(json.dumps(log_entry))

@app.post(
    "/enrich/legal-notice",
    response_model=EnrichmentOutput,
    status_code=status.HTTP_200_OK
)
async def enrich_legal_notice(payload: NoticeInput):
    # 1. Kill Switch: MUST BE FIRST to avoid any external calls or key checks
    if os.getenv("LLM_ENABLED", "true").lower() == "false":
        return EnrichmentOutput(
            category=CategoryEnum.OTHER,
            urgency=UrgencyEnum.LOW,
            confidence=0.0,
            summary="LLM feature is currently disabled via kill switch.",
            reason="LLM_ENABLED is set to false."
        )

    # 2. Stub mode check
    if os.getenv("LLM_STUB", "0") == "1":
        return EnrichmentOutput(
            category=CategoryEnum.BUG,
            urgency=UrgencyEnum.NORMAL,
            confidence=0.90,
            summary="Stub summary of the incoming support message.",
            reason="Stub response returned because LLM_STUB is set to 1."
        )

    system_prompt = load_prompt()
    
    # Check credentials explicitly
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)")
    model_name = os.getenv("LLM_MODEL", "openrouter/free")

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM_API_KEY is missing in environment variables."
        )

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=30.0,
        max_retries=2
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload.text}
    ]

    start_time = time.time()
    repairs_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    raw_output = ""

    # Attempt 1: Call LLM
    try:
        res = client.chat.completions.create(
            model=model_name,
            temperature=0.1,
            messages=messages
        )
        if res.usage:
            prompt_tokens += res.usage.prompt_tokens or 0
            completion_tokens += res.usage.completion_tokens or 0

        raw_output = res.choices[0].message.content or ""
        cleaned = clean_json_fence(raw_output)
        parsed_json = json.loads(cleaned)
        validated_output = EnrichmentOutput.model_validate(parsed_json)

        duration_ms = (time.time() - start_time) * 1000
        log_cost("enrich-v1.md", model_name, prompt_tokens, completion_tokens, duration_ms, repairs_count)
        return validated_output

    except AuthenticationError as auth_err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed with LLM Provider: {str(auth_err)}"
        )
    except APIError as api_err:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM Provider API Error: {str(api_err)}"
        )
    except (json.JSONDecodeError, ValidationError) as first_err:
        first_error_detail = str(first_err)

    # Attempt 2: Repair retry
    repairs_count = 1
    repair_prompt = (
        f"Your previous response was rejected due to this schema validation error:\n"
        f"{first_error_detail}\n\n"
        f"Previous rejected output was:\n{raw_output}\n\n"
        f"Return ONLY a corrected JSON object strictly matching the schema."
    )
    
    messages.append({"role": "assistant", "content": raw_output})
    messages.append({"role": "user", "content": repair_prompt})

    try:
        repair_res = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            messages=messages
        )
        if repair_res.usage:
            prompt_tokens += repair_res.usage.prompt_tokens or 0
            completion_tokens += repair_res.usage.completion_tokens or 0

        repaired_raw = repair_res.choices[0].message.content or ""
        cleaned_repaired = clean_json_fence(repaired_raw)
        repaired_json = json.loads(cleaned_repaired)
        validated_output = EnrichmentOutput.model_validate(repaired_json)

        duration_ms = (time.time() - start_time) * 1000
        log_cost("enrich-v1.md", model_name, prompt_tokens, completion_tokens, duration_ms, repairs_count)
        return validated_output

    except Exception as second_err:
        log_quarantine(payload.text, raw_output, f"Repair failed: {str(second_err)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Model output failed schema validation and could not be repaired."
        )