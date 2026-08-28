import json
import os
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, status
from dotenv import load_dotenv
from openai import OpenAI
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
    """Strip markdown code fences if present."""
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
    """Log unrepairable responses to quarantine.jsonl without crashing."""
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

@app.post(
    "/enrich/legal-notice",
    response_model=EnrichmentOutput,
    status_code=status.HTTP_200_OK
)
async def enrich_legal_notice(payload: NoticeInput):
    # Stub mode check
    if os.getenv("LLM_STUB", "0") == "1":
        return EnrichmentOutput(
            category=CategoryEnum.BUG,
            urgency=UrgencyEnum.NORMAL,
            confidence=0.90,
            summary="Stub summary of the incoming support message.",
            reason="Stub response returned because LLM_STUB is set to 1."
        )

    system_prompt = load_prompt()
    client = OpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"]
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload.text}
    ]

    raw_output = ""
    # Attempt 1: Initial call
    try:
        res = client.chat.completions.create(
            model=os.environ["LLM_MODEL"],
            temperature=0.1,
            messages=messages
        )
        raw_output = res.choices[0].message.content or ""
        cleaned = clean_json_fence(raw_output)
        parsed_json = json.loads(cleaned)
        validated_output = EnrichmentOutput.model_validate(parsed_json)
        return validated_output

    except (json.JSONDecodeError, ValidationError, Exception) as first_err:
        first_error_detail = str(first_err)

    # Attempt 2: Repair retry (1 single repair attempt)
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
            model=os.environ["LLM_MODEL"],
            temperature=0.0,
            messages=messages
        )
        repaired_raw = repair_res.choices[0].message.content or ""
        cleaned_repaired = clean_json_fence(repaired_raw)
        repaired_json = json.loads(cleaned_repaired)
        validated_output = EnrichmentOutput.model_validate(repaired_json)
        return validated_output

    except Exception as second_err:
        # Give up cleanly: Log to quarantine and return 422 Unprocessable Entity
        log_quarantine(payload.text, raw_output, f"Repair failed: {str(second_err)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Model output failed schema validation and could not be repaired."
        )