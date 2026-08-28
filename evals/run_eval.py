import json
import urllib.request
from pathlib import Path

EVAL_PATH = Path("evals/cases.json")
API_URL = "http://localhost:8000/enrich/legal-notice"

def run_eval():
    if not EVAL_PATH.exists():
        print(f"Eval file not found at {EVAL_PATH}")
        return

    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    total = len(cases)
    passed = 0

    print(f"Running eval suite across {total} test cases...\n")

    for case in cases:
        payload = json.dumps({"text": case["text"]}).encode("utf-8")
        req = urllib.request.Request(
            API_URL,
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                actual_cat = res_data.get("category")
                expected_cat = case["expected_category"]

                if actual_cat == expected_cat:
                    passed += 1
                    print(f"✅ Case {case['id']}: PASSED (Category: {actual_cat})")
                else:
                    print(f"❌ Case {case['id']}: FAILED (Expected: {expected_cat}, Got: {actual_cat})")
        except Exception as e:
            print(f"💥 Case {case['id']}: ERROR ({str(e)})")

    score_pct = (passed / total) * 100
    print(f"\nEval Score: {passed}/{total} ({score_pct:.1f}%)")

if __name__ == "__main__":
    run_eval()