"""
Main evaluator — sends golden dataset questions to LLMs
and collects responses for later analysis.

Uses Claude (Anthropic API) as the tested model.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import anthropic

load_dotenv()

# --- Config ---
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
TESTED_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MODEL = "claude-sonnet-4-6-20250627"
RESULTS_DIR = Path("reports")
RESULTS_DIR.mkdir(exist_ok=True)


def call_llm(prompt: str) -> str:
    """Send prompt to Claude and get response."""
    message = client.messages.create(
        model=TESTED_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def load_dataset(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --- ISTQB Evaluation ---

def eval_istqb_entry(entry: dict) -> dict:
    prompt = (
        "You are taking an ISTQB Foundation Level exam. "
        "Answer with ONLY the letter of the correct option (a, b, c, or d). "
        "Do not explain.\n\n"
        f"Question: {entry['question']}"
    )

    try:
        raw_answer = call_llm(prompt).lower()
        answer = raw_answer[0] if raw_answer and raw_answer[0] in "abcd" else raw_answer
        correct = answer == entry["correct_choice"]
    except Exception as e:
        answer = f"ERROR: {e}"
        correct = False

    return {
        "id": entry["id"],
        "expected": entry["correct_choice"],
        "actual": answer,
        "correct": correct,
        "difficulty": entry["difficulty"],
        "tags": entry["tags"],
    }


# --- Coding Evaluation ---

def eval_coding_entry(entry: dict) -> dict:
    prompt = (
        "Write ONLY the Python function. No explanation, no examples, "
        "no markdown fences. Just the function code.\n\n"
        f"{entry['prompt']}"
    )

    generated_code = "N/A"
    try:
        generated_code = call_llm(prompt)
        generated_code = generated_code.replace("```python", "").replace("```", "").strip()

        namespace = {}
        exec(generated_code, namespace)
        exec(entry["test_code"], namespace)

        return {
            "id": entry["id"],
            "generated_code": generated_code,
            "tests_passed": True,
            "error": None,
            "difficulty": entry["difficulty"],
            "tags": entry["tags"],
        }

    except AssertionError as e:
        return {
            "id": entry["id"],
            "generated_code": generated_code,
            "tests_passed": False,
            "error": f"Assertion failed: {e}",
            "difficulty": entry["difficulty"],
            "tags": entry["tags"],
        }
    except Exception as e:
        return {
            "id": entry["id"],
            "generated_code": generated_code,
            "tests_passed": False,
            "error": f"{type(e).__name__}: {e}",
            "difficulty": entry["difficulty"],
            "tags": entry["tags"],
        }


# --- Main ---

def run_evaluation():
    print(f"Model: {TESTED_MODEL}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {"timestamp": timestamp, "model": TESTED_MODEL, "istqb": [], "coding": []}

    # ISTQB
    print("\n=== ISTQB Evaluation ===")
    istqb_data = load_dataset("golden_datasets/istqb.json")
    for entry in istqb_data["entries"]:
        print(f"  {entry['id']}...", end=" ")
        result = eval_istqb_entry(entry)
        all_results["istqb"].append(result)
        status = "PASS" if result["correct"] else "FAIL"
        print(f"{status} (expected={result['expected']}, got={result['actual']})")
        time.sleep(1)

    # Coding
    print("\n=== Coding Evaluation ===")
    coding_data = load_dataset("golden_datasets/python_coding.json")
    for entry in coding_data["entries"]:
        print(f"  {entry['id']}...", end=" ")
        result = eval_coding_entry(entry)
        all_results["coding"].append(result)
        status = "PASS" if result["tests_passed"] else f"FAIL ({result['error']})"
        print(status)
        time.sleep(1)

    # Summary
    istqb_correct = sum(1 for r in all_results["istqb"] if r["correct"])
    istqb_total = len(all_results["istqb"])
    coding_passed = sum(1 for r in all_results["coding"] if r["tests_passed"])
    coding_total = len(all_results["coding"])

    print(f"\n=== Summary ===")
    print(f"ISTQB:  {istqb_correct}/{istqb_total} ({istqb_correct/istqb_total*100:.1f}%)")
    print(f"Coding: {coding_passed}/{coding_total} ({coding_passed/coding_total*100:.1f}%)")

    output_path = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    run_evaluation()