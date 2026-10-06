"""
LLM-as-judge — uses Claude to evaluate quality of code
generated in the eval run.
"""

import json
import os
import sys
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
TESTED_MODEL = "claude-haiku-4-5-20251001"
JUDGE_MODEL = "claude-sonnet-4-6-20250627"

JUDGE_PROMPT = """You are a senior Python code reviewer. 
Rate the following code on a scale of 1-5 for each criterion.

Criteria:
- correctness: Does the code do what the prompt asks?
- readability: Is the code clean and easy to understand?
- best_practices: Does it follow Python conventions (PEP 8, naming, etc.)?
- edge_cases: Does it handle edge cases (empty input, None, etc.)?

Respond with ONLY a JSON object, no explanation:
{{"correctness": N, "readability": N, "best_practices": N, "edge_cases": N}}

PROMPT: {prompt}

CODE:
{code}
"""


def judge_code(prompt: str, code: str) -> dict:
    full_prompt = JUDGE_PROMPT.format(prompt=prompt, code=code)

    try:
        message = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": full_prompt}],
        )
        raw = message.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        scores = json.loads(raw)
        scores["average"] = sum(scores.values()) / len(scores)
        return scores
    except Exception as e:
        return {
            "correctness": 0,
            "readability": 0,
            "best_practices": 0,
            "edge_cases": 0,
            "average": 0,
            "error": str(e),
        }


def run_judge(eval_results_path: str):
    print(f"Judge model: {JUDGE_MODEL}")

    with open(eval_results_path, "r", encoding="utf-8") as f:
        eval_results = json.load(f)

    coding_data = json.load(open("golden_datasets/python_coding.json", encoding="utf-8"))
    prompts = {e["id"]: e["prompt"] for e in coding_data["entries"]}

    judged = []
    for result in eval_results.get("coding", []):
        entry_id = result["id"]
        code = result.get("generated_code", "N/A")
        prompt = prompts.get(entry_id, "Unknown prompt")

        print(f"  Judging {entry_id}...", end=" ")
        scores = judge_code(prompt, code)
        scores["id"] = entry_id
        scores["tests_passed"] = result["tests_passed"]
        judged.append(scores)
        print(f"avg={scores['average']:.1f}")

    output_path = eval_results_path.replace(".json", "_judged.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(judged, f, indent=2)
    print(f"\nJudge results saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python llm_judge.py <eval_results.json>")
        sys.exit(1)
    run_judge(sys.argv[1])