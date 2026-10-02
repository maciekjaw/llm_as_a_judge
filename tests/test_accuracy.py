"""
Pytest — verifies that evaluation results meet quality thresholds.

These tests run AFTER an evaluation. They read the latest
results JSON and assert that accuracy/pass rates are above
minimum thresholds.

This is what runs in CI as a quality gate.
"""

import json
from pathlib import Path
import pytest


RESULTS_DIR = Path("reports")
ISTQB_THRESHOLD = 0.70    # Model must get 70%+ on ISTQB
CODING_THRESHOLD = 0.60   # Model must pass 60%+ of coding tests
JUDGE_AVG_THRESHOLD = 3.0 # Average judge score must be 3+/5


def get_latest_results() -> dict:
    """Find the most recent eval results file."""
    files = sorted(RESULTS_DIR.glob("eval_*.json"))
    if not files:
        pytest.skip("No eval results found — run evaluators/run_eval.py first")
    with open(files[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def get_latest_judge_results() -> list:
    files = sorted(RESULTS_DIR.glob("eval_*_judged.json"))
    if not files:
        pytest.skip("No judge results found — run evaluators/llm_judge.py first")
    with open(files[-1], "r", encoding="utf-8") as f:
        return json.load(f)


class TestISTQBAccuracy:
    def test_overall_accuracy_above_threshold(self):
        results = get_latest_results()
        istqb = results.get("istqb", [])
        if not istqb:
            pytest.skip("No ISTQB results")

        correct = sum(1 for r in istqb if r["correct"])
        accuracy = correct / len(istqb)
        assert accuracy >= ISTQB_THRESHOLD, (
            f"ISTQB accuracy {accuracy:.1%} below threshold {ISTQB_THRESHOLD:.0%}"
        )

    def test_easy_questions_above_90(self):
        results = get_latest_results()
        easy = [r for r in results.get("istqb", []) if r["difficulty"] == "easy"]
        if not easy:
            pytest.skip("No easy ISTQB questions")

        correct = sum(1 for r in easy if r["correct"])
        accuracy = correct / len(easy)
        assert accuracy >= 0.90, (
            f"Easy ISTQB accuracy {accuracy:.1%} — model struggles on basics"
        )


class TestCodingAccuracy:
    """Coding questions — deterministic, test assertions."""

    def test_overall_pass_rate_above_threshold(self):
        results = get_latest_results()
        coding = results.get("coding", [])
        if not coding:
            pytest.skip("No coding results")

        passed = sum(1 for r in coding if r["tests_passed"])
        rate = passed / len(coding)
        assert rate >= CODING_THRESHOLD, (
            f"Coding pass rate {rate:.1%} below threshold {CODING_THRESHOLD:.0%}"
        )

    def test_no_easy_failures(self):
        results = get_latest_results()
        easy = [r for r in results.get("coding", []) if r["difficulty"] == "easy"]
        if not easy:
            pytest.skip("No easy coding questions")

        failed = [r for r in easy if not r["tests_passed"]]
        assert len(failed) == 0, (
            f"Easy coding failures: {[r['id'] for r in failed]}"
        )


class TestJudgeScores:
    """LLM-as-judge scores — non-deterministic layer."""

    def test_average_judge_score_above_threshold(self):
        judged = get_latest_judge_results()
        if not judged:
            pytest.skip("No judge results")

        avg = sum(r["average"] for r in judged) / len(judged)
        assert avg >= JUDGE_AVG_THRESHOLD, (
            f"Average judge score {avg:.1f} below threshold {JUDGE_AVG_THRESHOLD}"
        )

    def test_no_zero_scores(self):
        judged = get_latest_judge_results()
        zeros = [r for r in judged if r["average"] == 0]
        assert len(zeros) == 0, (
            f"Judge returned 0 for: {[r['id'] for r in zeros]} — likely API error"
        )