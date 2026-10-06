# LLM Eval Pipeline — Agent Rules

## Project overview
Automated LLM evaluation pipeline. Two golden datasets (ISTQB multiple choice + Python coding), 
deterministic verification + LLM-as-judge, pytest quality gates, GitHub Actions CI.

## Commands
- Run evaluation: `python evaluators/run_eval.py`
- Run judge: `python evaluators/llm_judge.py reports/<latest_eval>.json`
- Run tests: `pytest tests/ -v`
- Activate venv: `.venv\Scripts\Activate.ps1` (Windows) or `source .venv/bin/activate` (Linux)

## Rules
- Python 3.12, type hints on all new functions
- All test files in `tests/`, prefixed `test_`
- Golden dataset entries must have: id, difficulty, tags
- ISTQB entries: correct_choice must be a/b/c/d
- Coding entries: test_code must be runnable assertions
- Never hardcode API keys — use .env + python-dotenv
- Always add `time.sleep(1)` between API calls (rate limit)
- Reports go to `reports/` — never commit them (gitignored)
- Commit messages: `feat:`, `fix:`, `docs:`, `test:` prefixes

## Architecture
- `golden_datasets/` — JSON files with questions and ground truth
- `evaluators/` — scripts that call LLM APIs and verify answers
- `tests/` — pytest quality gates on eval results
- `reports/` — generated JSON results (gitignored)

## Known constraints
- exec() runs untrusted code — no sandbox yet
- Gemini only — Groq integration planned
- Judge not calibrated against human ratings