# LLM Evaluation Pipeline

Automated evaluation framework for testing LLM accuracy, hallucination,
and code generation quality.

## What this does

Sends questions from curated golden datasets to LLMs (Gemini, Llama, Mistral)
and measures how well they perform across two domains:

| Domain | Verification | Oracle |
|--------|-------------|--------|
| **ISTQB testing theory** | Exact match (A/B/C/D) | Official ISTQB CTFL v4.0 sample exam |
| **Python code generation** | Execute code + assert | Test assertions (deterministic) |
| **Code quality** | LLM-as-judge (1-5) | Claude (or any other model) rates correctness, readability, best practices |

### Three evaluation layers

1. **Deterministic** — exact match, keyword, assertion pass/fail. Zero ambiguity.
2. **Execution-based** — generated code runs against test cases. Works or crashes.
3. **LLM-as-judge** — semantic evaluation where deterministic checks aren't enough.

This layered approach means we don't trust the judge blindly — the deterministic
layer validates what it can, the judge handles only what it must.

## Set up

git clone 
cd llm-eval-pipeline
python -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt

# Configure API keys
cp .env.example .env

# Run evaluation
python evaluators/run_eval.py

# Run judge on results
python evaluators/llm_judge.py reports/eval_YYYYMMDD_HHMMSS.json

# Run quality gate tests
pytest tests/ -v
