import json
import re
import time
import ollama

MODEL = "deepseek-r1:7b"
DATA_FILE = "gsm8k_sample.jsonl"
SYSTEM_PROMPT = ""  # baseline: no system prompt at all

def load_problems(path):
    problems = []
    with open(path) as f:
        for line in f:
            problems.append(json.loads(line))
    return problems

def extract_ground_truth(answer_field):
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))

def extract_model_answer(text):
    after_think = text.split("</think>")[-1] if "</think>" in text else text
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", after_think)
    if not numbers:
        return None
    return float(numbers[-1].replace(",", ""))

def solve_problem(problem, system_prompt=SYSTEM_PROMPT):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": problem["question"]})

    response = ollama.chat(model=MODEL, messages=messages)
    content = response["message"]["content"]

    think_len = 0
    if "<think>" in content and "</think>" in content:
        think_text = content.split("<think>")[1].split("</think>")[0]
        think_len = len(think_text)

    return {"content": content, "eval_count": response["eval_count"], "think_char_len": think_len}

def run_baseline(limit=None):
    problems = load_problems(DATA_FILE)
    if limit:
        problems = problems[:limit]
    results = []

    for i, problem in enumerate(problems):
        gt = extract_ground_truth(problem["answer"])
        start = time.time()
        out = solve_problem(problem)
        elapsed = time.time() - start

        pred = extract_model_answer(out["content"])
        correct = pred is not None and gt is not None and abs(pred - gt) < 1e-4

        results.append({
            "index": i, "ground_truth": gt, "predicted": pred, "correct": correct,
            "eval_count": out["eval_count"], "think_char_len": out["think_char_len"], "elapsed": elapsed,
        })

        status = "✓" if correct else "✗"
        print(f"[{i+1}/{len(problems)}] {status} pred={pred} gt={gt} tokens={out['eval_count']} time={elapsed:.1f}s")

    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_tokens = sum(r["eval_count"] for r in results) / len(results)
    print(f"\nAccuracy: {accuracy:.1%}")
    print(f"Avg tokens/response: {avg_tokens:.1f}")

    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    run_baseline(limit=5)  # sanity check on 5 before committing to all 50