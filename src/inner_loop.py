import json
import re
import time
import datetime
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
    boxed_match = re.search(r"\\boxed\{([^}]*)\}", text)
    search_text = boxed_match.group(1) if boxed_match else text

    cleaned = re.sub(r"\\[!,;:]", "", search_text)
    cleaned = cleaned.replace("\\", "").replace("$", "")

    numbers = re.findall(r"-?\d[\d,]*\.?\d*", cleaned)
    if not numbers:
        return None, text
    return float(numbers[-1].replace(",", "")), text


def solve_problem(problem, system_prompt, max_tokens=1500):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": problem["question"]})

    response = ollama.chat(
        model=MODEL,
        messages=messages,
        options={"num_predict": max_tokens},
    )
    return {
        "content": response.message.content,
        "think_text": response.message.thinking or "",
        "think_char_len": len(response.message.thinking or ""),
        "eval_count": response.eval_count,
        "prompt_eval_count": response.prompt_eval_count,
        "total_duration_s": response.total_duration / 1e9,
    }


def run_baseline(limit=None, system_prompt=SYSTEM_PROMPT, model=MODEL, out_file="baseline_results.json"):
    problems = load_problems(DATA_FILE)
    if limit:
        problems = problems[:limit]
    results = []

    for i, problem in enumerate(problems):
        gt = extract_ground_truth(problem["answer"])
        start = time.time()
        out = solve_problem(problem, system_prompt)
        elapsed = time.time() - start

        pred, after_think = extract_model_answer(out["content"])
        correct = pred is not None and gt is not None and abs(pred - gt) < 1e-4

        results.append({
            "index": i,
            "question": problem["question"],
            "ground_truth_raw": problem["answer"],
            "ground_truth": gt,
            "predicted": pred,
            "correct": correct,
            "full_response": out["content"],
            "think_text": out["think_text"],
            "think_char_len": out["think_char_len"],
            "answer_after_think": after_think,
            "eval_count": out["eval_count"],
            "prompt_eval_count": out["prompt_eval_count"],
            "total_duration_s": out["total_duration_s"],
            "elapsed_wallclock_s": elapsed,
        })

        status = "✓" if correct else "✗"
        print(f"[{i+1}/{len(problems)}] {status} pred={pred} gt={gt} "
              f"tokens={out['eval_count']} think_chars={out['think_char_len']} time={elapsed:.1f}s")

    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_tokens = sum(r["eval_count"] for r in results) / len(results)
    avg_think_chars = sum(r["think_char_len"] for r in results) / len(results)
    print(f"\nAccuracy: {accuracy:.1%}")
    print(f"Avg tokens/response: {avg_tokens:.1f}")
    print(f"Avg think chars: {avg_think_chars:.1f}")

    output = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": model,
            "system_prompt": system_prompt,
            "data_file": DATA_FILE,
            "n_problems": len(problems),
            "accuracy": accuracy,
            "avg_tokens": avg_tokens,
            "avg_think_chars": avg_think_chars,
        },
        "results": results,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    return output


if __name__ == "__main__":
    run_baseline(limit=2)
