import json
import re
import ollama
from inner_loop import run_baseline

MODEL = "deepseek-r1:7b"
LAMBDA = 0.0005  # weight on token cost vs accuracy — tune this

def generate_variants(current_best_prompt, n_variants=3):
    instruction = (
        f"Create {n_variants} rewordings of the following system prompt. "
        f"Keep the same content and intent, but vary the wording, structure, and order. "
        f"Return ONLY a JSON array of {n_variants} strings, nothing else."
    )
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": current_best_prompt or "(no system prompt — baseline)"},
    ]
    response = ollama.chat(model=MODEL, messages=messages)
    try:
        variants = json.loads(response.message.content)
    except json.JSONDecodeError:
        # fallback: fragile, but a starting point — revisit if this fires often
        variants = re.findall(r'"([^"]+)"', response.message.content)
    return variants

def score_prompt(system_prompt, n_problems=5):
    output = run_baseline(limit=n_problems, system_prompt=system_prompt, out_file="_scratch.json")
    meta = output["metadata"]
    weighted = meta["accuracy"] - LAMBDA * meta["avg_tokens"]
    return {"accuracy": meta["accuracy"], "avg_tokens": meta["avg_tokens"], "weighted_score": weighted}

def run_outer_loop(n_generations=2, n_variants=3, n_problems=2, out_file="outer_loop_results.json"):
    current_best_prompt = ""
    current_best = score_prompt(current_best_prompt, n_problems)
    history = [{"generation": 0, "prompt": current_best_prompt, **current_best}]

    for gen in range(n_generations):
        variants = generate_variants(current_best_prompt, n_variants)
        for variant in variants:
            result = score_prompt(variant, n_problems)
            history.append({"generation": gen + 1, "prompt": variant, **result})
            if result["weighted_score"] > current_best["weighted_score"]:
                current_best = result
                current_best_prompt = variant

    output = {"final_best_prompt": current_best_prompt, "final_best": current_best, "history": history}
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    return output

if __name__ == "__main__":
    run_outer_loop()