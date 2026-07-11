import json
import random

random.seed(42)

with open("gsm8k_test.jsonl") as f:
    lines = f.readlines()

sample = random.sample(lines, 50)

with open("gsm8k_sample.jsonl", "w") as f:
    f.writelines(sample)

print(f"Wrote {len(sample)} problems to gsm8k_sample.jsonl")