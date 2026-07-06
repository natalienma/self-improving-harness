import time
import ollama

start = time.time()

response = ollama.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "What is 17 * 23? Answer with just the number."}]
)

wall_clock = time.time() - start

print("Response:", response["message"]["content"])
print()
print(f"Wall clock time: {wall_clock:.2f}s")
print(f"Total duration (Ollama): {response['total_duration'] / 1e9:.2f}s")
print(f"Load duration: {response['load_duration'] / 1e9:.2f}s")
print(f"Prompt tokens: {response['prompt_eval_count']}")
print(f"Response tokens: {response['eval_count']}")
print(f"Tokens/sec: {response['eval_count'] / (response['eval_duration'] / 1e9):.1f}")