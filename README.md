# Self Improving Reasoning Compression via Prompt-Space Autoresearch
In this project, the agent's task is to optimize a system prompt for solving math word problems from the GSM8K dataset.

**Status: in progress.** Inner loop (solve + score) built and validated. Outer loop (explore/exploit search) built, currently being tuned — an initial run surfaced a real bug (exploiting an empty seed prompt produces unrelated variants, not refinements) that's informing the next iteration.


## Inner Loop
The inner loop of this project is Claude + a system prompt for reasoning through math problems. 
We run this system on 50 GSM8K problems, and measure 
1) Accuracy (# correct/50)
2) Token usage 

Correctness is measured by extracting the final numerical answer from Claude's response and comparing it to the GSM8K ground truth.

## Outer Loop  
The agent finds points where the run failed and where reasoning was wrong. Then, the outer loop is optimized by LLM suggestions. 

In this prompt-based agent harness, this is essentially prompt optimization in 2 ways:
1) Exploration: Noise injection via instruction dropout/ablation. If a system prompt has multiple clauses, drop one or more clauses and measure the effect. 

2) Exploitation: Take the highest scoring prompt, compare it to the other prompts to find the difference. Then, create a similar prompt with the same content, but with different structure, wording, and order. 

After each run, the highest scoring prompt replaces the current best prompt. The loop runs for N iterations or until accuracy plateaus.

## Scoring and Validation
- **Scoring:** each candidate produces an (accuracy, avg. reasoning tokens) pair. Across generations this traces a Pareto frontier — for any token budget, the best achievable accuracy; for any accuracy floor, the cheapest prompt that clears it.
- **Validation:** the search sample is fixed; a held-out sample is used once at the end to check the winning prompt generalizes rather than overfitting to the search set.

## Model: deepseek-r1:7b
Why? 
1. This is an open-source, cheap model that can run on a Mac with an M5 chip. Cost of compute becomes an important constraint when it comes to autonomous research.

2. Deepseek-r1 has a reasoning block. We are trying to minimize the length of the chain-of-thought reasoning during decode, which dominates the cost for LLM compute. 

In open source models without reasoning blocks, such as Qwen 2.5:7b, the cost is just token in + token out, neither of which should be minimized while optimizing performance. 

## Why compress reasoning?
Here is an example of a thinking block through a math problem:
``` thinking="\nAlright, so I need to figure out how much money the girls raised in total for the carnival. Let me read the problem again carefully. Hmm, okay. I need to find the total amount raised by all four girls.\n\nLet me break it down step by step.... Therefore, after careful consideration and multiple methods of addition, I'm confident the total money raised by all four girls is $2180.```

This block ran through 4 different reasoning methods and consumed ~1800 tokens. Yet the answer was correct from the beginning.

### Findings so far
- Manual review of failures surfaced a labeling error in the GSM8K reference dataset itself (a ground-truth answer that doesn't match its own worked solution) — caught by inspecting a mismatch rather than trusting the accuracy number blindly.
- Observed at least one case of runaway reasoning: a response that generated ~600 output tokens over 21 minutes and still answered incorrectly — evidence that more reasoning tokens don't reliably buy more accuracy, motivating the compression objective directly.




