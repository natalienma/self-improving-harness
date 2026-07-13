# Self Improving Agent Harness Using Autonomous Prompt Optimization
In this project, the agent's task is to optimize a system prompt for solving math word problems from the GSM8K dataset.

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

## Quick Start
Download Ollama and pull deepseek-r1:7b model.


### Model: deepseek-r1:7b
Why? 
1. This is an open-source, cheap model that can run on a Mac with an M5 chip. Cost of compute becomes an important constraint when it comes to autonomous research.

2. Deepseek-r1 has a reasoning block. We are trying to minimize the length of the chain-of-thought reasoning during decode, which dominates the cost for LLM compute. 

In open source models without reasoning blocks, such as Qwen 2.5:7b, the cost is just token in + token out, neither of which should be minimized while optimizing performance. 

## What is Autoresearch
## What is an Agent Harness

## Why compress reasoning?
Here is an example of a thinking block through a math problem:
``` thinking="\nAlright, so I need to figure out how much money the girls raised in total for the carnival. Let me read the problem again carefully. Hmm, okay. So we have two pairs here: Kim and Alexandra, and Maryam and Sarah. I need to find the total amount raised by all four girls.\n\nLet me break it down step by step....                                                                                                                                                \n\nWhich is the same as adding them one by one or grouping in any way, resulting in $2180. So that should be our total.\n\nTherefore, after careful consideration and multiple methods of addition, I'm confident the total money raised by all four girls is $2180.```

This block ran through 4 different reasoning methods and consumed ~1800 tokens. Yet the answer was correct from the beginning.