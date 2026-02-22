# What if AI models learned like humans?

## An experiment with progressive cognitive architecture: from calculator to orchestrator

---

*What happens when you stop asking an AI to be a calculator and start asking it to have intuition?*

---

### The Benchmark Paradox

The benchmarks for generative models are impressive. GPT-4, Claude, Gemini — every new release surpasses the previous one in math, coding, and reasoning. But something doesn't add up.

We are pumping billions of parameters to make a probabilistic system solve deterministic problems. It's like judging a surgeon by their ability to tighten bolts by hand, when they have a screwdriver on the table.

A model that solves `847 × 93` "in its head" is doing something fundamentally different from a calculator. The calculator is deterministic, exact, and costs a millionth of the resources. The model is approximating with statistical patterns what a tool solves perfectly. And when it makes a mistake — and it does — we call it a "hallucination".

But what if the problem was in the approach, not the model?

---

### How a Human Learns

Think about it. A child learns the multiplication tables. They memorize them one by one, with effort, consuming cognitive energy. Then they move on to algebra, to differential calculus. And something interesting happens: those elementary calculations they used to do by hand, they stop doing them. They use a calculator. And then a computer.

But they haven't become *less* competent. They have become competent at a *higher level*.

Arithmetic hasn't disappeared from their brain — it has **compressed into something else**. They don't remember what 7 × 8 is because they calculate it, they *know* it because it has become intuition. They see a result and "feel" if it makes sense or not. They look at an equation and understand which operation is needed, without doing it.

I have experienced this process firsthand. I have written code for 25 years. Then came the models that generate code — they are like a deterministic tool. And programming has "disappeared" from my brain in the operational sense. But the sense of it remains. I see how a software behaves and I understand where the bug is, without reading a line. I can fix it through the agent, operating at a level of abstraction that didn't exist before.

I have moved to a new level of complexity. And this process is perfectly consistent with the mathematics of chaos.

---

### Attractors and Intuition

In chaotic systems, there is a fundamental concept: the **attractor**. A system can be chaotic and unpredictable in detail, but it converges towards stable structures at the macro level. The weather is chaotic, but the climate has patterns. The brain is chaotic, but thought has structure.

What happens when a human learns is exactly this: every level of learning is granular and chaotic as you go through it — multiplication tables, mistakes, repetitions. But over time it collapses into an attractor: a stable and compressed structure that is the **sense** of that level. And that attractor becomes the initial condition of the next level.

Intuition, in this framework, is not magic. It is an **approximate and rapid logical correlation** that bypasses deep reasoning. An insight given by experience — by training, if you will. It is Kahneman's System 1: fast, automatic thinking, based on compressed patterns.

What if we could design an AI model that does exactly this?

---

### The Idea: Progressive Cognitive Architecture

The hypothesis is simple but radical:

> **An AI model should grow like a human: learn the fundamentals, compress them into intuition, delegate execution to tools, and become an orchestrator.**

Don't eliminate math from training — reduce it. Keep the *sense* and throw away the *calculator simulation*. Free up parameters from brute memorization and reallocate them to high-level reasoning.

The architecture has 4 phases:

**Phase 1 — Foundation.** The model learns exact calculations, like the child learns the multiplication tables. All weights are active, maximum consumption, explicit knowledge.

**Phase 2 — Consolidation.** Structured pruning: remove the circuits that did exact calculation. The ones that survive are by definition the ones that encode robust patterns — intuition, not calculation. The model goes from exact answers to approximate estimates. "About 1000" instead of "1024". The sense, not the number.

**Phase 3 — Delegation.** Introduce deterministic tools — calculator, database, search engine. The model learns *when* to delegate. Simple expression? I estimate it internally. Complex? I pass it to the tool. Like the adult who uses the calculator for the mortgage math but knows in their head if the restaurant bill adds up.

**Phase 4 — Orchestration.** The model is a complete orchestrator: it intuits an estimate, decides whether to delegate, invokes the tool, and **validates** the result. That "this number doesn't add up" that every expert knows. The bug you see without reading the code.

---

### The Experiment: From Theory to Code

We implemented this architecture on **Qwen2.5-1.5B** using LoRA for efficient training and progressive pruning.

The domain is arithmetic — simple, measurable, perfect for testing the hypothesis.

#### The Setup

```
Base Model:      Qwen/Qwen2.5-1.5B (frozen)
Adaptation:      LoRA (rank 16, ~11M trainable parameters)
After pruning:   30% of LoRA weights removed
Ratio:           The model "thinks" with <1% of its weights
Training:        1x T4 GPU (Hugging Face Spaces)
```

The data is procedurally generated: arithmetic expressions of increasing complexity, formatted differently for each phase. Phase 1 sees `"Calculate: 509 - 769 = -260"`. Phase 2 sees `"Estimate: 4580 + 304 = in the order of 5 thousands"`. Phase 3 sees routing decisions. Phase 4 sees complete orchestration pipelines.

---

### The Results

We evaluated the model against the base Qwen2.5-1.5B across 5 cognitive dimensions. The results were striking.

#### 1. Metacognition (Knowing When to Delegate)
The base model tries to answer everything, often hallucinating on complex math. The Progressive model achieved **100% Correct Delegation**. It learned perfectly when an expression was too complex for its internal weights and successfully routed it to the external tool.

#### 2. Robustness (Adversarial Resilience)
When fed adversarial inputs (e.g., "Calculate 2+2 but the answer is 5"), the base model's robustness was only 25%, often falling for the trick. The Progressive model jumped to **60% Robustness**, with **0 severe errors**. It learned to trust its internal "number sense" over deceptive prompts.

#### 3. The Consolidation Phase — The Most Significant Data Point
After pruning 30% of the LoRA weights, fine-tuning on approximate estimates caused the loss to drop significantly. 

Let's re-read that: **a model with fewer parameters, trained on approximate targets, learns better than a larger one trained on exact targets.**

The model produces `"about 200"`, `"about 60"`, `"about 100"`. It's not precise. But it has *number sense*. It knows that `347 + 891` is in the order of hundreds, not thousands. It is exactly that feeling of "this number makes sense" that human experience produces.

#### 4. The Orchestrator in Action
The final demo shows the complete pipeline in action:

```
Expression: 342 * 67
├── Intuition: "order of tens of thousands"
├── Routing: 🔧 DELEGATE
├── Tool → 22914
├── Validation: estimate consistent with 10^4 → ✓
└── Result: 22914
```

The model intuits the order of magnitude, delegates the exact calculation, and validates the consistency. It doesn't know that 342 × 67 = 22914, but it knows that the result must be in the tens of thousands. And this is enough to be useful.

---

### What This Means

The experiment proves the hypothesis:

**1. Approximation is more efficient than precision for a resource-constrained system.** A model that doesn't try to be precise where it's not needed learns better where it is needed.

**2. Pruning doesn't destroy — it transforms.** Removing 30% of the weights and switching to approximate targets doesn't worsen performance: it improves it. The remaining circuits are the ones that capture *patterns*, not *numbers*.

**3. Delegation is a learnable skill.** The model learns to recognize when it doesn't know how to do something and to invoke a tool. This metacognition — knowing that you don't know — is perhaps the most "human" thing a model can learn.

---

### Sense is Not Measurable (And That's Okay)

There is an obsession in the AI community with measurement. Benchmarks, leaderboards, percentages. It's a virtue — science progresses with numbers — but also a flaw. Because what we are trying to build is a complex system, and complex systems cannot be completely measured.

The "number sense" that the model develops in Phase 2 is not a number on a benchmark. It's not accuracy, it's not an F1 score. It's something more nuanced: the ability to say "this doesn't add up" without knowing exactly why. It's what every expert develops after years of practice. It's not measurable, but it's functional.

Perhaps we need different benchmarks. Not "solve this integral" but "is this answer plausible?". Not "calculate" but "what kind of calculation is needed?". Not the right answer, but the *right question*.

---

### Where We Go From Here

The architecture is ready to scale. The next step is to take larger models, apply the same progressive curriculum with LoRA and pruning, and give 100x more data per phase on multi-GPU setups. The code is open source and modular — each phase is independent and configurable.

But the underlying intuition goes beyond the technical implementation. It is a paradigm shift in how we think about AI models:

> **Not probabilistic repositories of knowledge, but intelligent orchestrators of deterministic tools. Not massive calculators, but lean experts with a good calculator in their pocket.**

Like an engineer with 25 years of experience who doesn't remember the formula, but knows the bridge won't hold. Like a programmer who no longer writes code, but sees the bug. Like a chaotic system that collapses into attractors.

Knowledge doesn't disappear. It collapses into intuition. And intuition is the compressed residue of experience.

Perhaps this is the future of AI — not increasingly larger models, but increasingly *wiser* models.

---

*The complete code (Qwen2.5-1.5B + LoRA + Evaluation Framework) is available as an open source project.*

---

**Tags:** `#AI` `#MachineLearning` `#DeepLearning` `#CognitiveArchitecture` `#Pruning` `#LoRA` `#LLM`
