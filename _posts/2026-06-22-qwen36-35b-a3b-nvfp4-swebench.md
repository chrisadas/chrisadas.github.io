---
layout: post
title: I ran SWE-bench Verified on a 35B model at home.
date: 2026-06-22 10:00:00 +0000
categories: general
---

# I ran SWE-bench Verified on a 35B model at home. It scored 60%.

I have two RTX PRO 2000 Blackwell cards at home, 16GB of GDDR7 each, 32GB in total. These are not H100s, or anything close to one. They're the kind of hardware you buy when you want to run local inference but you're paying for it yourself.

![2X RTX Pro 2000 Blackwell in a 2U server](/assets/images/proxmox-rack-2000blackwell.webp)

I want to answer a simple question. Can a model running entirely on my own hardware do real coding work? It is obvious it will be a few notches below frontier models. But even the leaderboards do not cover all the quantization levels, and on 32GB of VRAM I can only run quantized models, so the only honest way to find out was to measure it myself.

I ran SWE-bench Verified against Qwen3.6-35B-A3B in NVIDIA's NVFP4 4-bit build, with a 256K context window. No cloud, no API key. Below is the scorecard, and what I took from it.

## What SWE-bench Verified actually tests, and why I used it

I'll admit something. For years I've read SWE-bench scores in model release posts and treated them as an accepted measuring tape, without really knowing what the test does. Running it myself forced me to learn.

SWE-bench Verified is 500 real GitHub issues, human-validated for quality, drawn from twelve popular open-source Python projects such as Django, sympy, scikit-learn, matplotlib, astropy and sphinx. Each task hands the model an issue description plus the repository at the commit just before the fix landed. The model has to produce a patch, an actual diff, that resolves the issue. Grading isn't a judgment call. The harness applies your patch and runs the repo's own unit tests. The tests tied to the bug have to flip from failing to passing, and the tests that were already passing have to stay green. Clear both and the instance counts as resolved.

Two things clicked once I saw this. First, it's entirely Python. Every repo in the set is a Python library, so a strong SWE-bench number tells me about Python bug-fixing in particular, not coding in general. Second, it's a bug-fixing benchmark rather than a code-quality one. It asks whether the tests pass, not whether it produces good code.

That second point is why the frontier has moved past it. The headline models are now measured on harder, quality-aware evaluations like Scale's SWE-bench Pro and Cognition's FrontierCode. FrontierCode grades whether a change is actually mergeable, meaning correct and in scope and maintainable, and it is hard enough that the best model in the world scores around 13%. Some of the motivation came from METR's finding that a large share of SWE-bench "solutions" is code no maintainer would actually accept.

I can't run any of that at home, and for this exercise I don't need to. SWE-bench Verified has years of widely reported, directly comparable results across nearly every model that matters. If you just want to place your local box against numbers everyone already knows, the old, well-worn ruler is the one to use.

## Why this model

My model-selection method wasn't sophisticated. I started at the top of the [SWE-bench Verified leaderboard](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified) and walked down until I found something that would fit on 32GB of VRAM. The top is a wall of models I have no hope of running, like DeepSeek-V4-Pro, Kimi-K2.6, GLM-5 and the 400B-class Qwens, all hundreds of billions of parameters and built for datacenters. So you keep scrolling. The first entry that scores in the 70s and also has any path onto two 16GB cards is Qwen3.6-35B-A3B, through NVIDIA's NVFP4-quantized checkpoint. And "has a path" is generous. It barely fits.

## Making it fit

"Barely" is doing real work in that sentence. NVIDIA's serving recipe for the 4-bit build assumes a single DGX Spark. I had two separate 16GB cards, so most of what follows is me re-deriving that recipe for hardware it wasn't written for. The first thing I wanted to know was whether the full native 256K context would fit at all, or just blow up the cards on startup. The back-of-envelope is worth showing.

The weights are the easy half. Thirty-five billion parameters at four bits is about 17.5 GB, and all 35 billion have to sit in memory even though only 3 billion activate per token, because mixture-of-experts saves compute, not memory. That leaves roughly 14.5 GB of the 32 for everything else.

The KV cache is the half that decides things. That is the running memory of attention, and it grows with every token of context, so at 256K tokens, left unchecked, it's what runs you out of memory. Two things keep it affordable. The first is storing it at eight bits instead of sixteen, which halves the cost outright. The second is the model itself. Qwen3.6 is a hybrid, where only every fourth layer (10 of its 40) runs full attention with a growing cache, while the other 30 use linear attention, whose state stays a small fixed size no matter how long the context gets. So the cache that actually scales with context comes from ten layers, not forty.

![How VRAM is used](/assets/images/qwen36-vram-budget.svg)

The arithmetic that falls out is friendly. Each full-attention layer holds a key and a value for two KV-heads of 256 dimensions, which at eight bits works out to roughly 2 × 10 × 2 × 256, or about 10 KB per token. Across the full 262,144 tokens the KV cache comes to under 2.5 GB. Stacked on the 17.5 GB of weights, the part of the budget I can size on the back of an envelope sits under 20 GB. (A naive estimate that treated all 40 layers as full attention would have predicted around 10 GB of cache and still fit. The hybrid design makes it considerably roomier.) But the envelope is optimistic about the unglamorous part. The rest of the budget goes to CUDA graphs, activation buffers, NCCL tensor-parallel buffers, and the server's own workspace, and those are what the exact number turns on. The envelope said 256K was safe, and the measured allocations confirmed it. The table below is the live breakdown, pulled from the running container. The cross-check closes cleanly: 0.90 × 16,311 MiB = 14,680 MiB, and nvidia-smi reports each TP worker holding 14,684 MiB.

| Component | Per card | × 2 cards (TP=2) | Source |
|---|---:|---:|---|
| Model weights (NVFP4, 35B total) | 10.28 GiB | 20.56 GiB | startup log |
| KV cache (fp8) | 1.53 GiB | 3.06 GiB | startup log |
| CUDA graphs | 0.50 GiB | 1.00 GiB | startup log |
| Overhead (CUDA ctx, NCCL buffers, FlashInfer workspace) | ~2.03 GiB | ~4.06 GiB | derived |
| **vLLM total** | **14.34 GiB** | **28.68 GiB** | nvidia-smi: 14,684 MiB/rank ✓ |

The KV pool comes to 303,848 tokens, or 1.16× a full 256K request, meaning a single 256K context fits with room for one additional sequence in flight. That the cache is only 1.53 GiB per card despite the large window is a direct consequence of the two properties already noted: fp8 halves the per-token cost, and only 10 of the 40 layers accumulate a growing per-token cache.

For anyone with the same hardware, the complete working vLLM Docker Compose service definition is below. The `<<: *vllm-common` and `<<: *vllm-env` blocks are shared anchors covering the image, restart policy, and environment. The model-specific settings are entirely in `command:`.

```yaml
command:
    - nvidia/Qwen3.6-35B-A3B-NVFP4
    - --served-model-name
    - qwen3.6-35b
    - --trust-remote-code
    - --dtype
    - auto
    - --quantization
    - modelopt              # NVIDIA ModelOpt NVFP4 checkpoint format
    - --kv-cache-dtype
    - fp8                   # halve KV memory so more context fits
    - --attention-backend
    - flashinfer            # recipe-required backend for this model on Blackwell
    - --moe-backend
    - flashinfer_b12x       # NATIVE NVFP4 MoE for sm_120 (FlashInfer CuteDSL "SM12x" path).
    - --tensor-parallel-size
    - "2"                   # split across GPU 0 + GPU 1 (recipe targets a single DGX Spark)
    - --disable-custom-all-reduce  # CUDA IPC peer access DEADLOCKS on TP=2;
    - --max-num-seqs
    - "4"
    - --max-num-batched-tokens
    - "8192"
    - --enable-prefix-caching      # agent conversations are append-only; caching the unchanged
                            # prefix avoids re-prefilling the full context each turn (~10-30x
                            # faster at long context). Changes reuse, not allocation.
    - --max-model-len
    - "262144"
    - --enable-auto-tool-choice
    - --tool-call-parser
    - qwen3_xml             # parse the model's <tool_call><function=..> XML into OpenAI tool_calls
    - --gpu-memory-utilization
    - "0.90"                # reserves ~14.7 GB/card. 
    - --reasoning-parser
    - qwen3                 # parse the model's <think> reasoning blocks
```

The decision that justified the hardware was native 4-bit math. Out of the box, the recipe ran the mixture-of-experts layers by decompressing the 4-bit weights back to 16-bit before multiplying, and it printed a confident warning that the hardware had "no native FP4." It crawled along at about 4 tokens per second. The warning was simply wrong, and why it was wrong is the whole reason I bought these cards. Native 4-bit tensor cores are exactly what the Blackwell generation added over the one before it. Once I pointed the server at a compute path actually built for the architecture, it ran the math in native 4-bit, and the model went from a science experiment to something usable in a single change. Running at native 4-bit on Blackwell, throughput lands around 1,400 tokens per minute sustained, bursting to around 60 tokens per second. KV cache utilization across a typical run sits at 70 to 87%, so the pool is nearly saturated by the large contexts but never overflows.

There were other snags, like a two-GPU deadlock I cleared by switching off an over-eager optimization, and a translation layer so the agent could read the model's tool-call format, but both were footnotes. The real lesson is that none of this was "download and run." On local hardware, the model is rarely the hard part. The stack around it is.

## The number

![The outcome, 61% on swe-bench-verified](/assets/images/qwen36-outcomes.svg)

The model resolved 307 of the 500 problems, or 61.4%. For a sparse MoE that activates only 3 billion parameters per token, quantized to 4 bits to fit in 32GB and run through a generic open-source scaffold on my own machine, I'm happy with that. For context, on the public SWE-bench Verified leaderboard, 61% sits above GPT-4.1 (54.6%) and just under Gemini 2.5 Pro (63.8%). That is roughly last-generation frontier territory, at zero marginal cost per token, fully private, on hardware that fits on a shelf.

![The outcome compared to leaderboard](/assets/images/qwen36-leaderboard.svg)

The comparison that stings is with the model's own headline. Qwen's model card reports 73.4% for this exact model, which puts it level with Claude Haiku 4.5 and Sonnet 4. I left 12 points on the table against that number. Working out where those points went is most of what this exercise was about.

## Where the 12 points went

The easy assumption is that quantization made the model dumber. That's mostly wrong, and the data shows why.

Here is what happened to the 500, broken down:

| Outcome | Count | % of 500 |
|---|---:|---:|
| **Resolved** (patch passed tests) | 307 | 61.4% |
| Unresolved (patch ran, failed tests) | 106 | 21.2% |
| **Empty patch** (no diff produced) | 68 | 13.6% |
| Harness error | 19 | 3.8% |

The number that changed how I read all of this is the next one. Of the 413 instances where the model actually committed to a patch that applied and ran, 307 passed. That is 74.3%.

When this model produces a patch, it's right about three times out of four. That points to strong reasoning paired with weak follow-through, not a model that thinks poorly. Most of the gap between 61% and 73% is the 68 problems where the model never produced a diff at all.

That is 14% of the benchmark producing nothing, and almost all of it comes from the agent running out of room rather than the model failing to reason.

## The empty patches are really a VRAM problem

None of these empty patches is the model giving up. Each one carries an exit status, and they break down like this:

![Chart showing causes of no patch](/assets/images/qwen36-empty-patches.svg)

Almost all of these come from the agent running out of room before it writes a diff, not from the model failing to reason. Most of the time it exhausts its turn budget, the cap on how many steps it can take in a single run, and stops with nothing to submit. Another twenty fill the entire 256K context window, the model's full native limit, which means the run read so much that it had nothing left to work with. The rest are a few format errors and timeouts. None of this is the model reasoning badly. It is the agent running past the limits of the harness and the model's context length before it can finish.

The context-window failures, when they happen, concentrate in Django, which accounts for thirteen of the twenty. The four format errors are the only failures that match the "model can't follow the diff format" story from the Unsloth GGUF community thread. Everything else is the agent running out of time or steps, not the model running out of ideas.

So the 12-point gap from Qwen's headline comes down to three things, which I can now rank by the evidence:

1. **The agent's budget, and it is the big one.** Most of the empty patches are the agent exhausting its turn budget or filling the full 256K window before it produces a diff. A larger step budget, or a more efficient harness that finds the right spot without reading everything first, recovers these. More VRAM would not, since the model is already serving its full native context. None of it means a better model. That is where the largest block of points is waiting.
2. **The harness quality.** Qwen's 73.4% used their own internal, tuned scaffold, and I used a generic one. A scaffold that finds the relevant code before it reads everything, retries when the output comes back empty, and spends its budget carefully would help both the empty runs and the wrong patches, without changing a single weight.
3. **Quantization.** NVFP4 is 4-bit and almost certainly costs something against the BF16 the headline was measured on, but it's the price of admission for 32GB, and it's the smallest of the three factors, nowhere near 12 points on its own.

## What the wins and losses look like

A few more things surfaced in the per-instance data.

There is no cheap way to cut losses early. Resolved instances took a median of 46 API calls and empties took a median of 250, most of them right up against the step limit, but that gap isn't the model working harder on the failures. Empties run long because hitting the step or context limit is exactly what leaves them empty, and the gap between resolved runs and wrong-but-submitted ones (46 against 66) is tangled up with difficulty, since easy problems are both cheaper and more likely to be solved. So the call counts don't really measure effort, and they don't justify capping the budget as a shortcut either. A third of my actual solves took more than 60 calls, and some took 187. A long run isn't a doomed one.

Correct fixes are small, and wrong ones are bigger. The median resolved patch changes 4 lines, and the median unresolved one changes 10. Some of that is the same confound again, since harder bugs need bigger fixes and are more likely to come out wrong, but it still works as a rough signal. When this model is right, it's usually right in a few lines, and a diff that keeps growing is a fair sign that it has lost the thread rather than found it.

Some of the failures fixed the bug but broke something else. Nine of the 106 unresolved cases passed every test tied to the bug and still failed, because the patch broke a different test that had been passing. The fix itself worked, but it caused breakage elsewhere. That is exactly the kind of thing the newer "would you merge this?" benchmarks were built to catch.

## Would I use it for real work? Not yet

I came in asking whether a model on my own hardware could do my real coding work. The honest answer is no, at least not yet, and I want to be precise about why, because it isn't a knock on the model.

On its own terms, the result is impressive. A quantized 35B model on two 16GB cards, fully private, at zero marginal cost (the entire 500-instance run cost me electricity and nothing else), landing about where the best closed models sat 18 months ago. A few years ago I would have called that implausible. But "as good as frontier was 18 months ago" isn't the comparison that decides anything for me. The real question is why I would run 18-month-old capability when today's frontier is one API call away. Opus, Sonnet and GPT-5.5 resolve more and stall less, including on exactly the find-where-to-act work where my local setup tends to run out of room. With those sitting right there, the local model's real advantages, its privacy and its zero cost, don't outweigh the gap in capability.

The math is different for anyone who can't make that API call. If you are working under strict privacy, cost or offline constraints, a capable local model is already worth running today. I'm just not in that position. So I would put this down as promising rather than practical. For my work it becomes the obvious choice only when local catches up to current frontier, not the frontier of a year and a half ago. What gives me some optimism is the trajectory. This would have been science fiction not long ago, and once the next jump or two arrives, the privacy and the zero cost stop being consolation prizes and start being the real reason to switch.

## The one-line version

A 3B-active MoE, quantized to 4 bits to fit 32GB across two 16GB cards, resolved 61% of SWE-bench Verified on my own machine, and when it committed to a patch it was right about 74% of the time, with most of its misses coming down to running out of its budget rather than to flawed reasoning. It is impressive that it runs at all, and it is still a demo rather than a tool I would reach for, because it is matching where frontier sat eighteen months ago, not where it is now. The gap is closing quickly, and that trajectory, rather than today's score, is the thing I'm watching.
