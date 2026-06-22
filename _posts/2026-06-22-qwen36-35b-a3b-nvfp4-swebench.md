# I ran SWE-bench Verified on a 35B model at home. It scored 60%.

I have two RTX PRO 2000 Blackwell cards at home, 16GB of GDDR7 each, 32GB in total. These are not H100s, or anything close to one. They're the kind of hardware you buy when you want to run serious local inference and you're paying for it yourself.

The question I wanted answered was practical. Can a model running entirely on my own hardware do real coding work, fixing actual bugs in real codebases, or is local inference still a notch below the point where it becomes useful? I'd seen the leaderboard numbers for the open models, but I had no idea what they would translate to once a model was quantized down to fit my cards and pointed at problems I hadn't hand-picked. The only honest way to find out was to measure it.

So I ran the whole thing. All 500 problems of SWE-bench Verified, against Qwen3.6-35B-A3B in NVIDIA's NVFP4 4-bit build, served on vLLM with a 128K-token context window and an agent harness pointed at the local endpoint through LiteLLM. No cloud, no API key, just my own machine working through real GitHub issues from Django, sympy, scikit-learn, matplotlib and the rest.

Below is the scorecard, and what I took from it.

## What SWE-bench Verified actually tests, and why I used it

I'll admit something. For years I've read SWE-bench scores in model release posts and treated them as a single magic number, without really knowing what the test does. Running it myself forced me to learn, and the mechanics are worth spelling out.

SWE-bench Verified is 500 real GitHub issues, human-validated for quality, drawn from twelve popular open-source Python projects such as Django, sympy, scikit-learn, matplotlib, astropy and sphinx. Each task hands the model an issue description plus the repository at the commit just before the fix landed. The model has to produce a patch, an actual diff, that resolves the issue. Grading isn't a judgment call. The harness applies your patch and runs the repo's own unit tests. The tests tied to the bug have to flip from failing to passing, and the tests that were already passing have to stay green. Clear both and the instance counts as resolved. That is the whole game, and it explains why an empty patch scores zero, because there is nothing to run.

Two things clicked once I saw this. First, it's entirely Python. Every repo in the set is a Python library, so a strong SWE-bench number tells me about Python bug-fixing in particular, not coding in general, and not the languages I work in elsewhere. Second, it's a bug-fixing benchmark rather than a code-quality one. It asks whether the tests pass, not whether you would merge the change.

That second point is why the frontier has moved past it. The headline models are now measured on harder, quality-aware evaluations like Scale's SWE-bench Pro and Cognition's FrontierCode. FrontierCode grades whether a change is actually mergeable, meaning correct and in scope and maintainable, and it is hard enough that the best model in the world scores around 13%. Some of the motivation came from METR's finding that a large share of SWE-bench "solutions" is code no maintainer would actually accept.

I can't run any of that at home, and for this exercise I don't need to. SWE-bench Verified has years of widely reported, directly comparable results across nearly every model that matters. If you just want to place your local box against numbers everyone already knows, the old, well-worn ruler is the one to use.

## Why this model

My model-selection method wasn't sophisticated. I started at the top of the [SWE-bench Verified leaderboard](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified) and walked down until I found something that would fit on 32GB of VRAM. The top is a wall of models I have no hope of running, like DeepSeek-V4-Pro, Kimi-K2.6, GLM-5 and the 400B-class Qwens, all hundreds of billions of parameters and built for datacenters. So you keep scrolling. The first entry that scores in the 70s and also has any path onto two 16GB cards is Qwen3.6-35B-A3B, through NVIDIA's NVFP4-quantized checkpoint. And "has a path" is generous. It barely fits.

## Making it fit

"Barely" is doing real work in that sentence. NVIDIA's serving recipe for the 4-bit build assumes a single DGX Spark, meaning one device, a large pool of unified memory, and the whole model on one chip. I had two separate 16GB cards, so most of what follows is me re-deriving that recipe for hardware it wasn't written for. The first thing I wanted to know was whether 128K of context would fit at all, or just blow up the cards on startup. The back-of-envelope is worth showing.

The weights are the easy half. Thirty-five billion parameters at four bits is about 17.5 GB, and all 35 billion have to sit in memory even though only 3 billion activate per token, because mixture-of-experts saves compute, not memory. That leaves roughly 14.5 GB of the 32 for everything else.

The KV cache is the half that decides things. That is the running memory of attention, and it grows with every token of context, so at 128K tokens, left unchecked, it's what runs you out of memory. Two things keep it affordable. The first is storing it at eight bits instead of sixteen, which halves the cost outright. The second is the model itself. Qwen3.6 is a hybrid, where only every fourth layer (10 of its 40) runs full attention with a growing cache, while the other 30 use linear attention, whose state stays a small fixed size no matter how long the context gets. So the cache that actually scales with context comes from ten layers, not forty.

The arithmetic that falls out is friendly. Each full-attention layer holds a key and a value for two KV-heads of 256 dimensions, which at eight bits works out to roughly 2 × 10 × 2 × 256, or about 10 KB per token. Across 131,072 tokens the KV cache itself tops out near 1.3 GB. Stacked on the 17.5 GB of weights, the part of the budget I can size on the back of an envelope sits under 19 GB. (A naive estimate that treated all 40 layers as full attention would have predicted around 5 GB of cache and still fit comfortably. The hybrid design just makes it roomier.) But the cache was never the thing that bound me. The rest of the budget goes to unglamorous overhead, the CUDA graphs and activation buffers and the server's own workspace, and that overhead is what set the real ceiling. The envelope said a long context was safe. The allocator then settled the exact number at 128K, half the model's native 256K. That wasn't me being cautious. 128K is the most these two cards will physically hold once the overhead is paid, and I would have run the full 256K in a heartbeat if the VRAM allowed. It is worth holding onto that number, because the gap between it and the native window comes up again later, in a place I didn't expect.

The decision that justified the hardware was native 4-bit math. Out of the box, the recipe ran the mixture-of-experts layers by decompressing the 4-bit weights back to 16-bit before multiplying, and it printed a confident warning that the hardware had "no native FP4." It crawled along at about 4 tokens per second. The warning was simply wrong, and why it was wrong is the whole reason I bought these cards. Native 4-bit tensor cores are exactly what the Blackwell generation added over the one before it. Once I pointed the server at a compute path actually built for the architecture, it ran the math in native 4-bit, and the model went from a science experiment to something usable in a single change.

There were other snags, like a two-GPU deadlock I cleared by switching off an over-eager optimization, and a translation layer so the agent could read the model's tool-call format, but both were footnotes. The real lesson is that none of this was "download and run." On local hardware, the model is rarely the hard part. The stack around it is.

## The number

The model resolved 299 of the 500 problems, or 59.8%. For a sparse MoE that activates only 3 billion parameters per token, quantized to 4 bits to fit in 32GB and run through a generic open-source scaffold on my own machine, I'm happy with that. For context, on the public SWE-bench Verified leaderboard, 60% sits above GPT-4.1 (54.6%) and just under Gemini 2.5 Pro (63.8%). That is roughly last-generation frontier territory, at zero marginal cost per token, fully private, on hardware that fits on a shelf.

The comparison that stings is with the model's own headline. Qwen's model card reports 73.4% for this exact model, which puts it level with Claude Haiku 4.5 and Sonnet 4. I left 13.6 points on the table against that number. Working out where those points went is most of what this exercise was about.

## Where the 13 points went

The easy assumption is that quantization made the model dumber. That's mostly wrong, and the data shows why.

Here is what happened to the 500, broken down:

| Outcome | Count | % of 500 |
|---|---:|---:|
| **Resolved** (patch passed tests) | 299 | 59.8% |
| Unresolved (patch ran, failed tests) | 97 | 19.4% |
| **Empty patch** (no diff produced) | 86 | 17.2% |
| Harness error | 18 | 3.6% |

The number that changed how I read all of this is the next one. Of the 396 instances where the model actually committed to a patch that applied and ran, 299 passed. That is 75.5%.

When this model produces a patch, it's right about three times out of four. That points to strong reasoning paired with weak follow-through, not a model that thinks poorly. Most of the gap between 60% and 73% is those 86 problems where the model never produced a diff at all.

That is 17% of the benchmark producing nothing, and almost all of it comes from my setup hitting its own ceiling rather than the model failing to reason.

## The empty patches are really a VRAM problem

None of these empty patches is the model giving up. Each one carries an exit status, and they break down like this:

| Why the patch came back empty | Count | Share of empties |
|---|---:|---:|
| **Context window exceeded** | 59 | 69% |
| Turn budget exhausted | 24 | 28% |
| Repeated format errors | 3 | 3% |

Sixty-nine percent of my empty patches come from the agent running out of context window before it ever writes a diff. It reads a file, searches, reads another, reasons, reads another, and once there is enough material to read, it runs past 128K tokens and dies with nothing to submit. And this is where it all connects. The 128K ceiling is the one my 32GB of VRAM forced. The model ships with 256K of native context, and 128K is the most these 32GB will hold. It isn't a cautious setting I could have relaxed, it is the hardware's own limit. So the largest single chunk of my lost points traces straight back to the constraint that made the model barely fit in the first place. The memory budget and the empty patches turn out to be the same problem.

The pattern backs this up. The context-exceeded failures cluster in a handful of repos, with Django at 17, Sphinx at 13 and sympy at 11, all cases where, whatever the codebase is like internally, the runs read enough to overflow the window before landing a fix. The 3 format errors are the only failures that match the "model can't follow the diff format" story from the Unsloth GGUF community thread. The remaining 83 are about running out of room, not about the model's ability.

So the 13-point gap from Qwen's headline comes down to three things, which I can now rank by the evidence:

1. **Context budget, and it is the big one.** Most of my empty patches are context exhaustion, which follows from running 128K instead of the model's native 256K. There is no setting to fix this, since FP8 KV is already doing its job. It needs either more VRAM to lift the ceiling or a smarter, context-frugal harness that finds the right spot without reading everything first. That is where the largest block of points is waiting.
2. **The harness.** Qwen's 73.4% used their own internal, tuned scaffold, and I used a generic one. A scaffold that finds the relevant code before it reads everything, retries when the output comes back empty, and spends its context budget carefully would recover a chunk of those 59 without changing a single weight.
3. **Quantization.** NVFP4 is 4-bit and almost certainly costs something against the BF16 the headline was measured on, but it's the price of admission for 32GB, and it's the smallest of the three factors, nowhere near 13 points on its own.

## What the wins and losses look like

A few more things surfaced in the per-instance data.

There is no cheap way to cut losses early. Resolved instances took a median of 46 API calls and empties took 148, but that gap isn't the model working harder on the failures. Empties run long because hitting the context or turn limit is exactly what leaves them empty, and the gap between resolved runs and wrong-but-submitted ones (46 against 66) is tangled up with difficulty, since easy problems are both cheaper and more likely to be solved. So the call counts don't really measure effort, and they don't justify capping the budget as a shortcut either. A third of my actual solves took more than 60 calls, and some took 187. A long run isn't a doomed one.

Correct fixes are small, and wrong ones are bigger. The median resolved patch changes 4 lines, and the median unresolved one changes 10. Some of that is the same confound again, since harder bugs need bigger fixes and are more likely to come out wrong, but it still works as a rough signal. When this model is right, it's usually right in a few lines, and a diff that keeps growing is a fair sign that it has lost the thread rather than found it.

Some of the failures fixed the bug but broke something else. Eight of the 97 unresolved cases passed every test tied to the bug and still failed, because the patch broke a different test that had been passing. The fix itself worked, but it caused breakage elsewhere. That is exactly the kind of thing the newer "would you merge this?" benchmarks were built to catch.

## Would I use it for real work? Not yet

I came in asking whether a model on my own hardware could do my real coding work. The honest answer is no, at least not yet, and I want to be precise about why, because it isn't a knock on the model.

On its own terms, the result is impressive. A quantized 35B model on two 16GB cards, fully private, at zero marginal cost (the entire 500-instance run cost me electricity and nothing else), landing about where the best closed models sat 18 months ago. A few years ago I would have called that implausible. But "as good as frontier was 18 months ago" isn't the comparison that decides anything for me. The real question is why I would run 18-month-old capability when today's frontier is one API call away. Opus, Sonnet and GPT-5.5 resolve more and stall less, including on exactly the find-where-to-act work where my local setup runs out of context. With those sitting right there, the local model's real advantages, its privacy and its zero cost, don't outweigh the gap in capability.

The math is different for anyone who can't make that API call. If you are working under strict privacy, cost or offline constraints, a capable local model is already worth running today. I'm just not in that position. So I would put this down as promising rather than practical. For my work it becomes the obvious choice only when local catches up to current frontier, not the frontier of a year and a half ago. What gives me some optimism is the trajectory. This would have been science fiction not long ago, and once the next jump or two arrives, the privacy and the zero cost stop being consolation prizes and start being the real reason to switch.

## The one-line version

A 3B-active MoE, quantized to 4 bits to fit 32GB across two 16GB cards, resolved 60% of SWE-bench Verified on my own machine, and when it committed to a patch it was right about 75% of the time, with most of its misses coming down to context budget rather than reasoning. It is impressive that it runs at all, and it is still a demo rather than a tool I would reach for, because it is matching where frontier sat eighteen months ago, not where it is now. The gap is closing quickly, and that trajectory, rather than today's score, is the thing I'm watching.
