# Continual learning: the problem DDT works on

The goal behind DDT is continual learning: an agent that keeps learning while it works, without forgetting what it learned before. This repository is a first step, built in one day for a hackathon whose track fixed the task (deep research). The work continues after the hackathon.

## The problem

To learn on the fly, a model would normally update its weights. New training tends to overwrite old knowledge (catastrophic forgetting), and deep networks can also gradually lose their ability to learn in continual-learning settings (loss of plasticity; Dohare et al., 2024). Closed models, such as the ones behind Claude Code, Codex and Antigravity, give no access to their weights at all.

## DDT's bet

Do not update the weights. Keep the model fixed and put everything the agent learns into vectors outside it, in the twin. A metacognitive layer decides what gets written, protected or merged. This needs no weight access, so it works with closed and open models alike.

Keeping a model stable and adding a separate memory is not new in itself (see related work below). What DDT tests is a metacognitive gate on that memory: nothing is written unless its quoted words are found on the source page, and the twin predicts each run and measures its own error.

## Four open parts, and where DDT stands

| Open part | DDT today | Evidence so far | After the hackathon |
| --- | --- | --- | --- |
| 1. The rule for updating the vectors | A claim is written only if its quote is found on the source page. One versioned batch per run writes every store. | The citation control rejected a made-up quote (live run of the simple version, 15:09:07). | Learn which writes pay off later, from the episode and tool stores. |
| 2. Stopping new vectors from overwriting old ones | The model never changes. Claims are added; only near-duplicates (cosine 0.95 or more) merge. Rollback undoes the latest runs. | S4 passed live on the simple version: after 3 questions on a new topic, the first 3 questions were still answered from memory with 0 searches. | A long sequence of topics, with every earlier set asked again. |
| 3. How the model reads the vectors | Up to 5 claims at cosine 0.75 or more go into the planner's prompt as text. | S3 passed live on the simple version: every second run used 0 searches. | Level 2: vectors read inside an open-weight Nemotron model. |
| 4. A test that proves nothing was forgotten | S3 (learns) and S4 (keeps), with a brain-off control that must fail. | A short test only: 2 topics, 6 questions. | The same test at scale, compared with published methods. |

## What the twin's self-knowledge shows so far

- **S9, metacognition learns: passed, in stand-ins only.** In the stand-in run at 16:23:42, on the second half of 18 logged claims, the online weights' mean log loss was 0.0452, against 0.6931 with learning switched off.
- **S8, the twin predicts its own runs better over time: failed.** In the same run, the search-prediction error rose from 1.000 in round 1 to 1.597 in round 2. The predictor averages the searches of similar past episodes and ignores what the brain already knows, so after the twin has learned an answer it still expects to search.

Stand-in runs replace Token Factory and Tavily with fixed fakes; they are not live results.

## Plan after the hackathon

1. A predictor that uses what the twin already knows, judged by S8 with its rule unchanged.
2. A full live `verify.py` run of the current code, with every result copied into the README by script.
3. A long forgetting test: many topics in sequence, with every earlier set asked again.
4. Level 2: the same twin as vectors inside an open-weight Nemotron model on GPUs, judged by the same forgetting test.

## Related work, and how far each was read

| Work | What learns | Read |
| --- | --- | --- |
| [Titans: Learning to Memorize at Test Time](https://arxiv.org/html/2501.00663v1), Behrouz, Zhong, Mirrokni (Google), Dec 2024 | A neural long-term memory module whose parameters update at test time | abstract, introduction |
| [MemoryLLM: Towards Self-Updatable Large Language Models](https://arxiv.org/abs/2402.04624), Wang et al., 2024 | A fixed-size memory pool in the transformer's latent space | abstract |
| [Continual Learning via Sparse Memory Finetuning](https://arxiv.org/abs/2510.15103v1), Lin et al. (Meta FAIR), Oct 2025 | Only the memory slots most activated by the new knowledge. Reported NaturalQuestions F1 drop: 89% with full finetuning, 71% with LoRA, 11% with sparse memory finetuning | abstract |
| [Learning to Prompt for Continual Learning](https://arxiv.org/abs/2112.08654v2) (L2P), Wang et al. (Google), 2021 | Small learnable prompts kept in a memory space | abstract |
| [Introducing Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning) (Hope), Behrouz, Mirrokni (Google), NeurIPS 2025 | Modules that update at different frequencies | blog post |
| [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/pdf/1612.00796) (EWC) | Weights, slowed where they matter for old tasks | abstract |
| [Loss of plasticity in deep continual learning](https://www.amii.ca/research/loss-of-plasticity-in-deep-continual-learning), Dohare, Hernandez Garcia, Lan, Rahman, Mahmood, Sutton, Aug 2024 | (the problem, not a method) | research page |

These were found by targeted searches, not a systematic review, so this repository makes no claim of novelty.

Background, not read in this project: Richard Sutton's essay "The Bitter Lesson" ([Wikipedia summary](https://en.wikipedia.org/wiki/Bitter_lesson)).

Written by Claude (Cowork) from Hasnat's ideation notes and the project's results, 18 October 2026.
