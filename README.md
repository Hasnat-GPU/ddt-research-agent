# DDT: a research agent that learns across runs

DDT turns a research question into a cited report. It keeps only claims whose quoted words are really found on the source page, and stores them in a versioned brain outside the model, so the next run starts from what it already knows. A self-model tracks, per topic, how often claims pass the quote check and raises the evidence bar when they don't.

Track: **Agentic Deep Research with Nebius Token Factory and Tavily**. Also entered for **Best use of Nemotron**: every planning, drafting and writing call runs NVIDIA Nemotron on Token Factory.

## Setup

Needs Python 3.12 or newer. On Windows:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

On macOS or Linux: `source .venv/bin/activate` and `cp .env.example .env`.

Open `.env` and fill the two keys: `NEBIUS_API_KEY` (Nebius Token Factory) and `TAVILY_API_KEY` (Tavily). The model ids are already set.

## Run

```
python run.py "What model sizes does the NVIDIA Nemotron 3 family include?"
```

The trace prints live; the cited report is written to `reports/` and the trace to `traces/`. Run the same command a second time: the agent answers from its brain, with the same citations and no search.

Other commands:

```
python run.py "question" --topic "nvidia nemotron"   # file the run under a topic
python run.py "question" --no-brain                  # neither read nor write the brain
python run.py --stats                                # brain counts and the self-model table
python run.py --rollback 1                           # undo the last run (one brain version)
```

Offline, with deterministic stand-ins instead of the model and the web: set `DDT_FAKE=1`, then `pytest` or `python verify.py --fake`.

## How it works

One run is a loop of seven steps around a fixed model. Each step prints one or more trace lines with its tag.

| Step | Tag | What happens |
| --- | --- | --- |
| Read | [READ] | Retrieve the 5 closest checked claims from the brain (cosine similarity 0.75 or more) and the topic's evidence bar. |
| Plan | [PLAN] | Nemotron marks which known claims already answer the question and writes web queries only for what is missing: none if the brain covers it, exactly 3 if nothing is known, never more than 3. |
| Act | [SEARCH] [DECIDE] | Tavily searches (advanced depth, 5 results, raw page text). Nemotron drafts claims, each with 8 to 30 words quoted from its source, and decides whether the evidence is enough. If not, one more round of up to 2 queries, at most 6 searches per run. |
| Check | [CHECK] | A claim passes only if its quote (8 words or more) is found in the source page's text, ignoring case, spacing and curly quotes. Failed claims are dropped. |
| Write | [WRITE] | Passed claims go into the brain; a claim at cosine similarity 0.95 or more to a stored claim is merged into it. The run and its failed claims are logged. The whole write is one brain version. |
| Reflect | [REFLECT] | The self-model adds the run's pass and fail results to the topic's last 5 results. |
| Report | [REPORT] | Nemotron writes a short answer from the checked claims only, every sentence cited [n]. The report lists each source with its quoted words, what was known before, what is new, and what was dropped. |

The evidence bar rule (per topic, applied in this order after every run):

1. Raise to 2 if the last 5 results hold 3 or more results and fewer than 80% passed (with 5 results: fewer than 4 of 5).
2. Lower to 1 if the last 5 results are 5 passes.
3. Otherwise keep it.

With evidence bar 2, every part of the question needs claims from 2 different sources before the agent stops searching.

## Verification

`python verify.py` runs the checks below against Token Factory and Tavily, with at most 40 Tavily searches, prints PASS, FAIL or NOT TRIGGERED with the measured numbers, and appends them to the results file. A check counts only if it can fail, so there are two controls.

| Check | Pass when |
| --- | --- |
| S2 Citations hold | 3 random claims from the latest report, re-fetched with Tavily extract, all still contain their quoted words. |
| S3 Learns | Set A (3 questions on NVIDIA Nemotron), then set A again: every second run uses at most 1 search, uses known claims, and cites only sources from its first run. |
| Failure control | The same rule with the brain switched off must fail (the second run needs 3 or more searches). Otherwise every result is void. |
| S4 Keeps | After set B (3 questions on the EU AI Act), set A still passes the S3 rule. |
| S5 Knows itself | A topic where quotes often fail, run twice: if the first run raised the evidence bar, the second run's trace shows bar 2. If nothing failed: NOT TRIGGERED, never PASS. A stand-in version with forced failures must PASS. |
| Citation control | A made-up quote against a real fetched page must be rejected. |
| S6 Undo | On a copy of the brain, rollback 1 removes exactly the claims the last run added; earlier claims stay. |
| S7 Any model | With `NEBIUS_MODEL` set to a second Token Factory model and no code change, the S3 rule holds on one question. |

Latest live results (copied from the results file):

```
verify.py run 2026-10-18 15:09:07 | live | model nvidia/nemotron-3-super-120b-a12b | embedding Qwen/Qwen3-Embedding-8B | Tavily cap 40 searches
S5 stand-in PASS | first run: 1 passed, 2 dropped, bar raised yes; second run trace shows evidence bar 2: yes
S3 PASS | set A, 3 questions
  "What model sizes does the NVIDIA Nemotron 3 family include?": searches 3 -> 0, citations 4 -> 3, known claims used 3, subset yes, rule holds
  "Under which licence does NVIDIA release the Nemotron 3 models?": searches 3 -> 0, citations 1 -> 1, known claims used 1, subset yes, rule holds
  "Where can developers run or deploy NVIDIA Nemotron models?": searches 2 -> 0, citations 4 -> 3, known claims used 3, subset yes, rule holds
Failure control PASS | brain off, 1 question: S3 rule fails as it must; second run searches 3 (needs 3 or more)
  "What model sizes does the NVIDIA Nemotron 3 family include?": searches 3 -> 3, citations 3 -> 3, known claims used 0, subset no, rule fails
S4 PASS | after set B (3 questions), set A again:
  "What model sizes does the NVIDIA Nemotron 3 family include?": searches 3 -> 0, citations 4 -> 1, known claims used 1, subset yes, rule holds
  "Under which licence does NVIDIA release the Nemotron 3 models?": searches 3 -> 0, citations 1 -> 1, known claims used 1, subset yes, rule holds
  "Where can developers run or deploy NVIDIA Nemotron models?": searches 2 -> 0, citations 4 -> 3, known claims used 3, subset yes, rule holds
  S2 pass: https://www.knolli.ai/post/nvidia-nemotron-models | "developers can deploy Nemotron models using open frameworks such as vLLM, SGLang, Ollama, and llama.cpp on NVIDIA GPUs across edge, cloud, and data center environments."
  S2 pass: https://developer.nvidia.com/topics/ai/nemotron | "Easily deploy models using open frameworks like vLLM, SGLang, Ollama and llama.cpp on any NVIDIA GPUs—from the edge and cloud to the data center."
  S2 pass: https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron | "Yes, you can download and run NVIDIA Nemotron models from [Hugging Face](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3) for free in production."
S2 PASS | 3 of 3 random claims from 20261018-151053-where-can-developers-run-or-deploy-nvidia-nemotron.md found in their re-fetched source
Citation control PASS | made-up quote against https://www.knolli.ai/post/nvidia-nemotron-models (33206 characters fetched): check_claim returned False
S6 PASS | claims 13 -> 11 after rollback 1 (version 9 -> 8); the last run ("What are the maximum fines under the EU AI Act?") added 2; removed exactly its claims: yes; earlier claims kept: yes
S5 PASS | topic "european ai startup funding": first run: 2 passed, 1 dropped, bar raised yes; second run trace shows evidence bar 2: yes
S7 PASS | model nvidia/Nemotron-3.5-Super-VL
  "What model sizes does the NVIDIA Nemotron 3 family include?": searches 3 -> 0, citations 4 -> 2, known claims used 2, subset yes, rule holds
Tavily searches used: 30 of 40
Summary: S5 stand-in PASS, S3 PASS, Failure control PASS, S4 PASS, S2 PASS, Citation control PASS, S6 PASS, S5 PASS, S7 PASS
```

## Tools used

| Tool | Used for |
| --- | --- |
| Nebius Token Factory, chat completions | Planning, drafting claims, writing the answer. Base URL `https://api.tokenfactory.nebius.com/v1` (OpenAI-compatible). Model `nvidia/nemotron-3-super-120b-a12b` (`NEBIUS_MODEL`); S7 also uses `nvidia/Nemotron-3.5-Super-VL`. |
| Nebius Token Factory, embeddings | Claim and question vectors for the brain. Model `Qwen/Qwen3-Embedding-8B` (`NEBIUS_EMBED_MODEL`, 4096 dimensions). If the call fails or `DDT_EMBED=local`, a local 512-dimension hashed bag of words is used. |
| Tavily | `search` (advanced depth, 5 results, raw content) and `extract` (re-fetching a page for the quote check and S2). The agent's only access to the web. |
| Python packages | `openai`, `tavily-python`, `numpy`, `python-dotenv`, `pytest`, versions pinned in `requirements.txt`. |

## Data

Everything DDT learns is in `brain/` (or the folder in `DDT_BRAIN_DIR`). Nothing is sent anywhere except the model and search calls.

| File | Contents |
| --- | --- |
| `brain.jsonl` | One JSON entry per line, readable. |
| `brain.npy` | The claim vectors: row i belongs to the i-th claim entry. |
| `selfmodel.json` | Per topic: `runs`, `checked`, `passed`, `last5` (latest pass/fail results), `searches`, `evidence_bar`, `models`. |
| `meta.json` | `{"version": N}`, the current brain version. |
| `versions/v<N>/` | The brain files as they were at version N; `--rollback` restores one. Each run is one version. |

Claim entry fields: `id` (8 hex characters), `kind` ("claim"), `text`, `topic`, `source_url`, `also_sources` (sources of merged duplicates), `quote` (the checked words), `confidence` (0 to 1, the model's own), `origin` ("checked claim"), `version`, `created_at`, `model`, `stale`.

Run entry fields: `kind` ("run"), `question`, `topic`, `searches`, `passed` (count), `added`, `merged`, `failed` (each failed claim's text, url and quote), `evidence_bar`, `model`, `version`, `created_at`.

## Limits

- The brain is outside the model. DDT never changes model weights; it changes what the model is given.
- The check is an exact quote match. It proves the words are on the page, not that the claim reads them correctly. Paraphrased quotes are dropped, even when true.
- S5 depends on real quote failures. When every quote on the topic passes, it reports NOT TRIGGERED.
- Read retrieves only the 5 closest claims. In the example's second run, the claim about Nemotron 3 Ultra was not among the 5 retrieved, so the second answer leaves it out.
- Retrieval uses one similarity threshold (0.75), checked on one embedding model.
- The quote check confirms the quoted words, not the claim's reading of them: in the example's first run, claim [4] calls a model "Nemotron 3 Ultra" while its quote names "Nemotron Ultra".

## Credits

The starting idea is the Nebius cookbook [Deep research agent with Tavily](https://dev.nebius.com/cookbook/deep-research-agent-tavily). DDT replaces its framework with a small loop of its own so that every decision shows in the trace, and adds the quote check, the brain and the self-model. All code was written on 18 October 2026 at Nebius Build Week, Berlin.

## Example runs

Set A question 1, from the live verify.py run of 18 October 2026:

| Run | Report | Trace | What it shows |
| --- | --- | --- | --- |
| First | [1-first-run-report.md](examples/1-first-run-report.md) | [1-first-run-trace.log](examples/1-first-run-trace.log) | Nothing known, 3 searches, 5 claims drafted, 4 passed the quote check, 1 dropped (its quote was edited by the model). |
| Second | [2-second-run-report.md](examples/2-second-run-report.md) | [2-second-run-trace.log](examples/2-second-run-trace.log) | 5 known claims retrieved, 3 marked as covering the question, 0 searches, citations a subset of the first run's. |
