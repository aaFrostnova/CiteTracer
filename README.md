# CITETRACER: Cascading Multi-Agent Citation Hallucination Detection

[![Paper](https://img.shields.io/badge/arXiv-2605.08583-b31b1b.svg)](https://arxiv.org/abs/2605.08583) [![Dataset](https://img.shields.io/badge/HuggingFace-Hallucinated__Citation-yellow.svg)](https://huggingface.co/datasets/Afrostnova/Hallucinated_Citation)

CITETRACER detects fabricated citations in research papers and routes each
citation to one of a **12-code taxonomy** (R1-R3, P1-P3, H1-H6) so reviewers
see *which* field is wrong, not just whether the citation is fake. The
pipeline parses PDF or BibTeX input, retrieves evidence through a four-stage
cascade (Memory cache, URL Fetch, eight Scholar Connectors in parallel, Web
Agent fallback), runs deterministic field matching, and routes residual
cases to class-specialist judge agents.

![CITETRACER overview](figs/overview.png)

On a 2,450-citation synthetic benchmark CITETRACER attains class-level F1 of 97.0 / 95.8 / 98.5 for Real / Potential /
Hallucinated. On 957 real-world fabricated citations from ICLR 2026 and
ACM CCS 2026 desk-rejected submissions it catches 97.1% with no
abstentions. See [docs/taxonomy.md](docs/taxonomy.md) for the full
taxonomy and [docs/metric_guide.md](docs/metric_guide.md) for the scoring
protocol.

## Setup

```bash
# 1) clone
git clone https://github.com/aaFrostnova/Citation_Hallucination_Detection.git
cd Citation_Hallucination_Detection

# 2) install (Python 3.10+)
pip install -r requirements.txt

# 3) configure (edit the keys described in "Configuration" below)
cp config.example.json config.json
```


## Configuration

The pipeline reads `config.json`. Every key can also be overridden by an
environment variable of the same name prefixed with `CITATION_CHECKER_`.
`config.example.json` ships with documentation comments (any key whose
name starts with `_` is ignored by the loader, so feel free to leave them
in or strip them out).

| Block                   | What it controls                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `entry_extraction`      | OCR model for bibliography-region detection (PDF only).                                                                   |
| `ocr_vlm_extract`       | Parser Agent (cropped-block VLM reparse). Set `provider` to a cloud API and fill the matching sub-block.                  |
| `verification_llm`      | Matcher Agent + Class-Specialist Judgers. `max_candidates` caps top-K candidates per source.                              |
| `connectors`            | Connector cache path, DBLP mirror paths, eight academic sources, web-search provider, all related API keys.               |
| `citation_parse_method` | Fixed at `"ocr_vlm_extract"`.                                                                                             |

### How to fill `config.json`

After `cp config.example.json config.json`, edit four things in order:

**1. Pick LLM providers for `ocr_vlm_extract` and `verification_llm`.**
Each block has its own `provider` field. The two blocks are independent,
so you may mix providers (e.g. Bedrock for the Parser, OpenAI for the
Verifier).

  - `bedrock`  → set `bedrock.region`, `bedrock.model_id` (e.g.
    `"qwen.qwen3-vl-235b-a22b"`), and `bedrock.bearer_token`.
  - `openai`  → set `bedrock.model_id` (e.g. `"gpt-5"`) and export
    `OPENAI_API_KEY` in your shell.
  - `azure_openai` → set `bedrock.model_id` to your Azure deployment
    name (e.g. `"gpt-5.4"`) and export
    `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`.

The `bedrock.model_id` field name is historical; for `openai` and
`azure_openai` it is interpreted as the OpenAI model name or Azure
deployment name respectively.

**2. Configure `entry_extraction` if you will run on PDF input.**
Point `entry_extraction.local.model_path` at a downloaded
`DeepSeek-OCR-2` checkpoint. BibTeX-only users (`apps.bib_checker.run`)
can leave this block alone.

**3. Pick a Web Agent backend in `connectors.web_search_provider`.**
Either `tavily` (set `tavily_api_key`) or `serpapi` (set `serpapi_key`).
The two are functionally interchangeable.

**4. Fill optional Scholar Connector keys for higher quotas.**
Every Scholar Connector works without a key, but `semantic_scholar_api_key`
and `ncbi_api_key` (+ `ncbi_email`) materially raise the rate-limit cap.
`openalex_mailto` puts your traffic in the polite pool. Leave any field
empty to skip.

After editing, sanity-check the file is loadable:

```bash
python -c "from apps.pdf_checker.config import load_pdf_checker_config; cfg = load_pdf_checker_config(); print('OK', cfg.ocr_vlm_extract.provider, '/', cfg.verification_llm.provider)"
```

### Supported LLM backends

`ocr_vlm_extract` (Parser Agent) and `verification_llm` (Matcher Agent +
Class-Specialist Judgers) both go through the same
`packages.llm.client.build_chat_client(...)` factory, so the same provider
list applies to both. The OCR step (`entry_extraction`) is image-to-text
on the rendered PDF page and runs only on the in-tree DeepSeek-OCR vLLM
bundle.

| `provider`     | Used by                  | Required env / config                                                                                                                                                                              |
| -------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bedrock`      | parser, verifier         | `bedrock.bearer_token`, `bedrock.region`, `bedrock.model_id`. Tested IDs: `qwen.qwen3-vl-235b-a22b`, `moonshotai.kimi-k2.5`, `us.anthropic.claude-opus-4-7-20251101-v1:0`, `us.anthropic.claude-sonnet-4-5-v1:0`. |
| `openai`       | parser, verifier         | `OPENAI_API_KEY` env, `bedrock.model_id` (interpreted as OpenAI model name; multimodal models such as `gpt-4o` / `gpt-5` are supported because `OpenAIChatShim` translates image blocks to `image_url`). |
| `azure_openai` | parser, verifier         | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` env, `bedrock.model_id` (interpreted as deployment name, e.g. `gpt-5.4`).                                                                          |
| `local`        | OCR `entry_extraction`   | `entry_extraction.local.model_path` pointing at the DeepSeek-OCR-2 HF checkpoint; uses the in-tree multimodal vLLM bundle in `apps/pdf_checker/ingest/reference_segmenter.py`.                       |

Parser Agent and Verifier are cloud-API only. The local-LLM path was
removed because the local OCR vLLM bundle and a second local LLM tend to
fight for GPU memory and have incompatible `transformers` / `vllm`
version requirements. Per-block fields (`max_new_tokens`, `temperature`,
`max_candidates`) take effect under every supported provider.

### Scholar Connectors (academic data sources)

Each Scholar Connector queries one external bibliographic source. Most
are free without authentication; a few accept a key for higher quota.

| Connector          | Field in `connectors.*`                        | Key required?                    |
| ------------------ | ---------------------------------------------- | -------------------------------- |
| `arxiv`            | (none)                                         | No.                              |
| `dblp_online`      | (none)                                         | No.                              |
| `dblp_sqlite`      | `dblp_sqlite_path`                             | No (offline mirror).             |
| `crossref`         | (none)                                         | No.                              |
| `acl_anthology`    | (none)                                         | No.                              |
| `europepmc`        | (none)                                         | No.                              |
| `pubmed`           | `ncbi_api_key`, `ncbi_email`                   | Optional. Without a key, NCBI throttles to ~3 requests/s; with a key, ~10 requests/s. |
| `semantic_scholar` | `semantic_scholar_api_key`                     | Optional. Public tier is severely rate-limited; an API key (request at semanticscholar.org) lifts the cap. |
| `openalex`         | `openalex_mailto`, `openalex_api_key`          | Optional. `mailto` puts you in the polite pool; `api_key` is for premium quota. |
| `url_direct`       | (uses `tavily_api_key` as fallback)            | No, except for the Tavily URL-extract fallback when a citation has only a vague URL. |
| `web_search`       | `web_search_provider` + the matching API key   | **Yes.** See the Web Search table below.                                            |

### Web Search backends (`connectors.web_search_provider`)

The Web Agent needs one general-web search backend to close the cascade
for long-tail citations. The two options below are functionally
interchangeable: each returns the top-5 results as
`{url, title, snippet}` and the rest of the pipeline does not care which
one produced them. Pick whichever you have credentials for.

| `web_search_provider` | Required key(s)                                        | Notes                                                                                       |
| --------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `tavily`              | `tavily_api_key`                                       | Default. Tavily's own LLM-oriented index; snippets are markdown excerpts of the page body.  |
| `serpapi`             | `serpapi_key`                                          | Wraps Google Search through SerpAPI; snippets are Google's native snippets.                 |

If you set `web_search_provider` to a value whose key is not configured,
the Web Agent stage logs a warning and is skipped (the rest of the
cascade still runs).

CLI flags worth knowing (`python -m apps.pdf_checker.run --help` for the full list):

| flag                   | default | purpose                                                  |
| ---------------------- | ------- | -------------------------------------------------------- |
| `--paper-workers`      | 2       | parallel papers when `--input` is a directory            |
| `--citation-workers`   | 4       | parallel citations within each paper                     |
| `--connector-workers`  | 8       | parallel connector calls per citation                    |
| `--offline-only`       | off     | skip every online connector (use local DBLP only)        |
| `--extract-only`       | off     | run Stages 1-2 only and dump parsed citations            |
| `--resume`             | off     | skip papers whose `_report.md` already exists            |
| `--save-ocr-artifacts` | off     | persist OCR debug artifacts under `<out>/ocr_artifacts/` |

## Quick start

### From a PDF

```bash
mkdir -p artifacts/demo
python -m apps.pdf_checker.run \
  --input path/to/paper.pdf \
  --out  artifacts/demo \
  --paper-workers 1 --citation-workers 8 --connector-workers 6
```

### From a BibTeX file

```bash
python -m apps.bib_checker.run \
  --input path/to/refs.bib \
  --out  artifacts/demo \
  --paper-workers 1 --citation-workers 8 --connector-workers 6
```

Output per input file:

- `<stem>_report.json` — structured per-citation verdicts
- `<stem>_report.md`   — reviewer-friendly markdown report
- `<stem>_report.timing.json` — per-phase latency (PDF only)

`--input` accepts a single file or a directory. `--resume` skips inputs
whose `_report.md` already exists, useful for batch jobs.

## Verifier-only mode (skip PDF extraction)

If your input is already-parsed citations,
skip Stage 1 and run only the cascade plus judges. Inputs are JSON files
with the schema produced by Stage 1 (`results/eval_*_rule_based.json`).

```bash
python scripts/eval_judge_only.py \
  --inputs   results/eval_*_rule_based.json \
  --out      results/judge_only_run \
  --model    qwen.qwen3-vl-235b-a22b \
  --workers  16
```

This also accepts `--exclude-connectors` to ablate any connector group
(e.g. `--exclude-connectors web_search`).

## Datasets

Two datasets ship with the repository:

| Path                                       | What it is                                                                                                                | Size  |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | ----- |
| `data/synthetic_data/v2/`                  | 2,450 synthetic citations across the 12 codes (one JSON per subtype: `R1.json`, `R2.json`, `R3.json`, `P1.json`, `P3.json`, `H1.json`..`H6.json`; `P2` is reserved for non-academic citations and not in this snapshot), with `meta.json` carrying the per-citation ground-truth label. Built from real BibTeX seeds with controlled LLM mutations. | 2,450 |
| `data/iclr2026_hallucinated/`              | 957 real-world fabricated citations from ICLR 2026 desk-rejected submissions. `hallucinated_refs.json` is the raw list; `hallucinated_refs_structured.json` is the parsed structured-record version used by the verifier. | 957   |

The synthetic set is the primary benchmark used in every paper table; the
real-world set is the out-of-distribution test in Section 4.4.

A mirror of both datasets is also published on Hugging Face:
[Afrostnova/Hallucinated_Citation](https://huggingface.co/datasets/Afrostnova/Hallucinated_Citation).

## Reproduce paper numbers

```bash
# main verification on the 2,450-citation synthetic benchmark
bash scripts/eval_H_rule_based.sh
bash scripts/eval_R_rule_based.sh
```

## Repository layout

```
apps/pdf_checker/         end-to-end PDF entry point (run.py, config.py, ingest/)
apps/bib_checker/         BibTeX-only entry point (skips Stage 1, runs verifier directly)
packages/connectors/      bibliographic connectors + cache + orchestrator
packages/core/            verifier, cascading agents, field matcher, judges
packages/llm/             provider-agnostic LLM client (bedrock / openai / azure / local vLLM)
scripts/                  evaluation, benchmarking, scoring, sbatch templates
data/synthetic_data/v2/   2,450-citation synthetic benchmark, one JSON per subtype
data/iclr2026_hallucinated/  957 real-world fabricated citations from ICLR 2026 desk rejections
docs/                     taxonomy, metric guide, render-omission notes
tests/                    pytest unit tests for parser, normalizer, agents
```

## 📌Citation

If you find this repo helpful, please kindly cite the paper:

```bibtex
@article{li2026source,
  title={Source or It Didn't Happen: A Multi-Agent Framework for Citation Hallucination Detection},
  author={Li, Mingzhe and Lin, Zhiqiang and Ma, Shiqing},
  journal={arXiv preprint arXiv:2605.08583},
  year={2026}
}
```
