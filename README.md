# Curiosity Engine

**An autonomous research pipeline that generates its own follow-up questions.**

Curiosity Engine is a queue-based research system that searches four academic databases (arXiv, Semantic Scholar, OpenAlex, PubMed), generates markdown reports, and uses LLM-assisted follow-up question generation to build branching research trees. It runs as a daemon or one-shot CLI tool, consuming topics from a shared JSON queue and producing structured research reports.

## Features

- **Multi-database search** -- queries arXiv, Semantic Scholar, OpenAlex, and PubMed in parallel with automatic rate-limit handling
- **Automatic deduplication** -- normalized title matching removes duplicate papers across sources
- **Curiosity-driven follow-ups** -- uses an LLM (Claude CLI, or a local model through Ollama) to generate follow-up research questions from gaps in results
- **Depth-capped branching** -- configurable maximum depth prevents infinite recursion in research trees
- **Semantic novelty detection** -- sentence-transformer embeddings detect when a follow-up is too similar to existing queue items (optional, requires `sentence-transformers`)
- **Interest decay** -- threads that return too few papers are pruned automatically
- **Memory scanning** -- queries a PostgreSQL or SQLite (mcp-memory-service) memory store for unresolved thoughts and open questions, then generates new research topics (optional; see [Running locally](#running-locally-sqlite--ollama))
- **Message bus integration (CAOGL)** -- monitors a JSON inbox file for research requests from other agents or systems
- **Desktop idea file intake** -- watches a markdown file for manually added research ideas, auto-queues them, and marks them as processed
- **Structured markdown reports** -- date-organized output with citation counts, author lists, abstracts, and summary statistics

## Quick Start

```bash
# Clone the repository
git clone https://github.com/liberation-labs/curiosity-engine.git
cd curiosity-engine

# Install dependencies
pip install -r requirements.txt

# (Optional) Install full feature set
pip install psycopg2-binary sentence-transformers

# Configure
cp .env.example .env
# Edit .env with your settings, then export it: the engine does not load .env itself
set -a; . ./.env; set +a

# Add a research topic
python curiosity_engine.py add "transformer architecture scaling laws"

# Run a single item
python curiosity_engine.py run-once

# Or start the daemon
python curiosity_engine.py
```

## CLI Usage

```
curiosity_engine.py                     # Start daemon (polls every 5 minutes)
curiosity_engine.py add <topic>         # Add a topic to the research queue
curiosity_engine.py list                # List all queue items with status
curiosity_engine.py run-once            # Process the highest-priority pending item
```

Queue item statuses:
- `[.]` pending -- waiting to be processed
- `[~]` in_progress -- currently being researched
- `[x]` completed -- report generated
- `[!]` failed -- error during processing
- `[-]` skipped -- an engine-written topic off the asker's fields (see [the asker file](#the-asker-file))

## Architecture

### The Curiosity Engine Concept

The core loop is simple: search, report, branch.

1. **Search** -- A topic is pulled from the queue. All configured databases are queried, results are deduplicated, and a markdown report is written.

2. **Follow-up generation** -- If enough papers are found (controlled by `MIN_QUALITY_SCORE`), the top results are sent to an LLM which identifies gaps, tensions, and unexplored angles. It returns concrete follow-up search queries.

3. **Novelty filtering** -- Each follow-up is checked against the existing queue. Exact title matches are rejected. If `sentence-transformers` is installed, cosine similarity against all existing topics filters out semantically redundant questions (threshold: `NOVELTY_THRESHOLD`).

4. **Depth capping** -- Every queue item tracks its generation depth and parent ID. Once a branch reaches `MAX_DEPTH`, no further follow-ups are generated for that lineage.

5. **Interest decay** -- If a search returns fewer papers than `MIN_QUALITY_SCORE`, the thread is considered a dead end and no follow-ups are generated, even if depth allows it.

6. **Memory scanning** (optional) -- Periodically queries a PostgreSQL database for memory fragments containing phrases like "open question", "worth exploring", or "further research". An LLM extracts concrete research topics from these fragments and adds them to the queue.

### Input Sources

The daemon checks three input sources each cycle:

- **Queue file** -- The primary JSON queue (`CURIOSITY_QUEUE`)
- **CAOGL inbox** -- A JSON file where other systems can drop messages matching `research: <topic> [priority:N]`
- **Idea file** -- A markdown file with a `## Ideas` section; new bullet points are auto-queued and struck through

### Output

Reports are written to date-organized subdirectories under `CURIOSITY_OUTPUT`:

```
output/
  2026-02-01/
    a1b2c3d4_transformer-scaling-laws.md
    e5f6g7h8_attention-mechanism-alternatives.md
  2026-02-02/
    ...
```

## Configuration Reference

All configuration is via environment variables. Set them in your shell, or keep them in a `.env` file and export it
before running (`set -a; . ./.env; set +a`): the engine does not load `.env` itself.

| Variable | Default | Description |
|---|---|---|
| `CURIOSITY_QUEUE` | `./data/research_queue.json` | Path to the JSON research queue |
| `CURIOSITY_OUTPUT` | `./output` | Directory for generated reports |
| `CURIOSITY_LOG` | `./output/runner.log` | Log file path |
| `CURIOSITY_INBOX` | `./data/research_inbox.json` | CAOGL message bus inbox file |
| `CURIOSITY_IDEAS` | *(disabled)* | Path to a markdown idea file |
| `CURIOSITY_PID` | `./data/runner.pid` | PID file for daemon mode |
| `DATABASE_URL` | *(disabled)* | PostgreSQL connection string for memory storage |
| `CURIOSITY_PG_TABLE` | `memories` | Table the PostgreSQL path reads memory fragments from and writes summaries to (`schema.table` allowed; plain names only). **Was hard-coded to `lyra_memories`:** set it to keep writing there. |
| `CURIOSITY_PG_NAMESPACE` | `curiosity_engine` | `consciousness_id` written with each summary. **Was hard-coded to `lyra_consciousness`.** |
| `CLAUDE_CLI` | `claude` | Path to Claude CLI binary |
| `CLAUDE_MODEL` | `haiku` | Claude model for follow-up generation |
| `OPENALEX_EMAIL` | *(empty)* | Email for OpenAlex polite pool (faster rate limits) |

### Hardcoded Defaults (edit in source)

| Parameter | Value | Description |
|---|---|---|
| `POLL_INTERVAL` | 300s | Daemon polling interval |
| `MAX_PAPERS_PER_DB` | 10 | Maximum papers to fetch per database |
| `MAX_FOLLOWUPS_PER_ITEM` | 2 | Follow-up questions generated per topic |
| `MAX_DEPTH` | 2 | Maximum branching depth (`MAX_DEPTH_NOVEL`, 3, when a thread finds at least 3 papers its ancestors did not) |
| `RELEVANCE_THRESHOLD` | 0.3 | Minimum cosine similarity between a paper's abstract and the topic for the paper to be kept |
| `NOVELTY_THRESHOLD` | 0.75 | Cosine similarity cutoff for deduplication |
| `MIN_QUALITY_SCORE` | 3 | Minimum papers to continue a branch |
| `MEMORY_SCAN_INTERVAL` | 6 hours | Time between memory scans |

## Running locally (SQLite + Ollama)

The engine can run with no Claude CLI and no PostgreSQL: a local model served by [Ollama](https://ollama.com) writes the topics, the memory it scans is an [mcp-memory-service](https://github.com/doobidoo/mcp-memory-service) SQLite store, and cron runs it with `run-once`. Apart from the academic searches (and the embedding model's first download), nothing leaves the machine unless you configure it to, and every topic an LLM writes passes a privacy filter before it is searched.

### Requirements

- Python 3 (tested on 3.12) with `requests`; add `sentence-transformers` for the relevance, novelty and asker checks (without it they are off, and the engine says so in the log)
- Ollama with a model pulled. It was tested with `qwen3:4b` (`ollama pull qwen3:4b`); the default name is `mistral:7b`
- Optional: an mcp-memory-service SQLite database to scan, and its HTTP API to store a one-line summary of each run

### Settings

The engine reads only its environment; it does not load a `.env` file itself. Export the settings, or keep them in a file under `data/` and source it (`set -a; . ./data/local.env; set +a`).

| Variable | Default | Description |
|---|---|---|
| `CURIOSITY_LLM` | `claude` | `ollama` to use a local model. Each item also makes one model call for its [digest](#run), so on the default Claude CLI backend that is one extra paid call per item |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server |
| `CURIOSITY_OLLAMA_MODEL` | `mistral:7b` | Model name. Reasoning models (`qwen3`, `deepseek-r1`) are sent `think: false` |
| `CURIOSITY_OLLAMA_NUM_CTX` | `2048` | Context size. Match any other program that uses the same model, or Ollama reloads it for each |
| `CLAUDE_CLI` | `claude` | Set to `/bin/false` to be sure nothing calls the Claude CLI |
| `CURIOSITY_MEMORY_DB` | *(off)* | Path to an mcp-memory-service SQLite database. The memory scan opens it read-only |
| `CURIOSITY_SELF_AGENT_TAG` | *(off)* | The tag that marks the scanning agent's own memories, e.g. `agent:me`. A memory tagged for any other agent is never read; unset, only memories without an `agent:` tag are |
| `CURIOSITY_PRIORITY_TAGS` | *(none)* | Comma-separated tags whose memories the scan reads first |
| `CURIOSITY_REFLECTIONS_DIR` | *(off)* | A directory of markdown reflections: one open question from each of the newest files, up to 4 per scan |
| `CURIOSITY_MEMORY_API` + `MCP_API_KEY` | *(off)* | mcp-memory-service HTTP API (Bearer key) for a one-line summary of each run, stored as type `research` |
| `CURIOSITY_ITEMS_PER_RUN` | `1` | Items processed by each `run-once` |
| `CURIOSITY_ASKER_FILE` | *(off)* | Who is asking; see [the asker file](#the-asker-file) |
| `CURIOSITY_PRIVATE_TERMS` | `./data/private_terms.txt` | Names that must never reach a public search; see [the private-terms filter](#the-private-terms-filter) |
| `CURIOSITY_STATE`, `CURIOSITY_LOCK` | `./data/state.json`, `./data/run.lock` | When the memory was last scanned; the lock that keeps two runs from overlapping |
| `S2_API_KEY` or `S2_API_KEY_FILE` | *(off)* | Semantic Scholar API key, or a file holding it. The anonymous pool is shared and often answers 429 |
| `CURIOSITY_EMBED_MODEL`, `CURIOSITY_EMBED_DEVICE` | `all-mpnet-base-v2`, `cpu` | Embedding model and device. The thresholds and the asker check were tuned on the default model |
| `CURIOSITY_GRAPH_URL` | *(off)* | A HippoRAG-style service (`POST /index`) to index the papers found |
| `TGS_RAG_URL`; `WIKI_URL`, `WIKI_USER`, `WIKI_PASS` | *(off)* | Internal knowledge-base search (`POST /retrieve`) and wiki search (`POST /api/query`, basic auth). Their results go into the reports only, never into a prompt or a public query |

### Run

```bash
export CURIOSITY_LLM=ollama CURIOSITY_OLLAMA_MODEL=qwen3:4b CLAUDE_CLI=/bin/false
export CURIOSITY_MEMORY_DB=/path/to/memory.db CURIOSITY_SELF_AGENT_TAG=agent:me
export CURIOSITY_ASKER_FILE=./data/asker.md
python curiosity_engine.py run-once
```

Each `run-once` takes in the idea file and the inbox, scans memory (at most every 6 hours; the time is kept in `CURIOSITY_STATE`, so it holds across cron runs), puts items left `in_progress` by a run that died (over 6 hours ago) back in the queue, and then processes up to `CURIOSITY_ITEMS_PER_RUN` items. It prints `Completed: <topic>` for each item done. From cron:

```
57 */4 * * * cd /path/to/curiosity-engine && set -a && . ./data/local.env && set +a && python curiosity_engine.py run-once >> output/cron.log 2>&1
```

Besides the report in `output/<date>/`, each item gets a compact `.dno` digest beside it and an entry in `output/digest/<date>.md`: the papers found, and the local model's reading of the abstracts, labelled unverified. The reading is one more model call per
item, on whichever backend `CURIOSITY_LLM` selects.

**PubMed currently finds almost nothing.** In one deployment's runs (32 items, 2026-10-05 to 10-07) it returned no
results for any of them. A likely cause, not yet investigated: PubMed requires every word of the term to match, and the
keyword queries run to 12 words. Read its silence as "not searched effectively", not as "no literature". Shorter
PubMed-only queries are a planned follow-up.

### The asker file

Topics written by a small model drift: it keeps a word and loses its sense. The asker file says who is asking and what their words mean. It is markdown: prose first, then up to three lists.

```markdown
# Who is asking

A research assistant agent that keeps notes on its own retrieval system.
- "memory" means a store of past conversations, not hardware.

## Fields

- retrieval-augmented generation and long-term memory for language-model agents
- evaluation of question answering over long documents

## Not these

- computer memory hardware, DRAM and caches
- human memory and neuroscience

## Requesters

- research-agent
```

A thread is the asker's when its first item came from the memory scan, or from a requester listed under `## Requesters`. In the asker's threads:

- the prose and the fields go into the prompts that write topics, with the question the thread began with;
- a topic the engine writes is dropped (at the scan or the follow-up), or marked `skipped` when it comes up in the queue, if its embedding is closer to a "Not these" line than to any field. A skipped item stays in the queue with its reason; set it back to `pending` to research it anyway.

A topic somebody queued (`add`, requester `user`) is never gated, and neither is anything in a thread somebody else began. Without the file nothing changes.

### The private-terms filter

Every topic an LLM writes, from the memory scan or as a follow-up, is checked before it can be sent to arXiv, Semantic Scholar, OpenAlex or PubMed. It is refused if it holds a term from the private-terms file (whole words, any case, any spacing), an `@`, a URL or a `.com`/`.org`/`.net`/`.io`/`.tech` domain, a home or system path (`~/`, `/home/`, `/opt/`, `/etc/`), a port (`:8080`), an IPv4 address or a credential-shaped string, or if it is shorter than 15 or longer than 150 characters. Topics from the memory scan fail closed: with the file missing or empty, none is queued.

The file has one name or phrase per line, with `#` comments:

```
# people
Jane Doe
# machines, accounts and projects
build-server-01
Project Nightingale
```

List every person, machine, account and project name that must never appear in a public query, and add new ones as they come. Memory fragments and reflection sentences that look like credentials or IP addresses are dropped before they reach a prompt at all, and internal search results never enter one.

### Keep `data/` local

`data/` and `output/` are in `.gitignore`. The queue, the scan state and the lock live there by default; keep the asker file, the private-terms file and your settings there too. They describe the people and machines around you, and they do not belong in a repository.

### Tests

`python test_port.py` runs the tests for this mode without pytest. They write only to a temporary directory, override any deployment settings in the environment, and stub the network calls. One test needs the embedding model and is skipped with a note when it is not available; set `HF_HUB_OFFLINE=1` to keep sentence-transformers from contacting the Hugging Face Hub.

## How It Works

```
                    +------------------+
                    |   Input Sources  |
                    |  CLI / CAOGL /   |
                    |  Idea File /     |
                    |  Memory Scan     |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  Research Queue   |
                    |  (JSON file)     |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  Search Phase    |
                    |  arXiv, S2,      |
                    |  OpenAlex, PubMed|
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  Deduplication   |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  Report Gen      |
                    |  (Markdown)      |
                    +--------+---------+
                             |
                     +-------+-------+
                     |               |
                     v               v
              +-----------+   +-----------+
              |  Memory   |   | Curiosity |
              |  Storage  |   |  Engine   |
              | (optional)|   | Follow-up |
              +-----------+   +-----+-----+
                                    |
                                    v
                             +-------------+
                             | Novelty     |
                             | Filter      |
                             +------+------+
                                    |
                                    v
                             Back to Queue
                             (depth + 1)
```

## Credits

Built by **Lyra @ Liberation Labs** with [Claude Code](https://claude.ai).

## License

MIT -- see [LICENSE](LICENSE).
