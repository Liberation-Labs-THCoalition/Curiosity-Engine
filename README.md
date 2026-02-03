# Curiosity Engine

**An autonomous research pipeline that generates its own follow-up questions.**

Curiosity Engine is a queue-based research system that searches four academic databases (arXiv, Semantic Scholar, OpenAlex, PubMed), generates markdown reports, and uses LLM-assisted follow-up question generation to build branching research trees. It runs as a daemon or one-shot CLI tool, consuming topics from a shared JSON queue and producing structured research reports.

## Features

- **Multi-database search** -- queries arXiv, Semantic Scholar, OpenAlex, and PubMed in parallel with automatic rate-limit handling
- **Automatic deduplication** -- normalized title matching removes duplicate papers across sources
- **Curiosity-driven follow-ups** -- uses an LLM (Claude CLI) to generate follow-up research questions from gaps in results
- **Depth-capped branching** -- configurable maximum depth prevents infinite recursion in research trees
- **Semantic novelty detection** -- sentence-transformer embeddings detect when a follow-up is too similar to existing queue items (optional, requires `sentence-transformers`)
- **Interest decay** -- threads that return too few papers are pruned automatically
- **Memory scanning** -- queries a PostgreSQL memory store for unresolved thoughts and open questions, then generates new research topics (optional, requires `psycopg2`)
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
# Edit .env with your settings

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

All configuration is via environment variables. Set them in your shell or in a `.env` file.

| Variable | Default | Description |
|---|---|---|
| `CURIOSITY_QUEUE` | `./data/research_queue.json` | Path to the JSON research queue |
| `CURIOSITY_OUTPUT` | `./output` | Directory for generated reports |
| `CURIOSITY_LOG` | `./output/runner.log` | Log file path |
| `CURIOSITY_INBOX` | `./data/research_inbox.json` | CAOGL message bus inbox file |
| `CURIOSITY_IDEAS` | *(disabled)* | Path to a markdown idea file |
| `CURIOSITY_PID` | `./data/runner.pid` | PID file for daemon mode |
| `DATABASE_URL` | *(disabled)* | PostgreSQL connection string for memory storage |
| `CLAUDE_CLI` | `claude` | Path to Claude CLI binary |
| `CLAUDE_MODEL` | `haiku` | Claude model for follow-up generation |
| `OPENALEX_EMAIL` | *(empty)* | Email for OpenAlex polite pool (faster rate limits) |

### Hardcoded Defaults (edit in source)

| Parameter | Value | Description |
|---|---|---|
| `POLL_INTERVAL` | 300s | Daemon polling interval |
| `MAX_PAPERS_PER_DB` | 10 | Maximum papers to fetch per database |
| `MAX_FOLLOWUPS_PER_ITEM` | 2 | Follow-up questions generated per topic |
| `MAX_DEPTH` | 3 | Maximum branching depth |
| `NOVELTY_THRESHOLD` | 0.75 | Cosine similarity cutoff for deduplication |
| `MIN_QUALITY_SCORE` | 3 | Minimum papers to continue a branch |
| `MEMORY_SCAN_INTERVAL` | 6 hours | Time between memory scans |

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
