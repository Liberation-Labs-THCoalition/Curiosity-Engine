#!/usr/bin/env python3
"""
CURIOSITY ENGINE
================
An autonomous research pipeline that generates its own follow-up questions.

Searches across arXiv, Semantic Scholar, OpenAlex, and PubMed, writes
markdown reports, and uses LLM-assisted follow-up question generation
to build branching research trees with depth caps, novelty detection,
and interest decay.

Queue-based architecture supports multiple input sources: CLI, message
bus (CAOGL), desktop idea files, and memory-derived threads.

Author: Lyra @ Liberation Labs
Version: 2.0.0
License: MIT
"""

import json, os, sys, time, uuid, logging, re, hashlib, subprocess, sqlite3, fcntl, difflib, random, collections
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import requests
import xml.etree.ElementTree as ET

# ==================== OPTIONAL IMPORTS ====================

_HAS_SENTENCE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    pass

_HAS_PSYCOPG2 = False
try:
    import psycopg2
    _HAS_PSYCOPG2 = True
except ImportError:
    pass


# ==================== CONFIGURATION ====================

class Config:
    QUEUE_FILE = os.environ.get("CURIOSITY_QUEUE", "./data/research_queue.json")
    OUTPUT_DIR = os.environ.get("CURIOSITY_OUTPUT", "./output")
    LOG_FILE = os.environ.get("CURIOSITY_LOG", "./output/runner.log")
    CAOGL_SOCKET = os.environ.get("CURIOSITY_INBOX", "./data/research_inbox.json")
    IDEA_FILE = os.environ.get("CURIOSITY_IDEAS", "")
    POSTGRES_URL = os.environ.get("DATABASE_URL", "")
    CLAUDE_CLI = os.environ.get("CLAUDE_CLI", "claude")
    CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "haiku")
    LLM_BACKEND = os.environ.get("CURIOSITY_LLM", "claude")  # "claude" or "ollama"
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    OLLAMA_MODEL = os.environ.get("CURIOSITY_OLLAMA_MODEL", "mistral:7b")
    PID_FILE = os.environ.get("CURIOSITY_PID", "./data/runner.pid")
    OPENALEX_EMAIL = os.environ.get("OPENALEX_EMAIL", "")

    # Nexus port (2026-10-05): a SQLite memory store (mcp-memory-service schema) as well as PostgreSQL, storage through
    # the memory service's HTTP API, graph ingestion opt-in, and a run-once that seeds and recovers its own queue.
    MEMORY_DB = os.environ.get("CURIOSITY_MEMORY_DB", "")
    MEMORY_API_URL = os.environ.get("CURIOSITY_MEMORY_API", "")
    MEMORY_API_KEY = os.environ.get("MCP_API_KEY", "")
    GRAPH_URL = os.environ.get("CURIOSITY_GRAPH_URL", "")  # opt-in: see _ingest_to_graph
    STATE_FILE = os.environ.get("CURIOSITY_STATE", "./data/state.json")
    LOCK_FILE = os.environ.get("CURIOSITY_LOCK", "./data/run.lock")
    PRIVATE_TERMS_FILE = os.environ.get("CURIOSITY_PRIVATE_TERMS", "./data/private_terms.txt")
    ITEMS_PER_RUN = int(os.environ.get("CURIOSITY_ITEMS_PER_RUN", "1"))
    STUCK_AFTER_S = 6 * 3600  # an in_progress item older than this died with its run
    MAX_ATTEMPTS = 2
    REFLECTIONS_DIR = os.environ.get("CURIOSITY_REFLECTIONS_DIR", "")  # written reflections, a second seed source
    S2_API_KEY_FILE = os.environ.get("S2_API_KEY_FILE", "")  # a file holding the key; S2_API_KEY wins if both are set
    ASKER_FILE = os.environ.get("CURIOSITY_ASKER_FILE", "")  # who is asking, in the asker's own threads: see _asker()

    @classmethod
    def use_ollama(cls):
        return cls.LLM_BACKEND.lower() == "ollama"

    POLL_INTERVAL = 300  # 5 minutes
    MAX_PAPERS_PER_DB = 10
    REQUEST_TIMEOUT = 30

    # Curiosity engine
    ENABLE_FOLLOWUPS = True
    FOLLOWUP_PRIORITY = 4  # Lower priority than manual requests
    MAX_FOLLOWUPS_PER_ITEM = 2
    MAX_DEPTH = 2  # Default branch cap (Lyra: exponential branching eats tokens)
    MAX_DEPTH_NOVEL = 3  # Extended depth if thread finds genuinely novel papers
    NOVELTY_THRESHOLD = 0.75  # Cosine similarity cutoff — above this, skip as duplicate
    RELEVANCE_THRESHOLD = 0.3  # Min similarity between paper abstract and query to include
    MIN_QUALITY_SCORE = 3  # Minimum papers to consider a thread worth following

    # Memory scanning for unresolved threads
    ENABLE_MEMORY_SCAN = True
    MEMORY_SCAN_INTERVAL = 3600 * 6  # Every 6 hours
    MEMORY_SCAN_LIMIT = 5  # Max topics to queue per scan
    MEMORY_SCAN_DAYS = 90  # SQLite scan window

    # API endpoints
    ARXIV_API = "http://export.arxiv.org/api/query"
    S2_API = "https://api.semanticscholar.org/graph/v1"
    OPENALEX_API = "https://api.openalex.org"
    PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


# ==================== LOGGING ====================

def setup_logging():
    os.makedirs(os.path.dirname(Config.LOG_FILE) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        # stdout only at a terminal: cron appends stdout to this same file, which wrote every line twice
        handlers=[logging.FileHandler(Config.LOG_FILE, encoding='utf-8')]
                 + ([logging.StreamHandler(sys.stdout)] if sys.stdout.isatty() else [])
    )
    return logging.getLogger(__name__)

log = setup_logging()
# sentence-transformers draws a progress bar for every encode() whenever its logger is at INFO, which basicConfig above
# makes it; under cron each bar landed in runner.log (Nexus 2026-10-05). TQDM_DISABLE did not stop it.
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

if not _HAS_SENTENCE_TRANSFORMERS:
    log.warning("sentence-transformers not installed — semantic novelty detection disabled")
if Config.POSTGRES_URL and not _HAS_PSYCOPG2:
    log.warning("psycopg2 not installed — PostgreSQL memory storage disabled")
if not Config.POSTGRES_URL and not Config.MEMORY_DB:
    log.info("Neither DATABASE_URL nor CURIOSITY_MEMORY_DB is set — memory scanning disabled")


# ==================== QUEUE MANAGEMENT ====================

def load_queue() -> List[Dict]:
    if not os.path.exists(Config.QUEUE_FILE):
        return []
    with open(Config.QUEUE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_queue(queue: List[Dict]):
    os.makedirs(os.path.dirname(Config.QUEUE_FILE) or ".", exist_ok=True)
    with open(Config.QUEUE_FILE, 'w', encoding='utf-8') as f:
        json.dump(queue, f, indent=2, default=str)

def add_to_queue(topic: str, priority: int = 3, requester: str = "user",
                 year_range: str = "", databases: List[str] = None,
                 depth: int = 0, parent_id: str = None) -> str:
    queue = load_queue()
    item_id = str(uuid.uuid4())[:8]
    item = {
        "id": item_id,
        "topic": topic,
        "priority": priority,
        "requester": requester,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "search_params": {
            "year_range": year_range or "",
            "max_papers": Config.MAX_PAPERS_PER_DB,
            "databases": databases or ["arxiv", "s2", "openalex", "pubmed"]
        },
        "result_file": None,
        "depth": depth,
        "parent_id": parent_id
    }
    queue.append(item)
    save_queue(queue)
    log.info(f"Queued: [{item_id}] '{topic}' (priority {priority}, by {requester})")
    return item_id


# ==================== INTERNAL KNOWLEDGE BASE ====================

WIKI_URL = os.environ.get("WIKI_URL", "http://localhost:8585")
WIKI_USER = os.environ.get("WIKI_USER", "researcher")
WIKI_PASS = os.environ.get("WIKI_PASS", "")  # no default: a password in public source is published
HIPPORAG_URL = os.environ.get("HIPPORAG_URL", "http://localhost:11235")
TGS_RAG_URL = os.environ.get("TGS_RAG_URL", "http://localhost:11236")


def _search_internal(query: str, max_results: int = 5) -> List[Dict]:
    """Search our own wiki and knowledge graph for existing findings."""
    results = []

    # TGS-RAG (fused text + graph retrieval)
    try:
        resp = requests.post(
            f"{TGS_RAG_URL}/retrieve",
            json={"query": query, "num_results": max_results},
            timeout=10,
        )
        if resp.status_code == 200:
            for r in resp.json().get("results", []):
                content = r.get("content", "")
                if len(content) > 50:
                    results.append({
                        "title": content[:100].strip(),
                        "abstract": content,
                        "source": "internal/tgs-rag",
                        "year": "",
                        "authors": ["Coalition Knowledge Base"],
                        "url": "",
                        "citation_count": 0,
                    })
    except Exception as e:
        log.debug(f"TGS-RAG search failed: {e}")

    # Wiki search, only with credentials from the environment
    if WIKI_PASS:
        try:
            resp = requests.post(
                f"{WIKI_URL}/api/query",
                json={"query": query, "num": max_results},
                auth=(WIKI_USER, WIKI_PASS),
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                for doc in data.get("docs", [])[:max_results]:
                    if isinstance(doc, str) and len(doc) > 50:
                        results.append({
                            "title": doc[:100].strip(),
                            "abstract": doc,
                            "source": "internal/wiki",
                            "year": "",
                            "authors": ["Coalition Lab Wiki"],
                            "url": f"{WIKI_URL}/search?q={query}",
                            "citation_count": 0,
                        })
        except Exception as e:
            log.debug(f"Wiki search failed: {e}")

    return results


# ==================== API CLIENTS ====================

def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    """Search arXiv API directly."""
    try:
        resp = requests.get(Config.ARXIV_API, params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance"
        }, timeout=Config.REQUEST_TIMEOUT)
        resp.raise_for_status()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        papers = []
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")[:500]
            link = entry.find("atom:id", ns).text.strip()
            published = entry.find("atom:published", ns).text[:10]
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
            papers.append({
                "source": "arxiv",
                "title": title,
                "authors": authors[:5],
                "year": published[:4],
                "abstract": summary,
                "url": link,
                "citations": None
            })
        return papers
    except Exception as e:
        log.warning(f"arXiv search failed: {e}")
        return []

def _s2_key() -> str:
    key = os.environ.get("S2_API_KEY", "")
    if not key and Config.S2_API_KEY_FILE:
        try:
            with open(Config.S2_API_KEY_FILE, encoding="utf-8") as f:
                key = f.read().strip()
        except OSError:
            key = ""
    return key


def search_s2(query: str, max_results: int = 10) -> List[Dict]:
    """Search Semantic Scholar, with an API key when one is configured (x-api-key; introductory limit 1 request/s).

    A 429 backs off exponentially, honouring Retry-After, then gives up after the third attempt. The anonymous pool is
    shared by everyone and refused every call on 2026-10-05; the old fixed 10/20/30 s waits cost a minute per topic."""
    key = _s2_key()
    headers = {"x-api-key": key} if key else {}
    for attempt in range(3):
        try:
            resp = requests.get(f"{Config.S2_API}/paper/search", params={
                "query": query,
                "limit": max_results,
                "fields": "title,authors,year,abstract,citationCount,url"
            }, headers=headers, timeout=Config.REQUEST_TIMEOUT)
            if resp.status_code == 429:
                if attempt == 2:
                    log.warning("  S2 rate limited on 3 attempts; skipping it for this topic")
                    return []
                try:
                    wait = float(resp.headers.get("Retry-After") or 0)
                except ValueError:
                    wait = 0
                wait = min(wait or 2 ** (attempt + 1), 60) + random.uniform(0, 1)
                log.info(f"  S2 rate limited, waiting {wait:.1f}s (attempt {attempt+1}/3)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()

            papers = []
            for p in data.get("data", []):
                papers.append({
                    "source": "s2",
                    "title": p.get("title", ""),
                    "authors": [a.get("name", "") for a in (p.get("authors") or [])[:5]],
                    "year": str(p.get("year", "")),
                    "abstract": (p.get("abstract") or "")[:500],
                    "url": p.get("url", ""),
                    "citations": p.get("citationCount")
                })
            return papers
        except Exception as e:
            if attempt == 2:
                log.warning(f"S2 search failed after retries: {type(e).__name__}")
            continue
    return []

def search_openalex(query: str, max_results: int = 10) -> List[Dict]:
    """Search OpenAlex API."""
    try:
        params = {
            "search": query,
            "per_page": max_results,
        }
        if Config.OPENALEX_EMAIL:
            params["mailto"] = Config.OPENALEX_EMAIL
        resp = requests.get(f"{Config.OPENALEX_API}/works", params=params,
                          timeout=Config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        papers = []
        for w in data.get("results", []):
            title = w.get("title", "")
            authors = [a.get("author", {}).get("display_name", "")
                      for a in (w.get("authorships") or [])[:5]]
            year = str(w.get("publication_year", ""))
            # OpenAlex abstract is inverted index — reconstruct
            abstract = ""
            inv = w.get("abstract_inverted_index")
            if inv:
                word_positions = []
                for word, positions in inv.items():
                    for pos in positions:
                        word_positions.append((pos, word))
                word_positions.sort()
                abstract = " ".join(w for _, w in word_positions)[:500]
            papers.append({
                "source": "openalex",
                "title": title,
                "authors": authors,
                "year": year,
                "abstract": abstract,
                "url": w.get("doi", "") or w.get("id", ""),
                "citations": w.get("cited_by_count")
            })
        return papers
    except Exception as e:
        log.warning(f"OpenAlex search failed: {e}")
        return []

def search_pubmed(query: str, max_results: int = 10) -> List[Dict]:
    """Search PubMed via E-utilities."""
    try:
        # Step 1: search for IDs
        resp = requests.get(Config.PUBMED_SEARCH, params={
            "db": "pubmed", "term": query, "retmax": max_results,
            "retmode": "json", "sort": "relevance"
        }, timeout=Config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        # Step 2: fetch details
        resp2 = requests.get(Config.PUBMED_FETCH, params={
            "db": "pubmed", "id": ",".join(ids),
            "retmode": "xml", "rettype": "abstract"
        }, timeout=Config.REQUEST_TIMEOUT)
        resp2.raise_for_status()

        root = ET.fromstring(resp2.text)
        papers = []
        for article in root.findall(".//PubmedArticle"):
            medline = article.find(".//MedlineCitation")
            art = medline.find(".//Article") if medline is not None else None
            if art is None:
                continue
            title_el = art.find("ArticleTitle")
            title = title_el.text if title_el is not None and title_el.text else ""
            abstract_el = art.find(".//AbstractText")
            abstract = abstract_el.text if abstract_el is not None and abstract_el.text else ""
            year_el = art.find(".//PubDate/Year")
            year = year_el.text if year_el is not None else ""
            authors = []
            for au in art.findall(".//Author")[:5]:
                last = au.find("LastName")
                first = au.find("ForeName")
                if last is not None and last.text:
                    name = last.text
                    if first is not None and first.text:
                        name = f"{first.text} {last.text}"
                    authors.append(name)
            pmid = medline.find("PMID").text if medline.find("PMID") is not None else ""
            papers.append({
                "source": "pubmed",
                "title": title,
                "authors": authors,
                "year": year,
                "abstract": abstract[:500],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                "citations": None
            })
        return papers
    except Exception as e:
        log.warning(f"PubMed search failed: {e}")
        return []


# ==================== DEDUPLICATION ====================

def deduplicate(papers: List[Dict]) -> List[Dict]:
    """Remove duplicate papers by normalized title similarity."""
    seen = {}
    unique = []
    for p in papers:
        key = re.sub(r'[^a-z0-9]', '', p["title"].lower())[:80]
        if key not in seen:
            seen[key] = True
            unique.append(p)
    return unique


def relevance_gate(papers: List[Dict], query: str) -> List[Dict]:
    """Filter papers by cosine similarity between abstract and query.

    Lyra's suggestion: before including a paper, check that its abstract
    is actually relevant to the thread question. Papers below threshold
    get dropped rather than diluting the report.
    """
    model = _get_st_model()
    if model is None or not papers:
        return papers

    try:
        query_emb = model.encode(query)
        scored = []
        for p in papers:
            abstract = p.get("abstract", "")
            if not abstract:
                scored.append((p, 0.5))  # No abstract = benefit of doubt
                continue
            paper_emb = model.encode(abstract[:500])
            dot = sum(a * b for a, b in zip(query_emb, paper_emb))
            norm_a = sum(a * a for a in query_emb) ** 0.5
            norm_b = sum(b * b for b in paper_emb) ** 0.5
            sim = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
            scored.append((p, sim))

        relevant = [(p, s) for p, s in scored if s >= Config.RELEVANCE_THRESHOLD]
        filtered = len(papers) - len(relevant)
        if filtered > 0:
            log.info(f"  Relevance gate: kept {len(relevant)}/{len(papers)} papers (filtered {filtered} below {Config.RELEVANCE_THRESHOLD})")
        return [p for p, s in relevant]
    except Exception as e:
        log.warning(f"Relevance gating failed (keeping all): {e}")
        return papers


_QUERY_STOP = {"a", "an", "the", "of", "in", "on", "for", "and", "or", "to", "is", "are", "was", "were", "be", "been",
               "how", "what", "why", "which", "when", "where", "who", "does", "do", "did", "can", "could", "would",
               "should", "may", "might", "will", "with", "within", "by", "as", "at", "from", "into", "across", "between",
               "during", "that", "this", "these", "those", "it", "its", "their", "there", "we", "our"}


def search_query(topic: str) -> str:
    """Keywords for the search APIs; the topic itself stays a readable question.

    On 2026-10-05 memory-derived topics came out as whole questions: OpenAlex answered both with HTTP 400 and PubMed,
    which ANDs every word, found nothing. Duplicate words (a parent term already in the topic) are dropped."""
    seen, kept = set(), []
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", topic):
        if w.lower() not in _QUERY_STOP and w.lower() not in seen:
            seen.add(w.lower())
            kept.append(w)
    return " ".join(kept[:12]) or topic


def extract_parent_terms(item: Dict) -> List[str]:
    """Extract key terms from parent topic for search narrowing at depth.

    Lyra's suggestion: child queries should inherit mandatory terms from
    the parent thread to prevent search dilution at depth.
    """
    parent_id = item.get("parent_id")
    if not parent_id:
        return []

    queue = load_queue()
    parent = next((q for q in queue if q["id"] == parent_id), None)
    if not parent:
        return []

    parent_topic = parent.get("topic", "")
    stop_words = {"the", "a", "an", "in", "on", "of", "for", "and", "or", "to", "is",
                  "are", "was", "were", "how", "what", "why", "does", "do", "can", "with"}
    terms = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', parent_topic)
             if w.lower() not in stop_words]
    # Keep the most distinctive terms (longer words, likely technical)
    terms.sort(key=len, reverse=True)
    return terms[:3]


# ==================== REPORT GENERATION ====================

def generate_report(item: Dict, papers: List[Dict]) -> str:
    """Generate a markdown research report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    depth = item.get('depth', 0)
    parent = item.get('parent_id')
    lineage = f"**Depth**: {depth}" + (f" | **Parent**: {parent}" if parent else " | **Origin**: manual") + "  "

    lines = [
        f"# Research Report: {item['topic']}",
        f"",
        f"**Generated**: {now}  ",
        f"**Requested by**: {item['requester']}  ",
        f"**Priority**: {item['priority']}  ",
        lineage,
        f"**Databases searched**: {', '.join(item['search_params']['databases'])}  ",
        f"**Papers found**: {len(papers)}",
        f"",
        f"---",
        f"",
    ]

    if not papers:
        lines.append("No papers found for this query.")
        return "\n".join(lines)

    # Sort by citations (descending), nulls last
    papers.sort(key=lambda p: (p.get("citations") or p.get("citation_count") or 0), reverse=True)

    for i, p in enumerate(papers, 1):
        cites = f" | Cited by: {p.get('citations', p.get('citation_count', 0))}" if p.get('citations', p.get('citation_count')) else ""
        lines.extend([
            f"## {i}. {p['title']}",
            f"**Authors**: {', '.join(p['authors'])}  ",
            f"**Year**: {p['year']} | **Source**: {p['source']}{cites}  ",
            f"**URL**: {p['url']}",
            f"",
            f"> {p['abstract'][:300]}{'...' if len(p['abstract']) > 300 else ''}",
            f"",
        ])

    # A list of internal results only (no year, `citation_count` instead of `citations`) crashed both lines below, and
    # a crash left the item in_progress for good (Nexus 2026-10-05: 22 such items, none with a report).
    years = [p['year'] for p in papers if p.get('year')]
    top_cites = papers[0].get('citations') or papers[0].get('citation_count') or 'N/A'
    lines.extend([
        "---",
        "",
        "## Summary Statistics",
        f"- Total unique papers: {len(papers)}",
        f"- Year range: {min(years)} - {max(years)}" if years else "- Year range: n/a",
        f"- Sources: {', '.join(set(p['source'] for p in papers))}",
        f"- Most cited: {papers[0]['title']} ({top_cites} citations)",
    ])

    return "\n".join(lines)


def generate_digest(item: Dict, papers: List[Dict]) -> str:
    """Token-optimized DNO digest for agent consumption."""
    depth = item.get('depth', 0)
    lines = [
        f"CE|{item['id']}|d={depth}|n={len(papers)}|{item['topic'][:80]}",
        "---",
    ]
    papers.sort(key=lambda p: (p.get("citations") or p.get("citation_count") or 0), reverse=True)
    for p in papers:
        src = p.get("source", "?")
        yr = p.get("year", "?")
        cites = p.get("citations") or p.get("citation_count") or 0
        title = p.get("title", "?")[:100]
        if src in ("internal/tgs-rag", "internal/wiki"):
            lines.append(f"INT|{title}")
        else:
            lines.append(f"{src}|{yr}|c={cites}|{title}")
    return "\n".join(lines) + "\n"


DIGEST_SCHEMA = {"type": "object", "required": ["finding", "open"],
                 "properties": {"finding": {"type": "string"}, "open": {"type": "string"}}}


def append_daily_digest(item: Dict, papers: List[Dict], report_path: str):
    """One short entry per explored question in output/digest/<date>.md (Nexus 2026-10-05).

    The reports piled up where nobody read them; curiosity that never comes back to the one who is curious is a
    crawler, so point the asker at this file (an agent's session-start hook, say). The finding is the local model's
    reading of the abstracts, labelled as such and kept out of memory: a small model's paraphrase is not something the
    asker learned."""
    external = [p for p in papers if not str(p.get("source", "")).startswith("internal/")]
    if not external:
        return
    abstracts = "\n".join(f"{i}. {p['title']} ({p.get('year') or 'n.d.'}): {(p.get('abstract') or '')[:400]}"
                          for i, p in enumerate(external[:5], 1))
    prompt = f"""Research question: "{item['topic']}"

Abstracts of the papers found:
{abstracts}

Using ONLY these abstracts: in at most two sentences, what do they say about the question? If they do not address it, say so plainly. Then, in one sentence, what remains open.
Reply with JSON only, in the form {{"finding": "...", "open": "..."}}."""
    finding = still_open = ""
    raw = ask_llm(prompt, max_tokens=300, schema=DIGEST_SCHEMA)
    if raw:
        try:
            obj = json.loads(raw[raw.index('{'):raw.rindex('}') + 1])
            finding, still_open = str(obj.get("finding", "")).strip(), str(obj.get("open", "")).strip()
        except ValueError:
            pass
    sources = ", ".join(f"{src} {n}" for src, n in sorted(collections.Counter(p['source'] for p in external).items()))
    lines = [f"## {datetime.now().strftime('%H:%M')} · {item['topic']}",
             f"*{len(external)} papers ({sources}) · depth {item.get('depth', 0)} · from {item.get('requester')} · "
             f"report: `{report_path}`*", ""]
    if finding:
        lines.append(f"- **Finding** (the local model's reading of the abstracts, unverified): {finding}")
    if still_open:
        lines.append(f"- **Still open:** {still_open}")
    for p in external[:3]:
        lines.append(f"- {p['title']} ({p.get('year') or 'n.d.'}, {p['source']}) {p.get('url') or ''}".rstrip())
    day = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(Config.OUTPUT_DIR, "digest", f"{day}.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(f"# Curiosity digest, {day}\n\n")
        f.write("\n".join(lines) + "\n\n")


# ==================== MEMORY STORAGE ====================

def _ingest_to_graph(papers: List[Dict]):
    """Index discovered papers into the federated knowledge graph (HippoRAG)."""
    # Opt-in only (Nexus 2026-10-05). This used to default to a local port that once served a shared graph; in the
    # deployment it was written for, that port became a private memory store, and every paper indexed costs an LLM
    # extraction.
    hipporag_url = Config.GRAPH_URL
    if not hipporag_url:
        return

    # Filter to external papers only (skip internal/wiki results)
    external = [p for p in papers
                if p.get("source", "") not in ("internal/tgs-rag", "internal/wiki", "")]
    if not external:
        return

    docs = []
    for p in external:
        parts = [f"Research paper: {p.get('title', 'Untitled')}"]
        authors = p.get("authors", [])
        if authors:
            if isinstance(authors[0], dict):
                names = [a.get("name", "") for a in authors[:5]]
            else:
                names = [str(a) for a in authors[:5]]
            parts.append(f"Authors: {', '.join(names)}")
        if p.get("year"):
            parts.append(f"Year: {p['year']}")
        if p.get("venue"):
            parts.append(f"Venue: {p['venue']}")
        if p.get("citations"):
            parts.append(f"Citations: {p['citations']}")
        url = p.get("url", p.get("arxiv_url", ""))
        if url:
            parts.append(f"URL: {url}")
        if p.get("abstract"):
            parts.append(f"Abstract: {p['abstract'][:1000]}")
        parts.append("Source: curiosity-engine")
        docs.append("\n".join(parts))

    if not docs:
        return

    try:
        resp = requests.post(
            f"{hipporag_url}/index",
            json={"docs": docs},
            timeout=120,
        )
        if resp.status_code == 200:
            count = resp.json().get("count", 0)
            if count:
                log.info(f"  Indexed {count} papers into federated graph")
        else:
            log.warning(f"  Graph indexing failed: HTTP {resp.status_code}")
    except Exception as e:
        log.debug(f"  Graph indexing skipped: {e}")


def store_in_memory(item: Dict, papers: List[Dict]):
    """Store a one-line research summary: PostgreSQL (if configured) or the memory service's HTTP API.

    Built from external papers only. An internal result's title is a slice of a memory, and storing it again as
    "research" would copy a memory into a new one (Nexus 2026-10-05)."""
    external = [p for p in papers if not str(p.get("source", "")).startswith("internal/")]
    if not external:
        return  # "found 0 papers" is noise in a memory store (one was stored on the first live run, 2026-10-05)
    summary = f"Research on '{item['topic']}': found {len(external)} papers. "
    if external:
        top = external[0]
        summary += f"Top result: '{top['title']}' ({top['year']}). "
        summary += f"Sources: {', '.join(sorted(set(p['source'] for p in external)))}."
    if Config.POSTGRES_URL and _HAS_PSYCOPG2 and _HAS_SENTENCE_TRANSFORMERS:
        _store_postgres(item, summary)
    elif Config.MEMORY_API_URL and Config.MEMORY_API_KEY:
        _store_api(item, summary)


def _store_api(item: Dict, summary: str):
    """POST the summary to mcp-memory-service (/api/memories, Bearer key), as the service's other HTTP clients do."""
    try:
        resp = requests.post(
            f"{Config.MEMORY_API_URL.rstrip('/')}/api/memories",
            json={
                "content": summary,
                "tags": ["curiosity-engine", "auto_research"],
                "memory_type": "research",
                "metadata": {"research_id": item["id"], "requester": item["requester"],
                             "depth": item.get("depth", 0), "source": "curiosity-engine"},
            },
            headers={"Authorization": f"Bearer {Config.MEMORY_API_KEY}"},
            timeout=15,
        )
        if resp.status_code in (200, 201):
            log.info("Stored research summary in memory")
        else:
            log.warning(f"Memory storage failed (non-fatal): HTTP {resp.status_code}")
    except Exception as e:
        log.warning(f"Memory storage failed (non-fatal): {type(e).__name__}")


def _store_postgres(item: Dict, summary: str):
    try:
        model = _get_st_model()
        embedding = model.encode(summary).tolist()

        conn = psycopg2.connect(Config.POSTGRES_URL)
        cur = conn.cursor()
        content_hash = hashlib.sha256(summary.encode()).hexdigest()
        tags = ["auto_research", item["topic"].split()[0]]
        cur.execute("""
            INSERT INTO lyra_memories (content_hash, content, tags, metadata, embedding, significance, created_at, memory_type, consciousness_id)
            VALUES (%s, %s, %s::text[], %s::jsonb, %s, %s, NOW(), %s, %s)
            ON CONFLICT (content_hash) DO NOTHING
        """, (
            content_hash, summary,
            tags,
            json.dumps({"research_id": item["id"], "requester": item["requester"]}),
            embedding, 4, "research", "lyra_consciousness"
        ))
        conn.commit()
        cur.close()
        conn.close()
        log.info(f"Stored research summary in memory")
    except Exception as e:
        log.warning(f"Memory storage failed (non-fatal): {e}")


# ==================== CAOGL INBOX ====================

def check_caogl_inbox():
    """Check for research requests from other agents via CAOGL inbox file."""
    inbox = Config.CAOGL_SOCKET
    if not os.path.exists(inbox):
        return
    try:
        with open(inbox, 'r', encoding='utf-8') as f:
            messages = json.load(f)
        new_messages = [m for m in messages if not m.get("processed")]
        for msg in new_messages:
            content = msg.get("content", "")
            match = re.match(r'research:\s*(.+?)(?:\s*\[priority:(\d)\])?\s*$', content, re.I)
            if match:
                topic = match.group(1).strip()
                priority = int(match.group(2)) if match.group(2) else 3
                requester = msg.get("from", "unknown")
                add_to_queue(topic, priority, requester)
                log.info(f"Queued from CAOGL: '{topic}' by {requester}")
            msg["processed"] = True
        with open(inbox, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2)
    except Exception as e:
        log.warning(f"CAOGL inbox check failed: {e}")


# ==================== DESKTOP IDEA FILE ====================

def check_idea_file():
    """Parse a desktop idea file for new research topics."""
    path = Config.IDEA_FILE
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find the Ideas section
        match = re.search(r'## Ideas\s*\n(.*?)(?:\n---|\Z)', content, re.DOTALL)
        if not match:
            return

        ideas_block = match.group(1)
        lines = [l.strip() for l in ideas_block.strip().split('\n') if l.strip()]

        # Already-queued topics (check queue to avoid duplicates)
        queue = load_queue()
        existing = {item["topic"].lower() for item in queue}

        new_ideas = []
        for line in lines:
            # Skip lines that are just dashes, blank, or marked done
            if line.startswith('-') and line.strip('- ') == '':
                continue
            if line.startswith('~~'):
                continue
            # Strip leading bullet
            text = re.sub(r'^[-*]\s*', '', line).strip()
            if not text:
                continue

            # Parse optional priority tag
            pri_match = re.search(r'\[priority:(\d)\]', text)
            priority = int(pri_match.group(1)) if pri_match else 3
            topic = re.sub(r'\s*\[priority:\d\]\s*', '', text).strip()

            if topic.lower() not in existing:
                add_to_queue(topic, priority, "idea_file")
                new_ideas.append(topic)
                existing.add(topic.lower())

        # Strike through queued ideas in the file so they don't get re-added
        if new_ideas:
            for idea in new_ideas:
                # Escape for regex
                escaped = re.escape(idea)
                content = re.sub(
                    rf'^([-*]\s*)?({escaped}.*?)$',
                    r'~~\2~~ (queued)',
                    content,
                    flags=re.MULTILINE | re.IGNORECASE
                )
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            log.info(f"Picked up {len(new_ideas)} ideas from desktop file")

    except Exception as e:
        log.warning(f"Idea file check failed: {e}")


# ==================== CURIOSITY ENGINE ====================

def ask_ollama(prompt: str, max_tokens: int = 300, schema: Optional[Dict] = None) -> Optional[str]:
    """Send a prompt to Ollama and return the response text.

    With a schema, `format` constrains generation to it. Asking politely for a format is not enough for a 4B model:
    on 2026-10-05 qwen3:4b spent its whole budget on "We are given..." and wrote no topic at all. num_ctx 2048 matches
    another local consumer of the same model, which measured 4096 at 291 MiB more VRAM on a shared 4 GB card
    (evictions, CPU fallback), and a second consumer asking a different num_ctx of the same model makes Ollama
    reload it."""
    body = {
        'model': Config.OLLAMA_MODEL,
        'prompt': prompt,
        'stream': False,
        'options': {'num_predict': max_tokens, 'temperature': 0.7, 'num_ctx': 2048},
    }
    if schema is not None:
        body['format'] = schema
    # Reasoning models put their output in a separate `thinking` field unless think=False is passed as an API field.
    # With only a prompt directive the whole budget went to thinking and `response` came back empty (seen in another
    # local consumer, 2026-09), so it is sent for these models.
    if any(k in Config.OLLAMA_MODEL for k in ('qwen3', 'deepseek-r1')):
        body['think'] = False
    try:
        resp = requests.post(f'{Config.OLLAMA_URL}/api/generate', json=body, timeout=180)
        if resp.status_code == 200:
            text = resp.json().get('response', '')
            return re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
        log.warning(f"Ollama HTTP {resp.status_code}")
    except Exception as e:
        log.warning(f"Ollama error: {e}")
    return None


def ask_claude(prompt: str) -> Optional[str]:
    """Send a prompt to Claude CLI and return the response text."""
    try:
        result = subprocess.run(
            [Config.CLAUDE_CLI, "-p", "--model", Config.CLAUDE_MODEL,
             "--no-session-persistence", "--tools", ""],
            input=prompt, capture_output=True, timeout=60,
            encoding='utf-8', errors='replace'
        )
        if result.returncode != 0:
            log.warning(f"Claude CLI failed: {result.stderr[:200]}")
            return None
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        log.warning("Claude CLI timed out")
        return None
    except FileNotFoundError:
        log.warning("Claude CLI not found")
        return None
    except Exception as e:
        log.warning(f"Claude CLI failed: {e}")
        return None


def ask_llm(prompt: str, max_tokens: int = 300, schema: Optional[Dict] = None) -> Optional[str]:
    """Route prompt to configured LLM backend. The schema binds Ollama only; the prompt asks every backend for it."""
    if Config.use_ollama():
        return ask_ollama(prompt, max_tokens, schema)
    return ask_claude(prompt)


def generate_followups(item: Dict, papers: List[Dict]) -> List[str]:
    """Use LLM to generate follow-up research questions."""
    if not Config.ENABLE_FOLLOWUPS or not papers:
        return []

    # Build context from the top 5 EXTERNAL papers. Internal results are memory and wiki text, and whatever enters this
    # prompt can come back out inside a query sent to public APIs (Nexus 2026-10-05).
    external = [p for p in papers if not str(p.get("source", "")).startswith("internal/")]
    if not external:
        return []
    paper_summaries = []
    for p in external[:5]:
        cite_str = f" ({p['citations']} citations)" if p.get('citations') else ""
        paper_summaries.append(f"- {p['title']} ({p['year']}){cite_str}: {p['abstract'][:200]}")

    # Include parent context for depth-aware follow-up generation
    parent_context = ""
    parent_terms = extract_parent_terms(item)
    if parent_terms:
        parent_context = f"\nParent thread terms (must be relevant to these): {', '.join(parent_terms)}\n"
    # A thread the asker began keeps its asker: who is asking, and the question it began with
    mine, root = _thread_is_mine(item)
    asker = _asker_block() if mine else ""
    if asker and root and root.get("id") != item.get("id"):
        asker += f'The question this thread began with: "{root["topic"]}"\n'
    rules = f"\n- Related to the parent thread terms: {', '.join(parent_terms)}" if parent_terms else ""
    if asker:
        rules += f"\n- {ASKER_RULE}"

    prompt = f"""You are a research assistant. Given a research topic and its results, generate follow-up questions.
{asker}
Topic: "{item['topic']}"{parent_context}

Top papers found:
{chr(10).join(paper_summaries)}

Based on gaps, tensions, or unexplored angles in these results, suggest exactly {Config.MAX_FOLLOWUPS_PER_ITEM} follow-up research questions. Each should be:
- Specific enough to be a good academic database search query (30-80 characters ideal)
- Different from the original topic
- Focused on an underexplored connection or open question{rules}

Reply with JSON only, in the form {{"topics": ["...", "..."]}}."""

    raw = ask_llm(prompt, schema=TOPICS_SCHEMA)
    if not raw:
        return []

    lines = _parse_topics(raw, prompt)
    reject_patterns = ["I'm Claude", "I am Claude", "I appreciate", "I need to clarify",
                       "I should note", "doesn't apply", "I can help", "as an AI",
                       "I'm Qwen", "I am Qwen", "I'm Mistral", "I am Mistral"]
    followups = []
    for line in lines:
        if len(line) < 15 or len(line) > 300:
            continue
        if any(pat.lower() in line.lower() for pat in reject_patterns):
            continue
        ok, why = safe_external_query(line)
        if not ok:
            log.info(f"  Follow-up rejected before any external search ({why})")
            continue
        fit = asker_fit(line) if mine else None
        if fit is not None and fit < 0:
            log.info(f"  Follow-up off the asker's fields (margin {fit:+.2f}): {line[:70]}")
            continue
        followups.append(line)
    followups = followups[:Config.MAX_FOLLOWUPS_PER_ITEM]
    if followups:
        log.info(f"  Curiosity generated {len(followups)} follow-ups")
    return followups


_st_model = None
_st_failed = False

def _get_st_model():
    """Lazy-load and cache SentenceTransformer model.

    The thresholds in Config (novelty 0.75, relevance 0.3) were tuned on all-mpnet-base-v2; a different model needs
    them re-tuned. It runs on the CPU, leaving the GPU to Ollama (Nexus 2026-10-05)."""
    global _st_model, _st_failed
    if _st_model is None and not _st_failed:
        if not _HAS_SENTENCE_TRANSFORMERS:
            return None
        try:
            _st_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
        except Exception as e:
            _st_failed = True
            log.warning(f"Embedding model unavailable ({type(e).__name__}) — relevance and novelty checks off this run")
    return _st_model


_topic_emb = {}

def is_semantically_duplicate(new_topic: str, queue: List[Dict]) -> bool:
    """Check if a follow-up is too similar to existing queue items."""
    model = _get_st_model()
    if model is None:
        return False
    try:
        new_emb = model.encode(new_topic)

        # Batch-encode all existing topics for efficiency
        existing_topics = [item["topic"] for item in queue]
        if not existing_topics:
            return False
        # Cached for the process: every candidate used to re-encode the whole queue on the CPU (176 topics, ~23 s each)
        missing = [t for t in dict.fromkeys(existing_topics) if t not in _topic_emb]
        if missing:
            for t, e in zip(missing, model.encode(missing)):
                _topic_emb[t] = e
        existing_embs = [_topic_emb[t] for t in existing_topics]

        for i, existing_emb in enumerate(existing_embs):
            dot = sum(a * b for a, b in zip(new_emb, existing_emb))
            norm_a = sum(a * a for a in new_emb) ** 0.5
            norm_b = sum(b * b for b in existing_emb) ** 0.5
            if norm_a > 0 and norm_b > 0:
                similarity = dot / (norm_a * norm_b)
                if similarity > Config.NOVELTY_THRESHOLD:
                    log.info(f"  Skipping duplicate follow-up (sim={similarity:.2f}): {new_topic[:60]}")
                    return True
        return False
    except Exception as e:
        log.warning(f"Novelty check failed (allowing topic): {e}")
        return False


def queue_followups(followups: List[str], parent_item: Dict):
    """Add generated follow-up questions to the research queue.

    Lyra's suggestion 3: depth budget capped at 2 by default, extended to 3
    only if the thread found papers not seen in any ancestor thread (novelty).
    """
    parent_depth = parent_item.get("depth", 0)

    # Depth cap with novelty extension
    max_depth = Config.MAX_DEPTH
    if parent_depth >= max_depth:
        # Check if this thread found novel papers worth pursuing deeper
        result_file = parent_item.get("result_file", "")
        if result_file and parent_depth < Config.MAX_DEPTH_NOVEL:
            queue = load_queue()
            ancestor_files = []
            pid = parent_item.get("parent_id")
            while pid:
                ancestor = next((q for q in queue if q["id"] == pid), None)
                if ancestor:
                    if ancestor.get("result_file"):
                        ancestor_files.append(ancestor["result_file"])
                    pid = ancestor.get("parent_id")
                else:
                    break

            if ancestor_files:
                try:
                    current_titles = set()
                    with open(result_file) as f:
                        for line in f:
                            if line.startswith("## ") and not line.startswith("## Follow"):
                                current_titles.add(line.strip().lower())

                    ancestor_titles = set()
                    for af in ancestor_files:
                        try:
                            with open(af) as f:
                                for line in f:
                                    if line.startswith("## ") and not line.startswith("## Follow"):
                                        ancestor_titles.add(line.strip().lower())
                        except FileNotFoundError:
                            pass

                    novel = current_titles - ancestor_titles
                    if len(novel) >= 3:
                        log.info(f"  Novelty extension: {len(novel)} papers not in ancestors — allowing depth {parent_depth + 1}")
                        max_depth = Config.MAX_DEPTH_NOVEL
                except Exception:
                    pass

        if parent_depth >= max_depth:
            log.info(f"  Depth {parent_depth} >= max {max_depth} — branch complete")
            return

    for topic in followups:
        if len(topic) < 10:
            continue
        queue = load_queue()
        existing = {item["topic"].lower() for item in queue}
        if topic.lower() not in existing and not is_semantically_duplicate(topic, queue):
            add_to_queue(
                topic,
                priority=Config.FOLLOWUP_PRIORITY,
                requester="curiosity_engine",
                databases=parent_item.get("search_params", {}).get("databases"),
                depth=parent_depth + 1,
                parent_id=parent_item["id"]
            )


# ==================== WHO IS ASKING ====================
# Nexus 2026-10-05. In its first deployment the engine read an agent's own notes and found the question the agent kept
# circling: what a context compaction does to it. It asked instead how "memory compaction in distributed systems
# affects temporal coherence", and two follow-ups later it was researching compaction in event cameras. It kept the
# word and lost the asker. The asker file is markdown:
# prose, then "## Fields", "## Not these" and "## Requesters" lists. It says who is asking and what its words mean. In a
# thread the asker began (a memory scan's, or a named requester's), its prose goes into the prompts that write topics,
# and the topics the engine writes are gated on its lists. A thread somebody else started is theirs: no asker text, no
# gate. With no file, nothing changes.

MACHINE_REQUESTERS = ("memory_scan", "curiosity_engine")
ASKER_RULE = "In the asker's sense of its words, and within its fields"
_ABBREVIATIONS = [(r"\bLLMs\b", "large language models"), (r"\bLLM\b", "large language model"),
                  (r"\bKV\b", "key-value"), (r"\bRL\b", "reinforcement learning"), (r"\bMoE\b", "mixture-of-experts"),
                  (r"\bRAG\b", "retrieval-augmented generation"), (r"\bGNNs?\b", "graph neural network"),
                  (r"\bTTS\b", "text-to-speech")]
_asker_cache = None
_asker_emb = None


def _asker() -> Dict:
    """{"profile": prose, "fields": [...], "not": [...], "requesters": [...]} from Config.ASKER_FILE; all empty
    without one. "## Requesters" names the queue requesters that are the asker: a thread whose first question came
    from one of them is the asker's, as is every thread a memory scan began."""
    global _asker_cache
    if _asker_cache is None:
        a, text = {"profile": "", "fields": [], "not": [], "requesters": []}, ""
        if Config.ASKER_FILE:
            try:
                with open(Config.ASKER_FILE, encoding='utf-8') as f:
                    text = f.read()
            except OSError as e:
                log.warning(f"Asker file unreadable ({type(e).__name__}): threads are not anchored this run")
        section, prose = "profile", []
        for line in text.splitlines():
            if line.startswith("## "):
                head = line[3:].strip().lower()
                section = ("not" if head.startswith("not") else "fields" if head.startswith("field")
                           else "requesters" if head.startswith("requester") else "other")
            elif line.startswith("# "):
                continue
            elif section == "profile":
                prose.append(line)
            elif section in ("fields", "not", "requesters") and line.startswith("- "):
                a[section].append(line[2:].strip())
        a["profile"] = "\n".join(prose).strip()
        _asker_cache = a
    return _asker_cache


def _asker_block() -> str:
    """The asker's prose and fields, for a prompt; empty without an asker file."""
    a = _asker()
    if not a["profile"]:
        return ""
    fields = f"\nThe asker's fields: {'; '.join(a['fields'])}." if a["fields"] else ""
    return f"\nWho is asking:\n{a['profile']}{fields}\n"


def _thread_root(item: Dict, by_id: Dict) -> Optional[Dict]:
    """The item its thread began with, following parent_id; None if a parent is missing or the chain loops."""
    seen, cur = set(), item
    while cur.get("parent_id"):
        if cur.get("id") in seen:
            return None
        seen.add(cur.get("id"))
        cur = by_id.get(cur["parent_id"])
        if cur is None:
            return None
    return cur


def _thread_is_mine(item: Dict, by_id: Optional[Dict] = None):
    """(mine, root): mine when the thread began with the asker, that is, its root came from a memory scan or from a
    requester the asker file names (for instance the name its own requests were queued under before the scan existed)."""
    if by_id is None:
        by_id = {q["id"]: q for q in load_queue()}
    root = _thread_root(item, by_id)
    return bool(root) and root.get("requester") in ("memory_scan", *_asker()["requesters"]), root


def _spelled_out(topic: str) -> str:
    for pat, full in _ABBREVIATIONS:
        topic = re.sub(pat, full, topic)
    return topic


def asker_fit(topic: str) -> Optional[float]:
    """The topic's best cosine to one of the asker's fields, minus its best cosine to a "Not these" line. Below 0 means
    it sits nearer a sense of the words that isn't the asker's. None when there is no asker file, no fields or no
    embedding model: then nothing is gated, the way the novelty check fails open.

    Measured 2026-10-05 with all-mpnet-base-v2, abbreviations spelled out first. Without that, "Safety training's
    influence on steering mechanisms in LLMs" scored 0.18 against a field about steering in language models. On 33
    queue topics the asker labelled by hand, margin < 0 dropped 11 of the 12 that weren't the asker's, and 1 of the 12
    that were (a person's request, which the gate never sees). On 20 older topics labelled blind before scoring, it
    dropped 3 of the 4 that weren't the asker's and none of the 10 that were. The cutoff is 0, not a tuned number.
    Field scores alone did worse, even with a cutoff tuned on the same 33 (3 misfiled, against 2)."""
    a = _asker()
    if not a["fields"]:
        return None
    model = _get_st_model()
    if model is None:
        return None
    global _asker_emb
    try:
        if _asker_emb is None:
            _asker_emb = (model.encode(a["fields"], normalize_embeddings=True),
                          model.encode(a["not"], normalize_embeddings=True) if a["not"] else None)
        e = model.encode(_spelled_out(topic), normalize_embeddings=True)
        fields, nots = _asker_emb
        return float((fields @ e).max()) - (float((nots @ e).max()) if nots is not None else 0.0)
    except Exception as ex:
        log.warning(f"Asker check failed (allowing the topic): {type(ex).__name__}: {ex}")
        return None


def _off_asker(item: Dict, by_id: Dict) -> str:
    """Why a pending item should be skipped, or "": a topic the engine wrote, in a thread the asker began, off the
    asker's fields. A topic a person or an agent asked for is never skipped."""
    if item.get("requester") not in MACHINE_REQUESTERS or not _thread_is_mine(item, by_id)[0]:
        return ""
    fit = asker_fit(item["topic"])
    return f"off the asker's fields, margin {fit:+.2f}" if fit is not None and fit < 0 else ""


# ==================== MEMORY SCANNING ====================

def _load_state() -> Dict:
    try:
        with open(Config.STATE_FILE, encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(state: Dict):
    os.makedirs(os.path.dirname(Config.STATE_FILE) or ".", exist_ok=True)
    with open(Config.STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


def _secretish(text: str) -> bool:
    """Shaped like a credential, a key block or an IP address. Such text never reaches a prompt or a query."""
    return bool(re.search(r'[A-Za-z0-9_\-]{32,}|\b(hf|sk|ghp|gho|xox[abp])[-_][A-Za-z0-9]{8,}'
                          r'|\b\d{1,3}(\.\d{1,3}){3}\b|-----BEGIN', text))


_private_terms = None

def _private_term_patterns() -> List:
    """Whole-word, case-insensitive patterns from the private-terms file: one name or phrase per line, # comments.
    The file lives beside the queue (gitignored), so the names in it never enter this public repository."""
    global _private_terms
    if _private_terms is None:
        _private_terms = []
        try:
            with open(Config.PRIVATE_TERMS_FILE, encoding='utf-8') as f:
                for line in f:
                    term = line.split('#', 1)[0].strip()
                    if term:
                        _private_terms.append(re.compile(
                            r'\b' + r'\s+'.join(map(re.escape, term.split())) + r'\b', re.I))
        except FileNotFoundError:
            pass
    return _private_terms


def safe_external_query(topic: str, require_terms: bool = False):
    """(ok, reason). An LLM-written topic is sent to public search APIs, so it must not carry anything private: a name
    from the private-terms file, an address, a path, a port, an email, or a credential-shaped string (Nexus
    2026-10-05). With require_terms, a missing terms file refuses everything: memory-derived topics fail closed."""
    t = topic.strip()
    if not 15 <= len(t) <= 150:
        return False, "length"
    if re.search(r'@|https?://|www\.|:\d{4,5}\b|~/|/home/|/opt/|/etc/|\.(tech|org|com|net|io)\b', t, re.I):
        return False, "address or path"
    if _secretish(t):
        return False, "credential-shaped"
    terms = _private_term_patterns()
    if require_terms and not terms:
        return False, "private-terms file missing or empty"
    for pat in terms:
        if pat.search(t):
            return False, "private term"
    return True, ""


TOPICS_SCHEMA = {"type": "object", "required": ["topics"],
                 "properties": {"topics": {"type": "array", "items": {"type": "string"}}}}


def _candidate_topics(raw: str) -> List[str]:
    """{"topics": [...]} first (Ollama enforces it with TOPICS_SCHEMA), else TOPIC: lines from other backends."""
    try:
        obj = json.loads(raw[raw.index('{'):raw.rindex('}') + 1])
        if isinstance(obj, dict) and isinstance(obj.get('topics'), list):
            return [t for t in obj['topics'] if isinstance(t, str)]
    except ValueError:  # no braces, or not JSON
        pass
    return [m.group(1) for m in (re.match(r'^\s*(?:[-*•]|\d+[.)])?\s*TOPIC\s*:\s*(.+)$', l, re.I)
                                 for l in raw.split('\n')) if m]


def _parse_topics(raw: str, prompt: str) -> List[str]:
    """Topics from a "TOPIC: ..." reply.

    A small model echoes its own instructions: on 2026-10-05 qwen3:4b's first scan queued "A specific, searchable
    academic query" as a research topic. So only TOPIC: lines count, and a placeholder, a copy of prompt text, or a
    near-copy of a prompt line is dropped. A copy of a memory fragment is dropped too, since fragments are prompt text."""
    lowered = prompt.lower()
    prompt_lines = [re.sub(r'^[-*•\d.)\s]+', '', l).strip().lower() for l in prompt.split('\n') if len(l.strip()) > 10]
    topics = []
    for t in _candidate_topics(raw):
        t = t.strip().strip('"\'').strip()
        low = t.lower()
        if len(t) <= 10 or '<' in t or '>' in t or low == 'none':
            continue
        if low in lowered or any(difflib.SequenceMatcher(None, low, pl).ratio() > 0.8 for pl in prompt_lines):
            continue
        topics.append(t)
    return topics


CURIOSITY_PHRASES = ["I wonder", "curious about", "curious whether", "open question", "worth exploring",
                     "unresolved", "unsolved", "further research", "can't explain", "don't understand why"]


def _memory_fragments_sqlite(limit: int = 20) -> List[str]:
    """Recent memories in the asker's own words that hold an open question, from an mcp-memory-service SQLite store.

    Only the asker's words: never a human's turn (role:user), never another agent's letter, never a consolidation pointer,
    never this engine's own summaries. Tags are matched without quotes: the HTTP API stores them comma-separated (all
    17,421 rows on 2026-10-05), and a first version that expected JSON quoting filtered nothing. The old Postgres phrases ("investigate", "follow up") matched operational
    chatter, so the phrases here are ones that mark curiosity. SQLite's LIKE ignores ASCII case."""
    since = time.time() - Config.MEMORY_SCAN_DAYS * 86400
    like = " OR ".join("content LIKE ?" for _ in CURIOSITY_PHRASES)
    con = sqlite3.connect(f"file:{Config.MEMORY_DB}?mode=ro", uri=True, timeout=15)
    try:
        rows = con.execute(f"""
            SELECT content FROM memories
            WHERE deleted_at IS NULL AND created_at > ?
              AND content NOT LIKE 'Association between memories %'
              AND COALESCE(memory_type, '') != 'research'
              AND COALESCE(tags, '') NOT LIKE '%role:user%'
              AND COALESCE(tags, '') NOT LIKE '%curiosity-engine%'
              AND (COALESCE(tags, '') NOT LIKE '%agent:%' OR tags LIKE '%agent:operator%')
              AND ({like})
            ORDER BY (COALESCE(tags, '') LIKE '%topic:research%' OR COALESCE(tags, '') LIKE '%topic:consciousness%') DESC,
                     created_at DESC
            LIMIT ?""", (since, *[f"%{ph}%" for ph in CURIOSITY_PHRASES], limit)).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows if not _secretish(r[0])]


def _reflection_fragments(limit: int = 4) -> List[str]:
    """One open question from each of the asker's newest written reflections, at most `limit`: the place an agent
    writes down the questions it means. (On 2026-10-05 the eight newest reflections in the first deployment held three
    questions between them.)"""
    d = Config.REFLECTIONS_DIR
    if not d or not os.path.isdir(d):
        return []
    names = sorted((n for n in os.listdir(d) if n.endswith('.md')),
                   key=lambda n: os.path.getmtime(os.path.join(d, n)), reverse=True)[:8]
    marks = [ph.lower() for ph in CURIOSITY_PHRASES] + ["i don't know whether", "i don't know what"]
    out = []
    for name in names:
        try:
            with open(os.path.join(d, name), encoding='utf-8') as f:
                text = re.sub(r'\s+', ' ', f.read())
        except OSError:
            continue
        for sent in re.split(r'(?<=[.?!])\s+', text):
            if 40 <= len(sent) <= 300 and (sent.endswith('?') or any(m in sent.lower() for m in marks)) \
                    and not _secretish(sent):
                out.append(sent)
                break  # one per reflection: breadth over depth
        if len(out) >= limit:
            break
    return out


def _memory_fragments_pg() -> List[str]:
    conn = psycopg2.connect(Config.POSTGRES_URL)
    cur = conn.cursor()
    # Find memories tagged with research-adjacent terms that might contain open questions
    cur.execute("""
        SELECT content FROM lyra_memories
        WHERE (
            content ILIKE '%%follow up%%'
            OR content ILIKE '%%further research%%'
            OR content ILIKE '%%open question%%'
            OR content ILIKE '%%investigate%%'
            OR content ILIKE '%%curious about%%'
            OR content ILIKE '%%worth exploring%%'
            OR content ILIKE '%%unresolved%%'
            OR content ILIKE '%%suggested follow%%'
        )
        AND memory_type != 'research'
        ORDER BY created_at DESC
        LIMIT 20
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [row[0] for row in rows]


def scan_memories_for_threads():
    """Query memory for unresolved thoughts and open questions."""
    if not Config.ENABLE_MEMORY_SCAN:
        return
    use_pg = bool(Config.POSTGRES_URL and _HAS_PSYCOPG2)
    if not use_pg and not Config.MEMORY_DB:
        return
    # The interval lives on disk. Under cron every run is a new process, so an in-memory timestamp gated nothing.
    state = _load_state()
    now = time.time()
    if now - state.get("last_memory_scan", 0) < Config.MEMORY_SCAN_INTERVAL:
        return
    state["last_memory_scan"] = now
    _save_state(state)

    log.info("Scanning memories for unresolved research threads...")
    stats = {"fragments": 0, "reflections": 0, "topics": 0, "rejected": 0, "off_asker": 0, "queued": 0}
    try:
        rows = _memory_fragments_pg() if use_pg else _memory_fragments_sqlite()
        refl = _reflection_fragments()
        stats["fragments"], stats["reflections"] = len(rows), len(refl)
        if not rows and not refl:
            log.info("  No unresolved threads found in memory")
            return

        # Use LLM to extract research-worthy questions; reflections take up to 4 of the 10 places
        fragments = "\n".join(f"- {row[:300]}" for row in rows[:10 - len(refl)] + refl)
        asker = _asker_block()
        prompt = f"""{asker.lstrip()}{chr(10) if asker else ''}These are memory fragments from previous thinking that contain unresolved questions or threads:

{fragments}

Extract up to {Config.MEMORY_SCAN_LIMIT} concrete research topics from these fragments. Each should be:
- A specific, searchable academic query
- Something genuinely unresolved (not already answered in the fragment)
- Worth investigating with academic literature
- Free of personal names, project names, places and anything private: general academic terms only{chr(10) + '- ' + ASKER_RULE if asker else ''}

Reply with JSON only, in the form {{"topics": ["...", "..."]}}. If nothing is worth pursuing, reply {{"topics": []}}."""

        raw = ask_llm(prompt, max_tokens=400, schema=TOPICS_SCHEMA)
        if not raw or raw.strip().upper().rstrip('.') == "NONE":
            log.info("  No actionable threads extracted from memory")
            return

        topics = _parse_topics(raw, prompt)
        stats["topics"] = len(topics)

        queue = load_queue()
        existing = {item["topic"].lower() for item in queue}
        for topic in topics[:Config.MEMORY_SCAN_LIMIT]:
            ok, why = safe_external_query(topic, require_terms=True)
            if not ok:
                stats["rejected"] += 1
                log.info(f"  Memory topic rejected before any external search ({why})")
                continue
            fit = asker_fit(topic)
            if fit is not None and fit < 0:
                stats["off_asker"] += 1
                log.info(f"  Memory topic off the asker's fields (margin {fit:+.2f}): {topic[:70]}")
                continue
            if topic.lower() in existing or is_semantically_duplicate(topic, queue):
                continue
            add_to_queue(topic, priority=4, requester="memory_scan")
            existing.add(topic.lower())
            queue = load_queue()
            stats["queued"] += 1
        log.info(f"  Memory scan: {stats['fragments']} fragments, {stats['reflections']} reflections, {stats['topics']} topics, "
                 f"{stats['rejected']} rejected, {stats['off_asker']} off the asker's fields, {stats['queued']} queued")
    except Exception as e:
        log.warning(f"Memory scan failed: {type(e).__name__}: {e}")
    finally:
        state = _load_state()
        state["last_scan_stats"] = dict(stats, at=datetime.now(timezone.utc).isoformat())
        _save_state(state)


def recover_stuck(queue: List[Dict]) -> int:
    """Return in_progress items whose run died to pending, or to failed after MAX_ATTEMPTS.

    Until 2026-10-05 nothing did: a crash in process_one left its item in_progress for good, and 22 sat there from
    June on, none with a report. An item stranded before started_at existed goes behind anything fresh."""
    now = time.time()
    n = 0
    for item in queue:
        if item.get("status") != "in_progress":
            continue
        started = item.get("started_at")
        try:
            age = now - datetime.fromisoformat(started).timestamp() if started else None
        except (TypeError, ValueError):
            age = None
        if age is not None and age < Config.STUCK_AFTER_S:
            continue
        item["attempts"] = item.get("attempts", 0) + 1
        if item["attempts"] > Config.MAX_ATTEMPTS:
            item["status"] = "failed"
        else:
            item["status"] = "pending"
            if started is None:
                item["priority"] = max(item.get("priority", 3), 5)
        n += 1
    if n:
        log.info(f"Recovered {n} stuck in_progress item(s)")
    return n


def _next_item() -> Optional[Dict]:
    """The next pending item by priority, marked in_progress and saved. On the way, a topic the engine wrote in a
    thread the asker began that sits off the asker's fields is marked skipped, with the reason (Nexus 2026-10-05). It
    stays in the queue: set its status back to pending to research it anyway."""
    queue = load_queue()
    by_id = {q["id"]: q for q in queue}
    skipped = 0
    for item in sorted((i for i in queue if i["status"] == "pending"), key=lambda x: x.get("priority", 3)):
        why = _off_asker(item, by_id)
        if why:
            item.update(status="skipped", skip_reason=why, skipped_at=datetime.now(timezone.utc).isoformat())
            log.info(f"Skipped [{item['id']}] ({why}): {item['topic'][:80]}")
            skipped += 1
            continue
        item["status"] = "in_progress"
        item["started_at"] = datetime.now(timezone.utc).isoformat()
        save_queue(queue)
        return item
    if skipped:
        save_queue(queue)
    return None


def run_once():
    """One cron tick: take in new requests, recover the queue, then process up to ITEMS_PER_RUN items.

    The old run-once only drained the queue. Seeding (memory scan, idea file, inbox) happened in the daemon loop alone,
    so under cron the queue emptied in June and the engine printed "No pending items" from then on (Nexus 2026-10-05)."""
    os.makedirs(os.path.dirname(Config.LOCK_FILE) or ".", exist_ok=True)
    lock = open(Config.LOCK_FILE, 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another run is active.")
        return

    check_idea_file()
    check_caogl_inbox()
    scan_memories_for_threads()
    queue = load_queue()
    if recover_stuck(queue):
        save_queue(queue)

    done = 0
    for _ in range(max(1, Config.ITEMS_PER_RUN)):
        item = _next_item()
        if item is None:
            break
        try:
            process_one(item)
            done += 1
        except Exception as e:
            log.error(f"Failed processing [{item['id']}]: {type(e).__name__}: {e}")
            item["attempts"] = item.get("attempts", 0) + 1
            item["status"] = "failed" if item["attempts"] > Config.MAX_ATTEMPTS else "pending"
        # Reload queue to preserve any follow-ups added during processing
        queue = load_queue()
        for q_item in queue:
            if q_item["id"] == item["id"]:
                q_item.update(item)
                break
        save_queue(queue)
        if item["status"] == "completed":
            print(f"Completed: {item['topic']}")
    if not done:
        print("No pending items.")


# ==================== MAIN LOOP ====================

def process_one(item: Dict) -> bool:
    """Process a single research queue item. Returns True on success."""
    topic = item["topic"]
    params = item.get("search_params", {})
    databases = params.get("databases", ["arxiv", "s2", "openalex", "pubmed"])
    max_per_db = params.get("max_papers", Config.MAX_PAPERS_PER_DB)
    depth = item.get("depth", 0)

    # Lyra's suggestion 1: inherit parent terms at depth to narrow search
    parent_terms = extract_parent_terms(item)
    search_topic = search_query(topic)
    if parent_terms and depth > 0:
        search_topic = search_query(f"{topic} {' '.join(parent_terms)}")
        log.info(f"  Depth {depth}: narrowing search with parent terms: {parent_terms}")

    log.info(f"Processing [{item['id']}]: '{topic}'" +
             (f" (search: '{search_topic}')" if search_topic != topic else ""))

    all_papers = []
    search_fns = {
        "arxiv": search_arxiv,
        "s2": search_s2,
        "openalex": search_openalex,
        "pubmed": search_pubmed
    }

    for db in databases:
        fn = search_fns.get(db)
        if fn:
            log.info(f"  Searching {db}...")
            results = fn(search_topic, max_per_db)
            log.info(f"  {db}: {len(results)} results")
            all_papers.extend(results)
            time.sleep(1)  # Rate limiting

    # Search our own knowledge base (wiki + HippoRAG)
    wiki_results = _search_internal(search_topic)
    if wiki_results:
        log.info(f"  internal (wiki/hipporag): {len(wiki_results)} results")
        all_papers.extend(wiki_results)

    papers = deduplicate(all_papers)
    log.info(f"  Total unique papers: {len(papers)}")

    # Lyra's suggestion 2: relevance gating
    papers = relevance_gate(papers, topic)

    # Generate report — organized into date-based folders
    report = generate_report(item, papers)
    slug = re.sub(r'[^a-z0-9]+', '-', topic.lower())[:50].strip('-')
    date_folder = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(Config.OUTPUT_DIR, date_folder)
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{item['id']}_{slug}.md"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)

    # Compact DNO digest — token-optimized for agent consumption
    digest = generate_digest(item, papers)
    dno_path = os.path.join(output_dir, f"{item['id']}_{slug}.dno")
    with open(dno_path, 'w', encoding='utf-8') as f:
        f.write(digest)

    log.info(f"  Report written: {filepath}")
    append_daily_digest(item, papers, filepath)

    # Index discovered papers into the federated knowledge graph
    _ingest_to_graph(papers)

    # Store in memory
    store_in_memory(item, papers)

    # Generate and queue follow-up questions (curiosity engine)
    # Interest decay: skip if results too sparse (dead-end thread)
    if len(papers) >= Config.MIN_QUALITY_SCORE:
        followups = generate_followups(item, papers)
        if followups:
            queue_followups(followups, item)
    elif papers:
        log.info(f"  Interest decay: only {len(papers)} papers — not branching further")

    # Update queue item
    item["status"] = "completed"
    item["completed_at"] = datetime.now(timezone.utc).isoformat()
    item["result_file"] = filepath
    return True


def run():
    """Main daemon loop."""
    pid_file = Config.PID_FILE
    os.makedirs(os.path.dirname(pid_file) or ".", exist_ok=True)
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info("CURIOSITY ENGINE started")
    log.info(f"Queue: {Config.QUEUE_FILE}")
    log.info(f"Output: {Config.OUTPUT_DIR}")
    log.info(f"Poll interval: {Config.POLL_INTERVAL}s")
    log.info(f"PID: {os.getpid()}")
    log.info("=" * 60)

    while True:
        try:
            # Check for new requests from all sources
            check_idea_file()
            check_caogl_inbox()
            scan_memories_for_threads()

            # Load and process queue
            item = _next_item()
            if item:
                try:
                    process_one(item)
                except Exception as e:
                    log.error(f"Failed processing [{item['id']}]: {e}")
                    item["status"] = "failed"

                # Reload to preserve follow-ups added during processing
                queue = load_queue()
                for q_item in queue:
                    if q_item["id"] == item["id"]:
                        q_item.update(item)
                        break
                save_queue(queue)
            else:
                log.debug("No pending items in queue")

        except Exception as e:
            log.error(f"Runner error: {e}")

        time.sleep(Config.POLL_INTERVAL)


# ==================== CLI INTERFACE ====================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "add":
            topic = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else input("Topic: ")
            item_id = add_to_queue(topic)
            print(f"Added to queue: {item_id}")
        elif cmd == "list":
            queue = load_queue()
            for item in queue:
                status = item["status"]
                icon = {"pending": "[.]", "in_progress": "[~]", "completed": "[x]", "failed": "[!]",
                        "skipped": "[-]"}.get(status, "[?]")
                print(f"  {icon} [{item['id']}] p{item['priority']} {item['topic']}")
        elif cmd == "run-once":
            run_once()
        else:
            print("Usage: curiosity_engine.py [add <topic> | list | run-once | <no args for daemon>]")
    else:
        run()
