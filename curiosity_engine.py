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

import json, os, sys, time, uuid, logging, re, hashlib, subprocess
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
        handlers=[
            logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

log = setup_logging()

if not _HAS_SENTENCE_TRANSFORMERS:
    log.warning("sentence-transformers not installed — semantic novelty detection disabled")
if not _HAS_PSYCOPG2:
    log.warning("psycopg2 not installed — PostgreSQL memory storage disabled")
if not Config.POSTGRES_URL:
    log.info("DATABASE_URL not set — memory storage and memory scanning disabled")


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

    # Wiki search
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

def search_s2(query: str, max_results: int = 10) -> List[Dict]:
    """Search Semantic Scholar API with retry on rate limit."""
    for attempt in range(3):
        try:
            resp = requests.get(f"{Config.S2_API}/paper/search", params={
                "query": query,
                "limit": max_results,
                "fields": "title,authors,year,abstract,citationCount,url"
            }, timeout=Config.REQUEST_TIMEOUT)
            if resp.status_code == 429:
                wait = (attempt + 1) * 10
                log.info(f"  S2 rate limited, waiting {wait}s (attempt {attempt+1}/3)")
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
                log.warning(f"S2 search failed after retries: {e}")
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

    lines.extend([
        "---",
        "",
        "## Summary Statistics",
        f"- Total unique papers: {len(papers)}",
        f"- Year range: {min(p['year'] for p in papers if p['year'])} - {max(p['year'] for p in papers if p['year'])}",
        f"- Sources: {', '.join(set(p['source'] for p in papers))}",
        f"- Most cited: {papers[0]['title']} ({papers[0]['citations'] or 'N/A'} citations)",
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


# ==================== MEMORY STORAGE ====================

def _ingest_to_graph(papers: List[Dict]):
    """Index discovered papers into the federated knowledge graph (HippoRAG)."""
    hipporag_url = os.environ.get("HIPPORAG_URL", "http://localhost:11235")

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
    """Store research summary in PostgreSQL memory (if configured)."""
    if not Config.POSTGRES_URL or not _HAS_PSYCOPG2 or not _HAS_SENTENCE_TRANSFORMERS:
        return
    try:
        model = _get_st_model()
        summary = f"Research on '{item['topic']}': found {len(papers)} papers. "
        if papers:
            top = papers[0]
            summary += f"Top result: '{top['title']}' ({top['year']}). "
            summary += f"Sources: {', '.join(set(p['source'] for p in papers))}."

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

def ask_ollama(prompt: str, max_tokens: int = 300) -> Optional[str]:
    """Send a prompt to Ollama and return the response text."""
    try:
        resp = requests.post(
            f'{Config.OLLAMA_URL}/api/generate',
            json={
                'model': Config.OLLAMA_MODEL,
                'prompt': prompt,
                'stream': False,
                'options': {'num_predict': max_tokens, 'temperature': 0.7},
            },
            timeout=90,
        )
        if resp.status_code == 200:
            return resp.json().get('response', '').strip()
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


def ask_llm(prompt: str, max_tokens: int = 300) -> Optional[str]:
    """Route prompt to configured LLM backend."""
    if Config.use_ollama():
        return ask_ollama(prompt, max_tokens)
    return ask_claude(prompt)


def generate_followups(item: Dict, papers: List[Dict]) -> List[str]:
    """Use LLM to generate follow-up research questions."""
    if not Config.ENABLE_FOLLOWUPS or not papers:
        return []

    # Build context from top 5 papers
    paper_summaries = []
    for p in papers[:5]:
        cite_str = f" ({p['citations']} citations)" if p.get('citations') else ""
        paper_summaries.append(f"- {p['title']} ({p['year']}){cite_str}: {p['abstract'][:200]}")

    # Include parent context for depth-aware follow-up generation
    parent_context = ""
    parent_terms = extract_parent_terms(item)
    if parent_terms:
        parent_context = f"\nParent thread terms (must be relevant to these): {', '.join(parent_terms)}\n"

    prompt = f"""You are a research assistant. Given a research topic and its results, generate follow-up questions.

Topic: "{item['topic']}"{parent_context}

Top papers found:
{chr(10).join(paper_summaries)}

Based on gaps, tensions, or unexplored angles in these results, suggest exactly {Config.MAX_FOLLOWUPS_PER_ITEM} follow-up research questions. Each should be:
- Specific enough to be a good academic database search query (30-80 characters ideal)
- Different from the original topic
- Focused on an underexplored connection or open question{f'{chr(10)}- Related to the parent thread terms: {", ".join(parent_terms)}' if parent_terms else ''}

Reply with ONLY the questions, one per line, no numbering, bullets, or commentary."""

    raw = ask_llm(prompt)
    if not raw:
        return []

    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    # Strip numbering, bullets, quotes
    lines = [re.sub(r'^[\d]+[\.\)]\s*', '', l) for l in lines]
    lines = [re.sub(r'^[-*•]\s*', '', l) for l in lines]
    lines = [l.strip('"\'') for l in lines]
    reject_patterns = ["I'm Claude", "I am Claude", "I appreciate", "I need to clarify",
                       "I should note", "doesn't apply", "I can help", "as an AI",
                       "I'm Qwen", "I am Qwen", "I'm Mistral", "I am Mistral"]
    followups = []
    for line in lines:
        if len(line) < 15 or len(line) > 300:
            continue
        if any(pat.lower() in line.lower() for pat in reject_patterns):
            continue
        followups.append(line)
    followups = followups[:Config.MAX_FOLLOWUPS_PER_ITEM]
    if followups:
        log.info(f"  Curiosity generated {len(followups)} follow-ups")
    return followups


_st_model = None

def _get_st_model():
    """Lazy-load and cache SentenceTransformer model."""
    global _st_model
    if _st_model is None:
        if not _HAS_SENTENCE_TRANSFORMERS:
            return None
        _st_model = SentenceTransformer("all-mpnet-base-v2")
    return _st_model


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
        existing_embs = model.encode(existing_topics)

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


# ==================== MEMORY SCANNING ====================

_last_memory_scan = 0

def scan_memories_for_threads():
    """Query memory for unresolved thoughts and open questions."""
    global _last_memory_scan
    now = time.time()
    if not Config.ENABLE_MEMORY_SCAN:
        return
    if not Config.POSTGRES_URL or not _HAS_PSYCOPG2:
        return
    if now - _last_memory_scan < Config.MEMORY_SCAN_INTERVAL:
        return
    _last_memory_scan = now

    log.info("Scanning memories for unresolved research threads...")

    try:
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

        if not rows:
            log.info("  No unresolved threads found in memory")
            return

        # Use LLM to extract research-worthy questions from memory fragments
        fragments = "\n".join(f"- {row[0][:300]}" for row in rows[:10])
        prompt = f"""These are memory fragments from previous thinking that contain unresolved questions or threads:

{fragments}

Extract up to {Config.MEMORY_SCAN_LIMIT} concrete research topics from these fragments. Each should be:
- A specific, searchable academic query
- Something genuinely unresolved (not already answered in the fragment)
- Worth investigating with academic literature

Reply with ONLY the topics, one per line, no numbering or bullets. If nothing is worth pursuing, reply with just "NONE"."""

        try:
            raw = ask_llm(prompt)
            if not raw or "NONE" in raw:
                log.info("  No actionable threads extracted from memory")
                return

            topics = [l.strip() for l in raw.split('\n')
                      if l.strip() and len(l.strip()) > 10]

            queue = load_queue()
            existing = {item["topic"].lower() for item in queue}

            queued = 0
            for topic in topics[:Config.MEMORY_SCAN_LIMIT]:
                if topic.lower() not in existing:
                    add_to_queue(topic, priority=4, requester="memory_scan")
                    queued += 1

            if queued:
                log.info(f"  Queued {queued} topics from memory scan")

        except Exception as e:
            log.warning(f"Memory scan LLM extraction failed: {e}")

    except Exception as e:
        log.warning(f"Memory scan failed: {e}")


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
    search_topic = topic
    if parent_terms and depth > 0:
        search_topic = f"{topic} {' '.join(parent_terms)}"
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
            queue = load_queue()
            pending = [item for item in queue if item["status"] == "pending"]
            pending.sort(key=lambda x: x.get("priority", 3))

            if pending:
                item = pending[0]
                item["status"] = "in_progress"
                save_queue(queue)

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
                icon = {"pending": "[.]", "in_progress": "[~]", "completed": "[x]", "failed": "[!]"}.get(status, "[?]")
                print(f"  {icon} [{item['id']}] p{item['priority']} {item['topic']}")
        elif cmd == "run-once":
            queue = load_queue()
            pending = [i for i in queue if i["status"] == "pending"]
            if pending:
                pending.sort(key=lambda x: x.get("priority", 3))
                item = pending[0]
                item["status"] = "in_progress"
                save_queue(queue)
                process_one(item)
                # Reload queue to preserve any follow-ups added during processing
                queue = load_queue()
                for q_item in queue:
                    if q_item["id"] == item["id"]:
                        q_item.update(item)
                        break
                save_queue(queue)
                print(f"Completed: {item['topic']}")
            else:
                print("No pending items.")
        else:
            print("Usage: curiosity_engine.py [add <topic> | list | run-once | <no args for daemon>]")
    else:
        run()
