#!/usr/bin/env python3
"""Tests for the SQLite/Ollama port (Nexus, 2026-10-05). Standalone: `python test_port.py`, no pytest needed.

CE_DIR names the directory to import curiosity_engine from (default: this one). The mutation controls point it at a
copy with one guard removed, and the matching test must fail. A test that cannot fail is not a test.

Hermetic: every path the engine writes points into a fresh temp dir, and the settings below override whatever a
deployment has in the caller's environment, so a run never reads or writes a live queue, memory store or data/ dir.
"""
import json
import os
import sqlite3
import sys
import tempfile
import time
import traceback

TMP = tempfile.mkdtemp(prefix='ce-test-')
os.environ.update({
    'CURIOSITY_QUEUE': f'{TMP}/queue.json', 'CURIOSITY_OUTPUT': f'{TMP}/out', 'CURIOSITY_LOG': f'{TMP}/out/test.log',
    'CURIOSITY_STATE': f'{TMP}/state.json', 'CURIOSITY_LOCK': f'{TMP}/run.lock',
    'CURIOSITY_PRIVATE_TERMS': f'{TMP}/terms.txt', 'CURIOSITY_INBOX': f'{TMP}/inbox.json',
    'CURIOSITY_LLM': 'ollama', 'CLAUDE_CLI': '/bin/false', 'DATABASE_URL': '', 'WIKI_PASS': '',
    'CURIOSITY_GRAPH_URL': '', 'CURIOSITY_MEMORY_API': '', 'MCP_API_KEY': '',
    'CURIOSITY_MEMORY_DB': '', 'CURIOSITY_REFLECTIONS_DIR': '', 'CURIOSITY_ASKER_FILE': '', 'CURIOSITY_IDEAS': '',
    'CURIOSITY_PID': f'{TMP}/runner.pid', 'CURIOSITY_ITEMS_PER_RUN': '1', 'S2_API_KEY': '', 'S2_API_KEY_FILE': '',
    'TGS_RAG_URL': '', 'WIKI_URL': '', 'OLLAMA_URL': 'http://127.0.0.1:9', 'CURIOSITY_OLLAMA_NUM_CTX': '2048',
    'CURIOSITY_EMBED_MODEL': 'all-mpnet-base-v2', 'CURIOSITY_EMBED_DEVICE': 'cpu',
    'CURIOSITY_SELF_AGENT_TAG': 'agent:self', 'CURIOSITY_PRIORITY_TAGS': 'topic:research',
})
sys.path.insert(0, os.environ.get('CE_DIR', os.path.dirname(os.path.abspath(__file__))))
import curiosity_engine as ce  # noqa: E402

with open(f'{TMP}/terms.txt', 'w') as f:
    f.write('# test names\nAlice\nAcme Lab Server\nAL\n')

NOW = time.time()


class Calls:
    """Stands in for requests.post / ask_llm and records what it was given."""
    def __init__(self, status=200, payload=None, reply=None):
        self.calls, self.status, self.payload, self.reply = [], status, payload or {}, reply

    def post(self, url, **kw):
        self.calls.append((url, kw))
        outer = self

        class R:
            status_code = outer.status
            text = ''
            def json(self):
                return outer.payload
        return R()

    def llm(self, prompt, max_tokens=300, **kw):
        self.calls.append(prompt)
        self.kw = kw
        return self.reply


def patch(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    return lambda: setattr(obj, name, old)


def fixture_db():
    path = f'{TMP}/mem-{time.time_ns()}.db'
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT, tags TEXT, memory_type TEXT, '
                'created_at REAL, deleted_at REAL)')
    day = 86400
    rows = [
        ('MINE-RESEARCH I wonder whether attention sinks carry positional information', ['agent:self', 'topic:research'], 'observation', NOW - 2 * day, None),
        ('Association between memories a and b: I wonder (ASSOC)', [], 'observation', NOW - day, None),
        ('HUMAN-TURN I wonder if we should rest today', ['verbatim', 'role:user'], 'observation', NOW - day, None),
        ('OWN-SUMMARY research I wonder', ['auto_research'], 'research', NOW - day, None),
        ('DELETED I wonder about this', ['agent:self'], 'observation', NOW - day, NOW),
        ('OLD I wonder from long ago', ['agent:self'], 'observation', NOW - 200 * day, None),
        ('OTHER-LETTER an open question about cache geometry', ['agent:other'], 'observation', NOW - day, None),
        ('SECRET I wonder about hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456', ['agent:self'], 'observation', NOW - day, None),
        ('DREAMER an open question remains about consolidation timing', ['dreamer-enriched'], 'observation', NOW - 3 * day, None),
        ('ENGINE-TAGGED I wonder, says the engine', ['curiosity-engine'], 'observation', NOW - day, None),
        ('PLAIN nothing curious here at all', ['agent:self'], 'observation', NOW - day, None),
    ]
    con.executemany('INSERT INTO memories (content, tags, memory_type, created_at, deleted_at) VALUES (?,?,?,?,?)',
                    [(c, ','.join(t), m, ts, d) for c, t, m, ts, d in rows])
    con.execute('INSERT INTO memories (content, tags, memory_type, created_at, deleted_at) VALUES (?,?,?,?,?)',
                ('JSON-LETTER I wonder, writes another agent', json.dumps(['agent:another']), 'observation', NOW - 86400, None))
    con.commit()
    con.close()
    return path


def reset_queue(items=()):
    ce.save_queue(list(items))
    ce._private_terms = None


# ------------------------------------------------------------------------------------------------------------- tests

def test_scan_reads_only_my_curious_words():
    ce.Config.MEMORY_DB = fixture_db()
    got = [c.split()[0] for c in ce._memory_fragments_sqlite()]
    assert got == ['MINE-RESEARCH', 'DREAMER'], got


def test_scan_tags_are_configurable():
    ce.Config.MEMORY_DB = fixture_db()
    old = ce.Config.SELF_AGENT_TAG, ce.Config.PRIORITY_TAGS
    try:
        ce.Config.SELF_AGENT_TAG = ''                                 # no own tag: agent-tagged rows are all someone else's
        assert [c.split()[0] for c in ce._memory_fragments_sqlite()] == ['DREAMER']
        ce.Config.SELF_AGENT_TAG, ce.Config.PRIORITY_TAGS = 'agent:self', ['dreamer-enriched']
        assert [c.split()[0] for c in ce._memory_fragments_sqlite()] == ['DREAMER', 'MINE-RESEARCH']   # tagged first
        ce.Config.PRIORITY_TAGS = []
        assert [c.split()[0] for c in ce._memory_fragments_sqlite()] == ['MINE-RESEARCH', 'DREAMER']   # newest first
    finally:
        ce.Config.SELF_AGENT_TAG, ce.Config.PRIORITY_TAGS = old


def test_internal_search_is_opt_in():
    fake = Calls(status=500)
    undo = [patch(ce.requests, 'post', fake.post), patch(ce, 'TGS_RAG_URL', ''), patch(ce, 'WIKI_URL', ''),
            patch(ce, 'WIKI_PASS', 'test-pass')]
    try:
        ce._search_internal('anything')
        assert not fake.calls, f'searched without an address: {fake.calls[0][0]}'
        ce.TGS_RAG_URL, ce.WIKI_URL = 'http://127.0.0.1:9', 'http://127.0.0.1:8'
        ce._search_internal('anything')
        assert [u for u, _ in fake.calls] == ['http://127.0.0.1:9/retrieve', 'http://127.0.0.1:8/api/query']
    finally:
        [u() for u in undo]


def test_embedding_model_and_device_come_from_config():
    seen = []
    old = getattr(ce, 'SentenceTransformer', None)
    undo = [patch(ce, '_st_model', None), patch(ce, '_st_failed', False), patch(ce, '_HAS_SENTENCE_TRANSFORMERS', True),
            patch(ce.Config, 'EMBED_MODEL', 'some-model'), patch(ce.Config, 'EMBED_DEVICE', '')]
    ce.SentenceTransformer = lambda name, device=None: seen.append((name, device)) or 'MODEL'
    try:
        assert ce._get_st_model() == 'MODEL' and seen == [('some-model', None)], seen
    finally:
        [u() for u in undo]
        ce.SentenceTransformer = old


def test_search_query_is_keywords():
    q = ce.search_query('How does memory compaction in distributed systems affect temporal coherence?')
    assert q == 'memory compaction distributed systems affect temporal coherence', q
    assert ce.search_query('compaction memory compaction LLM') == 'compaction memory LLM'


def test_no_summary_without_external_papers():
    fake = Calls(status=200)
    undo = [patch(ce.requests, 'post', fake.post)]
    ce.Config.MEMORY_API_URL, ce.Config.MEMORY_API_KEY = 'http://127.0.0.1:9', 'k-test'
    try:
        ce.store_in_memory({'id': 'i3', 'topic': 'T', 'requester': 'x'},
                           [{'source': 'internal/tgs-rag', 'title': 'MEMORY-TEXT', 'year': ''}])
        assert not fake.calls, 'stored a summary with no external papers'
    finally:
        ce.Config.MEMORY_API_URL = ce.Config.MEMORY_API_KEY = ''
        [u() for u in undo]


def test_s2_sends_the_key_and_backs_off_exponentially():
    with open(f'{TMP}/s2key', 'w') as f:
        f.write('test-key-123\n')
    seen, sleeps = [], []

    class Resp:
        def __init__(self, code):
            self.status_code, self.headers = code, {}
        def raise_for_status(self):
            pass
        def json(self):
            return {'data': [{'title': 'T', 'authors': [{'name': 'A'}], 'year': 2025, 'abstract': 'x', 'url': 'u',
                              'citationCount': 3}]}
    codes = iter([429, 429, 200])

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.append(headers)
        return Resp(next(codes))
    undo = [patch(ce.requests, 'get', fake_get), patch(ce.time, 'sleep', sleeps.append)]
    old = ce.Config.S2_API_KEY_FILE
    try:
        ce.Config.S2_API_KEY_FILE = f'{TMP}/s2key'
        out = ce.search_s2('q')
        assert [h.get('x-api-key') for h in seen] == ['test-key-123'] * 3, seen
        assert 2 <= sleeps[0] < 3 and 4 <= sleeps[1] < 5, sleeps          # exponential, with jitter under a second
        assert out and out[0]['title'] == 'T'
        ce.Config.S2_API_KEY_FILE, codes, seen[:] = f'{TMP}/no-such-key', iter([200]), []
        ce.search_s2('q')
        assert seen == [{}], 'sent a key header without a key'
    finally:
        ce.Config.S2_API_KEY_FILE = old
        [u() for u in undo]


def test_digest_entry_comes_back_labelled_and_external_only():
    fake = Calls(reply='{"finding": "FINDING-TEXT", "open": "OPEN-TEXT"}')
    undo = [patch(ce, 'ask_llm', fake.llm)]
    try:
        item = {'id': 'd1', 'topic': 'TOPIC-X', 'requester': 'memory_scan', 'depth': 0}
        papers = [{'source': 'internal/tgs-rag', 'title': 'MEMORY-TEXT', 'abstract': 'MEMORY-TEXT', 'year': ''},
                  {'source': 'arxiv', 'title': 'Paper A', 'abstract': 'abs A', 'year': '2025', 'url': 'http://a'},
                  {'source': 'openalex', 'title': 'Paper B', 'abstract': 'abs B', 'year': '2024', 'url': 'http://b'}]
        ce.append_daily_digest(item, papers, 'out/r.md')
        day = ce.datetime.now().strftime('%Y-%m-%d')
        text = open(f'{TMP}/out/digest/{day}.md').read()
        for want in ('TOPIC-X', 'FINDING-TEXT', 'unverified', 'OPEN-TEXT', 'Paper A', 'Paper B', 'arxiv 1, openalex 1'):
            assert want in text, want
        assert 'MEMORY-TEXT' not in text and 'MEMORY-TEXT' not in fake.calls[0], 'internal text reached the digest'
        assert fake.kw.get('schema') is ce.DIGEST_SCHEMA
        n = len(open(f'{TMP}/out/digest/{day}.md').read())
        ce.append_daily_digest(item, papers[:1], 'out/r.md')            # internal only: nothing to say
        assert len(open(f'{TMP}/out/digest/{day}.md').read()) == n
    finally:
        [u() for u in undo]


def test_reflection_fragments_take_one_real_question_each():
    d = f'{TMP}/refl'
    os.makedirs(d, exist_ok=True)
    files = {'001_a.md': 'A long statement that is not a question at all. Does attention really carry the position it seems to carry here?',
             '002_b.md': 'Short one? What about hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456 in this question, does it leak?',
             '003_c.md': 'Only conclusions here. Nothing is asked in this one, only told to the reader plainly.'}
    for i, (n, t) in enumerate(files.items()):
        with open(f'{d}/{n}', 'w') as f:
            f.write(t)
        os.utime(f'{d}/{n}', (NOW - 100 * i, NOW - 100 * i))
    old = ce.Config.REFLECTIONS_DIR
    try:
        ce.Config.REFLECTIONS_DIR = d
        assert ce._reflection_fragments() == ['Does attention really carry the position it seems to carry here?']
        ce.Config.REFLECTIONS_DIR = f'{TMP}/absent'
        assert ce._reflection_fragments() == []
    finally:
        ce.Config.REFLECTIONS_DIR = old


def test_external_query_filter():
    ce._private_terms = None
    ok = lambda t, **kw: ce.safe_external_query(t, **kw)[0]
    assert ok('attention sink dynamics in long-context transformers')
    assert ok('algorithmic alignment in speech models')                    # "AL" is a whole word only
    assert not ok('Alice memory consolidation patterns')
    assert not ok('acme  lab   server design patterns')                      # case and spacing don't hide a term
    assert not ok('AL licensing of open datasets')
    assert not ok('write to someone@example.org about this')
    assert not ok('latency of the service on 192.0.2.20 today')
    assert not ok('ollama latency on port :11434 under load')
    assert not ok('layout of /home/alice directory conventions')
    assert not ok('mirror at example.tech for papers')
    assert not ok('short')
    assert not ok('x' * 151)


def test_memory_topics_fail_closed_without_terms():
    old = ce.Config.PRIVATE_TERMS_FILE
    ce.Config.PRIVATE_TERMS_FILE, ce._private_terms = f'{TMP}/absent.txt', None
    try:
        assert ce.safe_external_query('attention sink dynamics in transformers', require_terms=True)[0] is False
        assert ce.safe_external_query('attention sink dynamics in transformers')[0] is True
    finally:
        ce.Config.PRIVATE_TERMS_FILE, ce._private_terms = old, None


def test_followup_prompt_never_sees_internal_results():
    fake = Calls(reply='TOPIC: Alice memory consolidation patterns\nTOPIC: sparse autoencoder features in late layers')
    undo = [patch(ce, 'ask_llm', fake.llm)]
    try:
        papers = [{'source': 'internal/tgs-rag', 'title': 'MEMORY-TEXT', 'abstract': 'MEMORY-TEXT private', 'year': ''},
                  {'source': 'arxiv', 'title': 'Sparse features', 'abstract': 'about features', 'year': '2025'}]
        out = ce.generate_followups({'id': 'x', 'topic': 'features'}, papers)
        assert 'MEMORY-TEXT' not in fake.calls[0], 'internal text reached the prompt'
        assert out == ['sparse autoencoder features in late layers'], out    # the private-term line was dropped
        fake.calls.clear()
        assert ce.generate_followups({'id': 'y', 'topic': 't'}, papers[:1]) == [] and not fake.calls
    finally:
        [u() for u in undo]


def test_summary_goes_to_the_api_built_from_external_papers():
    fake = Calls(status=200)
    undo = [patch(ce.requests, 'post', fake.post)]
    ce.Config.MEMORY_API_URL, ce.Config.MEMORY_API_KEY = 'http://127.0.0.1:9', 'k-test'
    try:
        papers = [{'source': 'internal/tgs-rag', 'title': 'MEMORY-TEXT', 'year': ''},
                  {'source': 'arxiv', 'title': 'Real paper', 'year': '2024'}]
        ce.store_in_memory({'id': 'i1', 'topic': 'T', 'requester': 'memory_scan'}, papers)
        url, kw = fake.calls[0]
        assert url == 'http://127.0.0.1:9/api/memories'
        assert kw['headers'] == {'Authorization': 'Bearer k-test'}
        body = kw['json']
        assert 'MEMORY-TEXT' not in body['content'] and 'Real paper' in body['content'], body['content']
        assert body['memory_type'] == 'research' and 'curiosity-engine' in body['tags']
        fake.calls.clear()
        ce.Config.MEMORY_API_KEY = ''
        ce.store_in_memory({'id': 'i2', 'topic': 'T', 'requester': 'x'}, papers)
        assert not fake.calls, 'stored without a key'
    finally:
        ce.Config.MEMORY_API_URL = ce.Config.MEMORY_API_KEY = ''
        [u() for u in undo]


def test_graph_ingestion_is_opt_in():
    fake = Calls(payload={'count': 1})
    undo = [patch(ce.requests, 'post', fake.post)]
    try:
        papers = [{'source': 'arxiv', 'title': 'P', 'abstract': 'a', 'year': '2024', 'authors': ['A']}]
        ce.Config.GRAPH_URL = ''
        ce._ingest_to_graph(papers)
        assert not fake.calls, f'posted without opt-in: {fake.calls[0][0]}'
        ce.Config.GRAPH_URL = 'http://127.0.0.1:9'
        ce._ingest_to_graph(papers)
        assert fake.calls[0][0] == 'http://127.0.0.1:9/index'
    finally:
        ce.Config.GRAPH_URL = ''
        [u() for u in undo]


def test_wiki_needs_a_password_from_the_environment():
    fake = Calls(status=500)
    undo = [patch(ce.requests, 'post', fake.post), patch(ce, 'WIKI_PASS', '')]
    try:
        ce._search_internal('anything')
        assert [u for u, _ in fake.calls if '/api/query' in u] == [], 'wiki searched without WIKI_PASS'
    finally:
        [u() for u in undo]


def test_ollama_think_field_and_think_stripping():
    fake = Calls(payload={'response': '<think>hidden</think>  answer'})
    undo = [patch(ce.requests, 'post', fake.post), patch(ce.Config, 'OLLAMA_NUM_CTX', ce.Config.OLLAMA_NUM_CTX)]
    old = ce.Config.OLLAMA_MODEL
    try:
        ce.Config.OLLAMA_MODEL = 'qwen3:4b'
        assert ce.ask_ollama('p') == 'answer'
        assert fake.calls[-1][1]['json']['think'] is False
        assert fake.calls[-1][1]['json']['options']['num_ctx'] == 2048     # one size for every consumer of the model: no reloads
        assert 'format' not in fake.calls[-1][1]['json']
        ce.ask_ollama('p', schema=ce.TOPICS_SCHEMA)
        assert fake.calls[-1][1]['json']['format'] == ce.TOPICS_SCHEMA
        ce.Config.OLLAMA_MODEL = 'mistral:7b'
        ce.ask_ollama('p')
        assert 'think' not in fake.calls[-1][1]['json']
        ce.Config.OLLAMA_NUM_CTX = 4096
        ce.ask_ollama('p')
        assert fake.calls[-1][1]['json']['options']['num_ctx'] == 4096
    finally:
        ce.Config.OLLAMA_MODEL = old
        [u() for u in undo]


def test_recover_stuck():
    iso = lambda t: ce.datetime.fromtimestamp(t, ce.timezone.utc).isoformat()
    q = [{'id': 'legacy', 'status': 'in_progress', 'priority': 3},
         {'id': 'fresh', 'status': 'in_progress', 'priority': 3, 'started_at': iso(NOW - 3600)},
         {'id': 'twice', 'status': 'in_progress', 'priority': 3, 'started_at': iso(NOW - 7 * 3600), 'attempts': 2},
         {'id': 'done', 'status': 'completed', 'priority': 3}]
    assert ce.recover_stuck(q) == 2
    by = {i['id']: i for i in q}
    assert (by['legacy']['status'], by['legacy']['priority'], by['legacy']['attempts']) == ('pending', 5, 1)
    assert by['fresh']['status'] == 'in_progress'
    assert by['twice']['status'] == 'failed'
    assert by['done']['status'] == 'completed'


def test_report_survives_internal_only_results():
    item = {'id': 'r', 'topic': 't', 'requester': 'x', 'priority': 3, 'depth': 0,
            'search_params': {'databases': ['arxiv']}}
    rep = ce.generate_report(item, [{'source': 'internal/tgs-rag', 'title': 'x', 'abstract': 'y', 'year': '',
                                     'authors': ['Internal knowledge base'], 'url': '', 'citation_count': 0}])
    assert 'Year range: n/a' in rep


def test_topic_parsing_drops_echoed_instructions():
    # The four lines qwen3:4b's first live scan queued on 2026-10-05, all of them its own instructions echoed back
    prompt = ("Extract up to 5 concrete research topics from these fragments. Each should be:\n"
              "- A specific, searchable academic query\n"
              "- Something genuinely unresolved (not already answered in the fragment)\n"
              "- Worth investigating with academic literature\n"
              "- Free of personal names, project names, places and anything private: general academic terms only\n"
              "- I wonder whether attention sinks carry positional information\n")
    raw = ("A specific, searchable academic query\n"                                  # no TOPIC: prefix
           "TOPIC: Something genuinely unresolved (not already answered in the fragment)\n"
           "TOPIC: Worth investigating with academic literature\n"
           "TOPIC: free of personal names, project names, places and anything private\n"
           "TOPIC: <a specific academic query>\n"
           "TOPIC: attention sinks carry positional information\n"                     # copied from a fragment
           "2. TOPIC: sparse autoencoder feature splitting across model scale\n")
    assert ce._parse_topics(raw, prompt) == ['sparse autoencoder feature splitting across model scale']


def test_json_topics_drop_echoes_too():
    prompt = ("- A specific, searchable academic query\n- Worth investigating with academic literature\n"
              'Reply with JSON only, in the form {"topics": ["...", "..."]}.')
    raw = ('{"topics": ["A specific, searchable academic query", "...", "worth investigating with academic literature",'
           ' "sparse autoencoder feature splitting across model scale"]}')
    assert ce._parse_topics(raw, prompt) == ['sparse autoencoder feature splitting across model scale']
    assert ce._parse_topics('We are given memory fragments. Let us analyze them.', prompt) == []   # the 10-05 reply


def test_scan_interval_is_persisted():
    ce._save_state({'last_memory_scan': time.time()})
    undo = [patch(ce, '_memory_fragments_sqlite', lambda *a, **k: (_ for _ in ()).throw(AssertionError('scanned')))]
    ce.Config.MEMORY_DB = fixture_db()
    try:
        ce.scan_memories_for_threads()
    finally:
        [u() for u in undo]


def test_run_once_seeds_from_memory_and_processes():
    ce._save_state({})
    reset_queue()
    ce.Config.MEMORY_DB = fixture_db()
    fake = Calls(reply='1. TOPIC: attention sink positional information in transformers\n- TOPIC: Alice consolidation habits\nNONE')
    processed = []

    def fake_process(item):
        processed.append(item['topic'])
        item['status'] = 'completed'
        return True
    undo = [patch(ce, 'ask_llm', fake.llm), patch(ce, 'process_one', fake_process),
            patch(ce, 'is_semantically_duplicate', lambda *a: False)]
    try:
        ce.run_once()
        q = ce.load_queue()
        assert [(i['topic'], i['requester'], i['status']) for i in q] == \
            [('attention sink positional information in transformers', 'memory_scan', 'completed')], q
        assert processed == ['attention sink positional information in transformers']
        assert ce._load_state()['last_scan_stats']['rejected'] == 1
        assert 'HUMAN-TURN' not in fake.calls[0] and 'MINE-RESEARCH' in fake.calls[0]
        assert fake.kw.get('schema') is ce.TOPICS_SCHEMA, 'the scan did not ask for the schema' 
    finally:
        [u() for u in undo]


ASKER = """# Who is asking

The asker is an AI agent that writes about its own memory.
- "compaction" means its context window being summarized.

## Fields

- context window summarization in large language models
- steering vectors and representation engineering in language models

## Not these

- event cameras, sensors and signal processing
- industrial control systems and operational technology networks

## Requesters

Queue requesters that are the asker.
- my-agent
"""


def set_asker(text):
    """Point the engine at an asker file holding text (None: no file), clearing its caches."""
    path = ''
    if text is not None:
        path = f'{TMP}/asker-{time.time_ns()}.md'
        with open(path, 'w') as f:
            f.write(text)
    ce.Config.ASKER_FILE = path
    ce._asker_cache = ce._asker_emb = None


def test_asker_file_is_read_in_four_parts():
    set_asker(ASKER)
    try:
        a = ce._asker()
        assert a['profile'].startswith('The asker is an AI agent') and '"compaction" means' in a['profile'], a
        assert a['fields'] == ['context window summarization in large language models',
                               'steering vectors and representation engineering in language models'], a
        assert a['not'] == ['event cameras, sensors and signal processing',
                            'industrial control systems and operational technology networks'], a
        assert a['requesters'] == ['my-agent'] and 'Queue requesters' not in a['profile'], a
        set_asker(None)
        assert ce._asker() == {'profile': '', 'fields': [], 'not': [], 'requesters': []}
        assert ce.asker_fit('any topic at all') is None
    finally:
        set_asker(None)


def test_abbreviations_are_spelled_out_before_scoring():
    assert ce._spelled_out('Steering in LLMs and an LLM, KV caches, RL, MoE, RAG, GNNs, TTS') == \
        ('Steering in large language models and an large language model, key-value caches, reinforcement learning, '
         'mixture-of-experts, retrieval-augmented generation, graph neural network, text-to-speech')
    assert ce._spelled_out('LLMSecurity and SKV stay') == 'LLMSecurity and SKV stay'


def test_scan_prompt_carries_the_asker():
    ce.Config.MEMORY_DB = fixture_db()
    fake = Calls(reply='{"topics": []}')
    undo = [patch(ce, 'ask_llm', fake.llm)]
    try:
        for text, want in ((ASKER, True), (None, False)):
            set_asker(text)
            ce._save_state({})
            fake.calls.clear()
            ce.scan_memories_for_threads()
            p = fake.calls[0]
            got = 'The asker is an AI agent' in p and 'steering vectors and representation' in p and ce.ASKER_RULE in p
            assert got == want, p[:400]
    finally:
        [u() for u in undo]
        set_asker(None)


def test_followups_keep_the_asker_and_root_in_my_threads_only():
    fake = Calls(reply='{"topics": []}')
    undo = [patch(ce, 'ask_llm', fake.llm)]
    papers = [{'source': 'arxiv', 'title': 'A paper', 'abstract': 'about something', 'year': '2025'}]
    try:
        set_asker(ASKER)
        reset_queue([
            {'id': 'r1', 'topic': 'ROOT-MINE what summarizing a context window does to an agent',
             'requester': 'memory_scan', 'status': 'completed'},
            {'id': 'c1', 'topic': 'a child topic of mine', 'requester': 'curiosity_engine', 'status': 'in_progress',
             'parent_id': 'r1', 'depth': 1},
            {'id': 'r2', 'topic': 'ROOT-THEIRS a question somebody asked', 'requester': 'user', 'status': 'completed'},
            {'id': 'c2', 'topic': 'a child topic of theirs', 'requester': 'curiosity_engine', 'status': 'in_progress',
             'parent_id': 'r2', 'depth': 1},
        ])
        q = {i['id']: i for i in ce.load_queue()}
        ce.generate_followups(q['c1'], papers)
        mine = fake.calls[-1]
        assert 'Who is asking' in mine and 'ROOT-MINE' in mine and ce.ASKER_RULE in mine, mine[:600]
        ce.generate_followups(q['c2'], papers)
        theirs = fake.calls[-1]
        assert 'Who is asking' not in theirs and 'ROOT-' not in theirs and ce.ASKER_RULE not in theirs, theirs[:600]
    finally:
        [u() for u in undo]
        set_asker(None)


def test_gate_skips_only_topics_the_engine_wrote_in_my_threads():
    set_asker(ASKER)
    off = {'off-mine', 'off-person', 'off-child-of-person', 'off-child-of-my-request'}
    processed = []

    def fake_process(item):
        processed.append(item['id'])
        item['status'] = 'completed'
        return True
    old_n = ce.Config.ITEMS_PER_RUN
    undo = [patch(ce, 'asker_fit', lambda t: -0.3 if t in off else 0.3), patch(ce, 'process_one', fake_process),
            patch(ce, 'scan_memories_for_threads', lambda: None)]
    try:
        reset_queue([
            {'id': 'a', 'topic': 'off-mine', 'requester': 'memory_scan', 'status': 'pending', 'priority': 3},
            {'id': 'b', 'topic': 'off-person', 'requester': 'user', 'status': 'pending', 'priority': 3},
            {'id': 'r', 'topic': 'somebody asked this', 'requester': 'user', 'status': 'completed', 'priority': 3},
            {'id': 'c', 'topic': 'off-child-of-person', 'requester': 'curiosity_engine', 'status': 'pending',
             'priority': 3, 'parent_id': 'r'},
            {'id': 'd', 'topic': 'on-mine', 'requester': 'memory_scan', 'status': 'pending', 'priority': 3},
            {'id': 'n', 'topic': 'a question I queued myself', 'requester': 'my-agent', 'status': 'completed', 'priority': 3},
            {'id': 'e', 'topic': 'off-child-of-my-request', 'requester': 'curiosity_engine', 'status': 'pending',
             'priority': 3, 'parent_id': 'n'},
        ])
        ce.Config.ITEMS_PER_RUN = 4
        ce.run_once()
        q = {i['id']: i for i in ce.load_queue()}
        assert q['a']['status'] == 'skipped' and 'margin -0.30' in q['a']['skip_reason'], q['a']
        assert q['e']['status'] == 'skipped', q['e']
        assert processed == ['b', 'c', 'd'], processed
    finally:
        [u() for u in undo]
        ce.Config.ITEMS_PER_RUN = old_n
        set_asker(None)


def test_scan_gate_drops_off_asker_topics():
    ce.Config.MEMORY_DB = fixture_db()
    ce._save_state({})
    reset_queue()
    fake = Calls(reply='{"topics": ["off-field memory topic about sensors", "on-field memory topic about context windows"]}')
    undo = [patch(ce, 'ask_llm', fake.llm), patch(ce, 'asker_fit', lambda t: -0.3 if t.startswith('off') else 0.3),
            patch(ce, 'is_semantically_duplicate', lambda *a: False)]
    try:
        set_asker(ASKER)
        ce.scan_memories_for_threads()
        assert [i['topic'] for i in ce.load_queue()] == ['on-field memory topic about context windows'], ce.load_queue()
        assert ce._load_state()['last_scan_stats']['off_asker'] == 1
    finally:
        [u() for u in undo]
        set_asker(None)


def test_followup_gate_drops_off_asker_topics_in_my_threads_only():
    fake = Calls(reply='{"topics": ["off-field follow-up question here", "on-field follow-up question here"]}')
    undo = [patch(ce, 'ask_llm', fake.llm), patch(ce, 'asker_fit', lambda t: -0.3 if t.startswith('off') else 0.3)]
    papers = [{'source': 'arxiv', 'title': 'A paper', 'abstract': 'about something', 'year': '2025'}]
    try:
        set_asker(ASKER)
        reset_queue([
            {'id': 'r1', 'topic': 'a root of mine', 'requester': 'memory_scan', 'status': 'completed'},
            {'id': 'c1', 'topic': 'a child of mine', 'requester': 'curiosity_engine', 'status': 'in_progress',
             'parent_id': 'r1', 'depth': 1},
            {'id': 'r2', 'topic': 'a root somebody asked for', 'requester': 'user', 'status': 'completed'},
            {'id': 'c2', 'topic': 'a child of theirs', 'requester': 'curiosity_engine', 'status': 'in_progress',
             'parent_id': 'r2', 'depth': 1},
        ])
        q = {i['id']: i for i in ce.load_queue()}
        assert ce.generate_followups(q['c1'], papers) == ['on-field follow-up question here']
        assert ce.generate_followups(q['c2'], papers) == ['off-field follow-up question here',
                                                          'on-field follow-up question here']
    finally:
        [u() for u in undo]
        set_asker(None)


def test_asker_margin_on_real_topics():
    """The 10-05 measurement in small. Needs the embedding model (CPU, from the local cache)."""
    set_asker(ASKER)
    try:
        if ce._get_st_model() is None:
            print('        (no embedding model: nothing measured)')
            return
        drift = ce.asker_fit('How does memory compaction in event cameras affect temporal resolution in high-latency '
                             'industrial networks?')
        mine = ce.asker_fit("Safety training's influence on steering mechanisms in LLMs")
        assert drift < 0 < mine, (drift, mine)
    finally:
        set_asker(None)


# ------------------------------------------------------------------------------------------------------------ runner

if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'  ok    {name}')
        except Exception:
            failed += 1
            print(f'  FAIL  {name}\n' + ''.join('        ' + l for l in traceback.format_exc().splitlines(True)[-3:]))
    print(f'{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
