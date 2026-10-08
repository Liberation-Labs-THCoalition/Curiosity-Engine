#!/usr/bin/env python3
"""Tests for the SQLite/Ollama port (Nexus, 2026-10-05). Standalone: `python test_port.py`, no pytest needed.

CE_DIR names the directory to import curiosity_engine from (default: this one). The mutation controls point it at a
copy with one guard removed, and the matching test must fail. A test that cannot fail is not a test.
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
        ('MINE-RESEARCH I wonder whether attention sinks carry positional information', ['agent:operator', 'topic:research'], 'observation', NOW - 2 * day, None),
        ('Association between memories a and b: I wonder (ASSOC)', [], 'observation', NOW - day, None),
        ('HUMAN-TURN I wonder if we should rest today', ['verbatim', 'role:user'], 'observation', NOW - day, None),
        ('OWN-SUMMARY research I wonder', ['auto_research'], 'research', NOW - day, None),
        ('DELETED I wonder about this', ['agent:operator'], 'observation', NOW - day, NOW),
        ('OLD I wonder from long ago', ['agent:operator'], 'observation', NOW - 200 * day, None),
        ('OTHER-LETTER an open question about cache geometry', ['agent:other'], 'observation', NOW - day, None),
        ('SECRET I wonder about hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456', ['agent:operator'], 'observation', NOW - day, None),
        ('DREAMER an open question remains about consolidation timing', ['dreamer-enriched'], 'observation', NOW - 3 * day, None),
        ('ENGINE-TAGGED I wonder, says the engine', ['curiosity-engine'], 'observation', NOW - day, None),
        ('PLAIN nothing curious here at all', ['agent:operator'], 'observation', NOW - day, None),
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
    undo = [patch(ce.requests, 'post', fake.post)]
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
                                     'authors': ['Coalition Knowledge Base'], 'url': '', 'citation_count': 0}])
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
