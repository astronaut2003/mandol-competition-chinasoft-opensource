"""Real SQLite/API; explicitly injected model/HTTP protocols, not live E2E."""
import json
from types import SimpleNamespace
import httpx
import pytest

from mandol_competition.chinasoft.web_query_planner import plan_web_query, distinct
from mandol_competition.chinasoft.models import ModelError
from tests.chinasoft.test_v33 import env, provider


@pytest.mark.parametrize('question,anchors', [
    ('Qwen3-Embedding-0.6B 的模型规格是什么？', ['Qwen3-Embedding-0.6B','模型规格']),
    ('DeepSeek 最近发布了什么模型？', ['DeepSeek','2026','最近']),
    ('比较 GPT-4o-mini 与 GPT-4o 的能力差异', ['GPT-4o-mini','GPT-4o','能力']),
    ('查找 2025 年关于 Agent Memory 的研究论文', ['2025','Agent Memory','论文']),
    ('哪些开源向量数据库支持混合检索和持久化？', ['开源','混合检索','持久化']),
    ('SPLADE-v3 与 BM25 在 retrieval 中有什么区别？', ['SPLADE-v3','BM25','retrieval']),
])
def test_clear_questions_preserve_identifiers_and_constraints_without_llm(env,question,anchors,monkeypatch):
    p,*_=env
    monkeypatch.setattr(p.generation,'complete',lambda *a,**k:pytest.fail('clear query must not call LLM'))
    plan=plan_web_query(p,question,request_date='2026-10-19')
    assert all(a in plan['primary_query'] for a in anchors)
    assert plan['model_calls']==0 and len(plan['primary_query'])<=128


def test_context_model_uses_bounded_existing_turns_and_preserves_original(env,monkeypatch):
    p,*_=env;captured=[]
    def complete(messages,**kwargs):
        captured.append((messages,kwargs))
        return {'content':json.dumps({'primary_query':'Qdrant 最新更新 2026','additional_queries':[], 'language':'mixed','intent':'latest'}),'usage':{'total_tokens':40}}
    monkeypatch.setattr(p.generation,'complete',complete)
    turns=[{'question':'介绍一下 Qdrant。','answer':'A previous description'}]*8
    question='它最近有什么更新？';plan=plan_web_query(p,question,turns,request_date='2026-10-19')
    assert plan['primary_query']=='Qdrant 最新更新 2026' and plan['planner_method']=='context_model'
    value=json.loads(captured[0][0][-1]['content']);assert len(value['recent_turns'])==3 and value['question']==question
    assert captured[0][1]['attempts']==1 and captured[0][1]['foreground'] and captured[0][1]['timeout']==4


@pytest.mark.parametrize('failure',['timeout','bad_json','lost_version','new_version','bad_query'])
def test_rewrite_failures_fall_back_without_losing_antecedent(env,monkeypatch,failure):
    p,*_=env
    def complete(*a,**k):
        if failure=='timeout':raise ModelError('model_timeout','injected timeout')
        if failure=='bad_json':return {'content':'{invalid'}
        query={'lost_version':'Qwen 最新更新','new_version':'Qwen3-Embedding-0.6B V99.9 2026','bad_query':'line\nline'}[failure]
        return {'content':json.dumps({'primary_query':query,'additional_queries':[]})}
    monkeypatch.setattr(p.generation,'complete',complete)
    plan=plan_web_query(p,'Qwen3-Embedding-0.6B 它最近更新了吗？',[{'question':'介绍 Qwen3-Embedding-0.6B','answer':''}],request_date='2026-10-19')
    assert plan['planner_method']=='deterministic_fallback' and 'Qwen3-Embedding-0.6B' in plan['primary_query']
    assert plan['model_calls']==1


def test_legacy_override_and_query_dedup_budgets(env,monkeypatch):
    p,*_=env
    monkeypatch.setattr(p.generation,'complete',lambda *a,**k:pytest.fail('legacy query should bypass rewrite'))
    plan=plan_web_query(p,'讨论当前附件'*50,explicit='SQLite WAL docs')
    assert plan['primary_query']=='SQLite WAL docs' and plan['planner_method']=='legacy_explicit'
    assert distinct(['Qdrant latest updates','Qdrant latest updates!','Qdrant deployment memory'],2)==['Qdrant latest updates','Qdrant deployment memory']


def test_complementary_queries_share_result_budget_and_real_url_identity(env):
    seen=[]
    def handle(request):
        seen.append(json.loads(request.content))
        suffix='a' if len(seen)==1 else 'b'
        return httpx.Response(200,json={'results':[{'title':'shared','url':'https://docs.python.org/3/','content':'shared docs'}, {'title':suffix,'url':'https://example.com/'+suffix,'content':suffix}]})
    web,o,w=provider(env,handle);stages=[]
    trace=web.search(o,w,'first query',additional_queries=['second query','third query'],progress=lambda stage,data:stages.append(stage))
    assert len(seen)==2 and trace['calls']=={'search':2,'fetch':0} and stages==['search','search']
    assert len(trace['results'])==3 and len({r['url'] for r in trace['results']})==3
    assert trace['context_chars']<=env[0].settings.search_context_chars
    assert [r['payload']['query'] for r in trace['requests']]==['first query','second query']


def test_partial_search_failure_keeps_real_evidence_and_actual_counts(env):
    def handle(request):
        if json.loads(request.content)['query']=='second':raise httpx.ReadTimeout('injected',request=request)
        return httpx.Response(200,json={'results':[{'title':'real protocol fixture','url':'https://example.com/','content':'first result'}]})
    web,o,w=provider(env,handle);trace=web.search(o,w,'first',additional_queries=['second'])
    assert trace['status']=='partial' and trace['calls']['search']==2 and trace['warnings']==['web_timeout'] and len(trace['results'])==1


def chat_protocol(env,monkeypatch,intent='general'):
    p,c,o,w,s=env;requests=[];messages=[]
    def handler(r):
        requests.append(json.loads(r.content));return httpx.Response(200,json={'results':[{'title':'docs','url':'https://www.sqlite.org/wal.html','content':'WAL documentation'}]})
    provider(env,handler)
    from mandol_competition.chinasoft import conversation
    monkeypatch.setattr(conversation,'plan_turn',lambda *a,**k:{'intent':intent,'retrieve':False,'use_attachments':False,'query':''})
    async def stream(values,**kwargs):messages.extend(values);yield '已找到资料 [W1]。' if requests else '你好。'
    monkeypatch.setattr(p.generation,'stream',stream)
    return p,c,w,s,requests,messages


@pytest.mark.parametrize('enabled,intent',[(False,'general'),(True,'social'),(True,'identity')])
def test_web_off_or_unneeded_never_plans_or_searches(env,monkeypatch,enabled,intent):
    p,c,w,s,requests,_=chat_protocol(env,monkeypatch,intent)
    from mandol_competition.chinasoft import web_query_planner
    monkeypatch.setattr(web_query_planner,'plan_web_query',lambda *a,**k:pytest.fail('unexpected planning'))
    r=c.post('/api/chat',json={'workspace_id':w,'session_id':s,'request_id':'off','content':'你好','web_search':enabled})
    assert 'event: done' in r.text and not requests


def test_auto_api_without_second_query_saves_original_and_replays_once(env,monkeypatch):
    p,c,w,s,requests,messages=chat_protocol(env,monkeypatch)
    data={'workspace_id':w,'session_id':s,'request_id':'auto-web','content':'请查找 SQLite WAL 文档','web_search':True}
    r=c.post('/api/chat',json=data);assert 'event: done' in r.text and 'event: error' not in r.text
    assert '正在组织搜索查询' in r.text and '正在搜索网页' in r.text and '正在生成回答' in r.text
    assert requests[0]['query']=='SQLite WAL 文档'
    history=c.get('/api/chat/history',params={'workspace_id':w,'session_id':s}).json()['items'][0]
    assert history['question']==data['content'] and history['route']['web']['plan']['primary_query']!=data['content']
    assert history['route']['web_sources'][0]['url']=='https://www.sqlite.org/wal.html'
    assert data['content'] in json.dumps(messages,ensure_ascii=False)
    replay=c.post('/api/chat',json=data);assert '"cached":true' in replay.text and len(requests)==1


def test_failed_generation_retry_reuses_completed_search_and_plan(env,monkeypatch):
    p,c,w,s,requests,_=chat_protocol(env,monkeypatch)
    async def fail(*a,**k):
        yield 'partial'
        raise ModelError('injected','protocol failure')
    monkeypatch.setattr(p.generation,'stream',fail)
    data={'workspace_id':w,'session_id':s,'request_id':'retry-web','content':'SQLite WAL docs','web_search':True}
    assert 'event: error' in c.post('/api/chat',json=data).text
    async def good(*a,**k):yield 'WAL [W1]。'
    monkeypatch.setattr(p.generation,'stream',good)
    r=c.post('/api/chat',json=data)
    assert 'event: done' in r.text and '"reused_for_same_request":true' in r.text and len(requests)==1


def test_api_search_failure_is_reported_without_fake_web_sources(env,monkeypatch):
    p,c,w,s,_,_=chat_protocol(env,monkeypatch)
    provider(env,lambda r:(_ for _ in ()).throw(httpx.ReadTimeout('injected',request=r)))
    async def stream(*a,**k):yield '本轮联网未获得网页依据。'
    monkeypatch.setattr(p.generation,'stream',stream)
    r=c.post('/api/chat',json={'workspace_id':w,'session_id':s,'request_id':'timeout','content':'SQLite WAL docs','web_search':True})
    assert 'web_timeout' in r.text and 'event: done' in r.text
    h=c.get('/api/chat/history',params={'workspace_id':w,'session_id':s}).json()['items'][0]
    assert h['route']['web']['calls']['search']==1 and not h['route']['web_sources']


def test_attachment_question_needs_no_second_query_and_keeps_file_evidence(env,monkeypatch):
    p,c,w,s,requests,_=chat_protocol(env,monkeypatch)
    from mandol_competition.chinasoft import conversation
    monkeypatch.setattr(conversation,'plan_turn',lambda *a,**k:{'intent':'attachments','retrieve':False,'use_attachments':True,'query':''})
    monkeypatch.setattr(p.generation,'complete',lambda *a,**k:{'content':json.dumps({'primary_query':'Qdrant documentation','additional_queries':[]})})
    upload=c.post('/api/attachments',data={'workspace_id':w,'session_id':s},files={'file':('topic.md',b'# Qdrant\nDocumentation topic.','text/markdown')})
    assert upload.status_code==201
    payload={'workspace_id':w,'session_id':s,'request_id':'with-file','content':'请查找这个附件主题的文档','attachment_ids':[upload.json()['id']],'web_search':True}
    response=c.post('/api/chat',json=payload)
    assert 'event: done' in response.text and requests[0]['query']=='Qdrant documentation'
    h=c.get('/api/chat/history',params={'workspace_id':w,'session_id':s}).json()['items'][0]
    assert h['route']['web']['plan']['attachment_previews']==1 and h['route']['attachment_sources'][0]['path']=='topic.md'


def test_multigoal_planner_keeps_main_goals_and_limits_complementary_queries(env,monkeypatch):
    p,*_=env
    result={'primary_query':'Qwen3 Embedding BGE-M3 效果 模型规模 部署成本','additional_queries':['Qwen3 Embedding BGE-M3 benchmark','Qwen3 Embedding BGE-M3 benchmark!']}
    monkeypatch.setattr(p.generation,'complete',lambda *a,**k:{'content':json.dumps(result)})
    plan=plan_web_query(p,'比较 Qwen3 Embedding 和 BGE-M3 的效果、模型规模以及部署成本')
    assert plan['planner_method']=='context_model' and plan['additional_queries']==['Qwen3 Embedding BGE-M3 benchmark']


def test_incomplete_complex_rewrite_cannot_discard_a_requested_goal(env,monkeypatch):
    p,*_=env
    monkeypatch.setattr(p.generation,'complete',lambda *a,**k:{'content':json.dumps({'primary_query':'Qdrant 最近更新','additional_queries':[]})})
    plan=plan_web_query(p,'Qdrant 最近更新与替代方案有哪些？')
    assert plan['planner_method']=='deterministic_fallback' and '替代' in plan['primary_query'] and 'query_goal_lost' in plan['reason_codes']


def test_processing_request_cannot_start_duplicate_search(env,monkeypatch):
    import concurrent.futures,threading
    p,c,w,s,_,_=chat_protocol(env,monkeypatch);started=threading.Event();release=threading.Event();calls=[]
    def handler(r):
        calls.append(1);started.set();assert release.wait(5)
        return httpx.Response(200,json={'results':[{'title':'docs','url':'https://www.sqlite.org/wal.html','content':'WAL documentation'}]})
    provider(env,handler)
    data={'workspace_id':w,'session_id':s,'request_id':'concurrent','content':'SQLite WAL docs','web_search':True}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(c.post,'/api/chat',json=data)
        assert started.wait(5)
        try:assert c.post('/api/chat',json=data).status_code==409
        finally:release.set()
        assert 'event: done' in future.result().text
    assert len(calls)==1
