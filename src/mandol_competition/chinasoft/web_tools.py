"""Separate DeepSeek JSON tools; only bounded planned queries are sent.

Search evidence remains in this chat's bounded route, outside canonical facts.
Fetch is opt-in deployment configuration, restricted to curated public domains;
the actual request is sent to the provider, never to arbitrary URLs locally.
"""
from datetime import datetime,timezone
import ipaddress,json,os,re,socket,threading,time
from urllib.parse import urlsplit,urlunsplit
import httpx
from .models import ModelError
from .store import AppError,now


def public_url(value,*,resolve=False,hosts=None):
    if not isinstance(value,str) or len(value)>2048 or any(ord(c)<33 for c in value):return None
    try:
        u=urlsplit(value);host=u.hostname or ''
        if u.scheme not in ('http','https') or not host or u.username or u.password or u.port not in (None,80,443):return None
        if host=='localhost' or host.endswith(('.local','.localhost','.internal','.test','.invalid')) or '.' not in host and ':' not in host:return None
        if hosts is not None and (u.scheme!='https' or host not in hosts or u.query):return None
        try:
            a=ipaddress.ip_address(host)
            if not a.is_global or a.is_multicast or a.is_reserved:return None
        except ValueError:pass
        if resolve:
            for row in socket.getaddrinfo(host,u.port or 443,type=socket.SOCK_STREAM):
                a=ipaddress.ip_address(row[4][0]);a=a.ipv4_mapped if isinstance(a,ipaddress.IPv6Address) and a.ipv4_mapped else a
                if not a.is_global or a.is_multicast or a.is_reserved:return None
        return urlunsplit((u.scheme,u.netloc,u.path,u.query,''))
    except (ValueError,OSError):return None


def search_query(content,explicit='',*,attachments=False):
    from .attachments import SENSITIVE
    query=explicit.strip() or content.strip()
    if not 1<=len(query)<=128 or '\n' in query or '```' in query or SENSITIVE.search(query) or re.search(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b',query):
        raise AppError('搜索查询须为 1–128 字，不能包含代码、凭据或邮箱',422,code='web_query_rejected')
    return query


class WebTools:
    def __init__(self,settings,store,transport=None,retry_sleep=time.sleep):
        self.settings,self.store=settings,store
        self.client=httpx.Client(timeout=settings.search_timeout,trust_env=False,transport=transport,follow_redirects=False,
            limits=httpx.Limits(max_connections=2,max_keepalive_connections=2))
        self.slots=threading.BoundedSemaphore(2);self.retry_sleep=retry_sleep

    def public(self):
        return {'enabled':self.settings.search_enabled,'configured':bool(os.getenv(self.settings.search_api_key_env)),
                'provider':'DeepSeek Search / Fetch JSON API','max_results':self.settings.search_max_results,'max_fetch':self.settings.search_max_fetch,
                'max_queries':self.settings.search_max_queries,'query_planner':'auto-web-query-v1',
                'privacy':'依据问题与必要的近期上下文自动组织搜索词；网页来源与长期记忆分别保存。'}

    def charge(self,owner,workspace):
        with self.store.state.transaction() as db:
            account=self.store.access.account_for(db,owner)
            if not account:raise ModelError('web_account_required','联网工具需要已认证账号')
            try:
                self.store.access.consume(db,account['id'],'web_per_day',86400000)
                self.store.access.consume(db,account['id'],'web_per_minute',60000)
                for key,maximum in (('__web_platform__',self.settings.web_requests_per_day),(account['id']+'@'+workspace,self.settings.web_workspace_per_day)):
                    used=self.store.access.used(db,key,'web_per_day',86400000)
                    if used>=maximum:raise AppError('联网工具调用预算已用尽',429,code='web_quota')
                    db.execute('INSERT INTO cs_usage_buckets VALUES (?,?,?,1) ON CONFLICT(account,action,window) DO UPDATE SET used=used+1',(key,'web_per_day',now()//86400000))
            except AppError as e:raise ModelError(e.code,e.message) from e

    def post(self,kind,payload,owner,workspace,trace):
        key=os.getenv(self.settings.search_api_key_env)
        if not self.settings.search_enabled or not key:raise ModelError('web_not_configured','联网工具未启用或独立搜索凭据未配置')
        if not self.slots.acquire(timeout=2):raise ModelError('web_busy','联网工具并发已满，请稍后重试',retryable=True)
        started=time.monotonic()
        deadline=started+self.settings.search_timeout
        try:
            for attempt in range(2):
                if time.monotonic()>=deadline:raise ModelError('web_timeout','联网请求已超过本次截止时间')
                self.charge(owner,workspace);trace['calls'][kind]+=1
                trace.setdefault('requests',[]).append({'kind':kind,'payload':dict(payload),'attempt':attempt+1})
                try:
                    with self.client.stream('POST',self.settings.search_base_url.rstrip('/')+'/tools/'+kind,
                        headers={'Authorization':'Bearer '+key},json=payload) as response:
                        status=response.status_code;trace['http_statuses'].append(status)
                        if status in (408,429) or status>=500:
                            if attempt==0:self.retry_sleep(.5);continue
                            raise ModelError('web_http_'+str(status),'联网工具限流或暂不可用',retryable=True)
                        if status!=200:raise ModelError('web_http_'+str(status),{401:'联网工具凭据未获授权',402:'联网工具额度不足',403:'网页抓取被安全策略拒绝',404:'未能抓取可用网页'}.get(status,'联网接口拒绝请求'))
                        body=bytearray()
                        for part in response.iter_bytes():
                            if time.monotonic()>deadline:raise ModelError('web_timeout','联网内容读取超过截止时间')
                            body.extend(part)
                            if len(body)>1024*1024:raise ModelError('web_response_budget','联网内容超过响应预算')
                        try:data=json.loads(body)
                        except (ValueError,RecursionError):raise ModelError('web_invalid_json','联网接口返回了无效 JSON')
                        if not isinstance(data,dict):raise ModelError('web_invalid_json','联网结果不是对象')
                        return data
                except httpx.TimeoutException as e:
                    raise ModelError('web_timeout','联网请求超时，本轮未获得完整网页依据') from e
                except httpx.TransportError as e:raise ModelError('web_connection','联网服务连接失败') from e
        finally:
            trace['timings'][kind+'_ms']=trace['timings'].get(kind+'_ms',0)+round((time.monotonic()-started)*1000,2)
            self.slots.release()

    def fetch(self,url,owner,workspace,trace):
        hosts={h.strip() for h in self.settings.search_fetch_hosts.split(',') if h.strip()}
        url=public_url(url,resolve=True,hosts=hosts)
        if not url:raise ModelError('web_fetch_url_rejected','抓取地址不在受控公网域名范围内')
        data=self.post('fetch',{'url':url},owner,workspace,trace)
        if not isinstance(data.get('content'),str) or not data['content'].strip():raise ModelError('web_fetch_empty','网页抓取没有返回正文')
        return {'title':str(data.get('title',''))[:300],'content':data['content'][:6000]}

    def search(self,owner,workspace,query,*,additional_queries=(),progress=None):
        self.store.workspace(owner,workspace)
        queries=list(dict.fromkeys(search_query(q) for q in [query,*additional_queries]))[:self.settings.search_max_queries]
        trace={'requested':True,'status':'processing','query':queries[0],'queries':queries,'calls':{'search':0,'fetch':0},'http_statuses':[],'timings':{},'warnings':[],'results':[],
               'retrieved_at':datetime.now(timezone.utc).isoformat(),'billable_successes_observed':0,'privacy':'planned_queries_only'}
        started=time.monotonic()
        def stage(name,**values):
            if progress:progress(name,values)
        try:
            try:
                with self.store.state.transaction() as db:self.store.access.check(db,owner,'storage_bytes',self.settings.search_context_chars*4+4096)
            except AppError as e:raise ModelError(e.code,e.message) from e
            batches=[]
            for index,search in enumerate(queries):
                stage('search',query=search,index=index+1)
                try:
                    data=self.post('search',{'query':search,'count':self.settings.search_max_results,'need_content':True},owner,workspace,trace)
                    values=data.get('results')
                    if not isinstance(values,list):raise ModelError('web_result_schema','搜索结果不符合官方结构')
                    trace['billable_successes_observed']+=int(bool(values));batches.append(values[:self.settings.search_max_results])
                except ModelError as e:
                    if not batches:raise
                    trace['warnings'].append(e.code);trace['partial_failure']=True;break
            # Round-robin complementary results, with a single global evidence
            # budget and URL identity. A second query cannot double context.
            seen=set();remaining=self.settings.search_context_chars
            for rank in range(self.settings.search_max_results):
                for batch in batches:
                    if rank>=len(batch) or len(trace['results'])>=self.settings.search_max_results:continue
                    value=batch[rank]
                    if not isinstance(value,dict):continue
                    url=public_url(value.get('url'));title=value.get('title');summary=value.get('summary') or value.get('snippet') or ''
                    if not url or url in seen or not isinstance(title,str) or not isinstance(summary,str):continue
                    content=value.get('content') or summary
                    if not isinstance(content,str):continue
                    excerpt=content[:min(2500,max(0,remaining-700))]
                    if not excerpt:continue
                    seen.add(url);remaining-=len(excerpt)+min(300,len(title))+len(url)+150
                    trace['results'].append({'id':'W'+str(len(trace['results'])+1),'kind':'web','title':title[:300],'url':url,
                        'site_name':str(value.get('site_name') or urlsplit(url).hostname)[:150],
                        'summary':summary[:500],'published_at':value.get('date') if isinstance(value.get('date'),str) else None,
                        'retrieved_at':trace['retrieved_at'],'excerpt':excerpt,'method':'search'})
            if self.settings.search_max_fetch and trace['results']:
                candidate=next((r for r in trace['results'] if public_url(r['url'],hosts=set(self.settings.search_fetch_hosts.split(',')))),None)
                if candidate:
                    stage('fetch',url=candidate['url'])
                    try:
                        page=self.fetch(candidate['url'],owner,workspace,trace);candidate['excerpt']=page['content'][:2500];candidate['method']='fetch';trace['billable_successes_observed']+=1
                    except ModelError as e:trace['warnings'].append(e.code)
            packed=[];remaining=self.settings.search_context_chars
            for row in trace['results']:
                fixed=len(json.dumps(dict(row,excerpt=''),ensure_ascii=False,separators=(',',':')))
                if remaining-fixed<80:break
                row['excerpt']=row['excerpt'][:remaining-fixed];remaining-=len(json.dumps(row,ensure_ascii=False,separators=(',',':')));packed.append(row)
            trace['results']=packed
            trace['context_chars']=self.settings.search_context_chars-remaining
            trace['status']=('partial' if trace.get('partial_failure') else 'completed') if trace['results'] else 'empty'
        except ModelError as e:trace.update(status='failed',error_code=e.code,message=e.message)
        trace['timings']['total_ms']=round((time.monotonic()-started)*1000,2)
        return trace

    def close(self):self.client.close()
