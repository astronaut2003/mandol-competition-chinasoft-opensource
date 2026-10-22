"""Public HTTPS repository snapshots. No git client, archive, hooks or LFS.

Resolve allowlisted hosts, reject every non-public address and pin the chosen IP
to a TLS connection with the original hostname/SNI. Redirects and proxies are
never followed. Provider response URLs are data, never connection targets.
"""
import base64,hashlib,http.client,ipaddress,json,os,re,shutil,socket,ssl,time
from pathlib import Path,PurePosixPath
from urllib.parse import urlsplit,quote,urlencode
from mandol_substrate.state.models import digest,canonical_json,Conflict
from .store import AppError,now

HOSTS={'api.github.com','raw.githubusercontent.com','gitlab.com'}
REPOSITORY_HOSTS={'github.com','gitlab.com'}
SHA=re.compile(r'^[0-9a-f]{40}$')
REMOTE_DIRS={'vendor','third_party','third-party','external','generated','.github','.gitlab','site-packages','fixtures','testdata','data','datasets'}


def validate_url(value,ref=''):
    if not isinstance(value,str) or any(ord(c)<33 or ord(c)>126 for c in value):raise AppError('仅支持标准公开 HTTPS Git 仓库地址',422,code='git_url_rejected')
    try:u=urlsplit(value);port=u.port
    except ValueError:raise AppError('无效仓库端口',422,code='git_url_rejected')
    if u.scheme!='https' or u.hostname not in REPOSITORY_HOSTS or u.username or u.password or port is not None or u.query or u.fragment or '\\' in value or '%' in u.path:
        raise AppError('仅允许 github.com 或 gitlab.com 的公开 HTTPS 仓库；不接受凭据、端口和跳转参数',422,code='git_url_rejected')
    path=u.path.rstrip('/').removesuffix('.git').strip('/')
    parts=path.split('/')
    if (u.hostname=='github.com' and len(parts)!=2) or not 2<=len(parts)<=5 or any(not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}',p) or p in ('.','..') or p.endswith('.') for p in parts):
        raise AppError('请输入仓库根地址，而非文件、分支页面或相对路径',422,code='git_url_rejected')
    if ref and (not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_./-]{0,127}',ref) or '..' in ref or '//' in ref or ref.endswith(('/','.lock'))):
        raise AppError('分支或 Tag 名称无效',422,code='git_ref_rejected')
    return 'https://'+u.hostname+'/'+path,ref


def public_addresses(host):
    if host not in HOSTS:raise AppError('远程目标不在平台白名单',422,code='git_host_rejected')
    addresses=[]
    for family,_,_,_,address in socket.getaddrinfo(host,443,type=socket.SOCK_STREAM):
        ip=ipaddress.ip_address(address[0]);check=ip.ipv4_mapped if isinstance(ip,ipaddress.IPv6Address) and ip.ipv4_mapped else ip
        if not check.is_global or check.is_multicast or check.is_reserved or check.is_unspecified:
            raise AppError('远程域名解析到非公开地址，已拒绝连接',422,code='git_ssrf_rejected')
        if address[0] not in addresses:addresses.append(address[0])
    if not addresses:raise AppError('无法解析仓库平台域名',503,code='git_dns_failed')
    return sorted(addresses,key=lambda x:':' in x)


class PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self,host,address,timeout):super().__init__(host,443,timeout=timeout,context=ssl.create_default_context());self.address=address
    def connect(self):
        deadline=time.monotonic()+self.timeout;last=None
        addresses=(self.address if isinstance(self.address,list) else [self.address])[:4]
        single_ipv4=sum(':' not in address for address in addresses)==1
        for address in addresses:
            remaining=deadline-time.monotonic()
            if remaining<=0:break
            connection=None
            try:
                connection=socket.create_connection((address,443),remaining if single_ipv4 and ':' not in address else min(3,remaining))
                if hasattr(connection,'settimeout'):connection.settimeout(max(.1,deadline-time.monotonic()))
                self.sock=self._context.wrap_socket(connection,server_hostname=self.host)
                if hasattr(self.sock,'settimeout'):self.sock.settimeout(self.timeout)
                return
            except OSError as exc:
                last=exc
                if connection:connection.close()
        if last:raise last
        raise TimeoutError('Validated public endpoints timed out')


class PublicHTTP:
    def __init__(self,deadline,check):self.deadline,self.check,self.requests,self.bytes=deadline,check,0,0

    def get(self,url,maximum=4*1024*1024,*,as_json=True):
        # Immutable, read-only requests may retry once within the same total
        # deadline. Do not discard a whole scan because one TLS connection stalls.
        for attempt in range(2):
            try:return self._get_once(url,maximum,as_json=as_json)
            except AppError as exc:
                if attempt or exc.code!='git_connection' or time.monotonic()+.25>=self.deadline:raise
                self.check();time.sleep(.2)

    def _get_once(self,url,maximum=4*1024*1024,*,as_json=True):
        self.check()
        if time.monotonic()>=self.deadline:raise AppError('公开仓库获取超时，原有索引保留',408,code='git_timeout')
        u=urlsplit(url)
        if u.scheme!='https' or u.hostname not in HOSTS or u.username or u.password or u.port or u.fragment:raise AppError('非法远程读取目标',422,code='git_host_rejected')
        try:addresses=public_addresses(u.hostname)
        except OSError:raise AppError('无法解析公开 Git 平台域名',503,code='git_dns_failed')
        connection=None
        try:
            # Exactly one validated DNS resolution per request. No subsequent DNS lookup.
            connection=PinnedHTTPS(u.hostname,addresses[:4],min(10,max(.1,self.deadline-time.monotonic())))
            self.requests+=1
            if self.requests>1100:raise AppError('远程请求数量超过预算',413,code='git_request_limit')
            connection.request('GET',u.path+('?' +u.query if u.query else ''),headers={'User-Agent':'Mandol-ChinaSoft-Public-Code/3.2','Accept':'application/json' if as_json else 'application/octet-stream','Accept-Encoding':'identity'})
            response=connection.getresponse()
            if 300<=response.status<400:raise AppError('仓库端点发生重定向，出于安全限制未跟随；请使用规范公开地址',422,code='git_redirect_rejected')
            if response.status!=200:
                raise AppError('公开 Git 平台暂不可用或仓库不是公开仓库（HTTP '+str(response.status)+'）',503 if response.status in (403,429) or response.status>=500 else 422,code='git_http_'+str(response.status))
            if response.getheader('Content-Encoding','identity')!='identity':raise AppError('拒绝压缩传输的仓库内容',422,code='git_encoding_rejected')
            length=response.getheader('Content-Length')
            if length and (not length.isdecimal() or int(length)>maximum):raise AppError('远程文件或元数据超过预算',413,code='git_response_limit')
            chunks=[];size=0
            while True:
                self.check()
                if time.monotonic()>=self.deadline:raise AppError('公开仓库获取超过总超时',408,code='git_timeout')
                part=response.read(min(65536,maximum-size+1))
                if not part:break
                size+=len(part);self.bytes+=len(part)
                if size>maximum:raise AppError('远程内容超过大小预算',413,code='git_response_limit')
                chunks.append(part)
            content=b''.join(chunks)
            if not as_json:return content
            try:return json.loads(content)
            except (ValueError,RecursionError):raise AppError('公开 Git 平台返回无效元数据',422,code='git_invalid_json')
        except (OSError,TimeoutError,http.client.HTTPException):
            raise AppError('公开 Git 平台连接失败或超时；未读取任何内部目标',503,code='git_connection')
        finally:
            if connection:connection.close()


class RemoteGit:
    def __init__(self,manager):self.manager,self.p,self.store,self.settings=manager,manager.platform,manager.store,manager.settings

    def register(self,owner,workspace,url,ref,name,request_id):
        self.store.workspace(owner,workspace)
        if not self.settings.remote_git_enabled:raise AppError('远程仓库导入已关闭',403)
        url,ref=validate_url(url,ref);key='repo_'+digest([owner,workspace,'public-https-v1',url,ref]);job_id='job_'+digest([owner,workspace,'repository',request_id])
        root=str(self.settings.data_dir/'remote-repositories'/key/'uninitialized')
        with self.p.state.transaction() as db:
            old=db.execute('SELECT payload FROM cs_jobs WHERE id=?',(job_id,)).fetchone()
            if old and json.loads(old['payload'])['repository_id']!=key:raise Conflict('相同导入请求对应不同仓库')
            if not old:
                self.store.access.check(db,owner,'repository_imports')
                self.store.access.check(db,owner,'active_jobs')
                self.store.access.check(db,owner,'storage_bytes',2048)
                account=self.store.access.account_for(db,owner)
                if not account:raise AppError('公开仓库导入需要真实账号',403)
                if not db.execute('SELECT 1 FROM cs_repositories WHERE id=? AND deleted=0',(key,)).fetchone():self.store.access.check(db,owner,'repositories')
                if db.execute('SELECT COUNT(*) FROM cs_jobs WHERE status IN (\'pending\',\'running\',\'blocked\',\'retry_wait\')').fetchone()[0]>=self.settings.queue_capacity:raise AppError('任务队列已满',503,code='queue_full')
                db.execute("INSERT INTO cs_repositories(id,owner,workspace,name,root,status,created_at,source_kind,source_url,source_ref) VALUES (?,?,?,?,?,'pending',?,'remote',?,?) ON CONFLICT(id) DO UPDATE SET deleted=0,status='pending',error=NULL",(key,owner,workspace,name or url.rsplit('/',1)[-1],root,now(),url,ref))
                self.store._job(db,job_id,owner,workspace,None,'repository',{'repository_id':key,'source_kind':'remote'})
        result=self.manager.public(self.store.repository(owner,workspace,key));result['job_id']=job_id;return result

    def root(self,repo,value=None):
        path=Path(value or repo['root']);base=self.settings.data_dir/'remote-repositories'/repo['id']
        if not path.is_absolute() or path!=path.resolve() or path.parent!=base or not path.name.startswith('snapshot-') or not path.is_dir():
            raise AppError('远程源码快照尚未完成或已不可用',409,code='remote_snapshot_unavailable')
        return path

    def metadata(self,http,repo):
        u=urlsplit(repo['source_url']);path=u.path.strip('/');ref=repo['source_ref']
        if u.hostname=='github.com':
            api='https://api.github.com/repos/'+path
            meta=http.get(api)
            if meta.get('private') is not False or meta.get('size',0)>100*1024:raise AppError('仓库非公开或平台报告的大小超过 100 MiB 预算',413,code='git_repository_limit')
            ref=ref or meta.get('default_branch','')
            commit=http.get(api+'/commits/'+quote(ref,safe=''),maximum=2*1024*1024);head=commit.get('sha','')
            if not SHA.fullmatch(head):raise AppError('无法确定不可变 Commit SHA',422)
            tree=http.get(api+'/git/trees/'+head+'?recursive=1')
            if tree.get('truncated'):raise AppError('仓库文件树被平台截断，未开始文件下载',413,code='git_tree_limit')
            entries=tree.get('tree',[])
            raw=lambda entry:'https://raw.githubusercontent.com/'+path+'/'+head+'/'+quote(entry['path'],safe='/')
        else:
            api='https://gitlab.com/api/v4/projects/'+quote(path,safe='')
            meta=http.get(api)
            if meta.get('visibility')!='public':raise AppError('只支持公开 GitLab 仓库',422)
            ref=ref or meta.get('default_branch','')
            commit=http.get(api+'/repository/commits/'+quote(ref,safe=''));head=commit.get('id','')
            if not SHA.fullmatch(head):raise AppError('无法确定不可变 Commit SHA',422)
            entries=[]
            for page in range(1,102):
                batch=http.get(api+'/repository/tree?'+urlencode({'ref':head,'recursive':'true','per_page':100,'page':page}))
                if not isinstance(batch,list):raise AppError('GitLab 文件树格式无效',422)
                entries.extend(dict(e,sha=e['id']) for e in batch)
                if len(batch)<100:break
            else:raise AppError('仓库文件树超过 10,000 条预算',413,code='git_tree_limit')
            raw=lambda entry:api+'/repository/files/'+quote(entry['path'],safe='')+'/raw?ref='+head
        if not isinstance(entries,list) or len(entries)>10000:raise AppError('仓库文件树超过预算',413,code='git_tree_limit')
        return head,ref,entries,raw

    def prepare(self,job,repo,check):
        existing=self.store.rows('SELECT * FROM cs_remote_snapshots WHERE job_id=?',(job['id'],))
        if existing:
            row=existing[0];return self.root(repo,row['root']),json.loads(row['manifest'])
        from .repository import excluded,safe_read,SECRET_CONTENT,LANGUAGES
        from .attachments import SENSITIVE
        deadline=time.monotonic()+self.settings.remote_git_timeout;http=PublicHTTP(deadline,check)
        def progress(stage,**values):
            check()
            with self.p.state.transaction() as db:db.execute('UPDATE cs_jobs SET progress=?,updated_at=? WHERE id=? AND lease=?',(canonical_json({'stage':stage,'repository_id':repo['id'],**values}),now(),job['id'],job['lease']))
        progress('获取公开仓库元信息与文件树')
        head,ref,entries,url_for=self.metadata(http,repo)
        # Equal Git commits do not imply a live memory projection. Clearing
        # retires file/symbol knowledge while preserving the checked snapshot.
        cleared=self.store.rows("SELECT 1 FROM cs_repository_files WHERE repository=? AND status='cleared' LIMIT 1",(repo['id'],))
        if repo['head']==head and repo['generation']>0 and not cleared:
            current=self.root(repo);names=[r['path'] for r in self.store.rows("SELECT path FROM cs_repository_files WHERE repository=? AND status='indexed' ORDER BY path",(repo['id'],))]
            manifest={'head':head,'ref':ref,'names':names,'skipped':json.loads(repo['statistics']).get('skipped',[]),'skipped_count':json.loads(repo['statistics']).get('skipped_count',0),'network_requests':http.requests,'network_bytes':http.bytes,'reused_commit':True}
            with self.p.state.transaction() as db:db.execute('INSERT INTO cs_remote_snapshots VALUES (?,?,?,?,?,?)',(job['id'],repo['id'],head,str(current),canonical_json(manifest),now()))
            return current,manifest
        selected=[];skipped=[];known_bytes=0
        for e in entries:
            path=e.get('path','');p=PurePosixPath(path)
            if e.get('type')=='tree':continue
            reason=excluded(path,self.settings)
            if not path or '\\' in path or len(path)>300 or any(ord(c)<32 for c in path) or any(len(x)>100 for x in p.parts) or str(p)!=path:reason='unsafe_path'
            if e.get('mode') not in ('100644','100755') or e.get('type')!='blob':reason='symlink_or_submodule'
            if any(x.lower() in REMOTE_DIRS for x in p.parts) or p.name.endswith(('.min.js','.lock','-lock.json','.map')) or p.name in ('pnpm-lock.yaml','bun.lockb'):reason='dependency_or_generated'
            if not reason and e.get('size',0)>self.settings.repository_max_file_bytes:reason='file_too_large'
            if reason:skipped.append({'path':path[:300],'reason':reason});continue
            if not SHA.fullmatch(e.get('sha','')):raise AppError('平台返回无效 Blob 身份',422)
            selected.append(e);known_bytes+=e.get('size',0)
        if len(selected)>self.settings.remote_git_max_files or known_bytes>self.settings.remote_git_max_bytes:
            raise AppError('筛选后的源码超过文件数或总内容预算，未下载源码',413,code='git_repository_limit')
        if not selected:raise AppError('仓库没有符合安全规则的源码文件',422,code='repository_empty')
        base=self.settings.data_dir/'remote-repositories'/repo['id'];base.mkdir(parents=True,exist_ok=True)
        destination=base/('snapshot-'+digest([job['id'],head])[:24]);temporary=base/('tmp-'+digest(job['id'])[:24])
        if temporary.exists():shutil.rmtree(temporary)
        temporary.mkdir(mode=0o700);files=[];total=0;github_blob_fallback=False
        try:
            cache=self.settings.data_dir/'remote-repositories'
            occupied=sum(max(p.stat().st_size,p.stat().st_blocks*512,1024) for p in cache.rglob('*') if not p.is_symlink())
            if occupied+max(known_bytes,self.settings.remote_git_max_bytes)+len(selected)*4096>self.settings.remote_git_cache_bytes or shutil.disk_usage(cache).free<1024**3:
                raise AppError('远程源码缓存空间不足，请由管理员释放缓存或调整预算',429,code='git_cache_limit')
            with self.p.state.transaction() as db:self.store.access.check(db,repo['owner'],'storage_bytes',known_bytes)
            previous={r['path']:r for r in self.store.rows("SELECT path,hash FROM cs_repository_files WHERE repository=? AND status IN ('indexed','cleared')",(repo['id'],))}
            old_root=None
            try:old_root=self.root(repo)
            except AppError:pass
            for index,e in enumerate(sorted(selected,key=lambda e:e['path'])):
                check()
                if time.monotonic()>=deadline:raise AppError('仓库下载超过总超时',408,code='git_timeout')
                path=e['path'];data=None
                if old_root and path in previous:
                    try:
                        old=safe_read(old_root,path,self.settings.repository_max_file_bytes)
                        if hashlib.sha1(b'blob '+str(len(old)).encode()+b'\0'+old).hexdigest()==e['sha']:data=old
                    except (OSError,AppError):pass
                if data is None:
                    try:
                        github=urlsplit(repo['source_url']).hostname=='github.com'
                        if not (github and github_blob_fallback):
                            try:data=http.get(url_for(e),maximum=self.settings.repository_max_file_bytes,as_json=False)
                            except AppError as exc:
                                if not github or exc.code not in ('git_connection','git_timeout','git_http_403','git_http_404','git_http_429'):raise
                                github_blob_fallback=True
                        if data is None and github and github_blob_fallback:
                            blob=http.get('https://api.github.com/repos/'+urlsplit(repo['source_url']).path.strip('/')+'/git/blobs/'+e['sha'],maximum=self.settings.repository_max_file_bytes*4//3+8192)
                            if blob.get('encoding')!='base64' or blob.get('sha')!=e['sha']:raise AppError('GitHub Blob 响应格式或身份不匹配',422,code='git_blob_mismatch')
                            try:data=base64.b64decode(blob['content'].replace('\n',''),validate=True)
                            except (ValueError,KeyError,TypeError):raise AppError('GitHub Blob 编码无效',422,code='git_blob_mismatch')
                            if len(data)>self.settings.repository_max_file_bytes:raise AppError('远程文件超过预算',413,code='git_response_limit')
                    except AppError as exc:
                        if exc.code=='git_response_limit':skipped.append({'path':path,'reason':'file_too_large'});continue
                        raise
                if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=e['sha']:raise AppError('源码内容不匹配指定 Commit 的 Blob 哈希',409,code='git_blob_mismatch')
                reason=None
                try:text=data.decode('utf-8')
                except UnicodeDecodeError:text='';reason='non_utf8_text'
                if b'\0' in data or any(ord(c)<9 or 13<ord(c)<32 for c in text):reason='binary_or_control_characters'
                if not text.strip():reason='empty_text'
                if text.startswith('version https://git-lfs.github.com/spec/v1'):reason='git_lfs_pointer'
                if SENSITIVE.search(text) or SECRET_CONTENT.search(text):reason='sensitive_content'
                if reason:skipped.append({'path':path,'reason':reason});continue
                total+=len(data)
                if total>self.settings.remote_git_max_bytes:raise AppError('源码内容超过总字节预算',413,code='git_repository_limit')
                with self.p.state.transaction() as db:self.store.access.check(db,repo['owner'],'storage_bytes',total)
                target=temporary/path;target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
                files.append(path)
                progress('下载并校验选中源码',files_done=index+1,files_total=len(selected),downloaded_bytes=total)
            if not files:raise AppError('所有选中文件均被安全内容审核排除',422,code='repository_empty')
            # Flush nested directory entries before committing a recoverable snapshot receipt.
            for folder,_,_ in os.walk(temporary,topdown=False):
                fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
            if destination.exists():shutil.rmtree(destination)
            os.replace(temporary,destination)
            fd=os.open(base,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
            manifest={'head':head,'ref':ref,'names':files,'skipped':skipped[:100],'skipped_count':len(skipped),'network_requests':http.requests,'network_bytes':http.bytes,'total_bytes':total,'index_mode':'selected-safe-source-files','source_recipe':'public-https-bounded-v32','reused_commit':False,'github_blob_api_fallback':github_blob_fallback}
            with self.p.state.transaction() as db:db.execute('INSERT INTO cs_remote_snapshots VALUES (?,?,?,?,?,?)',(job['id'],repo['id'],head,str(destination),canonical_json(manifest),now()))
            return destination,manifest
        finally:
            if temporary.exists():shutil.rmtree(temporary)
            # Before durable receipt: failed/cancelled transfers leave no orphan snapshot.
            if destination.exists() and not self.store.rows('SELECT 1 FROM cs_remote_snapshots WHERE job_id=?',(job['id'],)):
                shutil.rmtree(destination)

    def cleanup(self,repo):
        """Keep the current snapshot and retryable jobs; never touch another repository."""
        current=self.store.repository(repo['owner'],repo['workspace'],repo['id'])
        roots={current['root']}
        roots.update(r['root'] for r in self.store.rows("SELECT s.root FROM cs_remote_snapshots s JOIN cs_jobs j ON j.id=s.job_id WHERE s.repository=? AND j.status IN ('pending','running','retry_wait','failed')",(repo['id'],)))
        base=self.settings.data_dir/'remote-repositories'/repo['id']
        if base.is_dir():
            for child in base.iterdir():
                if child.is_dir() and not child.is_symlink() and str(child) not in roots:shutil.rmtree(child)
