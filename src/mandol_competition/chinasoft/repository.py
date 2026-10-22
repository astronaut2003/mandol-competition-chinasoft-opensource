"""Bounded, read-only Git repository ingestion into the existing substrate.

No repository program, dependency, hook, filter, fetch or clone is executed.
Every publication is restart-idempotent. Frozen scan metadata precedes parsing;
the source file must still match its hash before a partial scan can resume.
"""
from dataclasses import asdict
from pathlib import Path, PurePosixPath
import fnmatch, hashlib, json, os, re, selectors, stat, subprocess, time
from tree_sitter_language_pack import get_parser
from mandol_substrate.state import CanonicalMemory, Mutation, Edge
from mandol_substrate.state.models import digest, canonical_json
from .store import AppError, now
from mandol_substrate.state.models import Conflict

RECIPE = digest(['repository-parser-v31', 'tree-sitter-language-pack-1.14.3',
                 '80-lines-5000-chars', 'exact-import-only', 'no-symlinks', 'sensitive-exclusion-v2'])
LANGUAGES = {'.py':'python','.ts':'typescript','.tsx':'tsx','.js':'javascript','.jsx':'javascript','.mjs':'javascript','.cjs':'javascript',
             '.rs':'rust','.go':'go','.java':'java','.c':'c','.h':'c','.cpp':'cpp','.hpp':'cpp','.cc':'cpp'}
TEXT_EXTENSIONS = set(LANGUAGES) | {'.md','.rst','.txt','.toml','.yaml','.yml','.json','.sql','.rs','.go','.java','.c','.h','.cpp','.hpp','.sh','.cj','.css','.html','.vue'}
EXCLUDED_DIRS = {'.git','.venv','venv','node_modules','__pycache__','.cache','.data','.runtime','dist','build','target','coverage','runs','benchmark_raw','logs'}
SECRET_NAME = re.compile(r'(^\.env($|\.)|(^|[._-])(credentials?|secrets?|tokens?|passwords?|id_rsa|id_ed25519)([._-]|$)|\.(pem|key|p12|pfx|sqlite3?|db|log)$|^\.(npmrc|pypirc|netrc)$)',re.I)
SECRET_CONTENT = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16})\b')

def validate_root(settings, value):
    path=Path(value).expanduser()
    if not path.is_absolute() or '..' in path.parts: raise AppError('请输入白名单内的服务端绝对仓库路径',403,code='repository_path_rejected')
    resolved=path.resolve()
    roots=[Path(p).expanduser().resolve() for p in settings.repository_roots.split(os.pathsep) if p]
    if not any(resolved==r or r in resolved.parents for r in roots):
        raise AppError('仓库路径不在管理员配置的白名单中',403,code='repository_path_rejected')
    if resolved==Path('/') or any(p in EXCLUDED_DIRS for p in resolved.parts):
        raise AppError('运行数据、系统目录及依赖缓存不能注册为仓库',403,code='repository_path_rejected')
    if path != resolved: raise AppError('仓库根路径必须是规范路径，不能使用符号链接',403,code='repository_path_rejected')
    if not resolved.is_dir() or not (resolved/'.git').is_dir() or (resolved/'.git').is_symlink():
        raise AppError('仓库不存在、无法读取或不是普通本地 Git 仓库（暂不支持 Git worktree）',400,code='repository_unavailable')
    return resolved

def safe_read(root, relative, maximum):
    """Use dirfds/O_NOFOLLOW for every path component, including root ancestors."""
    rel=PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts: raise AppError('无效源码路径',403)
    descriptor=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in root.parts[1:]+rel.parts[:-1]:
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=descriptor)
            os.close(descriptor); descriptor=child
        file_fd=os.open(rel.parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=descriptor)
        try:
            before=os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode): raise AppError('只读取普通源码文件',400,code='not_regular_file')
            if before.st_size>maximum: raise AppError('源码文件超过单文件限制',413,code='file_too_large')
            with os.fdopen(os.dup(file_fd),'rb') as stream: data=stream.read(maximum+1)
            after=os.fstat(file_fd)
            if len(data)>maximum: raise AppError('源码文件超过单文件限制',413,code='file_too_large')
            if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
                raise AppError('读取期间源码发生变化，请重新扫描',409,code='repository_changed')
            return data
        finally: os.close(file_fd)
    finally: os.close(descriptor)

def excluded(path, settings):
    p=PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or any(x in EXCLUDED_DIRS for x in p.parts): return 'unsafe_or_runtime_path'
    if any(SECRET_NAME.search(x) for x in p.parts): return 'sensitive_name'
    if any(fnmatch.fnmatch(path,pattern) for pattern in settings.repository_excludes.split(',') if pattern): return 'operator_exclusion'
    if p.suffix.lower() not in TEXT_EXTENSIONS and p.name not in ('Dockerfile','Makefile','LICENSE','NOTICE'): return 'unsupported_file_type'
    return None

def git_read(root, args, maximum=2*1024*1024):
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL='/dev/null',GIT_TERMINAL_PROMPT='0')
    command=['/usr/bin/git','--no-optional-locks','--git-dir='+str(root/'.git'),'--work-tree='+str(root),
             '-c','core.fsmonitor=false','-c','core.untrackedCache=false','-c','core.hooksPath=/dev/null',*args]
    proc=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    chunks=[];size=0;deadline=time.monotonic()+10
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout,selectors.EVENT_READ)
            while selector.get_map():
                if time.monotonic()>deadline: raise AppError('Git 文件发现超时',408,code='repository_git_timeout')
                for key,_ in selector.select(.1):
                    part=os.read(key.fd,65536)
                    if not part: selector.unregister(key.fileobj);continue
                    size+=len(part)
                    if size>maximum: raise AppError('Git 文件清单超出安全预算，请缩小仓库',413,code='repository_limit')
                    chunks.append(part)
        if proc.wait(timeout=1)!=0: raise AppError('Git 元数据读取失败；未执行仓库代码',400,code='repository_git_error')
        return b''.join(chunks)
    finally:
        if proc.poll() is None: proc.kill()
        proc.wait();proc.stdout.close()

def walk_nodes(root):
    todo=[root]
    while todo:
        node=todo.pop();yield node;todo.extend(reversed(node.named_children))

def parse_source(path, text):
    language=LANGUAGES.get(Path(path).suffix.lower(),'text')
    if language=='text': return {'language':'text','parser':'text-chunks','symbols':[],'imports':[],'calls':[],'error':None}
    data=text.encode();tree=get_parser(language).parse(data);symbols=[];imports=[];calls=[]
    value=lambda n:data[n.start_byte:n.end_byte].decode() if n else ''
    symbol_nodes={'function_definition','class_definition','function_declaration','generator_function_declaration','class_declaration','method_definition',
                  'function_item','struct_item','enum_item','trait_item','method_declaration','constructor_declaration','interface_declaration','type_spec'}
    for node in walk_nodes(tree.root_node):
        is_arrow=node.type=='variable_declarator' and node.child_by_field_name('value') is not None and node.child_by_field_name('value').type in ('arrow_function','function_expression')
        if node.type in symbol_nodes or is_arrow:
            name=node.child_by_field_name('name')
            if not name and node.type=='function_definition':
                declarator=node.child_by_field_name('declarator')
                for candidate in walk_nodes(declarator) if declarator else []:
                    if candidate.type in ('identifier','field_identifier'):name=candidate;break
            if not name:continue
            owners=[];parent=node.parent
            while parent:
                if parent.type in symbol_nodes:
                    parent_name=parent.child_by_field_name('name')
                    if parent_name:owners.append(value(parent_name))
                parent=parent.parent
            kind='class' if node.type in ('class_definition','class_declaration','struct_item','enum_item','trait_item','interface_declaration','type_spec') else 'method' if node.type in ('method_definition','method_declaration','constructor_declaration') or (node.parent and any(p.type in ('class_definition','class_declaration','impl_item') for p in _parents(node))) else 'function'
            symbols.append({'name':value(name),'qualified_name':'.'.join(list(reversed(owners))+[value(name)]),'kind':kind,'start_line':node.start_point.row+1,'end_line':node.end_point.row+1})
        if language=='python' and node.type in ('import_statement','import_from_statement'):
            source=value(node)
            if node.type=='import_from_statement':
                match=re.match(r'from\s+([\w.]+)\s+import\s+(.+)',source,re.S)
                if match:
                    module=match[1]
                    for entry in match[2].strip('() \n').split(','):
                        parts=entry.strip().split(' as ')
                        if parts and re.fullmatch(r'\w+|\*',parts[0]): imports.append({'module':module,'symbol':parts[0],'alias':parts[-1],'line':node.start_point.row+1})
            else:
                for entry in source[6:].split(','):
                    parts=entry.strip().split(' as ')
                    if re.fullmatch(r'[\w.]+',parts[0]):imports.append({'module':parts[0],'symbol':None,'alias':parts[-1],'line':node.start_point.row+1})
        elif language in ('typescript','tsx','javascript') and node.type=='import_statement':
            module=node.child_by_field_name('source');module=value(module).strip('\"\'')
            clause=next((c for c in node.named_children if c.type=='import_clause'),None)
            found=False
            if clause:
                for spec in walk_nodes(clause):
                    if spec.type=='import_specifier':
                        name=value(spec.child_by_field_name('name'));alias=value(spec.child_by_field_name('alias')) or name
                        imports.append({'module':module,'symbol':name,'alias':alias,'line':node.start_point.row+1});found=True
            if not found:imports.append({'module':module,'symbol':None,'alias':'','line':node.start_point.row+1})
        if node.type in ('call','call_expression'):
            fn=node.child_by_field_name('function')
            if fn and len(value(fn))<160:calls.append({'expression':value(fn),'line':node.start_point.row+1,'resolved':False})
    return {'language':language,'parser':'tree-sitter-1.14.3','symbols':symbols[:800],'imports':imports[:500],'calls':calls[:500],
            'error':'Tree-sitter 存在语法错误；仅提供文本索引，结构结果未发布' if tree.root_node.has_error else None}

def _parents(node):
    result=[];p=node.parent
    while p: result.append(p);p=p.parent
    return result

def chunks(text):
    lines=text.splitlines(keepends=True);result=[];index=0
    while index<len(lines):
        end=index;size=0
        while end<len(lines) and end-index<80 and size+len(lines[end])<=5000:
            size+=len(lines[end]);end+=1
        if end==index:
            # One oversized line is represented by a bounded, explicitly truncated snippet.
            result.append((index+1,index+1,lines[index][:5000],True));index+=1
        else:result.append((index+1,end,''.join(lines[index:end]),False));index=end
    return result

class RepositoryManager:
    def __init__(self,platform):
        self.platform=platform;self.store=platform.store;self.state=platform.state;self.settings=platform.settings
        from .remote_git import RemoteGit
        self.remote=RemoteGit(self)

    def source_root(self,repo):
        if repo.get('deleted'):raise AppError('仓库已移除',404)
        return self.remote.root(repo) if repo.get('source_kind')=='remote' else validate_root(self.settings,repo['root'])

    def register(self,owner,workspace,root,name,request_id):
        self.store.workspace(owner,workspace);path=validate_root(self.settings,root)
        key='repo_'+digest([owner,workspace,str(path)])
        job_id='job_'+digest([owner,workspace,'repository',request_id])
        with self.state.transaction() as db:
            previous=db.execute('SELECT id,payload FROM cs_jobs WHERE id=?',(job_id,)).fetchone()
            if previous and json.loads(previous['payload'])['repository_id']!=key:raise Conflict('仓库扫描请求对应不同路径')
            if not previous:
                self.store.access.check(db,owner,'repository_imports')
                self.store.access.check(db,owner,'active_jobs')
                if not db.execute('SELECT 1 FROM cs_repositories WHERE id=? AND deleted=0',(key,)).fetchone():self.store.access.check(db,owner,'repositories')
                pending=db.execute("SELECT 1 FROM cs_jobs WHERE kind='repository' AND owner=? AND workspace=? AND json_extract(payload,'$.repository_id')=? AND status IN ('pending','running','retry_wait')",(owner,workspace,key)).fetchone()
                if pending:raise AppError('该仓库已有扫描任务，完成后再重新扫描',409,code='repository_busy')
                if db.execute("SELECT COUNT(*) FROM cs_jobs WHERE status IN ('pending','running','blocked','retry_wait')").fetchone()[0]>=self.settings.queue_capacity:
                    raise AppError('后台队列已满，仓库尚未入队',503,code='queue_full')
                if db.execute('SELECT COUNT(*) FROM cs_repositories WHERE owner=? AND workspace=?',(owner,workspace)).fetchone()[0]>=20 and not db.execute('SELECT 1 FROM cs_repositories WHERE id=?',(key,)).fetchone():
                    raise AppError('每个工作区最多 20 个仓库',429)
                db.execute("INSERT OR IGNORE INTO cs_repositories(id,owner,workspace,name,root,status,created_at) VALUES (?,?,?,?,?,'pending',?)",(key,owner,workspace,name,str(path),now()))
                db.execute("UPDATE cs_repositories SET status='pending',error=NULL,deleted=0 WHERE id=?",(key,))
                self.store._job(db,job_id,owner,workspace,None,'repository',{'repository_id':key})
        result=self.public(self.store.repository(owner,workspace,key));result['job_id']=job_id;return result

    def rescan(self,owner,workspace,key,request_id):
        repo=self.store.repository(owner,workspace,key)
        if repo.get('source_kind')=='remote':return self.remote.register(owner,workspace,repo['source_url'],repo['source_ref'],repo['name'],request_id)
        return self.register(owner,workspace,repo['root'],repo['name'],request_id)

    def public(self,repo):
        result={**repo,'repository_id':repo['id'],'statistics':json.loads(repo['statistics']),'worktree':json.loads(repo['worktree'])}
        active=self.store.rows("SELECT id,status,progress FROM cs_jobs WHERE owner=? AND workspace=? AND kind='repository' AND json_extract(payload,'$.repository_id')=? AND status IN ('pending','running','retry_wait') ORDER BY created_at DESC LIMIT 1",(repo['owner'],repo['workspace'],repo['id']))
        result['active_job']={**active[0],'progress':json.loads(active[0]['progress'])} if active else None
        if repo.get('source_kind')=='remote':result['root']=repo['source_url']
        try:self.source_root(repo);result['source_available']=True
        except (AppError,OSError):
            result['source_available']=False
            if repo['status'] not in ('pending','scanning','failed','cancelled','cleared'):
                result.update(status='source_unavailable',error='授权源码或远程快照不可用；历史证据保留')
        return result

    def current(self,owner,workspace,key):
        repo=self.store.repository(owner,workspace,key)
        result=self.public(repo)
        try:self.source_root(repo);result['source_available']=True
        except (AppError,OSError):result['source_available']=False
        return result

    def scan(self,job):
        def check_cancel():
            if self.platform.stop_event.is_set():raise AppError('仓库任务随服务关闭安全中断，可恢复',503,code='repository_interrupted')
            current=self.store.rows('SELECT status,error_code FROM cs_jobs WHERE id=?',(job['id'],))
            if current and (current[0]['status']=='cancelled' or current[0]['error_code']=='cancel_requested'):
                raise AppError('仓库扫描已按请求停止，已发布来源保留',409,code='repository_cancelled')
        check_cancel()
        repo=self.store.repository(job['owner'],job['workspace'],job['payload']['repository_id']);remote_manifest=None
        if repo.get('source_kind')=='remote':
            with self.state.transaction() as db:db.execute("UPDATE cs_repositories SET status='scanning',error=NULL WHERE id=?",(repo['id'],))
            root,remote_manifest=self.remote.prepare(job,repo,check_cancel)
        else:root=self.source_root(repo)
        scope=self.store.scope(job['owner'],job['workspace'],'code',repo['id'])
        saved=self.store.rows('SELECT * FROM cs_repository_scans WHERE job_id=?',(job['id'],))
        if saved:
            snapshot=json.loads(saved[0]['snapshot']);statistics=json.loads(saved[0]['statistics'])
            if saved[0]['state']=='completed':
                if statistics['changed_files'] or statistics['deleted_files']:
                    self.store.enqueue(job['owner'],job['workspace'],'index',{'scopes':[asdict(scope)],'repository_id':repo['id']},[job['id'],'repository-index'])
                return {**statistics,'repository_id':repo['id'],'stage':'已完成扫描的持久回执已恢复'}
            if snapshot['generation']<=repo['generation']:raise AppError('该扫描已被更新版本替代，请重新扫描当前仓库',409,code='outdated_scan')
        else:
            old={r['path']:r for r in self.store.rows('SELECT * FROM cs_repository_files WHERE repository=?',(repo['id'],))}
            names=sorted(remote_manifest['names']) if remote_manifest else sorted(set(git_read(root,['ls-files','--cached','--others','--exclude-standard','-z']).decode('utf-8','strict').split('\0'))- {''})
            files=[];skipped=list(remote_manifest.get('skipped',[])) if remote_manifest else [];total=0
            for path in names:
                check_cancel()
                reason=excluded(path,self.settings)
                if not reason:
                    try:
                        data=safe_read(root,path,self.settings.repository_max_file_bytes)
                        if b'\0' in data:reason='binary_file'
                        else:
                            text=data.decode('utf-8')
                            if SECRET_CONTENT.search(text):reason='sensitive_content'
                    except UnicodeDecodeError:reason='non_utf8_text'
                    except FileNotFoundError:reason='deleted_file'
                    except (OSError,AppError) as exc:reason=getattr(exc,'code','symlink_or_permission_rejected')
                if reason:
                    skipped.append({'path':path,'reason':reason});continue
                total+=len(data)
                if len(files)>=self.settings.repository_max_files or total>self.settings.repository_max_total_bytes:
                    raise AppError('仓库超出文件数量或总字节预算，请配置排除规则后重扫',413,code='repository_limit')
                fingerprint=hashlib.sha256(data).hexdigest();previous=old.get(path)
                unchanged=bool(previous and previous['hash']==fingerprint and previous['status']=='indexed' and previous['parser'].endswith(RECIPE))
                files.append({'path':path,'hash':fingerprint,'bytes':len(data),'unchanged':unchanged,'version':previous['version'] if unchanged else (previous['version']+1 if previous else 1),
                              'old_uids':json.loads(previous['uids']) if previous and previous['status']=='indexed' else []})
            if not files:raise AppError('没有可安全索引的源码文件，请查看白名单、排除规则和仓库内容',400,code='repository_empty')
            try:head=remote_manifest['head'] if remote_manifest else git_read(root,['rev-parse','--verify','HEAD'],maximum=128).decode().strip()
            except AppError as exc:
                if exc.code!='repository_git_error':raise
                head='' # An initialized repository can legitimately have no commit yet.
            deleted=[{'path':path,'uids':json.loads(row['uids'])} for path,row in old.items() if row['status']=='indexed' and path not in {f['path'] for f in files}]
            snapshot={'files':files,'deleted':deleted,'head':head,'generation':repo['generation']+1}
            statistics={'files':len(files),'total_bytes':total,'changed_files':sum(not f['unchanged'] for f in files),'unchanged_files':sum(f['unchanged'] for f in files),'deleted_files':len(deleted),
                        'skipped_count':len(skipped),'skipped':skipped[:100],'recipe_fingerprint':RECIPE,'parser_errors':[]}
            if remote_manifest:
                statistics.update({k:v for k,v in remote_manifest.items() if k not in ('names','skipped','head')})
            with self.state.transaction() as db:
                db.execute("INSERT INTO cs_repository_scans VALUES (?,?,?,?,'running',?)",(job['id'],repo['id'],canonical_json(snapshot),canonical_json(statistics),now()))
        with self.state.transaction() as db:db.execute("UPDATE cs_repositories SET status='scanning',error=NULL WHERE id=?",(repo['id'],))
        produced=0
        for index,file in enumerate(snapshot['files']):
            if self.platform.stop_event.is_set():raise AppError('仓库扫描在服务关闭时中断；可安全恢复',503,code='repository_interrupted')
            check_cancel()
            if file['unchanged']:continue
            data=safe_read(root,file['path'],self.settings.repository_max_file_bytes)
            if hashlib.sha256(data).hexdigest()!=file['hash']:raise AppError('扫描期间源码已变化；请显式重新扫描',409,code='repository_changed')
            parsed=parse_source(file['path'],data.decode());file['parsed']=parsed['parser']
            if parsed['error']:
                statistics['parser_errors'].append({'path':file['path'],'error':parsed['error']});parsed['symbols']=[];parsed['imports']=[];parsed['calls']=[]
            mutation,symbols=self.file_mutation(repo,file,data.decode(),parsed,scope,snapshot['head'],job['id'])
            produced+=len(mutation.memories)
            current_count=self.store.rows("SELECT COUNT(*) n FROM memories WHERE scope=? AND active=1",(scope.key,))[0]['n']
            if current_count+len(mutation.memories)-len(mutation.delete_uids)>self.settings.repository_max_chunks:
                raise AppError('仓库知识片段超过预算，请缩小扫描范围',413,code='repository_limit')
            self.platform.coordinator.apply(scope,job['id']+':'+digest([file['path'],file['hash']]),digest(asdict(mutation)),mutation,{'path':file['path']})
            with self.state.transaction() as db:
                db.execute('INSERT INTO cs_repository_files VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(repository,path) DO UPDATE SET hash=excluded.hash,bytes=excluded.bytes,language=excluded.language,parser=excluded.parser,status=excluded.status,error=excluded.error,version=excluded.version,uids=excluded.uids,updated_at=excluded.updated_at',
                           (repo['id'],file['path'],file['hash'],file['bytes'],parsed['language'],parsed['parser']+':'+RECIPE,'indexed',parsed['error'],file['version'],canonical_json([m.logical_uid for m in mutation.memories]),now()))
                db.execute('DELETE FROM cs_repository_symbols WHERE repository=? AND path=?',(repo['id'],file['path']))
                for symbol in symbols:
                    db.execute('INSERT INTO cs_repository_symbols VALUES (?,?,?,?,?,?,?,?,?)',(repo['id'],file['path'],symbol['uid'],symbol['name'],symbol['qualified_name'],symbol['kind'],symbol['start_line'],symbol['end_line'],file['hash']))
                db.execute('UPDATE cs_jobs SET progress=?,updated_at=? WHERE id=?',(canonical_json({'stage':'解析和词法索引','files_done':index+1,'files_total':len(snapshot['files']),'repository_id':repo['id']}),now(),job['id']))
        for deleted in snapshot['deleted']:
            mutation=Mutation(delete_uids=tuple(deleted['uids']))
            self.platform.coordinator.apply(scope,job['id']+':remove:'+digest(deleted),digest(asdict(mutation)),mutation,{})
            with self.state.transaction() as db:
                db.execute("UPDATE cs_repository_files SET status='deleted',updated_at=? WHERE repository=? AND path=?",(now(),repo['id'],deleted['path']))
                db.execute('DELETE FROM cs_repository_symbols WHERE repository=? AND path=?',(repo['id'],deleted['path']))
        changed=bool(statistics['changed_files'] or statistics['deleted_files'])
        if not changed and not self.state._rows("SELECT logical_uid FROM memories WHERE scope=? AND active=1 AND memory_type='repository_map' AND json_extract(metadata,'$.repository_generation')=? LIMIT 1",(scope.key,repo['generation'])):
            # Recover Maps invalidated by older no-op scans without re-parsing files.
            changed=True;statistics['structure_repaired']=True
        if changed: self.publish_structure(repo,scope,snapshot,job)
        statistics['symbols']=self.store.rows('SELECT COUNT(*) n FROM cs_repository_symbols WHERE repository=?',(repo['id'],))[0]['n']
        statistics['generated']=produced
        # A no-op resync must not invalidate the existing Repository Map by
        # advancing its generation, nor turn an already-ready projection pending.
        generation=snapshot['generation'] if changed else repo['generation']
        frontier=self.state._rows('SELECT published_seq,status FROM projection_frontiers WHERE scope=? AND channel=?',(scope.key,self.platform.projection_channel))
        projected=bool(not changed and frontier and frontier[0]['status']=='ready' and frontier[0]['published_seq']==self.state.scope_state(scope)['published_seq'])
        with self.state.transaction() as db:
            db.execute("UPDATE cs_repositories SET status=?,head=?,worktree=?,generation=?,scanned_at=?,statistics=?,error=NULL WHERE id=?",('ready' if projected else 'lexical_ready',snapshot['head'],canonical_json({'working_tree_hash':digest([(f['path'],f['hash']) for f in snapshot['files']]),'changed_from_last_scan':statistics['changed_files']}),generation,now(),canonical_json(statistics),repo['id']))
            if remote_manifest:db.execute('UPDATE cs_repositories SET root=? WHERE id=?',(str(root),repo['id']))
            db.execute("UPDATE cs_repository_scans SET state='completed',statistics=? WHERE job_id=?",(canonical_json(statistics),job['id']))
        if changed or not projected:self.store.enqueue(job['owner'],job['workspace'],'index',{'scopes':[asdict(scope)],'repository_id':repo['id']},[job['id'],'repository-index'])
        if remote_manifest:self.remote.cleanup(repo)
        return {**statistics,'stage':'仓库解析及词法索引完成；向量投影单独构建','repository_id':repo['id']}

    def file_mutation(self,repo,file,text,parsed,scope,head,job_id):
        base={'user_id':scope.user_id,'owner_id':repo['owner'],'workspace_id':repo['workspace'],'mode':'code','memory_family':'code','repository_id':repo['id'],
              'file_hash':file['hash'],'file_path':file['path'],'file_version':file['version'],'recipe_fingerprint':RECIPE,'git_head':head,'parser':parsed['parser']}
        records=[];symbol_rows=[];edges=[];raw=[]
        for start,end,part,truncated in chunks(text):
            key='src_'+digest([repo['id'],file['path'],file['version'],start,end])
            locator={'repository_id':repo['id'],'path':file['path'],'start_line':start,'end_line':end,'file_hash':file['hash'],'git_head':head}
            memory=CanonicalMemory(key,f"{file['path']}:{start}-{end}\n{part}",raw_data={'code':part,'source_locator':locator,**({'imports':parsed['imports'],'unresolved_calls':parsed['calls'][:100]} if not raw else {})},
                       metadata=dict(base,source_locator=locator,truncated=truncated,verification='CURRENT_SOURCE'),request_id=job_id,role='source',memory_tower='engineering',memory_type='repository_chunk',spaces=('code_raw',))
            records.append(memory);raw.append(memory)
        lines=text.splitlines(keepends=True)
        for symbol in parsed['symbols']:
            start=symbol['start_line'];end=min(symbol['end_line'],start+79);part=''.join(lines[start-1:end]);
            if len(part)>5000:part=part[:5000];end=start+part.count('\n')
            key='sym_'+digest([repo['id'],file['path'],file['version'],symbol['qualified_name'],start])
            supports=[r for r in raw if r.metadata['source_locator']['start_line']<=start<=r.metadata['source_locator']['end_line']]
            locator=dict(raw[0].metadata['source_locator'],start_line=start,end_line=end)
            memory=CanonicalMemory(key,f"{symbol['kind']} {symbol['qualified_name']} · {file['path']}:{start}-{end}\n{part}",raw_data={**symbol,'code':part},
                     metadata=dict(base,source_versions={r.logical_uid:r.version for r in supports},source_locator=locator,verification='STATIC_DEFINITION',definition_end_line=symbol['end_line']),
                     request_id=job_id,role='derived',memory_tower='engineering',memory_level='L1_Summary',memory_type='code_symbol',spaces=('code_raw',))
            records.append(memory);symbol_rows.append(dict(symbol,uid=key))
            edges.extend(Edge('support_'+digest([key,r.logical_uid]),key,r.logical_uid,'SUPPORTED_BY',provenance={'file_hash':file['hash'],'source_version':r.version}) for r in supports)
        return Mutation(memories=tuple(records),delete_uids=tuple(file['old_uids']),edges=tuple(edges),spaces=('code_raw',)),symbol_rows

    def resolve_import(self,path,module,language,files):
        if language=='python':
            if module.startswith('.'):
                dots=len(module)-len(module.lstrip('.'));base=PurePosixPath(path).parent
                for _ in range(dots-1):base=base.parent
                stem=str(base/module.lstrip('.').replace('.','/'))
            else:stem=module.replace('.','/')
            options=[stem+'.py',stem+'/__init__.py','src/'+stem+'.py','src/'+stem+'/__init__.py']
        elif module.startswith('.'):
            stem=os.path.normpath(str(PurePosixPath(path).parent/module));options=[stem]+[stem+x for x in ('.ts','.tsx','.js','.jsx','/index.ts','/index.js')]
        else:return None
        matches=[p for p in options if p in files]
        return matches[0] if len(matches)==1 else None

    def publish_structure(self,repo,scope,snapshot,job):
        files={r['path']:r for r in self.store.rows("SELECT * FROM cs_repository_files WHERE repository=? AND status='indexed'",(repo['id'],))}
        symbols=self.store.rows('SELECT * FROM cs_repository_symbols WHERE repository=?',(repo['id'],));by_name={}
        for row in symbols:by_name.setdefault((row['path'],row['name']),[]).append(row)
        old=self.state._rows("SELECT logical_uid FROM memories WHERE scope=? AND active=1 AND memory_type IN ('code_dependency','repository_map')",(scope.key,))
        records=[];edges=[];links=[];map_support={};map_support_by_path={}
        for path,file in sorted(files.items()):
            raw=self.state.get_memories(scope,json.loads(file['uids']));raws=[m for m in raw.values() if m.memory_type=='repository_chunk']
            if not raws:continue
            map_support[raws[0].logical_uid]=raws[0].version
            map_support_by_path[path]={raws[0].logical_uid:raws[0].version}
            for imported in raws[0].raw_data.get('imports',[]):
                target=self.resolve_import(path,imported['module'],file['language'],files)
                found=by_name.get((target,imported['symbol']),[]) if imported['symbol'] else []
                target_symbol=found[0]['uid'] if len(found)==1 else None
                line=imported['line'];supports=[r for r in raws if r.metadata['source_locator']['start_line']<=line<=r.metadata['source_locator']['end_line']]
                key='dep_'+digest([repo['id'],path,file['version'],imported,target,target_symbol])
                previous=self.state.get_memories(scope,[key]).get(key)
                if previous and not previous.active:key='dep_'+digest([key,snapshot['generation']])
                description=f"{path}:{line} 导入 {imported['module']}"+(f" 符号 {imported['symbol']}" if imported['symbol'] else '')+(f" → {target}" if target else '（外部模块或未解析；不能证明仓库内调用关系）')
                versions={r.logical_uid:r.version for r in supports}
                if target_symbol:versions[target_symbol]=1
                records.append(CanonicalMemory(key,description,raw_data={**imported,'target_path':target,'target_symbol':target_symbol,'resolved':bool(target),'relation':'IMPORTS_SYMBOL' if target_symbol else 'IMPORTS_MODULE'},
                         metadata={'user_id':scope.user_id,'repository_id':repo['id'],'file_path':path,'file_hash':file['hash'],'source_versions':versions,'source_locator':dict(raws[0].metadata['source_locator'],start_line=line,end_line=line),
                                   'recipe_fingerprint':RECIPE,'verification':'STATIC_IMPORT' if target else 'UNRESOLVED_IMPORT'},
                         request_id=job['id'],role='derived',memory_tower='engineering',memory_level='L1_Summary',memory_type='code_dependency',spaces=('code_raw',)))
                edges.extend(Edge('importsupport_'+digest([key,r.logical_uid]),key,r.logical_uid,'SUPPORTED_BY') for r in supports)
                if target_symbol:edges.append(Edge('import_'+digest([key,target_symbol]),key,target_symbol,'IMPORTS_SYMBOL',provenance={'match':'exact-import-and-unique-definition'}))
                links.append((repo['id'],path,key,target,target_symbol,'IMPORTS_SYMBOL' if target_symbol else 'IMPORTS_MODULE',int(bool(target)),snapshot['generation']))
        # Map is a bounded navigation view, never an asserted design rationale.
        map_lines=['仓库结构（静态导航，无设计原因或测试结论）']
        degree={path:0 for path in files}
        for _,path,_,target,_,_,resolved,_ in links:
            if resolved:
                degree[path]+=1;degree[target]+=1
        symbolic_files={row['path'] for row in symbols}
        ranked_files=sorted(files,key=lambda path:(-degree[path],0 if path in symbolic_files else 1,path.count('/'),path))
        map_files=set(ranked_files[:64]);positions={p:i for i,p in enumerate(ranked_files)}
        map_support={u:v for path in map_files for u,v in map_support_by_path.get(path,{}).items()}
        for row in sorted(symbols,key=lambda r:(positions[r['path']],r['start_line'])):
            if row['path'] not in map_files:continue
            line=f"{row['path']}:{row['start_line']}-{row['end_line']} {row['kind']} {row['qualified_name']}"
            if sum(len(x)+1 for x in map_lines)+len(line)>10000:break
            map_lines.append(line)
        if len(map_lines)==1:map_lines.extend(ranked_files[:60])
        key='repomap_'+digest([repo['id'],snapshot['generation'],job['id']])
        records.append(CanonicalMemory(key,'\n'.join(map_lines),raw_data={'files':len(files),'symbols':len(symbols),'context_budget_chars':10000,'ranking':'exact-import-degree / symbolic-file / path-depth','dynamic_calls':'unresolved'},
                     metadata={'user_id':scope.user_id,'repository_id':repo['id'],'source_versions':dict(list(map_support.items())[:64]),'repository_generation':snapshot['generation'],'verification':'STATIC_NAVIGATION','recipe_fingerprint':RECIPE},
                     request_id=job['id'],role='derived',memory_tower='engineering',memory_level='L1_Summary',memory_type='repository_map',spaces=('code_raw',)))
        new_ids={m.logical_uid for m in records}
        existing=self.state.get_memories(scope,list(new_ids))
        new_records=tuple(m for m in records if m.logical_uid not in existing or not existing[m.logical_uid].active)
        new_edges=tuple(e for e in edges if e.source_uid in {m.logical_uid for m in new_records})
        mutation=Mutation(memories=new_records,edges=new_edges,delete_uids=tuple(m['logical_uid'] for m in old if m['logical_uid'] not in new_ids),spaces=('code_raw',))
        # Freeze the full structural plan before its bounded publications. A retry
        # must not recalculate the retirement set from partially published rows.
        row=self.store.rows('SELECT snapshot FROM cs_repository_scans WHERE job_id=?',(job['id'],))[0]
        frozen=json.loads(row['snapshot'])
        if 'structure_plan' in frozen:
            plan=frozen['structure_plan']
            mutation=Mutation(memories=tuple(CanonicalMemory(**dict(m,spaces=tuple(m['spaces']))) for m in plan['memories']),
                              edges=tuple(Edge(**e) for e in plan['edges']),delete_uids=tuple(plan['delete_uids']),spaces=('code_raw',))
        else:
            count=self.state._rows('SELECT COUNT(*) n FROM memories WHERE scope=? AND active=1',(scope.key,))[0]['n']
            if count+len(mutation.memories)-len(mutation.delete_uids)>self.settings.repository_max_chunks:
                raise AppError('仓库结构知识超出对象预算',413,code='repository_limit')
            frozen['structure_plan']=asdict(mutation)
            with self.state.transaction() as db:db.execute('UPDATE cs_repository_scans SET snapshot=? WHERE job_id=?',(canonical_json(frozen),job['id']))
        for index in range(0,len(mutation.memories),300):
            batch=mutation.memories[index:index+300];ids={m.logical_uid for m in batch}
            part=Mutation(memories=batch,edges=tuple(e for e in mutation.edges if e.source_uid in ids),spaces=('code_raw',))
            self.platform.coordinator.apply(scope,job['id']+':structure:put:'+str(index//300),digest(asdict(part)),part,{})
        for index in range(0,len(mutation.delete_uids),500):
            part=Mutation(delete_uids=mutation.delete_uids[index:index+500])
            self.platform.coordinator.apply(scope,job['id']+':structure:retire:'+str(index//500),digest(asdict(part)),part,{})
        with self.state.transaction() as db:
            db.execute('DELETE FROM cs_repository_links WHERE repository=?',(repo['id'],))
            for row in links:db.execute('INSERT INTO cs_repository_links VALUES (?,?,?,?,?,?,?,?)',row)

    def eligible(self,memory):
        meta=memory.metadata;key=meta.get('repository_id')
        if not key:return True
        repos=self.store.rows('SELECT * FROM cs_repositories WHERE id=?',(key,))
        if not repos or repos[0].get('deleted'):return False
        repo=repos[0]
        if memory.memory_type=='repository_map' and meta.get('repository_generation')!=repo['generation']:return False
        if memory.memory_type in ('repository_map','repository_chunk','code_symbol','code_dependency'):
            try:self.source_root(repo)
            except (AppError,OSError):return False
        if not meta.get('file_path'):return True # Engineering history is independent of current code.
        rows=self.store.rows("SELECT * FROM cs_repository_files WHERE repository=? AND path=? AND status='indexed'",(key,meta['file_path']))
        if not rows or rows[0]['hash']!=meta['file_hash']:return False
        try:
            root=self.source_root(repo)
            return hashlib.sha256(safe_read(root,meta['file_path'],self.settings.repository_max_file_bytes)).hexdigest()==meta['file_hash']
        except (AppError,OSError):return False

    def files(self,owner,workspace,key,offset,limit,query=''):
        self.store.repository(owner,workspace,key)
        where="repository=? AND status='indexed'";params=[key]
        if query:where+=' AND instr(path,?)>0';params.append(query)
        total=self.store.rows('SELECT COUNT(*) n FROM cs_repository_files WHERE '+where,params)[0]['n']
        rows=self.store.rows('SELECT path,hash,bytes,language,parser,status,error,version,updated_at FROM cs_repository_files WHERE '+where+' ORDER BY path LIMIT ? OFFSET ?',(*params,limit,offset))
        return {'items':rows,'total':total,'offset':offset,'limit':limit}

    def symbols(self,owner,workspace,key,offset,limit,query='',path=None):
        self.store.repository(owner,workspace,key);where='repository=?';params=[key]
        if query:where+=' AND (instr(name,?)>0 OR instr(qualified_name,?)>0)';params.extend([query,query])
        if path:where+=' AND path=?';params.append(path)
        total=self.store.rows('SELECT COUNT(*) n FROM cs_repository_symbols WHERE '+where,params)[0]['n']
        rows=self.store.rows('SELECT * FROM cs_repository_symbols WHERE '+where+' ORDER BY path,start_line LIMIT ? OFFSET ?',(*params,limit,offset))
        return {'items':rows,'total':total,'offset':offset,'limit':limit}

    def exact_candidates(self,repository_id,query):
        terms=set(re.findall(r'[A-Za-z_][A-Za-z0-9_.\/]*',query))
        result=[]
        for term in sorted(terms)[:12]:
            result.extend(r['uid'] for r in self.store.rows('SELECT uid FROM cs_repository_symbols WHERE repository=? AND (name=? OR qualified_name=? OR path=?) LIMIT 12',(repository_id,term,term,term)))
            result.extend(r['uid'] for r in self.store.rows('SELECT uid FROM cs_repository_links WHERE repository=? AND (target_symbol IN (SELECT uid FROM cs_repository_symbols WHERE repository=? AND name=?) OR path=?) LIMIT 12',(repository_id,repository_id,term,term)))
        if any(x in query.lower() for x in ('主要模块','仓库结构','repository map')):
            result.extend(r['logical_uid'] for r in self.state._rows("SELECT logical_uid FROM memories WHERE active=1 AND memory_type='repository_map' AND json_extract(metadata,'$.repository_id')=? LIMIT 1",(repository_id,)))
        return list(dict.fromkeys(result))[:24]

    def history(self,owner,workspace,key,path,offset,limit):
        self.store.repository(owner,workspace,key)
        scope=self.store.scope(owner,workspace,'code',key)
        params=[scope.key,path]
        predicate="scope=? AND memory_type='repository_chunk' AND json_extract(metadata,'$.file_path')=?"
        total=self.state._rows('SELECT COUNT(*) n FROM memories WHERE '+predicate,params)[0]['n']
        rows=self.state._rows('SELECT logical_uid FROM memories WHERE '+predicate+' ORDER BY created_at_ms DESC LIMIT ? OFFSET ?',params+[limit,offset])
        items=[self.platform.serialize(scope,m,detail=True) for m in self.state.get_memories(scope,[r['logical_uid'] for r in rows]).values()]
        return {'items':items,'total':total,'offset':offset,'limit':limit}
