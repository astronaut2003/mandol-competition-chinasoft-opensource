"""Bounded UTF-8 attachments: immutable content, explicit active versions, no execution."""
import csv, io, json, re, hashlib, tomllib
from pathlib import PurePosixPath
from .store import AppError, now
from mandol_substrate.state.models import digest

CODE={'.py','.ts','.tsx','.js','.jsx','.mjs','.cjs','.rs','.go','.java','.c','.h','.cpp','.hpp','.cc'}
TEXT={'.txt','.md','.rst','.json','.csv','.yaml','.yml','.toml','.sql','.css'}
SENSITIVE=re.compile(r"""(?ix)
    -----BEGIN\ [^\r\n]*PRIVATE\ KEY-----
    | \b(?:sk|sk-proj)-[a-zA-Z0-9_-]{16,}
    | \bgh[pousr]_[a-zA-Z0-9]{25,}
    | \bAKIA[A-Z0-9]{16}\b
    | (?:[A-Za-z_][A-Za-z_0-9]*_)?(?:api[_-]?key|secret(?:_key)?|password|(?:access[_-]?)?token)[\"']?
      \s*[:=]\s*(?:\"[^\"\r\n]{8,}\"|'[^'\r\n]{8,}'|[A-Za-z0-9_+/=-]{12,}(?=[\s,;}}\]]|$))
""")


def save_attachment(platform, owner, workspace, session, name, content, mime=''):
    platform.store.session(owner, workspace, session)
    if not name or '/' in name or '\\' in name or name in ('.','..') or any(ord(c)<32 or ord(c)==127 for c in name):
        raise AppError('附件仅接受文件名，不接受目录或相对路径',415)
    suffix=PurePosixPath(name).suffix.lower()
    if len(name)>100 or name.startswith('.') or suffix not in CODE|TEXT:
        raise AppError('支持 UTF-8 源码、Markdown、TXT、JSON、CSV、YAML、TOML；不支持 PDF 或压缩包',415)
    if not 0<len(content)<=platform.settings.attachment_max_bytes:
        raise AppError('文本附件超过配置的大小限制（最大 256 KiB）',413)
    mime=(mime or '').split(';')[0].strip().lower()
    allowed={'application/octet-stream','application/json','application/yaml','application/x-yaml','application/toml','application/javascript','application/typescript','application/x-python-code'}
    if mime and not (mime.startswith('text/') or mime in allowed):
        raise AppError('附件 MIME 与支持的文本/源码内容不一致',415)
    try:text=content.decode('utf-8-sig')
    except UnicodeDecodeError:raise AppError('附件必须为 UTF-8 文本',415)
    if '\x00' in text or any(ord(c)<9 or 13<ord(c)<32 for c in text):
        raise AppError('二进制或控制字符内容不可作为文本导入',415)
    if len(text)>32000:raise AppError('文本附件最多 32,000 字符',413)
    if SENSITIVE.search(text):raise AppError('附件疑似包含凭据，请脱敏后重试',422,code='sensitive_attachment')
    if suffix=='.json':
        try:json.loads(text,parse_constant=lambda _:(_ for _ in ()).throw(ValueError()))
        except (ValueError,RecursionError):raise AppError('JSON 内容无效',422)
    if suffix=='.toml':
        try:tomllib.loads(text)
        except (ValueError,RecursionError):raise AppError('TOML 内容无效',422)
    if suffix=='.csv':
        try:
            for i,row in enumerate(csv.reader(io.StringIO(text),strict=True)):
                if i>10000 or len(row)>200:raise ValueError()
        except (ValueError,csv.Error):raise AppError('CSV 结构超出限制',422)
    # YAML stays text. No unsafe object loader or configuration execution.
    sha=hashlib.sha256(content).hexdigest()
    key='att_'+digest([owner,workspace,session,sha,name]);kind='code' if suffix in CODE else 'document'
    from .repository import parse_source
    parsed=parse_source(name,text) if kind=='code' else {'parser':'utf8-validation-v31','error':None}
    parser=parsed['parser'] if not parsed.get('error') else 'text-chunks:syntax-error'
    with platform.state.transaction() as db:
        previous=db.execute('SELECT * FROM cs_attachments WHERE id=?',(key,)).fetchone()
        if not previous or not previous['active']:
            if not previous:platform.store.access.check(db,owner,'storage_bytes',len(content))
            if not previous:platform.store.access.check(db,owner,'attachment_files')
            if not previous and not platform.store.access.account_for(db,owner) and db.execute('SELECT COALESCE(SUM(bytes),0) FROM cs_attachments WHERE owner=?',(owner,)).fetchone()[0]+len(content)>20*1024*1024:
                raise AppError('账号附件容量已达 20 MiB',429)
            version=db.execute('SELECT COALESCE(MAX(version),0)+1 FROM cs_attachments WHERE owner=? AND workspace=? AND session=? AND name=?',(owner,workspace,session,name)).fetchone()[0]
            db.execute('UPDATE cs_attachments SET active=0,superseded_by=? WHERE owner=? AND workspace=? AND session=? AND name=? AND active=1',(key,owner,workspace,session,name))
            db.execute("INSERT INTO cs_attachments(id,owner,workspace,session,name,mime,kind,hash,text,parser,bytes,created_at,version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET active=1,superseded_by='',version=excluded.version,parser=excluded.parser",
                       (key,owner,workspace,session,name,mime or 'text/plain',kind,sha,text,parser,len(content),now(),version))
    return {k:v for k,v in platform.store.attachment(owner,key,workspace,session).items() if k not in ('owner','text','hash')}


def current_attachment(platform,memory):
    key=memory.metadata.get('attachment_id')
    if not key:return True
    rows=platform.store.rows('SELECT active,version FROM cs_attachments WHERE id=?',(key,))
    return bool(rows and rows[0]['active'] and rows[0]['version']==memory.metadata.get('attachment_version',1))
