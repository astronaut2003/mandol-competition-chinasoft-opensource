"""Private accounts over the existing application state. No legacy guest login."""
import hashlib, hmac, re, secrets, sqlite3
from threading import BoundedSemaphore
from argon2 import PasswordHasher, Type
from argon2.exceptions import VerificationError, InvalidHashError
from .store import AppError, now, uid

HASHER=PasswordHasher(time_cost=3,memory_cost=65536,parallelism=1,type=Type.ID)
HASH_LIMIT=BoundedSemaphore(2)
DUMMY=HASHER.hash(secrets.token_urlsafe(32))
COOKIE='chinasoft_account'

def hashed(value): return hashlib.sha256(value.encode()).hexdigest()

def password_hash(password):
    if not 12<=len(password)<=128: raise AppError('密码长度须为 12–128 个字符')
    if len(set(password))<5 or sum(bool(re.search(p,password)) for p in (r'[a-zA-Z]',r'[0-9]',r'[^a-zA-Z0-9]'))<2:
        raise AppError('密码请使用至少两类字符，并避免重复字符',422)
    with HASH_LIMIT:return HASHER.hash(password)

class Accounts:
    def __init__(self,store):self.store,self.settings=store,store.settings

    def limit(self,key,maximum=10):
        timestamp=now();key=hashed(key)
        with self.store.state.transaction() as db:
            db.execute('DELETE FROM cs_auth_limits WHERE resets_at<?',(timestamp,))
            old=db.execute('SELECT * FROM cs_auth_limits WHERE key=?',(key,)).fetchone()
            if old and old['hits']>=maximum:raise AppError('尝试过于频繁，请稍后再试',429,code='rate_limited')
            if not old:
                if db.execute('SELECT COUNT(*) FROM cs_auth_limits').fetchone()[0]>=10000:raise AppError('认证繁忙，请稍后重试',429)
                db.execute('INSERT INTO cs_auth_limits VALUES (?,1,?)',(key,timestamp+300000))
            else:db.execute('UPDATE cs_auth_limits SET hits=hits+1 WHERE key=?',(key,))

    def create(self,username,password,display_name='',*,admin=False,actor=None,approved=False):
        username=username.strip().lower()
        if not re.fullmatch(r'[a-z0-9][a-z0-9_.@+-]{2,63}',username):raise AppError('账号须为 3–64 位字母、数字或 . _ @ + -')
        encoded=password_hash(password);account=secrets.token_hex(16)
        try:
            with self.store.state.transaction() as db:
                if actor:
                    from .access import Access
                    Access.require_admin(db,actor)
                if db.execute('SELECT COUNT(*) FROM cs_accounts').fetchone()[0]>=self.settings.max_accounts:
                    raise AppError('平台账号容量已满，请联系管理员',429,code='account_capacity')
                db.execute('INSERT INTO cs_accounts(id,username,password_hash,display_name,role,status,local_repository,created_at) VALUES (?,?,?,?,?,?,?,?)',(account,username,encoded,(display_name or username)[:60],'admin' if admin else 'user','active' if admin or approved or not self.settings.registration_approval else 'pending',int(admin),now()))
                db.execute('INSERT INTO cs_account_owners VALUES (?,?,?)',(account,account,now()))
                db.execute('INSERT INTO cs_workspaces VALUES (?,?,?,?,?)',(uid('ws_'),account,'我的记忆工作区',0,now()))
                from .access import Access
                Access.audit(db,actor or account,'account_create',account,{'role':'admin' if admin else 'user'})
        except sqlite3.IntegrityError:raise AppError('无法创建该账号，请检查信息或联系管理员',409,code='registration_failed')
        # A private initial workspace; existing guest workspaces are never selected.
        return self.public(self.store.rows('SELECT * FROM cs_accounts WHERE id=?',(account,))[0])

    @staticmethod
    def public(row):return {k:row[k] for k in ('id','username','display_name','role','status','local_repository')}

    def login(self,username,password,peer):
        normalized=username.strip().lower();self.limit('ip:'+peer,60)
        rows=self.store.rows('SELECT * FROM cs_accounts WHERE username=?',(normalized,));row=rows[0] if rows else None
        try:
            with HASH_LIMIT:valid=HASHER.verify(row['password_hash'] if row else DUMMY,password)
        except (VerificationError,InvalidHashError):valid=False
        if not row or not valid or row['status']!='active':
            self.limit('login:'+normalized,10)
            raise AppError('账号或密码错误，或账号尚未获准登录',401,code='invalid_credentials')
        token=secrets.token_urlsafe(48);csrf=secrets.token_urlsafe(32);expires=now()+self.settings.session_ttl*1000
        with self.store.state.transaction() as db:
            db.execute('DELETE FROM cs_auth_sessions WHERE expires_at<? OR revoked=1',(now(),))
            db.execute('INSERT INTO cs_auth_sessions VALUES (?,?,?,?,0,?)',(hashed(token),row['id'],csrf,expires,now()))
            db.execute('UPDATE cs_accounts SET last_login_at=? WHERE id=?',(now(),row['id']))
            db.execute('UPDATE cs_auth_sessions SET revoked=1 WHERE account=? AND token_hash IN (SELECT token_hash FROM cs_auth_sessions WHERE account=? ORDER BY created_at DESC LIMIT -1 OFFSET 20)',(row['id'],row['id']))
        return token,csrf,self.public(row)

    def authenticate(self,token):
        if not token or len(token)>128:return None
        rows=self.store.rows('SELECT a.*,s.csrf,s.expires_at FROM cs_auth_sessions s JOIN cs_accounts a ON a.id=s.account WHERE s.token_hash=? AND s.revoked=0 AND s.expires_at>? AND a.status=\'active\'',(hashed(token),now()))
        return rows[0] if rows else None

    def revoke(self,token):
        with self.store.state.transaction() as db:db.execute('UPDATE cs_auth_sessions SET revoked=1 WHERE token_hash=?',(hashed(token or ''),))

    def owners(self,account):return [r['owner'] for r in self.store.rows('SELECT owner FROM cs_account_owners WHERE account=?',(account,))]

    def owner_for(self,account,workspace=None):
        if not workspace:return account
        rows=self.store.rows('SELECT w.owner FROM cs_workspaces w JOIN cs_account_owners a ON a.owner=w.owner WHERE a.account=? AND w.id=?',(account,workspace))
        if not rows:raise AppError('资源不存在或无权访问',404,code='not_found')
        return rows[0]['owner']

    def workspaces(self,account):return self.store.rows('SELECT w.* FROM cs_workspaces w JOIN cs_account_owners a ON a.owner=w.owner WHERE a.account=? ORDER BY w.created_at',(account,))
