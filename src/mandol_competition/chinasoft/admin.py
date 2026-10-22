"""Local operator only: explicit provisioning and reversible guest ownership claims.

Passwords never appear in arguments or logs. This command never starts workers.
"""
import argparse, getpass, os, sqlite3
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from .config import Settings
from .auth import Accounts
from .store import AppStore,now,uid,AppError

class LocalState:
    def __init__(self,path):
        self.db=sqlite3.connect(path,timeout=10,isolation_level=None);self.db.row_factory=sqlite3.Row
    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:yield self.db;self.db.execute('COMMIT')
        except Exception:self.db.execute('ROLLBACK');raise
    def _rows(self,sql,args=()):return [dict(r) for r in self.db.execute(sql,args)]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['create-admin','create-user','approve','legacy-list','claim','unclaim','grant-repository','disable','reset-password'])
    parser.add_argument('--username');parser.add_argument('--password-file',type=Path);parser.add_argument('--owner');parser.add_argument('--confirm-owner');parser.add_argument('--admin-username')
    args=parser.parse_args();settings=Settings();state=LocalState(settings.data_dir/'state.sqlite3')
    from .migrations import migrate
    migrate(state)
    store=AppStore.__new__(AppStore);store.state=state;store.settings=settings
    from .access import Access
    store.access=Access(store);accounts=Accounts(store)
    if args.command in ('create-admin','create-user'):
        if args.password_file:
            if args.password_file.stat().st_mode&0o077:parser.error('password file must be private (0600)')
            password=args.password_file.read_text().strip()
        else:password=getpass.getpass('密码（至少12字符）: ')
        a=accounts.create(args.username or '',password,admin=args.command=='create-admin');print('账号已创建:',a['username'],a['status']);return
    if args.command=='legacy-list':
        for r in store.rows('SELECT w.owner,COUNT(*) workspaces FROM cs_workspaces w LEFT JOIN cs_account_owners a ON a.owner=w.owner WHERE a.owner IS NULL GROUP BY w.owner'):print(r)
        return
    rows=store.rows('SELECT * FROM cs_accounts WHERE username=?',(args.username or '',))
    if not rows:parser.error('account not found')
    account=rows[0]
    if args.command in ('claim','unclaim'):
        actor=store.rows("SELECT id FROM cs_accounts WHERE username=? AND role='admin' AND status='active'",(args.admin_username or '',))
        if not actor or not args.owner or args.confirm_owner!=args.owner:parser.error('explicit admin and matching --owner / --confirm-owner required')
        if args.owner==account['id']:parser.error('cannot remove primary owner')
        with state.transaction() as db:
            if args.command=='claim':
                if not db.execute('SELECT 1 FROM cs_workspaces WHERE owner=?',(args.owner,)).fetchone():parser.error('legacy owner not found')
                if db.execute('SELECT 1 FROM cs_accounts WHERE id=?',(args.owner,)).fetchone():parser.error('cannot claim another account')
                existing=db.execute('SELECT account FROM cs_account_owners WHERE owner=?',(args.owner,)).fetchone()
                if existing and existing['account']!=account['id']:parser.error('already claimed by another account')
                db.execute('INSERT OR IGNORE INTO cs_account_owners VALUES (?,?,?)',(account['id'],args.owner,now()))
            else:db.execute('DELETE FROM cs_account_owners WHERE account=? AND owner=?',(account['id'],args.owner))
            db.execute('INSERT INTO cs_admin_audit(id,account,action,subject,created_at) VALUES (?,?,?,?,?)',(uid('audit_'),actor[0]['id'],args.command,args.owner,now()))
        print('认领关系已更新；原始记忆、Scope 和来源未改写。');return
    if args.command=='reset-password':
        from .auth import password_hash
        if args.password_file:
            if args.password_file.stat().st_mode&0o077:parser.error('password file must be private (0600)')
            password=args.password_file.read_text().strip()
        else:password=getpass.getpass('新密码（至少12字符）: ')
        encoded=password_hash(password)
        with state.transaction() as db:
            db.execute('UPDATE cs_accounts SET password_hash=? WHERE id=?',(encoded,account['id']))
            db.execute('UPDATE cs_auth_sessions SET revoked=1 WHERE account=?',(account['id'],))
            Access.audit(db,'local-operator:'+str(os.getuid()),'password_reset',account['id'])
        print('密码已更新，旧登录会话已撤销');return
    with state.transaction() as db:
        if args.command=='disable' and account['role']=='admin' and account['status']=='active' and db.execute("SELECT COUNT(*) FROM cs_accounts WHERE role='admin' AND status='active'").fetchone()[0]<=1:
            raise AppError('不能禁用最后一位有效管理员',409,code='last_admin')
        if args.command=='grant-repository':db.execute('UPDATE cs_accounts SET local_repository=1 WHERE id=?',(account['id'],))
        else:
            db.execute('UPDATE cs_accounts SET status=? WHERE id=?',('active' if args.command=='approve' else 'disabled',account['id']))
            if args.command=='disable':db.execute('UPDATE cs_auth_sessions SET revoked=1 WHERE account=?',(account['id'],))
        Access.audit(db,'local-operator:'+str(os.getuid()),args.command,account['id'])
    print('账号状态已更新')

if __name__=='__main__':main()
