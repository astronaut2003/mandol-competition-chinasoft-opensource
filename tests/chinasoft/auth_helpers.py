"""V1/V2 regression tests now use real V3 credentials and CSRF cookies."""
import uuid
from mandol_competition.chinasoft.auth import COOKIE

def login_test(client,platform):
    username='regression-'+uuid.uuid4().hex[:12];a=platform.accounts.create(username,'regression-password-123',admin=True)
    w=platform.accounts.workspaces(a['id'])[0]
    platform.store.create_session(a['id'],w['id'],'textual','旧格式回归会话')
    client.headers['X-Mandol-Client']='studio'
    r=client.post('/api/auth/login',json={'username':username,'password':'regression-password-123'});assert r.status_code==200
    data=r.json();client.headers['X-CSRF-Token']=data['csrf_token'];return data

def api_owner(client,platform):return platform.accounts.authenticate(client.cookies.get(COOKIE))['id']
