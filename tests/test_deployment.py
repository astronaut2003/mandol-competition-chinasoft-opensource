"""Release isolation regressions; no private checkout or external model required."""
from pathlib import Path
import importlib.util,os,socket,subprocess
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('release_manage',ROOT/'scripts/manage.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_application_does_not_adopt_unknown_listener():
 with socket.socket() as s:
  s.bind(('127.0.0.1',0));s.listen();assert not module.port_free(s.getsockname()[1])

def test_guard_rejects_reserved_local_deployments():
 from mandol_competition.chinasoft.config import Settings
 import pytest
 with pytest.raises(RuntimeError,match='reserved local'):
  module.guard(Settings(embedding_url='http://127.0.0.1:19631/v1'))

def test_reused_pid_is_not_owned():
 record={'pid':os.getpid(),'ticks':'0','boot':'wrong'}
 assert not module.process(record)

def test_explicit_environment_wins_without_reading_private_config():
 result=subprocess.run(['bash','-c','source scripts/_common.sh; printf "%s" "$CHINASOFT_PORT"'],cwd=ROOT,env={**os.environ,'CHINASOFT_PORT':'20777'},text=True,capture_output=True,check=True)
 assert result.stdout=='20777'


def test_public_demo_contains_only_fresh_synthetic_sources(tmp_path):
 from mandol_competition.chinasoft.config import Settings
 from mandol_competition.chinasoft.service import Platform
 from mandol_competition.chinasoft.demo import seed_demo
 p=Platform(Settings(data_dir=tmp_path,allow_retrieval_degraded=True))
 try:
  account=p.accounts.create('synthetic-demo','Synthetic_Example_42',admin=True)
  result=seed_demo(p,account['id'])
  assert len(result['sessions'])==3 and len(result['receipts'])==3
  assert all(x['memory_ids'] for x in result['receipts'])
 finally:p.close()


def test_model_download_help_does_not_depend_on_application_environment():
 import sys
 result=subprocess.run([sys.executable,'-I','-S',str(ROOT/'scripts/download_models.py'),'--help'],cwd=ROOT,text=True,capture_output=True,check=True)
 assert '--channel' in result.stdout
