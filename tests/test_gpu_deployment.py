"""GPU deployment mechanics, not substitutes for real model inference."""
from pathlib import Path
import hashlib,importlib.util,json
import pytest
from mandol_competition.chinasoft.retrieval_contract import MODELS,verify_artifact,cache_snapshot

ROOT=Path(__file__).resolve().parents[1]

def test_hub_xet_cache_names_do_not_replace_content_verification(tmp_path,monkeypatch):
    # A Hub blob identifier is not necessarily its content SHA256. Both classic
    # and Xet cache layouts must validate actual bytes, including tamper rejection.
    recipe=dict(MODELS['dense']);data=b'explicitly synthetic checkpoint bytes'
    recipe['weights_sha256']=hashlib.sha256(data).hexdigest();monkeypatch.setitem(MODELS,'dense',recipe)
    cache=tmp_path/'hub';snap=cache_snapshot(cache,'dense');snap.mkdir(parents=True)
    blob=cache/'blobs'/'opaque-xet-id';blob.parent.mkdir();blob.write_bytes(data)
    (snap/recipe['weights']).symlink_to(blob)
    for n in ('config.json','tokenizer.json','tokenizer_config.json'):(snap/n).write_text('{}')
    assert verify_artifact(cache,'dense')['weights_sha256']==recipe['weights_sha256']
    blob.write_bytes(data+b'tampered')
    with pytest.raises(ValueError,match='pinned SHA256'):verify_artifact(cache,'dense')

def test_gpu_installer_isolates_workspace_constraints_and_checks_dependencies():
    script=(ROOT/'scripts/inference-setup.sh').read_text()
    assert 'uv pip sync --no-config' in script and '--require-hashes' in script
    assert 'uv pip check --no-config' in script and '--no-deps' not in script
    lock=(ROOT/'deploy/inference/vllm-unified.lock').read_text()
    assert '--hash=sha256:' in lock and 'protobuf==7.34.1' not in lock
    assert 'vllm==0.32.0' in lock and 'nvidia-cutlass-dsl==4.8.0' in lock
    assert "(3, 12)" in script and '--only-binary :all:' in script
    assert {m['version'] for m in MODELS.values()} == {'0.32.0'}


def test_start_failure_rolls_back_only_new_owned_channels(monkeypatch,tmp_path):
    spec=importlib.util.spec_from_file_location('model_lifecycle',ROOT/'scripts/chinasoft_vllm.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'RUN',tmp_path);monkeypatch.setattr(module,'preflight',lambda channels:[])
    monkeypatch.setattr(module,'owned',lambda c:{'pid':123} if c=='dense' else None)
    monkeypatch.setattr(module,'deployment',lambda c:{'managed':True});calls=[]
    def start(channel):
        if channel=='sparse':raise RuntimeError('explicit protocol startup failure')
        return {'channel':channel,'ready':True}
    monkeypatch.setattr(module,'start_one',start);monkeypatch.setattr(module,'stop_one',lambda c:calls.append(c))
    monkeypatch.setattr('sys.argv',['model_lifecycle','start'])
    with pytest.raises(RuntimeError,match='protocol startup'):module.main()
    assert calls==['sparse'] # preexisting dense must not be stopped
