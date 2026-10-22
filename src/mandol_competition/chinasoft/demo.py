"""Freshly authored synthetic examples; no production users or model answers."""
from .config import ROOT
from .schemas import AddMemory

TEXTUAL=[('偏好与更正',[
 {'role':'user','content':'这是一组虚构演示资料：演示用户在工作日使用蓝色笔记本，周末使用绿色笔记本。'},
 {'role':'user','content':'更正这组虚构资料：从2026年1月10日起，工作日改用黄色笔记本；周末仍使用绿色笔记本。'}])]
CODE=[{'role':'user','content':'合成工程记录：示例接口通过request_id与内容摘要识别请求重试，相同ID且内容不同应拒绝。该文本描述一个演示场景，不代表执行过真实测试。'}]

def seed_demo(platform,owner):
    store=platform.store;w=store.create_workspace(owner,'公开合成示例',demo=True);sessions=[];receipts=[]
    for i,(title,messages) in enumerate(TEXTUAL):
        s=store.create_session(owner,w['id'],'textual',title,unified=True);sessions.append(s)
        receipts.append(store.accept(owner,AddMemory(workspace_id=w['id'],session_id=s['id'],request_id=f'public-text-{i}',messages=messages)))
    s=store.create_session(owner,w['id'],'textual','合成工程记录',unified=True);sessions.append(s)
    receipts.append(store.accept(owner,AddMemory(workspace_id=w['id'],session_id=s['id'],request_id='public-code',messages=CODE)))
    s=store.create_session(owner,w['id'],'textual','合成状态板',unified=True);sessions.append(s)
    media=platform.save_media(owner,w['id'],s['id'],(ROOT/'demo/status_board.png').read_bytes())
    receipts.append(store.accept(owner,AddMemory(workspace_id=w['id'],session_id=s['id'],request_id='public-image',messages=[{'role':'user','content':'这是一张合成状态板，请按实际图片识别文字；它不是当前服务器状态。','media_ids':[media['id']]}])))
    return {'workspace':w,'sessions':sessions,'receipts':receipts,'questions':['虚构用户当前工作日用哪种颜色的笔记本？周末呢？','合成状态板上三个状态词分别是什么？']}
