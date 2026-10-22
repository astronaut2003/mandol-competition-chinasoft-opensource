"""Deterministic, multi-label routing of original evidence, never inferred truth."""
import re
from mandol_substrate.state.models import digest

ROUTING_RECIPE=digest(['conversation-first-multilabel-v31', 'assertion-candidates-not-questions', 'user-utterance-only-profile','attachment-type-validated'])
ENGINEERING=re.compile(r'(修复|报错|bug|traceback|回归测试|幂等|并发|设计决策|测试通过|单元测试|实现了|数据库|\bfix(?:ed)?\b|\btest(?:ed)?\b)',re.I)
CODE=re.compile(r'```(?:python|py|javascript|typescript|ts|js|rust|go|java|cpp)?\s*\n|^\s*(?:def |class |import |from \S+ import |export (?:function|class)|function \w)',re.M)

def assertion_candidate(text):
    # Conservative candidate routing only. Downstream schema/provenance review
    # still decides whether any fact is eligible; this rule proves no fact.
    statement=re.search(r'(我(?:叫|喜欢|偏好|买了|刚买|已经|昨天|今天|上周|正在|计划|准备)|我们(?:决定|已|选择|计划)|修复了|测试通过|验证结果|原因是|\bI (?:am|like|prefer|bought|fixed)\b)',text,re.I)
    if statement:return True
    if re.fullmatch(r'\s*(你好|您好|嗨|hello|hi|谢谢|thanks)[!！。,.，\s]*',text,re.I):return False
    if re.search(r'[?？]\s*$',text) or re.match(r'\s*(你是什么|你是哪个|你用的|请|什么是|如何|怎么|为什么|能否|是否|帮我|告诉我|介绍一下|解释一下|总结一下|我之前说过|what |how |why |can you )',text,re.I):return False
    return bool(text.strip())

def routes(message):
    text=message.get('content','');role=message.get('role','user');result=[]
    if role=='user' and text.strip():
        # A fenced/pasted source file is evidence, never a source of user attributes.
        if not CODE.search(text) and assertion_candidate(text):result.append('conversation')
        if ENGINEERING.search(text) and assertion_candidate(text):result.append('engineering')
        if CODE.search(text):result.append('source_code')
    if message.get('attachment_id'):result.append('source_code' if message.get('attachment_kind')=='code' else 'document')
    if message.get('media_ids'):result.append('visual')
    return result

def query_mode(text,repository=None):
    if repository or re.search(r'(函数|源码|代码|模块|仓库|依赖|修复|幂等|\bclass\b|\bfunction\b|\bimport\b)',text,re.I):return 'code'
    if re.search(r'(图片|图像|照片|看板|ocr|视觉|可见|物体)',text,re.I):return 'multimodal'
    return 'textual'
