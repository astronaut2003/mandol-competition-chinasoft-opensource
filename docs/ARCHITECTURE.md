# 架构与数据流

## 统一交互与类型化记忆

`studio/src/pages/Chat.tsx` 提供一个 Conversation Composer，支持文字、图片、源码和受支持的文档。内部 Session 是对话组织单位；Memory Type 是语义表示，两者不绑定为互斥场景。

`src/mandol_competition/chinasoft/api.py` 处理认证、Scope、幂等请求、SSE 和历史恢复；`conversation.py` 组织最近真实对话、当前附件、可选长期证据与自然语言生成。普通闲聊、身份询问和当前上下文追问有快速路径。联网查询规划在 `web_query_planner.py`，Search/Fetch 在 `web_tools.py`。

## Canonical 与投影

`packages/substrate/src/mandol_substrate/state/` 保存 CanonicalMemory、Scope、Mutation、显式 Edge、commit 与来源版本；`chinasoft/store.py` 保存账号、工作区、会话、消息、附件、仓库和任务。SQLite 是权威状态，Qdrant 只保存可重建 Dense/Sparse 投影。

`PublicationCoordinator` 与投影游标协调版本发布；已提交但投影未就绪不等于记录丢失。基础来源更新会使依赖旧来源版本的高阶记忆退出当前可用知识；历史记录仍可在授权审计视图查询。记忆有效性、索引就绪和某条 Lane 的候选资格是不同状态。

## 后台处理

`chinasoft/service.py` 的 Platform 启动持久化 ingest、index、derive、repository 任务处理。前台可靠接收消息和后台索引、高阶构建分开；模型失败不会撤销已可靠接收的输入。任务有幂等键、尝试次数、取消/恢复状态和配额。当前 Worker 是单 API 进程中的后台线程，模型调用通过 HTTP，不在线程内加载权重。

类型化构建复用 textual、code、multimodal 模块：画像只接受适格用户陈述；事件、关系和工程经验需要各自校验；视觉描述/OCR 保留真实图片来源。高阶记忆表示加工结果，不意味着事实可信度高于基础证据。

## 查询执行

`query_planner.py` 与 `typed_retrieval.py` 按多标签需求选择 Lane、候选预算、结构条件、范围和兜底。当前默认 `adaptive_v33`，保留 `baseline` 回退开关，不将所有研究候选称为普遍最优。

对话候选采用 Qwen Dense + SPLADE Sparse + BM25，经 RRF 合并和 BGE 统一重排序；代码优先路径/符号与已存结构，视觉区分 OCR 与语义描述。画像使用生命周期条件，事件只使用具有真实语义的时间字段，实体关系扩展有界且不得越过 Scope。

`native_retrieval.py` 验证独立 vLLM 模型和数值协议。`retrieval_contract.py` 将 revision、pooling、dtype、维度和 recipe 纳入 Profile。端口和部署地址不代表模型身份；不同稀疏表征不能写入同一个字段。

最终证据经过来源 UID、版本、有效性和权限验证。高阶候选负责定位知识，引用回溯真实基础材料；去重和上下文预算避免多个摘要与同一来源挤占窗口。检索 Trace 由实际执行记录，诊断数字不会从最终回答反推。

## 仓库与附件

`remote_git.py` 使用批准的公开 HTTPS 托管平台接口，有界下载源码，不执行代码、依赖安装、Hook、Submodule 或 LFS。仓库扫描和上传源码复用已有解析器与 Code Chunk/符号逻辑。无 AST 语言按真实能力降级为文本证据，不宣称完整结构分析。

## 授权与可视化

`auth.py`、`access.py`、`migrations.py` 维护账号、Argon2id、Cookie Session、CSRF、角色、配额和正式迁移。账号→Workspace→Conversation/Memory/Attachment/Repository/Job 的范围由服务端验证，管理员账户管理权不授予私有聊天读取权。

`semantic_schema.py` 将系统能力与当前范围实例统计分开。`visualization.py`、`memory_quality.py` 使用共享有效性和保守视图聚合；不改变实际 Memory UID。来源图与语义关系图分开，聚合展开仍能定位全部真实成员和已存边。

## 推理部署边界

Web 环境与统一推理环境各有独立锁文件。Dense、SPLADE、BGE 使用 vLLM 0.32.0 / Python 3.12，分别运行独立进程；模型权重从指定官方 revision 获取并校验。启动清单与实际 vLLM 配置共同形成元信息，正式客户端再用真实 HTTP 推理验证就绪。运行时版本参与版本化投影身份，升级时保留原 Canonical 状态与旧 Collection；进程地址和 GPU 预算不改变 Profile。

原生应用和 Linux host-network 应用容器均可访问同一组显式配置的回环 GPU 服务。SQLite 与 Qdrant 继续由每个应用部署独立持有；推理服务不持有用户数据或记忆索引。
