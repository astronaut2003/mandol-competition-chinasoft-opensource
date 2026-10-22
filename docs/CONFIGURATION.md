# 配置参考

下表从 `chinasoft/config.py` 的真实 Settings 声明生成。`.env` 由本项目脚本加载，已有进程环境优先；直接调用 Python 时需自行加载配置。所有默认值仍受 Settings 的范围验证。空 API Key 不会产生模拟回答。

| 环境变量 | 配置字段/用途 | 默认值 |
|---|---|---|
| `CHINASOFT_DATA_DIR` | `data_dir` | `'data'` |
| `CHINASOFT_HOST` | `host` | `'127.0.0.1'` |
| `CHINASOFT_PORT` | `port` | `20600` |
| `CHINASOFT_QDRANT_URL` | `qdrant_url` | `'http://127.0.0.1:20633'` |
| `CHINASOFT_QDRANT_COLLECTION` | `collection` | `'chinasoft_opensource_memory_v1'` |
| `CHINASOFT_LLM_BASE_URL` | `llm_base_url` | `'https://api.deepseek.com/v1'` |
| `CHINASOFT_LLM_API_KEY_ENV` | `llm_api_key_env` | `'DEEPSEEK_API_KEY'` |
| `CHINASOFT_LLM_MODEL` | `llm_model` | `'deepseek-flash'` |
| `CHINASOFT_LLM_REQUEST_MODEL` | `llm_request_model` | `'deepseek-flash'` |
| `CHINASOFT_VLM_MODEL` | `vlm_model` | `'deepseek-flash'` |
| `CHINASOFT_VLM_REQUEST_MODEL` | `vlm_request_model` | `'deepseek-flash'` |
| `CHINASOFT_LLM_TIMEOUT` | `llm_timeout` | `90` |
| `CHINASOFT_LLM_MAX_CONCURRENCY` | `llm_max_concurrency` | `2` |
| `CHINASOFT_LLM_MAX_TOKENS` | `llm_max_tokens` | `4096` |
| `CHINASOFT_JSON_OUTPUT_MODE` | `json_output_mode` | `'json_object'` |
| `CHINASOFT_GENERATION_RETRY_POLICY` | `generation_retry_policy` | `3` |
| `CHINASOFT_LLM_ACCEPTED_RESPONSE_MODELS` | `accepted_response_models` | `'deepseek-flash'` |
| `CHINASOFT_REASONING_PROTOCOL` | `reasoning_protocol` | `'auto'` |
| `CHINASOFT_SEARCH_ENABLED` | `search_enabled` | `'false'` |
| `CHINASOFT_SEARCH_BASE_URL` | `search_base_url` | `'https://api.deepseek.com/v1'` |
| `CHINASOFT_SEARCH_API_KEY_ENV` | `search_api_key_env` | `'DEEPSEEK_API_KEY'` |
| `CHINASOFT_SEARCH_TIMEOUT` | `search_timeout` | `12` |
| `CHINASOFT_SEARCH_MAX_QUERIES` | `search_max_queries` | `2` |
| `CHINASOFT_SEARCH_PLANNER_ENABLED` | `search_planner_enabled` | `'true'` |
| `CHINASOFT_SEARCH_PLANNER_TIMEOUT` | `search_planner_timeout` | `4` |
| `CHINASOFT_SEARCH_MAX_RESULTS` | `search_max_results` | `5` |
| `CHINASOFT_SEARCH_MAX_FETCH` | `search_max_fetch` | `0` |
| `CHINASOFT_SEARCH_CONTEXT_CHARS` | `search_context_chars` | `10000` |
| `CHINASOFT_SEARCH_FETCH_HOSTS` | `search_fetch_hosts` | `'api-docs.deepseek.com,docs.python.org'` |
| `CHINASOFT_WEB_REQUESTS_PER_DAY` | `web_requests_per_day` | `1000` |
| `CHINASOFT_WEB_WORKSPACE_PER_DAY` | `web_workspace_per_day` | `100` |
| `CHINASOFT_VLM_ACCEPTED_RESPONSE_MODELS` | `vlm_accepted_response_models` | `'deepseek-flash'` |
| `CHINASOFT_RETRIEVAL_BACKEND` | `retrieval_backend` | `'vllm'` |
| `CHINASOFT_EMBEDDING_BASE_URL` | `embedding_url` | `'http://127.0.0.1:20631/v1'` |
| `CHINASOFT_EMBEDDING_API_KEY_ENV` | `embedding_api_key_env` | `''` |
| `CHINASOFT_EMBEDDING_MODEL` | `embedding_model` | `'Qwen/Qwen3-Embedding-0.6B'` |
| `CHINASOFT_EMBEDDING_DIMENSION` | `embedding_dimension` | `1024` |
| `CHINASOFT_RERANKER_BASE_URL` | `reranker_url` | `'http://127.0.0.1:20632'` |
| `CHINASOFT_RERANKER_API_KEY_ENV` | `reranker_api_key_env` | `''` |
| `CHINASOFT_RERANKER_MODEL` | `reranker_model` | `'BAAI/bge-reranker-v2-m3'` |
| `CHINASOFT_SPLADE_URL` | `sparse_url` | `'http://127.0.0.1:20636'` |
| `CHINASOFT_SPLADE_API_KEY_ENV` | `sparse_api_key_env` | `''` |
| `CHINASOFT_DENSE_TIMEOUT` | `dense_timeout` | `30` |
| `CHINASOFT_SPARSE_TIMEOUT` | `sparse_timeout` | `30` |
| `CHINASOFT_RERANKER_TIMEOUT` | `reranker_timeout` | `30` |
| `CHINASOFT_DENSE_CONCURRENCY` | `dense_concurrency` | `2` |
| `CHINASOFT_SPARSE_CONCURRENCY` | `sparse_concurrency` | `2` |
| `CHINASOFT_RERANKER_CONCURRENCY` | `reranker_concurrency` | `2` |
| `CHINASOFT_ALLOW_RETRIEVAL_DEGRADED` | `allow_retrieval_degraded` | `'false'` |
| `CHINASOFT_MEMORY_QUERY_STRATEGY` | `memory_query_strategy` | `'adaptive_v33'` |
| `CHINASOFT_CHAT_HISTORY_CHARS` | `chat_history_chars` | `'16000'` |
| `CHINASOFT_CHAT_ATTACHMENT_CHARS` | `chat_attachment_chars` | `'16000'` |
| `CHINASOFT_ATTACHMENT_MAX_BYTES` | `attachment_max_bytes` | `'262144'` |
| `CHINASOFT_ATTACHMENT_MAX_FILES` | `attachment_max_files` | `'4'` |
| `CHINASOFT_QUEUE_CAPACITY` | `queue_capacity` | `192` |
| `CHINASOFT_WORKER_MAX_ATTEMPTS` | `worker_max_attempts` | `3` |
| `CHINASOFT_SECURE_COOKIE` | `secure_cookie` | `'false'` |
| `CHINASOFT_REGISTRATION_ENABLED` | `registration_enabled` | `'false'` |
| `CHINASOFT_REGISTRATION_APPROVAL` | `registration_approval` | `'true'` |
| `CHINASOFT_SESSION_TTL` | `session_ttl` | `604800` |
| `CHINASOFT_ACCOUNT_JOB_QUOTA` | `account_job_quota` | `96` |
| `CHINASOFT_MAX_ACCOUNTS` | `max_accounts` | `100` |
| `CHINASOFT_QUOTA_WORKSPACES` | `quota_workspaces` | `10` |
| `CHINASOFT_QUOTA_CONVERSATIONS` | `quota_conversations` | `300` |
| `CHINASOFT_QUOTA_REPOSITORIES` | `quota_repositories` | `5` |
| `CHINASOFT_QUOTA_STORAGE_BYTES` | `quota_storage_bytes` | `100 * 1024 * 1024` |
| `CHINASOFT_QUOTA_CHAT_PER_MINUTE` | `quota_chat_per_minute` | `10` |
| `CHINASOFT_QUOTA_MODEL_PER_DAY` | `quota_model_per_day` | `300` |
| `CHINASOFT_MODEL_REQUESTS_PER_DAY` | `model_requests_per_day` | `2000` |
| `CHINASOFT_REMOTE_GIT_ENABLED` | `remote_git_enabled` | `'true'` |
| `CHINASOFT_REMOTE_GIT_MAX_FILES` | `remote_git_max_files` | `500` |
| `CHINASOFT_REMOTE_GIT_MAX_BYTES` | `remote_git_max_bytes` | `20 * 1024 * 1024` |
| `CHINASOFT_REMOTE_GIT_TIMEOUT` | `remote_git_timeout` | `180` |
| `CHINASOFT_REMOTE_GIT_CACHE_BYTES` | `remote_git_cache_bytes` | `256 * 1024 * 1024` |
| `CHINASOFT_DEMO_ENABLED` | `demo_enabled` | `'false'` |
| `CHINASOFT_REPOSITORY_ROOTS` | `repository_roots` | `''` |
| `CHINASOFT_REPOSITORY_EXCLUDES` | `repository_excludes` | `'.release-private/*'` |
| `CHINASOFT_REPOSITORY_MAX_FILES` | `repository_max_files` | `2500` |
| `CHINASOFT_REPOSITORY_MAX_FILE_BYTES` | `repository_max_file_bytes` | `256 * 1024` |
| `CHINASOFT_REPOSITORY_MAX_TOTAL_BYTES` | `repository_max_total_bytes` | `12 * 1024 * 1024` |
| `CHINASOFT_REPOSITORY_MAX_CHUNKS` | `repository_max_chunks` | `6000` |

## 凭据与启动器

`DEEPSEEK_API_KEY` 为用户自己的生成/视觉凭据，默认空。`*_API_KEY_ENV` 配置的是环境变量名称，不是密钥本身。可为 Search、Dense、Sparse、BGE 分别使用独立凭据环境变量。不要把凭据写进 URL、前端或 Git。

| 变量 | 用途与默认值 |
|---|---|
| `MANDOL_COMPOSE_PROJECT` | 本机生命周期脚本专用命名空间，默认 mandol-os-native，必须以 mandol-os- 开头 |
| `MANDOL_QDRANT_PORT` | Qdrant 主机端口，20633 |
| `CHINASOFT_CONTAINER_DENSE_URL` / `...SPARSE_URL` / `...RERANKER_URL` | 容器连接的独立模型地址，见模板 |
| `CHINASOFT_MODEL_CACHE` | 模型缓存，data/models/hub |
| `CHINASOFT_DENSE_MANAGED` | false；只在显式 inference-start 中启动 |
| `CHINASOFT_DENSE_GPU` | 0；CUDA 设备标识 |
| `CHINASOFT_DENSE_GPU_MEMORY` | 见 retrieval_contract.py 的该通道预算 |
| `CHINASOFT_DENSE_MAX_SEQS` | 2；vLLM 最大并发序列 |
| `CHINASOFT_DENSE_MAX_BATCHED_TOKENS` | 见该通道 max_length |
| `CHINASOFT_DENSE_VLLM_PYTHON` | 项目 .venv-vllm-029/bin/python；BGE 为 .venv-vllm-025/bin/python |
| `CHINASOFT_SPARSE_MANAGED` | false；只在显式 inference-start 中启动 |
| `CHINASOFT_SPARSE_GPU` | 0；CUDA 设备标识 |
| `CHINASOFT_SPARSE_GPU_MEMORY` | 见 retrieval_contract.py 的该通道预算 |
| `CHINASOFT_SPARSE_MAX_SEQS` | 2；vLLM 最大并发序列 |
| `CHINASOFT_SPARSE_MAX_BATCHED_TOKENS` | 见该通道 max_length |
| `CHINASOFT_SPARSE_VLLM_PYTHON` | 项目 .venv-vllm-029/bin/python；BGE 为 .venv-vllm-025/bin/python |
| `CHINASOFT_RERANKER_MANAGED` | false；只在显式 inference-start 中启动 |
| `CHINASOFT_RERANKER_GPU` | 0；CUDA 设备标识 |
| `CHINASOFT_RERANKER_GPU_MEMORY` | 见 retrieval_contract.py 的该通道预算 |
| `CHINASOFT_RERANKER_MAX_SEQS` | 2；vLLM 最大并发序列 |
| `CHINASOFT_RERANKER_MAX_BATCHED_TOKENS` | 见该通道 max_length |
| `CHINASOFT_RERANKER_VLLM_PYTHON` | 项目 .venv-vllm-029/bin/python；BGE 为 .venv-vllm-025/bin/python |

## 使用示例

```dotenv
CHINASOFT_REGISTRATION_ENABLED=true
CHINASOFT_REGISTRATION_APPROVAL=false
CHINASOFT_SEARCH_ENABLED=true
# 使用单独的凭据环境变量名称
CHINASOFT_SEARCH_API_KEY_ENV=MANDOL_SEARCH_SECRET
```

公开服务必须自行配置 HTTPS 与 Secure Cookie。端点需使用不含认证信息的 HTTP(S) URL；Search 要求 HTTPS。默认本机只监听回环。绑定容器内部 0.0.0.0 与开放主机公网是两回事。

用户/Workspace 限额在后端执行。`MEMORY_QUERY_STRATEGY=baseline` 是查询策略回退；回退不修改记忆身份、来源版本或索引数值协议。默认自适应策略不代表每种 Lane 都必须调用全部三个模型。

`CHINASOFT_STUDIO_DIST` 由 API 静态文件入口读取，默认项目 `studio/dist`；容器设置为 `/app/studio/dist`。它不会改变 Vite 开发代理。`CHINASOFT_VLLM_MANIFEST_FILE` 由模型启动器为其拥有的服务设置，供元信息中间件校验，不是用户绕过模型身份验证的开关。

`CHINASOFT_APP_ROOT` 指定安装后的资源根目录，默认由源码位置推导。容器使用 `/app`，让非 editable 安装仍能找到公开合成图片和前端；不要指向其他项目。

## 推理安装与公开模型访问

`HF_TOKEN` 仅由显式模型下载命令读取，用于已经获得访问权限的 Hugging Face 模型。默认不读取全局 HF 登录，不保存 Token；SPLADE-v3 的门禁条款需由模型使用者在官方页面接受。下载器固定使用官方 Hugging Face 源，不继承全局 `HF_ENDPOINT` 镜像。它不参与 FastAPI 或检索 Profile。`UV_CACHE_DIR` 与 `UV_PYTHON_INSTALL_DIR` 由脚本固定到本项目 `.cache/uv`、`.toolchains`；推理安装使用独立 `--no-config` 和 hash lock。

Linux `docker-compose.loopback.yml` 使用 host network 访问宿主回环模型，读取 `CHINASOFT_EMBEDDING_BASE_URL`、`CHINASOFT_SPLADE_URL`、`CHINASOFT_RERANKER_BASE_URL`。基础 bridge Compose 使用 `CHINASOFT_CONTAINER_*_URL`；两种网络模式不要混用地址假设。
