# GPU 检索模型部署

Mandol 在独立 HTTP 进程中运行模型，FastAPI 只调用端点。三个通道沿用固定模型、revision 和数值配方；应用环境与推理环境分别安装。

## 模型与数值契约

| 通道 | 模型 / revision | 固定数值语义 | 运行时 / 默认端口 |
|---|---|---|---|
| Dense | `Qwen/Qwen3-Embedding-0.6B` / `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | BF16、LAST pooling、1024 维、L2 normalize、4096 tokens；查询使用既有 instruction | vLLM 0.29.0 / 20631 |
| Sparse | `naver/splade-v3` / `fdfeceb91d7b9de7985b38addd3ba9f53a59a355` | BF16、全 token `max(log1p(ReLU(logits)))`、30522 词表、512 tokens；查询与文档使用同一算法 | vLLM 0.29.0 / 20636 |
| Reranker | `BAAI/bge-reranker-v2-m3` / `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | BF16、成对输入、CLS logit + sigmoid、4096 tokens；分数越高越相关 | vLLM 0.25.1 / 20632 |

`src/mandol_competition/chinasoft/retrieval_contract.py` 保存权重 SHA256、tokenizer/config 身份、pooling 和各 memory family 的 Profile。版本、算法与权重参与 Profile 身份，端点地址不参与。更换端口不需要重建；更换数值配方不能直接复用旧投影。

## 运行条件

本机安装支持 Linux x86_64、Python 3.12 和 NVIDIA GPU。先确认 `nvidia-smi` 可用、目标 GPU 有足够空闲显存，并为 Python wheel、模型与缓存预留数十 GB 磁盘。锁文件提供官方 PyPI wheel 的完整依赖与哈希，不要求宿主安装 CUDA Toolkit；PyTorch wheel 携带 CUDA runtime，宿主驱动仍必须兼容。

安装器会实际执行 CUDA 张量运算。成功安装依赖不能替代模型验证。设备显存需求包含权重、CUDA 上下文、运行时与激活；本项目不承诺所有低显存 GPU 均可运行三个服务。启动前检查 GPU_MEMORY 总预算与额外 2048 MiB 余量，资源不足时报告诊断，保留其他项目进程。

## 1. 安装独立环境

```bash
./scripts/setup.sh
./scripts/inference-setup.sh
```

推理安装使用 `deploy/inference/vllm-029.lock` 和 `vllm-025.lock`，分别创建 `.venv-vllm-029` 与 `.venv-vllm-025`，执行 `uv pip sync --no-config --require-hashes`、`uv pip check --no-config` 和实际 CUDA 运算。重复执行可复用本工程公开下载缓存，不从 Web 环境复制包。

若只安装某一组，可使用 `./scripts/inference-setup.sh 029` 或 `025`。共享 uv 缓存的首次大型下载建议顺序安装，避免并行安装等待分发缓存锁。

### 为什么需要 `--no-config`

Web 项目的 uv 配置固定了 protobuf 7.34.1。uv 即使通过 `--python` 指向另一个虚拟环境，仍可能自动读取当前 workspace 的约束。vLLM 0.29.0 的官方依赖并没有固定 protobuf 7.34.1，而其 CUTLASS 4.6.2 依赖链要求 protobuf <7。独立推理锁选择 protobuf 6.33.6，隔离 workspace 配置后可正常解析。BGE 的 0.25.1 依赖集合使用自己的锁文件，不强制套用另一版本的 protobuf。

此处修正了早期文档将 workspace 约束误判为上游依赖冲突的描述。没有使用 `--no-deps`、强制忽略冲突或修改模型配方。Web 自己的锁文件保持独立。

## 2. 下载固定模型

```bash
./scripts/download-models.sh
# 单独重试一个通道
./scripts/download-models.sh --channel sparse
```

默认缓存 `data/models/hub`，可通过 `CHINASOFT_MODEL_CACHE` 指定明确授权的公共模型缓存。脚本明确使用官方 `huggingface.co`，避免继承第三方 `HF_ENDPOINT` 镜像并向其发送凭据。只下载固定 revision 的必要文件，逐个检查真实权重 SHA256。Hub/Xet 的缓存文件名不一定等于内容哈希，因此验证实际文件字节，而不依赖缓存文件名。

**SPLADE-v3 的官方仓库要求接受访问条款。** 先在 [模型页面](https://huggingface.co/naver/splade-v3) 获得访问权限，再将有读取权限的 `HF_TOKEN` 注入当前进程。脚本只使用显式环境变量，不读取全局 Hugging Face 登录，不保存 Token。不要把 Token 写入命令参数、日志或 Git。401 会显示明确指引，其余通道仍可下载，整体命令以失败状态退出，直至所需模型齐备。

## 3. 启动与验证

在本项目 `.env` 中设置以下三项，或在当前 shell 导出：

```bash
export CHINASOFT_DENSE_MANAGED=true
export CHINASOFT_SPARSE_MANAGED=true
export CHINASOFT_RERANKER_MANAGED=true
./scripts/inference-start.sh
./scripts/inference-status.sh
.venv/bin/python scripts/verify-gpu.py --output .release-private/gpu-verification.json
./scripts/start.sh
```

普通 `start.sh` 不会隐式安装或启动 GPU 模型。模型启动器验证公开缓存内容，按通道启动独立 vLLM，等待真实加载并调用推理。`/mandol/metadata` 读取服务实际 vLLM 配置、模型文件与启动清单；Ready 还要求实际编码或排序成功。仅 `/health` 返回 200 不足以通过。

`verify-gpu.py` 通过正式 Mandol 客户端验证 1024 维归一化、重复/批量结果、30522 稀疏空间、正有限权重、BGE 成对排序方向和元信息，并记录顺序与并发 2 的真实 HTTP P50/P95。该检查不写入记忆库，不使用 Mock。

启动失败会清理本次新建且所有权验证通过的进程；本次调用前已经运行的服务仍保留。PID 所有权包括 boot ID、启动时间、命令和进程组。不要用批量 kill 或全局 Docker 清理来恢复。

```bash
./scripts/inference-stop.sh
```

停止不会删除权重、SQLite、Qdrant 或用户配置。

## 资源与远端配置

每个通道支持 `CHINASOFT_<DENSE|SPARSE|RERANKER>_GPU`、`_GPU_MEMORY`、`_MAX_SEQS`、`_MAX_BATCHED_TOKENS`、`_VLLM_PYTHON`；默认设备 0，最大序列数 2，显存利用率分别 0.12 / 0.10 / 0.14。HTTP timeout 与客户端并发由对应 Settings 项控制。修改这些资源配置前先正常停止自己的模型进程；已有部署参数不一致时启动器拒绝接管。

连接已有独立服务时，保持 `*_MANAGED=false`，设置：

```dotenv
CHINASOFT_EMBEDDING_BASE_URL=http://your-gpu-host:20631/v1
CHINASOFT_SPLADE_URL=http://your-gpu-host:20636
CHINASOFT_RERANKER_BASE_URL=http://your-gpu-host:20632
```

端点必须满足上述模型数值契约，部署 `services/chinasoft_vllm_metadata.py` 并使用自己的真实启动清单。使用 `*_API_KEY_ENV` 指定凭据环境变量名。不能用固定 metadata JSON 伪造兼容性。

## Docker 应用访问 GPU

推荐的本机 GPU 路径是独立原生 vLLM 服务 + 原生应用。Linux 上也可让容器化应用通过 host network 访问仅监听回环的模型：

```bash
docker compose -f docker-compose.yml -f docker-compose.loopback.yml -p mandol-os-compose up -d --build
```

应用仍绑定 127.0.0.1；Qdrant 仍使用自己的容器与卷，并只映射回环端口。此方案容器化应用与 Worker，GPU 模型运行在上述原生环境中。它不声称在官方 vLLM 镜像内完成模型验证。基础 bridge Compose 适用于容器可达的远端 GPU 地址，通过 `CHINASOFT_CONTAINER_*_URL` 配置；host-gateway 无法连接只监听宿主 127.0.0.1 的服务。

[官方 vLLM 镜像](https://docs.vllm.ai/en/latest/deployment/docker/)和 [NVIDIA vLLM 容器](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html)是可选部署基础。它们需要 NVIDIA Container Toolkit，且仍需核对模型支持、固定运行时版本、数值协议与中间件；普通镜像启动不能替代本项目验收。不默认拉取大型 GPU 镜像，也不使用浮动 latest 作为安装方案。

## 许可

源码、外部权重、驱动与 API 分别适用各自条件。Qwen/BGE 权重模型卡标注 Apache-2.0；SPLADE-v3 权重标注 **CC-BY-NC-SA-4.0**，不由本项目 Apache 许可重新授权。仓库与应用镜像均不包含模型权重。

在同一机器复验多个干净检出时，可显式设置 `MANDOL_DOWNLOAD_CACHE_DIR` 复用本工程已从公开源下载的包缓存，`CHINASOFT_MODEL_CACHE` 复用已授权、固定哈希的公共模型缓存。每个检出仍创建自己的推理虚拟环境和进程清单，不复制已安装环境，也不读取私有工程。
