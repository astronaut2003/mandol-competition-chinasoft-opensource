# 检索模型的独立 HTTP 部署

应用只通过 HTTP 调用以下固定模型，不在 FastAPI 进程中加载模型权重：

| 通道 | 模型与 revision | 数值语义 | 服务 |
|---|---|---|---|
| Dense | Qwen/Qwen3-Embedding-0.6B，见 retrieval_contract.py 完整 revision | BF16、LAST pooling、1024维、归一化、长度4096 | vLLM 0.29.0，20631 |
| Sparse | naver/splade-v3，同文件记录精确 revision | masked-LM、全 token max(log1p(ReLU(logits)))、词表30522、长度512 | vLLM 0.29.0，20636 |
| Reranker | BAAI/bge-reranker-v2-m3，同文件记录精确 revision | BF16、CLS score、sigmoid、长度4096 | vLLM 0.25.1，20632 |

`src/mandol_competition/chinasoft/retrieval_contract.py` 是配方身份的唯一代码来源，包含 artifact hashes、query/document 和不同 memory family 的 Profile。启动脚本与客户端共享此契约；不能只修改环境变量模型名称后沿用旧投影。

## 已有 HTTP 服务

`.env` 中配置 `CHINASOFT_EMBEDDING_BASE_URL`、`CHINASOFT_SPLADE_URL`、`CHINASOFT_RERANKER_BASE_URL`。可分别通过 `*_API_KEY_ENV` 指定存放访问凭据的环境变量名称。端点必须是明确授权的独立服务。

服务需要提供 vLLM 原始推理接口和 `services/chinasoft_vllm_metadata.py` 的 `/mandol/metadata`。该中间件通过本服务启动清单暴露配方与模型文件身份；不能用固定“ready”JSON 替代。

客户端会验证数值契约并做真实探测。SPLADE 使用 `BertSpladeSparseEmbeddingModel`，不是把 Dense 向量或词频伪装成 Sparse；查询与文档的稀疏编码必须相同。Qdrant 的 Sparse 字段只接受对应 Profile。

## 显式本机服务

```bash
./scripts/inference-setup.sh
./scripts/download-models.sh
# 修改 .env：为要管理的通道设置 *_MANAGED=true
./scripts/inference-start.sh
./scripts/inference-status.sh
./scripts/inference-stop.sh
```

默认 `*_MANAGED=false`，普通 `start.sh` 不下载或启动模型。`CHINASOFT_MODEL_CACHE` 默认 `data/models/hub`；可显式选择公开模型缓存以减少重复下载。每个通道有 `*_GPU`、`*_GPU_MEMORY`、`*_MAX_SEQS`、`*_MAX_BATCHED_TOKENS` 和 `*_VLLM_PYTHON` 配置。服务 HTTP 超时和应用端并发另由 Settings 配置，不能混为一项。

GPU 预算使用 vLLM memory utilization；并发默认保守。所需显存不仅取决于参数量，还包含运行时、激活、批处理和预留。当前没有跨设备的最低显存保证；应检查启动预检和实际设备空闲量，不应挤占其他工程的模型服务。

## 本轮安装限制

2026-10-22，在全新 Python 3.12 环境从公共 PyPI 安装 `vllm==0.29.0` 时，解析器报告：vLLM 要求 `protobuf==7.34.1`，其依赖 `nvidia-cutlass-dsl[cu13]==4.6.2` 的依赖链要求 `protobuf>=6.30.2,<7`，无法同时满足。

本项目没有使用 `--no-deps` 或强制覆盖来隐藏冲突，也没有换用其他模型。0.29.0 本机安装因此为 BLOCKED，后续真实 Dense/SPLADE/BGE E2E 不能据此标记通过。BGE 的 0.25.1 环境也未作为独立生产兼容性结论发布。保留这些显式入口，便于在上游依赖或已有兼容服务可用后按同一契约验证。

基础模式的真实自然对话、SQLite、Qdrant、模型协议故障注入测试与真实 GPU 检索是不同证据。部署者应先运行 `inference-status.sh` 和真实检索验收，再以不带 `--basic` 的方式启动。

## 外部条件

vLLM 使用 Apache-2.0 源码许可；CUDA/驱动、权重和托管服务各自适用其条件。Qwen 与 BGE 模型卡标注 Apache-2.0；[SPLADE-v3 模型卡](https://huggingface.co/naver/splade-v3)标注 CC-BY-NC-SA-4.0。仓库不分发权重，不替权利人授予额外权利。
