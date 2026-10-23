# 测试与验证

测试使用合成账号、临时 SQLite 与测试 Scope，不读取私有数据库。协议故障注入与真实外部端点验收分开记录。

## 无真实模型的回归

```bash
uv sync --locked --group dev
.venv/bin/pytest -q tests
npm --prefix studio ci
npm --prefix studio run build
```

`tests/chinasoft` 是 SQLite/API 和明确注入的模型协议回归，`tests/test_deployment.py` 检查发布隔离与合成示例。`tests/integration/test_qdrant.py` 使用真实独立 Qdrant；未提供 `MANDOL_TEST_QDRANT` 时跳过。该测试使用明确标注的合成向量，仅证明数据库协议及过滤，不代表真实 Qwen/SPLADE 推理。所有测试按实际 pytest 统计判断。

## 独立 Qdrant

```bash
docker compose -p mandol-os-native up -d qdrant
MANDOL_TEST_QDRANT=http://127.0.0.1:20633 .venv/bin/pytest -q tests
```

测试为自己的随机 collection/Scope 建立并清理数据；不要设置成其他工程的数据库端点。应用 collection 与测试 collection 隔离。

## HTTP 与浏览器

```bash
./scripts/start.sh --basic
./scripts/smoke.sh
./scripts/smoke.sh --write --username <你的账号> --password-file <0600密码文件>
# 浏览器依赖与浏览器二进制需另外安装
cd studio
npx playwright install chromium
cd ..
MANDOL_E2E_USERNAME=<你的账号> MANDOL_E2E_PASSWORD_FILE=<0600密码文件> npm --prefix studio run test:e2e
```

浏览器使用真实静态前端和后端，创建自己的合成会话。结果和截图存放 `.release-private/`，不会进入公开提交。完整生命周期、来源、权限、远程仓库安全、联网参数和推理模式已有测试；其中模型/网络 Mock 是协议验证，不作为真实外部服务成功。

本轮外部凭据与 GPU 安装状态在内部验收报告中逐项给出。仓库不公开该凭据、真实运行数据库、逐用户日志或内部报告。

## 真实 GPU 与 Agent 闭环

三个服务启动后执行正式 HTTP 客户端测试：

```bash
./scripts/verify-gpu.sh --samples 12 --output .release-private/gpu-http.json
```

默认必须三个通道全部通过。`--channel dense` 等选项仅用于逐通道诊断，单通道 PASS 不代表完整检索。结果包含实际 metadata、批量与重复误差、稀疏非零数量、BGE 排序，以及顺序/并发 2 的原始测量和 P50/P95。

进一步在模型环境中加载同一份固定权重，用 Transformers CUDA 运算对照真实 HTTP 输出，核对 LAST + normalize、全 token SPLADE pooling 和 BGE sigmoid：

```bash
.venv-vllm-029/bin/python scripts/verify-model-reference.py --channel dense --output .release-private/dense-reference.json
.venv-vllm-029/bin/python scripts/verify-model-reference.py --channel sparse --output .release-private/sparse-reference.json
.venv-vllm-025/bin/python scripts/verify-model-reference.py --channel reranker --output .release-private/bge-reference.json
```

这一步在独立验证进程中临时加载一份模型，需为参考运算额外预留 GPU 显存；FastAPI 仍只通过 HTTP 调用模型。

完整服务与 DeepSeek 就绪后，在新建的合成工作区执行真实保存、索引、高阶构建、跨会话三路召回、RRF、BGE、Agent 引用、幂等重放及来源修订：

```bash
.venv/bin/python examples/full_retrieval.py --username admin \
  --password-file /secure/password.txt --output .release-private/full-retrieval.json
```

该示例保留自己的工作区用于查看结果，不清理已有账号数据。服务或配方不合格会失败，不能把基础模式的词法召回当作三路成功。重启检查应比较原工作区与 Source UID，再执行真实查询；不要用重新生成数据代替持久化验证。

`./scripts/stop.sh --api-only` 仅停止所有权验证通过的应用进程，保留模型和 Qdrant，适合验证应用重启恢复；不带该选项的 stop 仍停止本项目专属的完整服务栈。
