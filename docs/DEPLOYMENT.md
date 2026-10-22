# 部署 Mandol

## 选择部署方式

本机方式使用项目自己的 `.venv`、`studio/node_modules`、`data/native` 和 Compose Qdrant 卷。容器方式把 API、内置 Worker 和已构建前端放入应用容器，SQLite 与上传存储在 `app_data` 卷。两种方式不读取其他工程的环境、数据库或虚拟环境。

默认应用 20600，Qdrant 20633，模型 20631/20636/20632。监听地址为回环；没有自动开放公网或初始化固定密码的步骤。启动脚本会拒绝接管不属于本工程的监听进程。

## 系统准备

本项目基于 Linux 系统验证，验收环境为 x86_64、Python 3.12.13、Node.js 24.21.0、Docker 29.6.1 和 Compose 插件。不同发行版应按依赖和驱动条件自行验证。Python 声明固定为 3.12；Node.js 建议 22.12 或更新版本。需要可访问 Python/npm 包源、容器镜像源以及用户选择的模型/API 服务。

先从各工具官方渠道安装 uv、Node.js LTS、Docker Engine 和 Compose 插件，然后检查：

```bash
uv --version
node --version
npm --version
docker info
docker compose version
```

不应为了运行应用把 Docker socket 暴露给不可信用户。可使用 rootless Docker 或由操作者管理的 Docker 权限。

## 一键安装与本机启动

```bash
cd mandol
./scripts/setup.sh
```

安装脚本只在 `.env` 不存在时复制模板；安装独立 Python 环境和锁定前端依赖，构建前端，并执行幂等 SQLite 初始化。它不启动模型，不下载模型权重，不覆盖已有用户配置。若本项目 API 正在运行，重复 setup 只读校验 SQLite，不争抢写入所有权；Schema 初始化推迟到正常停止并重新启动时执行。

编辑 `.env`，配置 Dense、SPLADE、BGE HTTP 端点和 `DEEPSEEK_API_KEY`。环境中显式导出的同名变量优先于 `.env`。也可将真实凭据仅通过进程环境注入。

```bash
./scripts/admin.sh create-admin --username admin
./scripts/start.sh
./scripts/status.sh
./scripts/smoke.sh
```

管理员命令交互式读取密码，至少 12 字符且符合现有密码规则；也可用权限 0600 的 `--password-file`。首次访问 http://127.0.0.1:20600 登录。新账号拥有自己的默认工作区，历史数据不会自动认领给新用户。

当前进程模型是 **一个 Uvicorn 进程 + 内置四类 Worker**，不是单独 Worker 容器。不要增加 `--workers` 数量来扩容；当前调度和任务恢复应先通过专门验证。

没有 GPU 依赖时，可运行 `./scripts/start.sh --basic`，明确启用降级检查。状态页会报告缺失依赖，索引和高阶任务可能等待或失败。模型可用后应停止自己的应用，再以完整配置启动；不能假称历史失败任务已经完成。

## 手动安装

```bash
cp -n .env.example .env
chmod 600 .env
uv sync --locked --python 3.12 --group dev
npm --prefix studio ci
npm --prefix studio run build
# 以下命令需要先加载自己的 .env，或由操作者导出相应变量
./scripts/admin.sh create-admin --username admin
./scripts/start.sh
```

如希望直接调用 Python 入口，先通过 `set -a; source .env; set +a` 加载本地配置，再运行：

```bash
.venv/bin/python scripts/manage.py init
# 单独启动项目的 Qdrant
docker compose -p mandol-os-native up -d qdrant
.venv/bin/python -m uvicorn mandol_competition.chinasoft.api:create_app --factory --host 127.0.0.1 --port 20600 --workers 1
```

直接使用 uvicorn 不会创建本机生命周期脚本的 PID 所有权记录，需由操作者在原终端正常中断；不要期待 `stop.sh` 接管手工进程。常规使用建议统一采用启动脚本。

## 独立模型服务

模式 A：连接已有的、被明确配置的独立 HTTP 端点。必须满足 `docs/INFERENCE.md` 中的模型 revision、数值配方、元信息与鉴权协议。普通 vLLM `/health` 成功不代表兼容；需要 `/mandol/metadata` 和真实推理探测。

模式 B：使用各自的哈希锁文件安装独立 vLLM 环境，下载指定 revision 并校验实际文件内容，再启动。安装器通过 `--no-config` 隔离 Web workspace 的 protobuf 约束，保留完整依赖检查。SPLADE-v3 需要在官方模型页接受访问条款并注入授权 `HF_TOKEN`。安装、下载和真实验证细节见 [GPU 部署](INFERENCE.md)。

```bash
./scripts/inference-setup.sh
./scripts/download-models.sh
# 按实际资源编辑 .env 的 *_GPU、*_GPU_MEMORY 等
export CHINASOFT_DENSE_MANAGED=true
export CHINASOFT_SPARSE_MANAGED=true
export CHINASOFT_RERANKER_MANAGED=true
./scripts/inference-start.sh
./scripts/inference-status.sh
.venv/bin/python scripts/verify-gpu.py
./scripts/start.sh
```

默认没有 CPU 假模型或自动模型替换。模型环境不共用应用的 `.venv`，权重不进入 Git。可显式指定已有的公开模型缓存，但不能指向私有工程或把其环境当成安装依赖。

## Docker Compose

```bash
cp -n .env.example .env
# 编辑 .env
docker compose -p mandol-os-compose config --quiet
docker compose -p mandol-os-compose up -d --build
docker compose -p mandol-os-compose ps
docker compose -p mandol-os-compose exec app python -m mandol_competition.chinasoft.admin create-admin --username admin
curl --fail http://127.0.0.1:20600/healthz
```

`config --quiet` 仅校验配置，避免把展开后的密钥打印到日志。完整 `docker compose config` 可能含凭据，不应公开。

镜像以非 root 用户运行。应用 API 同时提供前端静态文件和内置 Worker。Qdrant 在独立网络与独立命名卷中运行；应用使用内部 `http://qdrant:6333`。本机与容器的 SQLite 不共用。

配置容器 GPU 端点时使用 `CHINASOFT_CONTAINER_DENSE_URL`、`CHINASOFT_CONTAINER_SPARSE_URL` 和 `CHINASOFT_CONTAINER_RERANKER_URL`。Linux bridge 不能访问宿主机仅绑定 127.0.0.1 的服务。使用远端可达地址，或为本机 GPU 使用以下 Linux overlay，应用仍只监听回环：

```bash
docker compose -f docker-compose.yml -f docker-compose.loopback.yml -p mandol-os-compose up -d --build
docker compose -f docker-compose.yml -f docker-compose.loopback.yml -p mandol-os-compose exec app python -m mandol_competition.chinasoft.admin create-admin --username admin
```

更改端口时通过环境设置 `CHINASOFT_PORT`、`MANDOL_QDRANT_PORT`；本机和容器部署不应同时占用同一组端口。此 overlay 不启动模型容器，使用已独立验证的本机 GPU 服务。不要为了连通模型把端口开放给公网。

基础容器检查需要显式设置 `CHINASOFT_ALLOW_RETRIEVAL_DEGRADED=true`。容器 HTTP Health 只证明 API 和 SQLite 可访问，完整模型就绪状态以登录后的系统状态和真实查询为准。

```bash
docker compose -p mandol-os-compose down
# 不要加 -v；不应执行系统级 prune
```

## DeepSeek 与联网

生成与视觉默认使用当前绑定的 `deepseek-flash`，响应模型身份必须在配置的接受列表中。若供应商不提供该模型，需明确配置它实际提供的模型及支持的协议，不能把不匹配响应视为成功。

联网默认关闭。开启 `CHINASOFT_SEARCH_ENABLED=true` 后，可为 Search/Fetch 单独配置 `CHINASOFT_SEARCH_BASE_URL` 与 `CHINASOFT_SEARCH_API_KEY_ENV`。只有用户在本轮开启联网时才执行；网页证据与长期记忆来源分别保存。Fetch 默认关闭，启用后仍受域名、安全检查及调用预算限制。

## 注册与公开访问

默认公开注册关闭。演示需要自行注册时，可设置 `CHINASOFT_REGISTRATION_ENABLED=true`；默认新账号仍需审批，`CHINASOFT_REGISTRATION_APPROVAL=false` 才允许直接激活。账号总数、模型调用、上传、存储和任务配额仍由后端执行。

正式公开服务需由部署者设置可信反向代理、TLS、Origin/Host 转发和防火墙，启用 `CHINASOFT_SECURE_COOKIE=true`。本仓库不自动配置公网端口、域名或生产密钥。普通账号不能扫描任意本机路径；管理员本地仓库还需要显式 allowlist 与账号授权。

## 停止、重启与备份

```bash
./scripts/stop.sh
./scripts/start.sh --basic  # 或在完整依赖就绪后不带 --basic
```

停止脚本校验本项目进程的 PID、启动时间、进程组和命令；不会按进程名批量杀死其他项目。Compose down 不删除命名卷。

SQLite/媒体/暂存文件位于配置的 data 目录；模型缓存、日志和 PID 记录分别在 `data/models`、`.runtime`。本机备份可先正常停止自己的应用和任务，再完整备份 `data/native` 与所需 Qdrant 卷。运行中 SQLite 应使用 SQLite Backup API 生成一致副本，不能只复制主文件而忽略 WAL。恢复需同时保留源码版本、配置、媒体和检索 Profile，不能混用不同模型数值配方的旧投影。

索引重建必须基于 SQLite 有效 Canonical 记录、原逻辑 UID/版本和匹配的 Profile，先在独立 collection 验证再切换。当前没有承诺无条件执行的全库一键重建命令；不能通过清空 Qdrant 或修改来源版本制造就绪状态。
