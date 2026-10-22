# 故障排查

| 现象 | 检查与处理 |
|---|---|
| Docker 不可用/权限拒绝 | 检查 `docker info`、daemon 和当前用户权限；不应全局删除容器或开放 socket |
| Compose 命令不存在 | 安装官方 Compose 插件；脚本使用 `docker compose` |
| 端口占用 | `ss -ltnp` 检查；选择独立端口并同步 API/Vite/Compose 配置；脚本不会终止未知监听者 |
| Qdrant 不可达 | 查看本项目 `docker compose -p mandol-os-native ps` 和 qdrant 日志；核对 URL、端口和当前项目卷 |
| SQLite 初始化失败 | 核对 DATA_DIR 的写权限、磁盘空间和 `PRAGMA quick_check`；备份后再处理，不能删除数据库来绕过错误 |
| 没有账号 | 先 setup，再用 `scripts/admin.sh create-admin --username admin`；无默认密码 |
| 注册后无法登录 | 默认需要审批；检查注册开关和 REGISTRATION_APPROVAL，不要修改数据库密码字段 |
| GPU 不可见 | 检查 `nvidia-smi`、驱动及所选 GPU；应用本身不加载 CUDA 权重 |
| vLLM 安装冲突 | 当前 0.29.0 的 protobuf 与 CUTLASS DSL 依赖冲突会使新环境解析失败；保留错误，使用已兼容的独立端点，不能静默换模型或跳过依赖 |
| CUDA/torch/vLLM 版本不兼容 | 模型环境与 Web 环境隔离；核对 INFERENCE.md 的精确数值配方与版本，不共用其他项目虚拟环境 |
| 模型下载失败 | 检查网络、许可与可用空间；可显式配置公开共享缓存，下载中断后按同一 revision 续传 |
| 服务 /health 成功但模型未就绪 | 检查 `/mandol/metadata`、artifact hashes、版本、pooling、真实编码与排序；普通 OpenAI 兼容端点不自动符合本协议 |
| 缺少 Dense/SPLADE/BGE | 完整 start 会拒绝；基础检查需显式 --basic。SPLADE 的词法降级不等于真实 SPLADE 成功 |
| 容器连接不到本机模型 | host-gateway 不能访问只绑定 127.0.0.1 的服务；使用本机 API 或独立远端端点，保留访问控制 |
| DeepSeek 401/402/429 | 检查用户自己的凭据、额度和限流；不要把原始凭据或供应商响应公开 |
| 生成响应模型不匹配 | 核对 request_model 与接受列表；只有经验证的模型协议才能启用对应推理档位 |
| 联网不可用 | 默认 SEARCH_ENABLED=false；检查专用端点/凭据、用户本轮开关和 Trace 的实际错误；聊天代理不必然支持 Search/Fetch |
| Fetch 被拒绝 | 仅批准域名和公共目标可用，默认 max_fetch=0；不能关闭 SSRF 检查来修复 |
| 前端 API 地址错误 | 生产前端由 API 同源提供；开发 Vite 代理 127.0.0.1:20600，改端口需同步 vite.config.ts |
| Worker 无任务 | 检查系统状态中的线程存活、任务类型和错误；后台失败与可靠输入接收是两种状态 |
| 任务 pending/failed | 核对模型就绪、配额、重试次数；模型恢复后仅对相关失败任务执行界面支持的重试，不批量改状态 |
| 来源已变化 | 该高阶记忆依赖的版本不再有效；历史仍可审计，等待合法重建，不可手动伪造 eligible |
| 重启后数据不见 | 核对 DATA_DIR、Compose project 和卷名是否变化；`down` 保留卷，`down -v` 会删除，禁止用于日常重启 |

## 恢复与重建

停止自己的进程、保留数据库/媒体/配置备份，使用匹配版本恢复。运行中 SQLite 使用 Backup API；不应将仅有主库文件的非一致副本用于恢复。Qdrant 重建要从 Canonical 当前有效记录生成匹配 Profile 的新投影，独立验证后切换，不能混写 SPLADE 与其他 Sparse 协议。

`.runtime` 与 `.release-private` 是本机诊断目录，不应公开。排查时使用错误码、任务 ID、脱敏状态和耗时，不粘贴密码、Cookie、Token 或含认证信息的 URL。
