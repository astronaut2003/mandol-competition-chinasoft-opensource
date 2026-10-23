# 第三方来源与独立许可

Mandol 应用源码采用 Apache-2.0，原有源文件注释及 substrate 许可声明保持在原位置。根 LICENSE 不替代以下外部组件、模型、数据和 API 的使用条件。

## 已有研究与代码

- Mandol 研究代码、Canonical substrate 和 ChinaSoft 实现的复用说明见 [NOTICE](NOTICE)、[packages/substrate/NOTICE](packages/substrate/NOTICE) 与包内 upstream.json。该文件保留出处和内容哈希，不包含私有仓库认证信息。
- 本仓库只保留实际运行的依赖闭包；HTTP Provider 基类从离线评测模块提取到 textual/core/quality_provider.py，避免安装产品时依赖评测数据。核心记忆表示、来源和检索配方保留。
- demo/status_board.png 是为本仓库新制作的合成图片，按 demo/LICENSE.txt 的 CC0 条件提供；示例源码为新编写的 Apache-2.0 代码，不包含真实用户资料。

## 主要运行依赖

| 组件 | 许可 | 来源 |
|---|---|---|
| FastAPI / Argon2-cffi / Tree-sitter / jieba | MIT（各包原声明见安装包） | 对应 PyPI 上游 |
| Qdrant server / qdrant-client | Apache-2.0 | https://github.com/qdrant/qdrant |
| React / React DOM / react-markdown / remark-gfm | MIT | 对应 npm 上游 |
| Cytoscape.js | MIT | https://github.com/cytoscape/cytoscape.js |
| Lucide icons | ISC | https://github.com/lucide-icons/lucide |
| vLLM | Apache-2.0 | https://github.com/vllm-project/vllm |
| certifi | MPL-2.0 | https://github.com/certifi/python-certifi |
| tqdm | MPL-2.0 AND MIT | 对应安装包许可声明 |
| Python / Node.js / CUDA / 容器基础系统 | 各自发行与组件许可 | 不由本项目源码许可覆盖 |

完整 Python 与 npm 依赖版本、许可字段见 [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md)。这些包由包管理器安装，没有将其完整源码复制为 Mandol 的原创代码。源码树未附带商业字体文件；前端保留 CSS 字体族声明并使用系统可用字体。

## 模型权重

| 模型 | 官方模型卡许可 | 获取与适用条件 |
|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-Embedding-0.6B |
| naver/splade-v3 | **CC-BY-NC-SA-4.0** | https://huggingface.co/naver/splade-v3 |
| BAAI/bge-reranker-v2-m3 | Apache-2.0 | https://huggingface.co/BAAI/bge-reranker-v2-m3 |

SPLADE 权重的非商业与相同方式共享等条件独立于 Mandol 源码。仓库不打包或重新授权模型权重；公开模型卡声明不应被解释为其他权利人的额外授权。实际使用以对应 revision 的官方条款为准。

## API 服务

DeepSeek 生成、视觉、Search/Fetch 为用户自行配置的外部服务，需遵守供应商条款和计费要求。本项目不附带服务额度、用户 API Key 或默认账号，也不承诺第三方接口永久提供相同模型。

## README 中的产品截图

四张 `docs/assets/` 截图来自本项目独立 Studio，使用新建合成账号与资料。`repository-symbols.png` 显示 Pallets itsdangerous 的真实公开源码片段，来源为 https://github.com/pallets/itsdangerous ，适用以下 BSD-3-Clause 声明。截图不改变该源码的版权与许可。

```text
Copyright 2011 Pallets

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1.  Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

2.  Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

3.  Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
