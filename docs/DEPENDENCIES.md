# 锁定依赖与许可清单

本表记录本次独立安装解析的 Python wheel METADATA 和 npm 锁文件 license 字段，包含开发依赖。第三方源码和权重未重新授权；随安装包分发的 notices 仍应保留。原生扩展和 NumPy 等二进制包可能另含组件声明，实际分发 wheel/镜像时应保留其 LICENSE 文件。

## 独立 GPU 环境

GPU 依赖不安装到 Web 环境，也不随 Git 仓库分发 wheel 或模型。三个模型统一使用 Python 3.12 / vLLM 0.32.0，198 个已安装包的完整哈希锁为 `deploy/inference/vllm-unified.lock`，逐包版本与随包许可路径见 [GPU 依赖清单](../deploy/inference/dependencies.json)。

| 组件 | 统一环境 | 实际安装包的许可声明 |
|---|---|---|
| vLLM | 0.32.0 | Apache-2.0 |
| PyTorch | 2.13.0 | 保留安装包内的多组件许可表达式及第三方声明 |
| Transformers / Tokenizers | 5.19.0 / 0.23.3 | Apache-2.0 |
| Triton | 3.7.1 | MIT |
| NVIDIA CUTLASS DSL | 4.8.0 | 遵循安装包随附的 NVIDIA 许可 |

`cuda-toolkit` 元包没有在 wheel METADATA 声明许可，不据此推断其依赖也无条件开放。驱动、CUDA 组件、模型权重及外部 API 分别适用自己的条款；尤其 SPLADE-v3 权重采用 CC-BY-NC-SA-4.0。部署或重新分发依赖镜像时，应保留官方安装包与组件声明。

## Python

| 包 | 版本 | 安装包许可声明 |
|---|---|---|
| annotated-doc | 0.0.4 | MIT |
| annotated-types | 0.7.0 | MIT |
| anyio | 4.14.2 | MIT |
| argon2-cffi | 25.1.0 | MIT |
| argon2-cffi-bindings | 26.1.0 | MIT |
| attrs | 26.1.0 | MIT |
| certifi | 2026.10.19 | Mozilla Public License 2.0 (MPL 2.0) |
| cffi | 2.1.1 | MIT-0 |
| click | 8.3.3 | BSD-3-Clause |
| distro | 1.9.0 | Apache-2.0 |
| fastapi | 0.136.3 | MIT |
| grpcio | 1.80.0 | Apache-2.0 |
| h11 | 0.16.0 | MIT License |
| h2 | 4.4.1 | MIT |
| hpack | 4.2.0 | MIT |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD License |
| hyperframe | 6.1.0 | MIT |
| idna | 3.15 | BSD-3-Clause |
| iniconfig | 2.3.0 | MIT |
| jieba | 0.42.1 | MIT License |
| jiter | 0.14.0 | MIT |
| jsonschema | 4.26.0 | MIT |
| jsonschema-specifications | 2025.9.1 | MIT |
| mandol-agent-memory | 0.1.0 | Apache-2.0 |
| mandol-substrate | 0.1.0 | Apache-2.0 |
| networkx | 3.6.1 | BSD-3-Clause |
| numpy | 1.26.4 | BSD-3-Clause (with bundled component notices) |
| openai | 2.32.0 | Apache Software License |
| packaging | 26.1 | Apache-2.0 OR BSD-2-Clause |
| pillow | 13.0.0 | MIT-CMU |
| pluggy | 1.6.0 | MIT License |
| portalocker | 3.2.0 | BSD-3-Clause |
| protobuf | 7.34.1 | BSD-3-Clause |
| pycparser | 3.11 | BSD-3-Clause |
| pydantic | 2.13.3 | MIT |
| pydantic_core | 2.46.3 | MIT |
| Pygments | 2.20.0 | BSD-2-Clause |
| pytest | 9.0.3 | MIT |
| python-multipart | 0.0.32 | Apache-2.0 |
| qdrant-client | 1.19.0 | Apache Software License |
| referencing | 0.37.0 | MIT |
| rpds-py | 2026.9.1 | MIT |
| rustworkx | 0.17.1 | Apache-2.0 |
| sniffio | 1.3.1 | MIT License, Apache Software License |
| starlette | 1.3.1 | BSD-3-Clause |
| tqdm | 4.67.3 | MPL-2.0 AND MIT |
| tree-sitter | 0.25.2 | MIT |
| tree-sitter-language-pack | 1.14.3 | MIT |
| typing-inspection | 0.4.2 | MIT |
| typing_extensions | 4.15.0 | PSF-2.0 |
| urllib3 | 2.8.0 | MIT |
| uvicorn | 0.49.0 | BSD-3-Clause |

## Node.js

| 包 | 版本 | 锁文件许可 | 开发依赖 |
|---|---|---|---|
| @babel/code-frame | 7.29.7 | MIT | 是 |
| @babel/compat-data | 7.29.7 | MIT | 是 |
| @babel/core | 7.29.7 | MIT | 是 |
| @babel/generator | 7.29.8 | MIT | 是 |
| @babel/helper-compilation-targets | 7.29.7 | MIT | 是 |
| @babel/helper-globals | 7.29.7 | MIT | 是 |
| @babel/helper-module-imports | 7.29.7 | MIT | 是 |
| @babel/helper-module-transforms | 7.29.7 | MIT | 是 |
| @babel/helper-plugin-utils | 7.29.7 | MIT | 是 |
| @babel/helper-string-parser | 7.29.7 | MIT | 是 |
| @babel/helper-validator-identifier | 7.29.7 | MIT | 是 |
| @babel/helper-validator-option | 7.29.7 | MIT | 是 |
| @babel/helpers | 7.29.10 | MIT | 是 |
| @babel/parser | 7.29.9 | MIT | 是 |
| @babel/plugin-transform-react-jsx-self | 7.29.7 | MIT | 是 |
| @babel/plugin-transform-react-jsx-source | 7.29.7 | MIT | 是 |
| @babel/template | 7.29.7 | MIT | 是 |
| @babel/traverse | 7.29.10 | MIT | 是 |
| @babel/types | 7.29.8 | MIT | 是 |
| @esbuild/aix-ppc64 | 0.28.2 | MIT | 是 |
| @esbuild/android-arm | 0.28.2 | MIT | 是 |
| @esbuild/android-arm64 | 0.28.2 | MIT | 是 |
| @esbuild/android-x64 | 0.28.2 | MIT | 是 |
| @esbuild/darwin-arm64 | 0.28.2 | MIT | 是 |
| @esbuild/darwin-x64 | 0.28.2 | MIT | 是 |
| @esbuild/freebsd-arm64 | 0.28.2 | MIT | 是 |
| @esbuild/freebsd-x64 | 0.28.2 | MIT | 是 |
| @esbuild/linux-arm | 0.28.2 | MIT | 是 |
| @esbuild/linux-arm64 | 0.28.2 | MIT | 是 |
| @esbuild/linux-ia32 | 0.28.2 | MIT | 是 |
| @esbuild/linux-loong64 | 0.28.2 | MIT | 是 |
| @esbuild/linux-mips64el | 0.28.2 | MIT | 是 |
| @esbuild/linux-ppc64 | 0.28.2 | MIT | 是 |
| @esbuild/linux-riscv64 | 0.28.2 | MIT | 是 |
| @esbuild/linux-s390x | 0.28.2 | MIT | 是 |
| @esbuild/linux-x64 | 0.28.2 | MIT | 是 |
| @esbuild/netbsd-arm64 | 0.28.2 | MIT | 是 |
| @esbuild/netbsd-x64 | 0.28.2 | MIT | 是 |
| @esbuild/openbsd-arm64 | 0.28.2 | MIT | 是 |
| @esbuild/openbsd-x64 | 0.28.2 | MIT | 是 |
| @esbuild/openharmony-arm64 | 0.28.2 | MIT | 是 |
| @esbuild/sunos-x64 | 0.28.2 | MIT | 是 |
| @esbuild/win32-arm64 | 0.28.2 | MIT | 是 |
| @esbuild/win32-ia32 | 0.28.2 | MIT | 是 |
| @esbuild/win32-x64 | 0.28.2 | MIT | 是 |
| @jridgewell/gen-mapping | 0.3.13 | MIT | 是 |
| @jridgewell/remapping | 2.3.5 | MIT | 是 |
| @jridgewell/resolve-uri | 3.1.2 | MIT | 是 |
| @jridgewell/sourcemap-codec | 1.6.0 | MIT | 是 |
| @jridgewell/trace-mapping | 0.3.31 | MIT | 是 |
| @napi-rs/lzma-linux-x64-gnu | 1.5.1 | MIT | 是 |
| @playwright/test | 1.64.0 | Apache-2.0 | 是 |
| @rolldown/pluginutils | 1.0.0-rc.3 | MIT | 是 |
| @rollup/rollup-android-arm-eabi | 4.64.5 | MIT | 是 |
| @rollup/rollup-android-arm64 | 4.64.5 | MIT | 是 |
| @rollup/rollup-darwin-arm64 | 4.64.5 | MIT | 是 |
| @rollup/rollup-darwin-x64 | 4.64.5 | MIT | 是 |
| @rollup/rollup-freebsd-arm64 | 4.64.5 | MIT | 是 |
| @rollup/rollup-freebsd-x64 | 4.64.5 | MIT | 是 |
| @rollup/rollup-linux-arm-gnueabihf | 4.64.5 | MIT | 是 |
| @rollup/rollup-linux-arm-musleabihf | 4.64.5 | MIT | 是 |
| @rollup/rollup-linux-arm64-gnu | 4.64.5 | MIT | 是 |
| @rollup/rollup-linux-arm64-musl | 4.64.5 | MIT | 是 |
| @rollup/rollup-linux-loong64-gnu | 4.64.5 | MIT | 是 |
| @rollup/rollup-linux-loong64-musl | 4.64.5 | MIT | 是 |
| @rollup/rollup-linux-ppc64-gnu | 4.64.5 | MIT | 是 |
| @rollup/rollup-linux-ppc64-musl | 4.64.5 | MIT | 是 |
| @rollup/rollup-linux-riscv64-gnu | 4.64.5 | MIT | 是 |
| @rollup/rollup-linux-riscv64-musl | 4.64.5 | MIT | 是 |
| @rollup/rollup-linux-s390x-gnu | 4.64.5 | MIT | 是 |
| @rollup/rollup-linux-x64-gnu | 4.64.5 | MIT | 是 |
| @rollup/rollup-linux-x64-musl | 4.64.5 | MIT | 是 |
| @rollup/rollup-openbsd-x64 | 4.64.5 | MIT | 是 |
| @rollup/rollup-openharmony-arm64 | 4.64.5 | MIT | 是 |
| @rollup/rollup-win32-arm64-msvc | 4.64.5 | MIT | 是 |
| @rollup/rollup-win32-ia32-msvc | 4.64.5 | MIT | 是 |
| @rollup/rollup-win32-x64-gnu | 4.64.5 | MIT | 是 |
| @rollup/rollup-win32-x64-msvc | 4.64.5 | MIT | 是 |
| @types/babel__core | 7.20.5 | MIT | 是 |
| @types/babel__generator | 7.27.0 | MIT | 是 |
| @types/babel__template | 7.4.4 | MIT | 是 |
| @types/babel__traverse | 7.28.0 | MIT | 是 |
| @types/cytoscape | 3.21.9 | MIT | 是 |
| @types/debug | 4.1.13 | MIT | 否 |
| @types/estree | 1.0.9 | MIT | 否 |
| @types/estree-jsx | 1.0.5 | MIT | 否 |
| @types/hast | 3.0.5 | MIT | 否 |
| @types/mdast | 4.0.4 | MIT | 否 |
| @types/ms | 2.1.0 | MIT | 否 |
| @types/react | 19.3.0 | MIT | 否 |
| @types/react-dom | 19.3.0 | MIT | 是 |
| @types/unist | 3.0.3 | MIT | 否 |
| @types/unist | 2.0.11 | MIT | 否 |
| @ungap/structured-clone | 1.4.0 | ISC | 否 |
| @vitejs/plugin-react | 5.2.0 | MIT | 是 |
| bail | 2.0.2 | MIT | 否 |
| baseline-browser-mapping | 2.11.28 | Apache-2.0 | 是 |
| browserslist | 4.29.3 | MIT | 是 |
| caniuse-lite | 1.0.30001815 | CC-BY-4.0 | 是 |
| ccount | 2.0.1 | MIT | 否 |
| character-entities | 2.0.2 | MIT | 否 |
| character-entities-html4 | 2.1.0 | MIT | 否 |
| character-entities-legacy | 3.0.0 | MIT | 否 |
| character-reference-invalid | 2.0.1 | MIT | 否 |
| comma-separated-tokens | 2.0.3 | MIT | 否 |
| convert-source-map | 2.0.0 | MIT | 是 |
| csstype | 3.2.3 | MIT | 否 |
| cytoscape | 3.33.1 | MIT | 否 |
| debug | 4.4.3 | MIT | 否 |
| decode-named-character-reference | 1.3.0 | MIT | 否 |
| dequal | 2.0.3 | MIT | 否 |
| devlop | 1.1.0 | MIT | 否 |
| electron-to-chromium | 1.5.452 | ISC | 是 |
| esbuild | 0.28.2 | MIT | 是 |
| escalade | 3.2.0 | MIT | 是 |
| escape-string-regexp | 5.0.0 | MIT | 否 |
| estree-util-is-identifier-name | 3.0.0 | MIT | 否 |
| extend | 3.0.2 | MIT | 否 |
| fdir | 6.5.0 | MIT | 是 |
| fsevents | 2.3.3 | MIT | 是 |
| gensync | 1.0.0-beta.2 | MIT | 是 |
| hast-util-to-jsx-runtime | 2.3.6 | MIT | 否 |
| hast-util-whitespace | 3.0.0 | MIT | 否 |
| html-url-attributes | 3.0.1 | MIT | 否 |
| inline-style-parser | 0.2.7 | MIT | 否 |
| is-alphabetical | 2.0.1 | MIT | 否 |
| is-alphanumerical | 2.0.1 | MIT | 否 |
| is-decimal | 2.0.1 | MIT | 否 |
| is-hexadecimal | 2.0.1 | MIT | 否 |
| is-plain-obj | 4.1.0 | MIT | 否 |
| js-tokens | 4.0.0 | MIT | 是 |
| jsesc | 3.1.0 | MIT | 是 |
| json5 | 2.2.3 | MIT | 是 |
| longest-streak | 3.1.0 | MIT | 否 |
| lru-cache | 5.1.1 | ISC | 是 |
| lucide-react | 0.468.0 | ISC | 否 |
| markdown-table | 3.0.4 | MIT | 否 |
| mdast-util-find-and-replace | 3.0.3 | MIT | 否 |
| mdast-util-from-markdown | 2.1.0 | MIT | 否 |
| mdast-util-gfm | 3.1.0 | MIT | 否 |
| mdast-util-gfm-autolink-literal | 2.0.1 | MIT | 否 |
| mdast-util-gfm-footnote | 2.1.0 | MIT | 否 |
| mdast-util-gfm-strikethrough | 2.0.1 | MIT | 否 |
| mdast-util-gfm-table | 2.0.0 | MIT | 否 |
| mdast-util-gfm-task-list-item | 2.0.0 | MIT | 否 |
| mdast-util-mdx-expression | 2.0.1 | MIT | 否 |
| mdast-util-mdx-jsx | 3.2.0 | MIT | 否 |
| mdast-util-mdxjs-esm | 2.0.1 | MIT | 否 |
| mdast-util-phrasing | 4.1.0 | MIT | 否 |
| mdast-util-to-hast | 13.2.1 | MIT | 否 |
| mdast-util-to-markdown | 2.2.0 | MIT | 否 |
| mdast-util-to-string | 4.0.0 | MIT | 否 |
| micromark | 4.0.3 | MIT | 否 |
| micromark-core-commonmark | 2.0.4 | MIT | 否 |
| micromark-extension-gfm | 3.0.0 | MIT | 否 |
| micromark-extension-gfm-autolink-literal | 2.1.0 | MIT | 否 |
| micromark-extension-gfm-footnote | 2.1.0 | MIT | 否 |
| micromark-extension-gfm-strikethrough | 2.1.0 | MIT | 否 |
| micromark-extension-gfm-table | 2.1.2 | MIT | 否 |
| micromark-extension-gfm-tagfilter | 2.0.0 | MIT | 否 |
| micromark-extension-gfm-task-list-item | 2.1.0 | MIT | 否 |
| micromark-factory-destination | 2.0.1 | MIT | 否 |
| micromark-factory-label | 2.0.1 | MIT | 否 |
| micromark-factory-space | 2.1.0 | MIT | 否 |
| micromark-factory-title | 2.0.1 | MIT | 否 |
| micromark-factory-whitespace | 2.0.1 | MIT | 否 |
| micromark-util-character | 2.1.1 | MIT | 否 |
| micromark-util-chunked | 2.0.1 | MIT | 否 |
| micromark-util-classify-character | 2.0.1 | MIT | 否 |
| micromark-util-combine-extensions | 2.0.1 | MIT | 否 |
| micromark-util-decode-numeric-character-reference | 2.0.2 | MIT | 否 |
| micromark-util-decode-string | 2.0.1 | MIT | 否 |
| micromark-util-edit-map | 1.0.0 | MIT | 否 |
| micromark-util-encode | 2.0.1 | MIT | 否 |
| micromark-util-html-tag-name | 2.0.1 | MIT | 否 |
| micromark-util-normalize-identifier | 2.0.1 | MIT | 否 |
| micromark-util-resolve-all | 2.0.1 | MIT | 否 |
| micromark-util-sanitize-uri | 2.0.1 | MIT | 否 |
| micromark-util-subtokenize | 2.1.0 | MIT | 否 |
| micromark-util-symbol | 2.0.1 | MIT | 否 |
| micromark-util-types | 2.0.3 | MIT | 否 |
| ms | 2.1.3 | MIT | 否 |
| nanoid | 3.3.20 | MIT | 是 |
| node-releases | 2.0.58 | MIT | 是 |
| parse-entities | 4.0.2 | MIT | 否 |
| picocolors | 1.1.1 | ISC | 是 |
| picomatch | 4.0.7 | MIT | 是 |
| playwright | 1.64.0 | Apache-2.0 | 是 |
| playwright-core | 1.64.0 | Apache-2.0 | 是 |
| postcss | 8.5.29 | MIT | 是 |
| property-information | 7.2.0 | MIT | 否 |
| react | 19.3.0 | MIT | 否 |
| react-dom | 19.3.0 | MIT | 否 |
| react-markdown | 10.1.0 | MIT | 否 |
| react-refresh | 0.18.0 | MIT | 是 |
| remark-gfm | 4.0.1 | MIT | 否 |
| remark-parse | 11.0.0 | MIT | 否 |
| remark-rehype | 11.1.2 | MIT | 否 |
| remark-stringify | 11.0.0 | MIT | 否 |
| rollup | 4.64.5 | MIT | 是 |
| scheduler | 0.28.0 | MIT | 否 |
| semver | 6.3.1 | ISC | 是 |
| source-map-js | 1.2.2 | BSD-3-Clause | 是 |
| space-separated-tokens | 2.0.2 | MIT | 否 |
| stringify-entities | 4.0.4 | MIT | 否 |
| style-to-js | 1.1.21 | MIT | 否 |
| style-to-object | 1.0.14 | MIT | 否 |
| tinyglobby | 0.2.17 | MIT | 是 |
| trim-lines | 3.0.1 | MIT | 否 |
| trough | 2.2.0 | MIT | 否 |
| typescript | 5.9.3 | Apache-2.0 | 是 |
| unified | 11.0.5 | MIT | 否 |
| unist-util-is | 6.0.1 | MIT | 否 |
| unist-util-position | 5.0.0 | MIT | 否 |
| unist-util-stringify-position | 4.0.0 | MIT | 否 |
| unist-util-visit | 5.1.0 | MIT | 否 |
| unist-util-visit-parents | 6.0.2 | MIT | 否 |
| update-browserslist-db | 1.3.4 | MIT | 是 |
| vfile | 6.0.3 | MIT | 否 |
| vfile-message | 4.0.3 | MIT | 否 |
| vite | 7.3.7 | MIT | 是 |
| yallist | 3.1.1 | ISC | 是 |
| zwitch | 2.0.4 | MIT | 否 |
