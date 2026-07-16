# ask-llm 架构评审报告与重构方案

> 评审视角：Python 架构师 / 系统设计师
> 评审对象：ask-llm v2.15.1（源码 17,319 行，测试 7,501 行）
> 评审日期：2026-07-13
> 本文取代 `docs/REFACTOR_PLAN.md`（该文件为 v1→v2 argparse→typer 迁移历史，已与现状脱节）

## 重构进度

| 阶段 | 状态 | 版本 |
|------|------|------|
| **P0** 承载性 bug 止血 | ✅ 已完成 | v2.16.0 (2026-07-14) |
| **P1** 执行引擎统一 | ✅ 已完成 | v2.16.1–2.16.7 (2026-07-14) |
| **P2** 配置去全局 + 单一对象 | ✅ 已完成 | v2.16.8–2.16.17 (2026-07-16) |
| P3 Markdown 单一管线 | 🔄 进行中 | v2.17.0–2.17.4 (2026-07-16) |
| P4 服务层/引擎/导出器收尾 | ⏳ 待开始 | — |

**P1 进度（v2.16.1–2.16.2）**：
- ✅ P1.2 — 删除死的单模型 `BatchProcessor` 平行层级（~330 LOC，仅 shim 再导出，从未实例化；`GlobalBatchProcessor` 不继承它）。
- ✅ P1.5 — 合并 4 个统计聚合器为单一 `BatchStatistics.from_results(results)` 类方法；`batch_service._calculate_statistics`（逐字节重复）删除；`calculate_statistics_by_model` 收为薄委托。
- ✅ P1.7 — 删除 `batch_processor.py` 重复的 `TYPE_CHECKING` 块。死代码审计结论：`ProviderRetryRegistry.set` / `BoundedRetryRunner.run` / `core/batch.py` shim 均有调用方（单测覆盖或 ~20 模块导入），保留。
- ✅ **P1.1 / B1** — retry×fallback 调用放大统一为**共享预算升级**（用户确认采用省钱语义）：回退链与重试预算共用 `max_retries+1` 次总预算，第 k 次尝试 `configs[min(k,len-1)]`。单 config 任务行为不变；多 config 任务瞬时错误改为推进链而非重试同 provider。terminal 错误（auth/content/validation）短路。runner 级回归测试断言 `total_calls ≤ tasks×(max_retries+1)`。
- ✅ **P1.6** — 三处 `RequestMetadata(...)` 构造合并为 `RequestMetadata.from_execution(...)`；温度解析三元（v2.15.1 崩溃路径）收口到唯一工厂。B8 根因收尾。
- ✅ **B5** — Ctrl-C 不再丢全部进度：`BoundedRetryRunner` 装 SIGINT 处理器（仅主线程），首次中断停止调度新任务、排空在飞任务、返回部分结果（不 re-raise `KeyboardInterrupt`），二次中断硬杀。新增 `RunMetrics.interrupted`。`batch`/`trans` 服务检测中断后保留 checkpoint、打印 resume 提示，仅全成功才 unlink。
- 🔄 **P1.3（主体完成）** — 上帝类 834→347 LOC，拆出三个协作者：`StreamCollector`（流式+token 收集纯函数）、`ProgressPresenter`（`rich.Progress` + per-worker 槽位池）、`TaskExecutor`（单 config 执行：限流 acquire / adapter 查找 / 流式收集 / metadata / 鉴权错误去重）。`GlobalBatchProcessor` 收为瘦协调器（B1 升级 + 调度）。三者均可独立单测。
- ✅ **P1 完成** — 执行引擎统一阶段结束。剩余可选优化：Scheduler/FallbackPolicy 进一步拆分（可选）、per-`(provider,model)` 池 sizing（P1.4）。`FormatCheckpoint` 迁移（P1.9）已评估为领域不匹配，暂缓。

**P2 进度（v2.16.8–v2.16.12）**：
- ✅ P2.7 — 冲突 env 覆盖告警：多个 env 变量映射同一 config key 时（如 `ASK_LLM_TRANSLATION_THREADS` 与 `ASK_LLM_TRANSLATION_MAX_CONCURRENT_API_CALLS`），`_apply_env_overrides` 检测并告警指明胜出者（迭代序最后者），不再静默。
- ✅ P2（去全局，增量）— `TokenCounter._get_encoding` 不再要求 config 已加载：热路径改用 `get_config_or_none()` 并回退 `cl100k_base`，库化/嵌入式使用不再崩（§4.2.3）。
- ✅ P2.6 — `paper_explain_pipeline.py` 从 `config/` 移到 `core/`（453 LOC 领域逻辑，非配置）；延迟 `get_config()` 读不变；4 处导入方更新（§4.2.6）。
- ✅ P2 — 全部剩余 `get_config()` 消费者迁移完成：`file_handler`、`format_markdown_file`、`md_heading_formatter`、`processor`、`md_body_formatter`、`text_splitter`、`format_service`、`paper_explain_pipeline`、`paper_explain`、`prompt_resolver` 均改用 `get_config_or_none()` 并携带与 `default_config.yml` 一致的本地默认值。`config.context.get_config()` 仍保留给真正需要活动配置的调用方。
- ✅ P2.8 / B12（v2.16.13）— 修复生产级崩溃：`global_batch_runner` 与 `notebook_translator` 读取不存在的 `ConfigManager.unified_config`（AttributeError；测试用 MagicMock 掩盖）。`ConfigManager(app_config, unified_config)` 新增可选参数与只读属性，两处构造点（`cli_session`、`format_cmd`）均传入，限流配置真正到达 `GlobalBatchProcessor`。
- ✅ P2.9（v2.16.14）— 配置单一对象：`UnifiedConfig` 吸收 `providers`/`default_provider`/`default_model`，`ConfigLoader.load` 单次校验（原先同一 dict 校验两遍、互相静默忽略对方字段）；`AppConfig` 改为派生视图（`_app_config_from_unified`，保留首 provider 回退告警）。
- ✅ P2.10（v2.16.15）— `SecretStr` 迁移：`ProviderConfig.api_key` 静态存储为 `SecretStr`（repr/日志/json dump 全遮蔽）；新增 `get_api_key()` 与 `EngineConfigView`（llm_engine 边界一次性解包，repr 再遮蔽），6 处 `create_provider_adapter` 直连点全部收口。
- ✅ P2.11（v2.16.16）— `loader.py` 拆分（§4.2.4）：613→311 LOC，新增 `config/env.py`（`${VAR}`+`ASK_LLM_*`）、`config/merge.py`、`config/providers_catalog.py`；`pricing.py`/`provider_specs.py` 改从 `config.env` 导入 `resolve_env_vars`。无行为变化。
- ✅ P2.12（v2.16.17）— provenance 落地：`LoadResult.provenance` 记录每个叶子 key 的最终来源层（package default / providers.yml / 用户配置 / `env:<VAR>`，按优先级低→高覆盖记录，标签即胜出层）；`config show --debug-config` 按来源分组逐 key 报告。`_load_providers_yml` 返回来源路径。
- 🎯 **P2 完成** — 配置去全局 + 单一对象 + SecretStr + loader 拆分 + provenance 全部落地。

**P3 进度（v2.17.0）**：
- ✅ P3.1 — `MarkdownStructure` 单一解析器（`core/markdown_structure.py`）：一次解析产出 fence ranges（未闭合延伸至 EOF）、YAML frontmatter range（仅 offset 0）、带层级 heading spans。`HeadingExtractor` 与 `MarkdownTokenSplitter` 已迁移消费；`HEADING_PATTERN`/`CODE_FENCE_PATTERN` 单一定义，原三处副本改为再导出。**新增 frontmatter 保护**：frontmatter 内的 `# foo` 不再被当标题（§4.4.3）。13 个新单测。
- ✅ P3.2（v2.17.1）— `BinarySplitter(BudgetPolicy)`（`core/binary_splitter.py`）：算法单份，`TokenBudget(model, max_tokens, prompt_overhead)` 预算 prompt+content（§4.4.4）；`MarkdownTokenSplitter` 降为 64 行兼容包装。删除死代码 `MarkdownSplitter`/`PlainTextSplitter`/`create_splitter`（`text_splitter.py` 629→71 LOC，仅留 `TextChunk`+基类）。splitter pair ~935→564 LOC 且算法唯一。12 新单测；`test_trans.py` 迁移 token 预算。
- ✅ P3.3（v2.17.2）— `ChunkedLLMJob` 基类（`core/chunked_llm_job.py`，182 LOC）：init 回退、prompt 加载、runner 接线、checkpoint save/resume 骨架单份；`HeadingFormatter`/`BodyFormatter` 降为薄子类。**非对称 resume 消灭**：`HeadingFormatter.resume_from_checkpoint` 新增（此前 title checkpoint 只写不能恢复，§4.4.6），按标题序号合并，回归测试覆盖。
- ✅ P3.4（v2.17.3）— position-aware 重组：`BodyFormatter._join_chunks_position_aware` 消费 `TextChunk.start_pos/end_pos`（此前正文侧死重），恢复原块间空白而非强制 `\n\n`；硬切分（character/hard_token_split）邻接空分隔符原样拼接（列表项/紧凑表格不再被插入空行）。spans 不构成干净分区时回退 legacy `_join_chunks`。5 新测试。
- ✅ P3.5（v2.17.4）— `format_service.format_one` 单一调度器：title/body 分叉只存一处，sequential/parallel 两份 if/else 删除（各 ~30 LOC）。**title resume 端到端接线**：`FormatService.resume_from_checkpoint` 支持 title checkpoint（此前直接 raise），经 `HeadingFormatter.resume_from_checkpoint` + `HeadingApplier` 合并写回；标题数不匹配报清晰 RuntimeError。
- ⏳ P3 余项：prompt 外迁（`CONTEXT_BATCH_INSTRUCTION` 等）、chunk-id 统一。

**P0 已落地（v2.16.0）**：B2（CJK 令牌近似+安全系数）、B3（`${VAR}` 告警 + gate 覆盖 trans/paper）、B4（splitter 代码栅栏感知）、B6（per-worker 进度条）、B7（`attempt_history` 改为扁平 `AttemptRecord`）、B8（provider-cache 接缝类型化）、B9（限流超时可配置）、密钥轮换清缓存。完整说明见 `CHANGELOG.md` 2.16.0 条目。
**P0 延后**：完整 `SecretStr` 迁移 → P2（与配置重构 + 引擎接缝收口一同进行）。
**未在 P0**：B1（retry×fallback 调用放大）、B5（增量 checkpoint）→ P1；B10/B11（外观）→ 随相关阶段。

> 🎯 **里程碑（v2.16.12）**：§5 承载性 Bug 清单 B1–B11 **全部修复**；P2 `get_config()` 去全局化消费者迁移完成，核心库可无配置使用。剩余工作为结构性重构（P2 `UnifiedConfig`/`AppConfig` 双叉合并、`SecretStr`、loader 拆分；P3 Markdown 单一管线；P4 服务/引擎收尾），不再有已知承载性 bug。

---

## 0. 执行摘要（Executive Summary）

ask-llm 已经经历过 Phase A–G 七轮重构，整体处在"中高级成熟度"水平：分层骨架（cli / services / core / config / utils）正确，并发原语（`BoundedRetryRunner`、`GlobalRateLimiter`、`ProviderAdapterCache`）隔离得当，可观测性（`telemetry`/`execution_report`）编织进批处理主路径，测试覆盖广泛。这是值得肯定的部分。

但项目存在 **三个结构性根因**，它们是绝大多数具体问题与历史 bug（含 v2.15.1 两个崩溃修复）的共同来源：

1. **执行引擎没有单一所有者。** 同一条"加载任务 → 并发执行 → 重试/回退 → checkpoint → 导出"管线被 `GlobalBatchProcessor`（上帝类）与 batch/trans/paper 三个 service **各复制一遍**；重试逻辑被劈成 **两层**（`BoundedRetryRunner` 与回退链），二者零协同，导致 `max_retries × len(fallback_chain)` 的调用放大。
2. **配置是进程级可变全局，且身份"双叉"。** `get_config()/set_config()` 是 service-locator 反模式，12 个模块硬依赖；一份 YAML 被拆成 `UnifiedConfig`（行为）与 `AppConfig`（provider）两个对象，没有单一句柄；外部引擎 `llm-api-engine` 还越界持有部分配置模型。
3. **同一类管线被平行实现多次。** 标题/正文/翻译三条格式化管线各写一遍；两份 splitter 文件 ~80% 行对行重复，字符版的生产调用为零、已成死代码。

由此派生出一批**仍潜伏的承载性 bug**（已逐条在 §5 验证）：fallback×retry 调用放大、CJK provider 令牌计数偏低导致上下文溢出、未解析的 `${VAR}` 占位符被当成真实 API key 发出、正文格式化会切断代码块、声称"可恢复"但 Ctrl-C 丢全部进度、N 个任务渲染 N 条进度条。

**建议**：执行一次"以设计思想统一为目标的彻底重构"——重构执行引擎为"调度器 + 执行器 + 回退策略 + 进度展示"四协作者；消灭全局配置，改为显式注入；合并双叉配置为单一对象；统一三条格式化管线为"解析→保护→按 token 切分→格式化→重组"；把引擎依赖收口到唯一模块。预计净删 ~1500 行，6 个正确性 bug 结构性消失。分阶段（P0–P4）落地，每阶段可独立发版、可独立回滚。

---

## 1. 项目概览

### 1.1 设计思想（意图）

ask-llm 是一个多 LLM API 命令行工具，核心能力：

| 命令 | 语义 |
|------|------|
| `ask` | 单次请求 |
| `chat` | 交互式会话 |
| `batch` | 从 YAML 批量执行 |
| `trans` | Markdown / Jupyter 翻译 |
| `format` | Markdown 标题/正文格式化 |
| `paper` | 论文分节解读 |
| `config` / `diagnose` | 配置与诊断 |

底层把实际 LLM 调用委托给外部引擎 `llm-api-engine`（`create_provider_adapter`），自身只做编排：并发、重试/回退、限流、token 计数、checkpoint、报告。

### 1.2 技术栈与质量基建

Typer + Rich + Loguru + Pydantic v2 + tiktoken + tqdm + nbformat；Ruff + mypy + pydocstyle + bandit + pre-commit；pytest 单元/集成/基准三层。`pyproject.toml` 配置完整、约束严格（`strict_equality`、`no_implicit_optional` 等）。质量工具链本身是项目的强项。

### 1.3 模块规模热点（LOC）

```
core/batch_processor.py      1150   ← 头号嫌疑
core/paper_explain.py         867
services/translation_service  833
services/paper_service         629
core/text_splitter.py         620   ← 生产无调用
services/batch_service         596
config/loader.py               580
core/md_heading_formatter      537
core/chat.py                   513
utils/batch_exporter           508
```

超过 500 LOC 的文件普遍承载 ≥2 类职责，是"上帝对象/上帝函数"高发区。

### 1.4 值得肯定的设计

- **服务层意图正确**：`AGENTS.md` 明确"CLI 是薄适配器；service 不调 `typer.Exit`；返回结构化结果"。`AskService` 是干净范本（零 `console.print`、零 `typer`、返回纯 dataclass）。
- **并发原语隔离**：`BoundedRetryRunner`（调度+重试堆）、`GlobalRateLimiter`（per `(provider,model)` 令牌桶）、`ProviderAdapterCache`（HTTP 连接复用）各自独立，线程安全分析通过（限流器无死锁，见 §4.5）。
- **可观测性编织到位**：`bind_context` 在 `batch_processor.py` 有 11 处调用，并行批处理可关联；`ExecutionReport` 结构化。
- **校验最严的模块**：`batch_loader.py`（逐字段、逐条目、结构化校验）。
- **原子写**：`BaseCheckpoint.save` 用 `tmp + os.replace`。
- **近期重构方向正确**：提取 `ProviderManager`、`RetryPolicy`、`constants`、`telemetry` 等都在朝解耦走——但尚未走完（见下）。

---

## 2. 当前架构与数据流

### 2.1 总体分层（现状）

```
CLI (typer)                         cli/commands/*
   │  load_cli_session               config/cli_session.py
   ▼
Services (编排)                      services/*  ← 5 个，风格不一
   │
   ▼
Core (执行/模型/管线)               core/batch_processor.py  ← 上帝类
   │   GlobalBatchProcessor
   │       ├─ BoundedRetryRunner      core/concurrent.py
   │       ├─ ProviderManager         core/provider_manager.py
   │       ├─ GlobalRateLimiter       utils/rate_limiter.py
   │       ├─ RequestProcessor        core/processor.py
   │       └─ rich.Progress           (内联两次)
   ▼
Engine (外部)                       llm-api-engine  create_provider_adapter
Config (全局)                       config/context.py: get_config/set_config  ← 进程级可变
```

### 2.2 batch 数据流（典型主路径）

```
YAML
 → BatchConfigLoader.load → list[BatchTask]
 → _validate_models        (主线程 mutate ConfigManager!)
 → build_fallback_chain
 → 笛卡尔展开 model×task   (task_id 重排，原始序号丢失)
 → BatchCheckpoint.create/load
 → run_global_batch_tasks
     → GlobalBatchProcessor.process_global_tasks
         → sort (longest-first)
         → ProviderManager.build_provider_cache
         → _effective_max_workers (全局最小 burst 截断)
         → BoundedRetryRunner.run_with_metrics
             worker → _process_single_global_task
                 for cfg in [primary, *fallbacks]:   ← 回退链（第二层重试）
                     _try_run_with_config
                         rate_limiter.acquire(60s)
                         RequestProcessor.process
                         _stream_and_collect
 → checkpoint.merge/save   (仅运行结束后!)
 → 统计 / 导出 / 报告
```

**转换边界（每个都是 bug 栖息点）**：YAML→dict→`BatchTask`（`task_kind` 默认值丢失）、笛卡尔展开、`ModelConfig`→`ProviderConfig`（v2.15.1 dict-vs-object bug 的发源地）、`(stream,encoding)`→`(response,tokens,latency)`、`BatchResult.attempt_history`→JSON checkpoint（自引用风险）。

---

## 3. 评估方法

启动 5 个 CodeGraph 探查 agent，分别深审 5 个子系统（执行核心 / 服务层 / 配置 / 切分与格式化 / 提供者与 IO），每个返回"设计意图 + 数据流 + 具体问题（file:line）+ 风险 + 重构机会"。本文是五份评审的综合与提炼。所有列为 §5 的承载性 bug 均经主线程直接 Read 源码复核。

---

## 4. 核心问题深度分析

### 4.1 执行引擎：上帝类、重试×回退叠加、并发失配

#### 4.1.1 `GlobalBatchProcessor` 是吞回六类职责的上帝类
`batch_processor.py:462-1148`（~700 LOC）独占：线程池尺寸策略、`rich.Progress` UI 构建、重试回调连线、鉴权错误去重（`_auth_error_lock`）、流式 token 收集（`_stream_and_collect`）、两个任务专用 runner（翻译 chunk、论文解读）、回退编排（`_process_single_global_task`）、单配置执行（`_try_run_with_config`）。**调度、执行、UI 三者缠绕**。提取 `ProviderManager` 的方向正确，但上帝类又把这些职责吸了回去。

#### 4.1.2 `BatchProcessor` 与 `GlobalBatchProcessor` 平行重复
- 流式/token 循环重复：`batch_processor.py:223-252` vs `:537-595`。
- 进度条脚手架整段重复：`:320-370` ≈ `:1020-1082`。
- 重试回调闭包重复：`:377-423` ≈ `:1086-1137`（各 ~50 LOC）。
- **4 个统计聚合器**：`:103`、`batch_service.py:143`、`:431`、`:1145`，全是同一套 `[r.metadata.latency for r in successful if r.metadata]`。
- **3 处 `RequestMetadata(...)` 构造**：`:258 / :668 / :767`，字段解析逻辑相同（`temperature if temperature is not None else provider.config.api_temperature`）——正是 v2.15.1 崩溃的同一代码路径，被复制三份。
- **`BatchProcessor` 在线上 CLI 路径里实际是死的**：`batch_service.py:315` 无论模型数量都走 `GlobalBatchProcessor`。~330 LOC 平行层级纯属历史包袱。

#### 4.1.3 两层重试零协同 —— 调用放大（最高严重度设计缺陷）
- 第一层：`BoundedRetryRunner`（`concurrent.py:126-144`）按瞬时关键词把失败结果重试 `max_retries` 次。
- 第二层：`_process_single_global_task`（`:916-986`）在**一次** worker 调用内走完整条回退链 `[primary, *fallbacks]`。

当 runner 重新提交任务时，worker **从 primary 重新走整条回退链**。净效果：**每个任务最多 `max_retries × len(fallback_chain)` 次 API 调用**。`max_retries=3` × 3 个回退 = 最多 12 次。`retry_policy.py` 里 `RetryPolicy`/`ProviderRetryRegistry` 正是为统一这一逻辑而存在，却**未接线**（`concurrent.py:269` 的 `_is_transient_error` 兼容 shim 仍在 gate runner）。这是一个为统一而写、却没启用的模块。

#### 4.1.4 并发正确性 / 性能
- **进度条"谎言"**：`batch_processor.py:309/320` docstring 声称"单一总进度条 / 每个 worker 一条"，但 `:347-370` 与 `:1047-1078` 对**每个待处理任务**都 `progress.add_task(...)`。N=1000 任务 / 50 线程 → 1000 条 live `rich.Progress` 条。内存/渲染成本真实，且被误导性注释掩盖。
- **线程池截断是"全局最小"，非 per-provider**（`_effective_max_workers` `:498-510`）：跨所有任务取 `min(max_workers, min_burst)`。混合 provider A（burst=2）与 B（burst=20）时，整个池被压到 2，B 的任务被不必要地串行化。正确做法是 per-`(provider,model)` `Semaphore`。
- **限流 `.acquire` 阻塞 worker**（`rate_limiter.py:107`，`batch_processor.py:847` 60s 超时）：worker 最多在 `Condition.wait` 停 60s，池仍把它计为"在飞"。架构安全依赖 `_effective_max_workers` 正确，而它在混合 provider 批中不正确。
- **"可恢复"在 Ctrl-C 时不成立**：checkpoint 只在**运行结束后**保存（`batch_service.py:332`）。`BaseCheckpoint.save` 是原子写，但只被调用一次。中途 Ctrl-C → 全丢。CHANGELOG 2.12 把"resumable"当卖点，但中断场景不兑现——**设计空洞**，不是竞态。

#### 4.1.5 历史 bug 根因仍在
- **`attempt_history` 循环引用（v2.15.1）**：v2.15.1 靠 `batch_processor.py:959 / :984-986` 的切片 + `if final in attempt_history` 修补。**结构弱点**：`BatchResult.attempt_history: list["BatchResult"]`（`batch_models.py:102`，**已复核**）是**自引用 Pydantic 类型**，类型系统允许环；唯一防线是两个构造点的手工切片加一行注释。任何新代码路径把结果 append 进自己的 history 就会重现序列化崩溃。包里已有现成的扁平非递归类型 `AttemptRecord`（`execution_report.py:19`）。正确修法是把 `attempt_history` 改成 `list[AttemptRecord]`，**让 bug 在结构上不可能**，而非靠约定。
- **adapter dict-vs-object（v2.15.1）**：靠在 `_create_cached_adapter` 内构造真实 `ProviderConfig` 修补。**结构弱点**：`ProviderAdapterCache.get(config: Any, ...)` 仍 `isinstance(config, dict)` 分支（`provider_cache.py:70`）——致 bug 的 dict 路径仍 live；`get` 被 30+ 处调用，全标 `Any`；三处 `provider.config.api_temperature` 读取（`:262 / :672 / :772`）分散在代码库。**根弱点是"无类型接缝 + 分散属性访问"**，而非 dict 本身。

#### 4.1.6 死代码 / 兼容 shim
- `core/batch.py`（26 LOC）纯 re-export shim，自身 docstring 写"新代码应直接导入"，却仍被 `global_batch_runner.py:6` 等 3 处依赖。
- `batch_processor.py:12-14` 与 `:46-47` 两处 `TYPE_CHECKING` 导入 `RateLimitConfig`（重复）。
- `FormatCheckpoint`（`format_checkpoint.py`）**不**继承 `BaseCheckpoint`——v2.12 加了泛型基类，旧 format checkpoint 没迁。包里两套 checkpoint 框架。
- `BoundedRetryRunner.run()`（`concurrent.py:205`）是 `run_with_metrics` 的薄包装且丢弃 metrics；外加 `run_bounded_with_retries`（`:233`）再包一层。一个操作三个入口。
- `retry_policy.py:75` 的 `ProviderRetryRegistry.set()` **无调用方**（死）。

---

### 4.2 配置系统：全局可变状态 + 双叉身份 + 引擎越界

#### 4.2.1 两文件方案"顶层有原则、接缝处模糊"
- `default_config.yml`（行为：general/translation/batch/file/format_*/text_splitter/token/paper/rate_limits）与 `providers.yml`（provider+model 目录，含 pricing/spec）的分离**本是有原则的**（用户可调 vs 厂商策展，变更节奏不同）。
- 但 `default_config.yml` **也**接受 `providers:` 段（`docs/default_config.example.yml:15-24`），`_deep_merge` 让用户配置覆盖 `providers.yml`（`loader.py:374-380`）。provider 身份有两个 owner，仅靠合并顺序裁决。
- `_load_providers_yml`（`loader.py:222-231`）硬过滤 `runtime_fields` 白名单并**丢弃全部 pricing/spec**；pricing 在 `utils/pricing.py` 另起一次解析。**同一文件被两个 consumer 各解析一遍，各留不同字段。**
- `default_provider`/`default_model` 两文件都能设，且**都不是 `UnifiedConfig` 的字段**——身份归属真模糊。

#### 4.2.2 配置数据流是"会分叉的管线"
```
default_config.yml + providers.yml + ASK_LLM_* env
 → _deep_merge（多源）
 → _apply_env_overrides
 → UnifiedConfig.model_validate  ──► unified_config   (仅行为段)
 → _convert_providers_format ──► _parse_app_config ──► AppConfig  (仅 providers)
 → LoadResult(app_config, unified_config)   ← 临时捆绑，无单一句柄
 → set_config(LoadResult) → context._current  ← 进程级可变全局
```
分叉点是核心别扭：一份 YAML 产出**两个**按关注点切分的 pydantic 对象，临时塞进 `LoadResult`。每个 consumer 都得记清"哪个字段在哪个对象"（`batch.py:135` 读 `unified_config.batch`，`:170` 读 `app_config`）。

#### 4.2.3 全局状态问题（service-locator 反模式）
`context.py:5` 的 `_current: LoadResult | None` 是进程级可变单例。
- `get_config()` 被 **12 个模块、13 处**调用（`paper_explain_pipeline.py:396`、`format_markdown_file.py:80`、`md_body_formatter.py:66`、`md_heading_formatter.py:140`、`paper_explain.py:385`、`processor.py:38`、`format_service.py:274`、`file_handler.py:15/20/151`、`prompt_resolver.py:14`、`token_counter.py:163`、`text_splitter.py:67`）。
- **并发**：`_current` 无锁，任何线程可 `set_config` 重绑整棵树，last-writer-wins。
- **测试**：4 个测试必须先 `set_config`（`test_md_body_formatter.py:24` 等），跨测试串扰真实存在。
- **库化**：程序化嵌入必须先 `set_config`，否则深模块全抛 `RuntimeError`（`context.py:22`）。`token_counter._get_encoding`（`:173-174`）在**热路径**里调 `get_config()` 无回退。
- **隐藏耦合**：`processor.py:38`、`paper_explain.py:385`、`file_handler.py` 伸手进全局而非收参——典型 service-locator 反模式，per-call 换配置不可行，重构被锁死。

#### 4.2.4 `loader.py` 580 LOC 承载 ≥4 类职责
`${VAR}` 解析、`ASK_LLM_*` 覆盖、按名字启发式的类型强转（`_parse_env_value` 用 `"threads" in last_key` 子串匹配——新加的数值字段名不含这些子串就**静默存 str**）、`_deep_merge`、providers.yml 候选路径与加载、`ConfigLoader` 搜索序、`_convert_providers_format`、`_parse_app_config`。其中模型名 list-of-dicts→list-of-strings 归一化**写了两遍**（`:250-261` 与 `:508-517`）。

#### 4.2.5 具体 bug 与弱点
- **未解析 `${VAR}` 被当真实 API key 发出**（`loader.py:163-178`，**已复核**）：`resolve_env_vars` 在 env 未设时仅 `logger.debug`，把字面量 `"${DEEPSEEK_API_KEY}"` 原样返回；`ProviderConfig.validate_api_key`（`models.py:111-122`）只对**空串**告警——它接受 `"${DEEPSEEK_API_KEY}"` 并当作真实 key 发送。**高严重度，正确性+安全，且静默。**
- **provenance 误导**：`LoadResult.config_path` 永远是 `user_path`（`:422`），但数据是 4 路合并；`--debug-config` 不报告哪些 key 来自 `providers.yml`、哪些来自包默认。
- **两个 env 写同一字段**：`ASK_LLM_TRANSLATION_THREADS` 与 `ASK_LLM_TRANSLATION_MAX_CONCURRENT_API_CALLS` 都映射到 `("translation","max_concurrent_api_calls")`（`:36-41`），字典迭代序决胜负；叠加 `TranslationConfig._sync_threads_and_max_concurrent_api_calls`（`:107-123`）——三个源、一个值。
- **密钥明文驻留全局**：`api_key` 是 `ProviderConfig` 上的普通 `str`（`models.py:98`），无 `SecretStr`，任何 `get_config()` 调用方可达。
- **热路径里函数内 import 引擎**：`_convert_providers_format` 在缺 `base_url` 时按 provider 做 `from llm_engine.config_loader import load_providers_config`（`:484-491`），且 `except Exception: pass` 静默吞。
- **`UnifiedConfig` 无 `model_config`**（`:394`）：默认 `extra="ignore"`，顶层 key 笔误（`providrs:`）静默丢失；兄弟模型 `PaperConfig`/`TranslationConfig` 却显式设 `extra="ignore"`，策略不一致。
- **`min_chunk_merge_tokens` 已弃用**（`:75-79`）却仍占 env 映射（`:46`）与解析分支——死面积。

#### 4.2.6 `paper_explain_pipeline.py`（454 LOC）放在 config/ 里是 smell
它不是配置——它是论文解读管线的**领域模型与解析引擎**（7 个嵌套 pydantic 模型、15+ 方法、自由函数）。只因"启动时读个 YAML"就被塞进 config/，且通过**调用时延迟 import** 伸手回全局（`:396`）规避循环依赖。应移到 `core/paper_explain.py` 或新 `pipeline/` 包。

---

### 4.3 服务层：编排重复 + CLI 边界泄漏 + 上帝函数

#### 4.3.1 抽象"名实不一"
五个 service 五种"编排"诠释：
- **`AskService`**：唯一干净范本（零 print、零 typer、返回 dataclass）。
- **`BatchService`**：`run_batch_from_config`（`batch_service.py:171-358`，~190 LOC 上帝函数）返回结构化结果；`BatchService` 类的方法却**既打印又写文件**——混合契约。
- **`TranslationService`**（833 LOC）：副作用巨物，`translate_files` 返回结果但 CLI 几乎不用；打印/写文件/聚合全在里面。
- **`PaperService`**（629 LOC）：直接 `raise typer.Exit`（`:196/208/270`），五者契约最差。
- **`FormatService`**：大多是展示助手，类本身只是 66 LOC resume 包装。"service"实为住进 services/ 的 presenter。

#### 4.3.2 跨 service 重复（量化 ~60-70%）
batch/trans/paper 是同一条管线的三个实例。重复块：
- **pricing 加载打印块**：`batch.py:137-143` / `trans.py:244-250` / `paper.py:158-164`，逐字节相同 6 行。
- **provider/model 解析+override+gate**：`trans.py:252-264` / `paper.py:166-178`（ask/format 是裁剪版，共 4 个变体）。
- **`build_fallback_chain` + `ModelConfig` 构造**：三处同形。
- **`run_global_batch_tasks` 调用**：三处同签名。
- **checkpoint 生命周期（最大块，~80 LOC ×2）**：create→可选 load→过滤完成→run→`merge(successful)`→`mark_all_failed_for_retry(failed)`→save→成功则 unlink。`batch_service.py:281-339` vs `translation_service.py:414-495`，8 步同序，已经**微妙漂移**。
- **`export_report`**：三处近似实现。
- **cost 估算打印**：6 处同调用。
- **CLI 异常样板**：`batch.py:200-213`/`trans.py:316-329`/`paper.py:217-227`/`format_cmd.py:352-365` 同一五臂 except 抄 4 遍；而 `ask.py:162` 已用 `cli_errors` 上下文管理器——**去重抽象已存在，5 个命令里 4 个不用**。

合计 ~250-300 LOC 机械重复。

#### 4.3.3 CLI 边界泄漏（违反 AGENTS.md 规则）
- **(a) `PaperService` 在 service 里 `raise typer.Exit`**（`:146/196/208/270`），docstring 甚至**白纸黑字记录**这一违规（`:144`"typer.Exit: On dry-run..."）。使该 service 在任何非 CLI 入口（测试/库/future HTTP API）不可用，且迫使 CLI 用 `try/finally` 兜报告（`paper.py:212-215`）。
- **(b) `TranslationService` 反向 import 私有 CLI helper**：`:166` `from ask_llm.cli.common import _is_directory_output, _resolve_trans_input_paths`——`cli→service→cli` 依赖倒置，且伸手拿 `_` 前缀私有名。
- **(c) 展示嵌进 service**：`translation_service.py` ~40、`paper_service.py` ~20、`batch_service.py` ~15、`format_service.py` ~15 处 `console.print*`；`ask_service.py` 0。service 不抓 stdout 就无法单测，也无法切结构化/JSON 渲染。
- **(d) `cli_session.py` 自己泄漏**：住 config/ 却直接 `raise typer.Exit(1)`（`:61/93/99/105/111`）与 `console.print_error`（`:58/90/98/104/108`）；函数名 `resolve_*_or_exit` 自曝泄漏。

#### 4.3.4 隐藏可变跨线程状态
`TranslationService._batch_results`（`:132`）从主线程与 `ThreadPoolExecutor` worker 双向写入（`:251-294`、`:430/451/692/824`），与 `session_result` 双账本，全靠 CPython GIL 的 `list.extend` 原子性兜底。

#### 4.3.5 其它具体问题
- `translation_service.py:472-473/517-518`：`r.status.value == "failed"`（字符串比较）而非 `r.status == TaskStatus.FAILED`，重复 4 次。
- `:479/491`：`getattr(processor, "_auth_error_logged"/"last_metrics")`——依赖未文档化的 processor 内部。
- `paper_service.py:260`：`max_retries=3` 硬编码进 `run_global_batch_tasks`，而 batch/trans 从配置读——重试策略在一个 service 里被烤死。
- `batch_service.py:175`：`batch_config_unified: Any` 丢弃类型（调用方传的是 `UnifiedBatchConfig`）。
- `format_service.py:337-340`：`os.remove(checkpoint)` 包 `except OSError: pass`，用户看到"全部完成"但 checkpoint 可能残留。
- `translation_service.py:452-462`：`run_global_batch_tasks` 外套宽 `except Exception`，把一切失败压成单个失败结果，**吞掉部分已 checkpoint 的结果**。

---

### 4.4 文本切分与格式化：三套并行管线 + 死代码 + 代码块腐蚀

#### 4.4.1 一个概念管线，三套实现
| 维度 | 标题 path | 正文 path | 翻译 path |
|------|-----------|----------|-----------|
| 入口 | `format_one_markdown_file` | `format_body_markdown_file` | `translation_service` / `notebook_translator` |
| 切分 | `HeadingExtractor.extract`（取标题 span） | `MarkdownTokenSplitter` | `MarkdownTokenSplitter` + `rebalance_translation_chunks` |
| 重组 | `HeadingApplier.apply`（按位置拼回） | `_join_chunks`（`\n\n` 强制拼接） | 有序拼接 |

两个 formatter（`HeadingFormatter` 537、`BodyFormatter` 450）是**同一编排骨架的平行重写**：init from config → load prompt → build work units → `run_bounded_with_retries` → stats → 失败写 checkpoint。

#### 4.4.2 两份 splitter ~80% 行对行重复，字符版是死代码
`MarkdownTokenSplitter`（`markdown_token_splitter.py`）继承 `TextSplitter` 却**重写整套二分算法**而非复用：
- `split()`：`text_splitter.py:96-160` vs `markdown_token_splitter.py:33-72`（控制流相同，仅 fit 测试不同：`len(text)<=max` vs `self._fits`）。
- `_split_by_headings_binary`：`:162-256` vs `:74-141`（除 fit 测试逐字相同）。
- `_split_by_paragraphs_binary`：`:258-338` vs `:143-209`。
- `_split_long_paragraph`：`:340-435` vs `:211-276`。

**生产只 `MarkdownTokenSplitter` 有用**：字符版 `MarkdownSplitter`/`PlainTextSplitter`/`create_splitter` **生产调用为零**（仅 `test_text_splitter.py:27/32` 与 `core/__init__.py` re-export）。`text_splitter.py` 620 行基本是死镜像。

#### 4.4.3 代码块/inline 保护"只在标题侧有、正文侧缺失"——最高潜伏 bug
- 只有 `HeadingExtractor` 保护代码块（`md_heading_formatter.py:43-69`）。
- **`MarkdownTokenSplitter` 完全无代码栅栏意识**（`markdown_token_splitter.py:33-72` 仅按 `HEADING_PATTERN` 与段落边界切）。它唯一保护是 `:166-172` 的 `$$` display-math 合并——而该逻辑在 `chunk_balance.py:29-35` **第三次重复**。
- **无任何 frontmatter（`---`）保护**：frontmatter 里的 `# foo` 会被当标题。
- **后果**：正文格式化能把一个 fenced 代码块切成两半，`_join_chunks` 再用 `\n\n`（`md_body_formatter.py:320`）粘回——损坏的代码被 LLM "重新格式化"。**最可能的活 bug。**

#### 4.4.4 重组策略不一致，正文侧有损
- `HeadingApplier.apply` 用真实 `start_pos/end_pos` 拼接（`:532-534`）。
- `BodyFormatter._join_chunks` **无视** splitter 辛苦算出的 `TextChunk.start_pos/end_pos`，只 `rstrip+"\n\n"+lstrip`（`:320`）。于是 token splitter 的全部位置簿记对正文 path 是**死重**。
- 正文 `_join_chunks` 强制 `\n\n`、丢弃原始块间结构——连续列表项/紧凑表格可能被改动。
- token 预算只算 content、不算 prompt overhead（`markdown_token_splitter.py:28-31`）——大模板+近上下文窗口的模型，"放得下"的 chunk 可能溢出。

#### 4.4.5 formatter 间重复
- `_BatchResult`/`_ChunkResult` dataclass 结构同构（`md_heading_formatter.py:214` vs `md_body_formatter.py:50`）。
- `_load_prompt_from_file` 两处**逐字节相同**（`:471-490` vs `:432-450`）。
- 包 `run_bounded_with_retries` 的四个 lambda 两处相同。
- 失败后 `FormatCheckpoint` 同字段集。
- `HEADING_PATTERN` 定义 **3 次**（`text_splitter.py:90`、`markdown_token_splitter.py:16`、`md_heading_formatter.py:40`）。

#### 4.4.6 具体问题
- `format_service.py:109-137` 与 `:191-221`：标题/正文 if 分叉在 sequential 与 parallel runner 里**各抄一遍**。
- resume 仅正文支持（`format_service.py:311` 标题模式 raise），`HeadingFormatter` 却写 checkpoint（`:372-398`）但永不能 resume——非对称、误导。
- 两种 chunk-id 约定并存：`resume_from_checkpoint` 用原始 split 的 id，翻译侧 `rebalance_translation_chunks` 重编号（`chunk_balance.py:159`）。
- 提示词字符串混进类体（`CONTEXT_BATCH_INSTRUCTION` `:136-138`），与编排逻辑交织。

---

### 4.5 提供者抽象与 IO：引擎边界散布 + 令牌计数近似 + 导出器重复

#### 4.5.1 引擎边界不止一处，知识散落 8+ 模块
- 规范接缝：`provider_cache.py:14` `from llm_engine import create_provider_adapter`，`@lru_cache`（`:19-48`）。仅 batch/paper/trans 走它。
- **直连绕过**：`chat.py:131`、`ask.py:213`、`format_cmd.py:269`、`config.py:134`、`interactive_config.py:177` 直接 `create_provider_adapter`。
- **第二接缝（更糟）**：`loader.py:485` `from llm_engine.config_loader import load_providers_config`——引擎还是 provider 目录/base_url 的真相源。**深耦合**，不只是 adapter 工厂。
- 存在性探测：`trans.py:12`、`paper.py:21` 裸 `except ImportError`。

引擎 API 一变（改名/签名/`load_providers_config` 移位）要 patch 6+ 文件。接缝存在但**未被强制**。

#### 4.5.2 provider 三模块名实不符
- `provider_cache.py`：LRU 缓存 adapter 实例。
- `provider_router.py`：**名含 route 实则不路由**，只从 `fallback_to` 构造回退 `ModelConfig` 列表（`:14-50`）。应名 `fallback_chain.py`。
- `provider_specs.py`：加载模型上限 + 硬编码 DeepSeek `max_tokens` cap（`:17-32`）。应名 `model_limits.py`。
- 外加 `retry_policy.ProviderRetryRegistry` 与 `telemetry.classify_error`——"provider 行为"散在 5 模块无共同归属。

#### 4.5.3 `rate_limiter` 正确但有摩擦
per `(provider,model)` `_SyncTokenBucket`，线程安全分析**通过**（外锁护 dict，桶内 `Condition.wait`，无嵌套锁死锁）。但：
- 无 `notify`——thundering-herd 唤醒（实践中由 pool≤burst 限幅，低影响）。
- 配置变更时**重建桶、丢弃热令牌**（`:122-126`）。
- **60s acquire 超时硬编码两处**（`:111`、`batch_processor.py:847`）；低 RPM provider + 长队列 → 假性 `RuntimeError("Rate limit timeout")`（`:850-852`）。acquire 层无指数退避。

#### 4.5.4 `token_counter` 静默近似 —— 第二大正确性风险
- `ENCODING_MAP`（`:29-47`，**已复核**）把 `deepseek`/`deepseek-chat`/`deepseek-reasoner`/`qwen*` 全映射到 **`cl100k_base`**（GPT-3.5/4 分词器）。DeepSeek/Qwen 用各自 BPE；cl100k_base 对 CJK **明显少计**（中文约 1 token/字 vs cl100k ~0.5）。
- 该计数喂给 `split_hard_by_max_tokens`（`:189-229`）→ 切块尺寸 → **上下文窗口预算**。少计 → 块过大 → provider 端 `context length` 错误。`deepseek-reasoner` 是 `DEFAULT_FALLBACK_MODEL`（`constants.py:29`）——**直接打击项目主力 provider**。
- 三条静默 fallback（`:127-137`）全 DEBUG 级，用户拿"词数当 token 数"无可见告警。
- `truncate_to_tokens`（`:232-262`）无 tiktoken 时用 `max_tokens*4` 字符启发——与 `count_tokens` 的"词数"回退**不一致**，同一类两种"无 tiktoken 模型"。
- `_get_encoding` 未知模型路径（`:182-187`）返回全局默认编码——拼错的 model id 静默用默认分词器。
- `_count_tokens_cached` 的 1024 项 LRU **持有完整文本串**，长会话驻留内存。

#### 4.5.5 导出器：三个序列化器、三种 JSON 策略、无共享投影
| 模块 | JSON 策略 | 投影 |
|------|-----------|------|
| `batch_exporter.py:124-133` | **流式** `iterencode` 分块写盘 | `_prepare_data`（`:343-389`） |
| `translation_exporter.py:318-376` | **内存** `json.dumps` 后 `write_text` | `_export_json` 内联 |
| `execution_report.py:99-103` | Pydantic `model_dump_json` | `_batch_result_to_attempt_records` |

- `translation_exporter._export_json` **不流式**（`:372` 整文档物化）——大多 chunk 翻译的最坏内存路径。
- 三者各自手算 `provider/model/input_tokens/output_tokens/latency`，`BatchResult.metadata` schema 一变改三处。
- `_detect_format` 重复：`batch_exporter.py:73-83` 与 `cli/commands/batch.py:147-158`。
- `translation_exporter._unwrap_translation_payload`/`_try_parse_json_with_latex_escapes`（~90 行脆弱字符串手术）住在"导出器"里，应移到独立 `response_parser`。

#### 4.5.6 `console.py` 过度单例 + loguru 强耦合
- 三重单例保障（`__new__` `:40-43` + `_initialized` `:47-51` + 模块全局 `:310`），冗余。
- `setup`（`:71-107`）`logger.remove()` 会**擦掉宿主应用的 loguru sink**。
- `print` 的 `end != "\n"`（`:131-149`）建临时 `RichConsole` 捕获再写——`print_stream` 已直接写 stdout，同一思路两种慢实现。
- loguru 硬耦合，无日志抽象。

#### 4.5.7 `file_handler.py` 把进度混进 IO
`read`/`write` 在"是否带进度"上分叉，各两套实现（带 `tqdm` 的分块读写 vs 朴素 `read_text/write_text`）。`_write_with_progress` total 用 `len(content)`（字符），增量却用 `len(chunk.encode("utf-8"))`（字节）——多字节文本进度条**冲过 100%**（外观 bug）。`is_text_file` 硬编码扩展名集，与别处扩展名逻辑重复。

#### 4.5.8 密钥残留
`provider_cache._create_cached_adapter` 的 `@lru_cache` 把 `api_key` 进 key（`:23`），**轮换凭据残留**至驱逐；`clear()` 存在（`:108`）但 `api_key_gate.ensure_api_key_for_provider`（`:118-121`）应用新 key 时不调它。

---

## 5. 承载性 Bug 清单（已复核）

| # | 严重度 | 位置 | 问题 | 后果 |
|---|--------|------|------|------|
| B1 | 高 | `batch_processor.py:916-986` × `concurrent.py:126-144` | 两层重试零协同，fallback×retry 调用放大 | 成本失控、provider 加重限流 |
| B2 | 高 | `token_counter.py:39-46` | deepseek/qwen 用 cl100k_base | CJK 少计 → 块过大 → 上下文溢出 |
| B3 | 高 | `loader.py:163-178` + `models.py:111` | 未解析 `${VAR}` 原样当 API key | 占位符当凭据发出（安全+正确性，静默） |
| B4 | 高 | `markdown_token_splitter.py:33-72` + `md_body_formatter.py:320` | 正文 splitter 无代码栅栏意识 | 切断代码块、`\n\n` 粘回损坏 |
| B5 | 中高 | `batch_service.py:332` | checkpoint 仅运行结束后保存 | Ctrl-C 丢全部进度，"可恢复"不兑现 |
| B6 | 中 | `batch_processor.py:347-370`/`1047-1078` | 每任务一条进度条 | N 任务→N 条 live bar，内存/渲染 |
| B7 | 中 | `batch_models.py:102` | `attempt_history: list["BatchResult"]` 自引用 | 循环引用崩溃类（v2.15.1 已爆一次） |
| B8 | 中 | `provider_cache.py:60-99` | `get(config: Any)` dict 路径仍 live | adapter dict-vs-object 崩溃类可复发 |
| B9 | 中 | `rate_limiter.py:111` + `batch_processor.py:847` | 60s acquire 超时硬编码 | 紧 provider + 长队列假性超时 |
| B10 | 低中 | `file_handler.py:131,148` | 进度条 total 用字符、增量用字节 | 多字节文本冲过 100% |
| B11 | 低 | `format_service.py:339` | `except OSError: pass` 删 checkpoint | 残留 checkpoint + 误导"全部完成" |

---

## 6. 风险评估（排序）

| 级别 | 风险 | 触发条件 | 影响 |
|------|------|----------|------|
| 🔴 高 | B1 调用放大 | 任意带 fallback 的批处理 | 成本、限流、口碑 |
| 🔴 高 | B2 CJK 令牌少计 | trans/paper 处理中文（主力场景） | 高频 API 失败 |
| 🔴 高 | B3 占位符当 key | env 未设 | 静默鉴权失败/泄漏配置意图 |
| 🟠 中高 | B4 代码块腐蚀 | 正文格式化含代码的 md | 静默损坏用户代码 |
| 🟠 中高 | 全局可变配置 | 测试/库化/并发 | 串扰、不可嵌入、重构锁死 |
| 🟠 中高 | 引擎边界散布 | 引擎升级 | 多文件 hunt，回归面大 |
| 🟡 中 | B5 中断不可恢复 | 大批量 Ctrl-C | 进度丢失、信任受损 |
| 🟡 中 | 服务层重复+泄漏 | 维护期 | 改一处漏 N 处，已漂移 |
| 🟡 中 | 三管线 splitter 重复 | 维护期 | 修一边漏另一边，已漂移 |
| 🟢 低 | B6/B10/B11 | 大规模/多字节文本 | UX、外观 |

---

## 7. 重构方案：目标架构

### 7.1 设计原则（统一思想）
1. **单一所有者原则**：每条管线只有一处实现。批处理执行、checkpoint 生命周期、BatchResult 投影、CLI bootstrap 各收敛到一处。
2. **显式依赖注入**：消灭进程级配置全局；config 作为参数流过构造函数。
3. **类型即防线**：用类型让历史 bug（循环引用、dict-vs-object）**结构上不可能**。
4. **关注点正交**：调度 / 执行 / 回退 / 展示 / 持久化 五者分离，可独立测试与替换。
5. **引擎私有化**：`llm-api-engine` 是且仅是一个模块的私有依赖。
6. **提示词与代码分离**：prompt 进文件/资源，代码只含编排。

### 7.2 目标分层
```
CLI (typer, 薄)                     cli/commands/*   只做参数解析+退出码+渲染
   │ cli/presentation.py            纯展示层（per-command presenter）
   │ cli_errors (context manager)   统一异常→退出码
   ▼
UseCase Services (编排, 无副作用)   services/*
   AskService / BatchService / TranslationService / FormatService / PaperService
   收 Config + 返回 *SessionResult；零 typer；零 print
   ▼
Execution Engine (单一所有者)        core/engine/
   BatchCoordinator   ~50 LOC 协调器
   ├── Scheduler         (ThreadPoolExecutor + 重试堆, per-provider Semaphore)
   ├── TaskExecutor      (限流 acquire + adapter 查找 + stream-collect + metadata)
   ├── FallbackPolicy    ([primary,*fallbacks] + should_fallback 短路)
   ├── EscalationPolicy  (统一 retry×fallback, 接线 RetryPolicy)
   ├── ProgressPresenter (per-worker 条)
   └── StreamCollector   (唯一流式/token 实现)
   ▼
Domain Models                        core/models.py, core/batch_models.py
   BatchResult.attempt_history: list[AttemptRecord]   ← 非自引用
   BatchResult.project() / to_export_dict()           ← 唯一投影
   ▼
Providers (引擎私有化)               core/engine/engine_facade.py
   EngineAdapter   封装 create_provider_adapter + load_providers_config
   ProviderAdapterCache.get(config: ProviderConfig) -> LLMProviderProtocol
   fallback_chain.py / model_limits.py / error_policy.py (统一关键词表)
   ▼
Config (无全局, 单一对象)            config/
   ConfigLoader.load(path) -> Config   (纯, 无全局)
   Config = AppConfig ⊃ UnifiedConfig 段 + ProviderCatalog + provenance
   contextvars.ContextVar[Config]  ← 过渡期; 最终移除
Markdown Pipeline (单一管线)         core/markdown/
   MarkdownStructure   (fence/frontmatter/heading/paragraph 一次解析)
   BinarySplitter(BudgetPolicy)   (char|token, 唯一二分实现)
   ChunkedLLMJob(基类)            (Heading/Body formatter 薄子类)
   Reassembler                    (position-aware, 唯一重组)
```

### 7.3 关键类型重塑
```python
# core/batch_models.py —— 让 B7 结构性消失
class AttemptRecord(BaseModel):       # 扁平、非递归（已存在于 execution_report.py）
    provider: str; model: str; status: TaskStatus
    error: str | None; error_category: ErrorCategory | None
    latency_ms: float; timestamp: datetime

class BatchResult(BaseModel):
    ...
    attempt_history: list[AttemptRecord] = []   # 不再 list["BatchResult"]
    def project(self) -> dict: ...              # 唯一导出投影

# core/engine/engine_facade.py —— 让 B8 结构性消失
class EngineAdapter:
    def provider(self, cfg: ProviderConfig, *, default_model: str) -> LLMProviderProtocol: ...
    def catalog(self, path: Path) -> ProviderCatalog: ...   # 收编 load_providers_config

# ProviderAdapterCache —— 删 dict 分支
def get(cfg: ProviderConfig, *, default_model: str) -> LLMProviderProtocol: ...

# LLMProviderProtocol —— 钉 config 类型
class LLMProviderProtocol(Protocol):
    config: ProviderConfig
    def call(self, ...) -> ...: ...
```

---

## 8. 重构与改进计划（分阶段 P0–P4）

> 原则：每阶段独立发版、独立回滚；先修承载性 bug（P0），再结构重构（P1–P3），最后清理（P4）。每阶段配套测试先行。

### P0 —— 承载性 Bug 与安全止血（1–2 天，零行为破坏性）

目标：不重构架构，先把会咬人的 bug 修掉。每条均可独立 PR。

| 任务 | 文件 | 动作 |
|------|------|------|
| **B3 `${VAR}` 占位符** | `loader.py:163-178`、`models.py:111` | `resolve_env_vars` 对 `api_key`/敏感字段解析失败时**抛 `ConfigError`**；`validate_api_key` 拒绝含 `${` 的串 |
| **B2 CJK 令牌近似** | `token_counter.py:39-46` | 短期：cl100k_base 替换时**每模型 INFO 告警一次** + 切块前加 `×1.15` 安全系数；`truncate_to_tokens` 回退统一为词数 |
| **B9 限流超时** | `rate_limiter.py:111`、`batch_processor.py:847` | 60s 提为 `RateLimitConfig.acquire_timeout_seconds`；acquire 前一次短退避 |
| **B6 进度条** | `batch_processor.py:347-370,1047-1078` | 改 per-worker 条（按 worker slot），修误导 docstring |
| **B8 类型止血** | `provider_cache.py:60` | `get(config: ProviderConfig, ...)`；保留 dict 兼容入口但标 deprecated + warn |
| **B7 类型止血** | `batch_models.py:102` | `attempt_history: list[AttemptRecord]`；迁移两个构造点；删 `[:-1]` 切片 hack |
| **B4 代码块腐蚀** | `markdown_token_splitter.py:33` | splitter 调用 `MarkdownStructure` 的 fence range，拒绝在栅栏内切（P3 完整化的前置补丁：内联 fence 检测函数） |
| **密钥** | `models.py:98` | `api_key: SecretStr`；`model_dump` 自动脱敏；`api_key_gate` 轮换时调 `ProviderAdapterCache.clear()` |

验收：`pytest` 全绿 + 6 个回归测试（每 bug 一条）。

### P1 —— 执行引擎统一（核心重构，3–5 天）

目标：消灭 §4.1 全部问题，B1 结构性消失。

1. **接线 `EscalationPolicy`**：worker 拥有完整升级（primary→fallback→backoff→retry），runner 只调度、不重试；或 runner 把"耗尽回退链"作为重试单元。**消除调用放大**。启用闲置的 `RetryPolicy`/`ProviderRetryRegistry`，删 `_is_transient_error` shim 与 `ProviderRetryRegistry.set` 死代码。
2. **删除 `BatchProcessor`**：单模型任务列表即 `GlobalBatchProcessor` 的退化情形，删 ~330 LOC 平行层级。
3. **拆 `GlobalBatchProcessor` 为四协作者**（`Scheduler`/`TaskExecutor`/`FallbackPolicy`/`ProgressPresenter`），提取 `StreamCollector`（唯一流式实现）。协调器 ≤50 LOC。
4. **per-`(provider,model)` 池 sizing**：`_effective_max_workers` 的全局最小 cap → per-bucket `Semaphore`，全局池可 `sum(bursts)`。
5. **合并 4 个统计聚合器**为 `BatchStatistics.from_results(results)`。
6. **统一 `RequestMetadata(...)` 构造**为 `RequestMetadata.from_execution(...)`，消灭三处重复（B8 根因收尾）。
7. **checkpoint 增量化**：runner 完成回调每 K 个完成 + SIGINT 落盘（修 B5）。
8. 删 `core/batch.py` shim、重复 `TYPE_CHECKING` 块、`BoundedRetryRunner.run` 薄包装；统一入口为 `run_bounded_with_retries`。
9. `FormatCheckpoint` 迁移到 `BaseCheckpoint`（或删基类），二选一。

验收：批处理基准（`tests/benchmarks/test_performance.py`）调用数回归测试（断言 `total_calls <= tasks × (retries+1)`）；新并发正确性测试（混合 provider 不串行）。

### P2 —— 配置去全局 + 单一对象（3–4 天）

目标：消灭 §4.2。

1. **合并双叉为单一 `Config`**：`Config = AppConfig ⊃ UnifiedConfig 段 + ProviderCatalog`；`LoadResult` 退役。`UnifiedConfig` 设 `extra="forbid"`（顶层笔误即报错）。
2. **`providers.yml` 单次解析**为 `ProviderCatalog`，runtime view 与 pricing 共享同一模型（删 pricing 二次解析）。`default_config.yml` 的 `providers:` 段移除，provider 身份唯一 owner。`_convert_providers_format`（~90 LOC）整体删除。
3. **`loader.py` 拆分**：`config/env.py`（`${VAR}`+`ASK_LLM_*`）、`config/merge.py`、`config/providers_catalog.py`、瘦身 `loader.py`。`_parse_env_value` 改为按字段类型强转（非名字启发式）。
4. **全局→注入**：过渡用 `contextvars.ContextVar[Config]`（并发/测试隔离）；目标：service 构造函数收 `Config`，12 个深读者改为收所需字段参数（`TokenCounter(default_encoding=...)` 等）。
5. **provenance 落地**：`_deep_merge` 记录每叶来源文件，`--debug-config` 真实报告。
6. **`paper_explain_pipeline.py` 移出 config/**→ `core/paper_explain.py` 或 `core/pipeline/`。
7. **env 重复映射**：`ENV_TO_CONFIG` 导入期检测重复 key 并报错；`THREADS`/`MAX_CONCURRENT_API_CALLS` 合并为一个规范 key；删 `_sync_threads_and_max_concurrent_api_calls`。

验收：无 `get_config()` 调用的 service 单测全绿；`--debug-config` 输出含 per-key provenance；`extra="forbid"` 下 typo 即报错测试。

### P3 —— Markdown 单一管线（3–4 天）

目标：消灭 §4.4，B4 完整修复。

1. **`MarkdownStructure` 单一解析器**：一次产出 fence range、frontmatter range、heading span（带层级）、段落边界。`HeadingExtractor`、`MarkdownTokenSplitter`、`chunk_balance._split_by_token_budget` 全部消费它。集中代码块/frontmatter 保护，`HEADING_PATTERN` 只剩一处。
2. **`BinarySplitter(BudgetPolicy)`**：`CharBudget` / `TokenBudget`（接受 `prompt_overhead`，预算 `prompt+content`）。删字符版死代码（`text_splitter.py` 的 `MarkdownSplitter`/`PlainTextSplitter`/`create_splitter`，仅留测试夹具或全删）。`split_hard_by_max_tokens` 与 `rebalance` 复用。
3. **`ChunkedLLMJob` 基类**：`HeadingFormatter`/`BodyFormatter` 降为薄子类（override `build_work_units`/`call_llm`/`reassemble`）。基类拥有 init 回退、`_load_prompt_from_file`、runner wiring、stats、checkpoint save/resume。消灭非对称 resume。
4. **`Reassembler` position-aware**：消费 `TextChunk.start_pos/end_pos`，正文侧不再强制 `\n\n`（或显式文档化"顺序分区"并删死的位置字段）。
5. **`format_service` 分叉合并**：单一 `format_one(file, fmt, opts)` 调度器，删 sequential/parallel 两份 if/else。
6. **prompt 外迁**：`CONTEXT_BATCH_INSTRUCTION` 等进 prompt 文件机制。
7. chunk-id 约定统一。

验收：切分/格式化"代码块不被切断"回归测试；splitter pair LOC 净降（~935→~350）。

### P4 —— 服务层 + 引擎接缝 + 导出器收尾（2–3 天）

目标：消灭 §4.3、§4.5 剩余。

1. **`CommandRunner.execute` + 共享 checkpoint 生命周期**（`core/command_runner.py`），batch/trans/paper 三 service 迁入；删 ~250 LOC 重复；统一重试策略（清掉 paper `max_retries=3` 硬编码）。
2. **展示剥离**：service 返回 `*SessionResult`，`cli/presentation.py` 渲染；`PaperService` 的 `typer.Exit` 换成 `PaperSessionResult(status=NOTHING_TO_DO)` + `DryRunReport`，CLI 翻译为退出码。
3. **`_resolve_trans_input_paths`/`_is_directory_output` 移出 `cli/common.py`**→`utils/path_resolver.py`，service 从此 import。
4. **CLI bootstrap 统一**：`bootstrap_command()` 返回 `(Config, provider, model, pricing_map)`；五命令统一用 `cli_errors`。
5. **`TranslationService` 拆分**为 `TextFileTranslator` + `NotebookFileTranslator`，service 变不可变聚合器；删 `_batch_results` 跨线程可变累加。
6. **`EngineAdapter` facade**：6 处直连 `create_provider_adapter` 与 `loader.py:485` 的 `load_providers_config` 全收编；`llm_engine` 成唯一模块私有依赖。`provider_router`→`fallback_chain`、`provider_specs`→`model_limits` 命名修正。
7. **导出器统一**：`BatchResult.project()` 唯一投影；全流式 `iterencode`；`_unwrap_translation_payload` 移 `response_parser.py`。`_detect_format` 收敛一处。
8. **关键词表统一**：`retry_policy.DEFAULT_TRANSIENT_KEYWORDS` 与 `telemetry` 分类关键词合并为单表 `(keyword → retry? category)`。
9. `console.py` 删 `__new__`/`_initialized` 冗余单例；`setup(append=True)` 不擦宿主 sink。
10. `file_handler` 去进度耦合（`on_chunk` 回调），修 B10。

验收：service 单测无需捕获 stdout；引擎 import 只出现在 `engine_facade.py`（grep 不变量）；导出 schema 单一源。

---

## 9. 迁移、兼容与回滚

- **配置兼容**：P2 期间 `default_config.yml` 的 `providers:` 段支持但标 deprecated + warning；下一大版本移除。提供 `ask-llm config migrate` 子命令自动把 `providers:` 迁到 `providers.yml`。
- **API 入口**：保留 `ask-llm`/`askllm` 双脚本名与全部子命令签名不变；CLI 行为视为契约。
- **`get_config()` 弃用**：过渡期保留但发 `DeprecationWarning`；P2 末或下一大版本删除。
- **版本**：P0 发 patch（v2.15.2）；P1–P3 各发 minor（v2.16/2.17/2.18）；P2 的配置收口与 P4 的 `get_config` 删除若需破坏性，发 v3.0 并在 CHANGELOG 顶部置迁移指南。
- **回滚单元**：每阶段独立 PR + 独立 tag；P1/P2/P3 任一阶段可单独回退而不影响其它（除 P0 止血已在生产）。

---

## 10. 预期收益

| 维度 | 现状 | 目标 |
|------|------|------|
| 执行引擎 LOC | `batch_processor.py` 1150（含 ~330 死层级） | 协调器 ~50 + 四协作者各 ~150，净删 ~400 |
| splitter LOC | `text_splitter` 620 + `markdown_token_splitter` 315（~80% 重复） | 单一 `BinarySplitter` ~350 |
| 服务层重复 | ~250-300 LOC | 0（共享 `CommandRunner`） |
| 配置全局读取点 | 12 模块 13 处 `get_config()` | 0（注入） |
| 引擎 import 点 | 8+ 模块 | 1（`engine_facade`） |
| 导出投影 | 3 套手写 | 1（`BatchResult.project`） |
| 承载性 bug（§5） | 11 条 | 0（B1/B7/B8/B4 结构性消失；其余修复） |
| 进度条 | N 任务 N 条 | per-worker ≤ max_workers 条 |

粗估净删 ~1500 行；6 个正确性 bug 结构性消失；service 可单测、可库化；引擎升级从"多文件 hunt"变"改一处"。

---

## 11. 附录

### 11.1 关键 file:line 索引（便于落地时定位）
- 执行核心：`batch_processor.py:462-1148`（上帝类）、`:916-986`×`concurrent.py:126-144`（B1）、`:498-510`（全局最小 cap）、`:347-370`/`:1047-1078`（B6）、`:258/668/767`（metadata 三连）、`batch_models.py:102`（B7）、`provider_cache.py:60-99`（B8）、`batch_service.py:171-358`（上帝函数）、`:332`（B5）。
- 配置：`context.py:5`（全局）、`loader.py:163-178`（B3）、`:374-380`（双 owner）、`:485-491`（引擎越界）、`unified_config.py:394`（无 forbid）、`paper_explain_pipeline.py:396`（错位）。
- 服务：`paper_service.py:146/196/208/270`（Exit 泄漏）、`translation_service.py:166`（反 import CLI）、`:132`（跨线程可变）、`batch.py:200-213` 等（异常样板 ×4）、`cli_session.py:61/93/99/105/111`（Exit 泄漏）。
- 切分/格式化：`text_splitter.py:86-620`（死镜像）、`markdown_token_splitter.py:33-72`（B4）、`:166-172`×`chunk_balance.py:29-35`×（math 三连）、`md_heading_formatter.py:471-490`×`md_body_formatter.py:432-450`（逐字节同）、`md_body_formatter.py:320`（有损重组）、`format_service.py:109-137`/`:191-221`（分叉 ×2）。
- 提供者/IO：`provider_cache.py:14/19/23`（接缝+残留）、`loader.py:485`（第二接缝）、`token_counter.py:39-46`（B2）、`rate_limiter.py:111`（B9）、`translation_exporter.py:372-374`（非流式）、`file_handler.py:65-86/128-148`（进度耦合 + B10）、`console.py:40-43/71-107`（单例+sink 擦除）。

### 11.2 方法论说明
本评审基于 CodeGraph 索引（`.codegraph/`）的 5 路并行深审，覆盖 src 全部 17,319 行。所有 §5 承载性 bug 经主线程独立 Read 源码复核（`batch_models.py:102`、`token_counter.py:39-46`、`loader.py:163-178`）。死代码"零生产调用"结论基于 `codegraph_callers` 爆炸半径与 grep 交叉验证。并发正确性（限流器无死锁、GIL 影响）经锁层次分析。

### 11.3 相关文档
- `AGENTS.md`：架构约定（service 规则、并发原语、observability）——本文 §4.3 即对照其规则评估。
- `CHANGELOG.md`：v2.11–v2.15 的 Phase A–G 重构脉络。
- `docs/REFACTOR_PLAN.md`：**已过时**（v1→v2 迁移历史，与现状不符），建议归档或删除。
