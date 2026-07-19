# ask-llm 架构评审报告 V2（双重视角）与重构计划

> **评审视角**：数据处理领域专家 × Python 架构师 / 系统设计师
> **评审对象**：ask-llm v2.19.0（`src/` 18,419 行，测试 52 文件）
> **评审日期**：2026-07-19
> **与 V1 的关系**：本文 **取代** `docs/ARCHITECTURE_REVIEW.md` 作为"当前状态"的权威判断。
> V1（2026-07-13）提出的 P0–P4 重构 **已全部落地**（见 §1.1），其 §0–§11 + P-进度块作为历史
> 保留。本文聚焦 **v2.19.0 之上仍未解决的结构性与数据正确性问题**，并给出第二轮重构计划 R0–R4。
>
> **方法论**：`.codegraph/` 索引驱动，5 路并行 Explore agent 分别深审执行引擎 / 服务层 / 配置 /
> Markdown 管线 / 提供者与 IO，每路返回"设计意图 + 数据流 + 当前问题（file:line）+ 风险"。
> 本文是五份评审的综合与提炼。所有 §5 列出的高严重度结论均经主线程独立 `grep`/`Read` 复核。

---

## 0. 执行摘要（Executive Summary）

ask-llm 在 V1 重构后已从"中高级成熟度"迈入 **"高级成熟度，但存在两处一致性裂缝"** 的阶段：

- **值得肯定**：V1 识别的 11 条承载性 bug（B1–B11）已逐条修复，多数以 **结构性消除** 的方式落地
  （B1 调用放大、B7 自引用、B8 dict 接缝、B4 栅栏腐蚀、B6 进度条、B10 多字节进度等）。
  四协作者执行引擎拆分（Scheduler/TaskExecutor/FallbackPolicy/ProgressPresenter）真实落地，
  `GlobalBatchProcessor` 从 1150→347 LOC；`llm_engine` 收口为单模块私有依赖的不变量成立；
  `MarkdownStructure → BinarySplitter → ChunkedLLMJob` 主干干净。

- **本轮新论点（双重视角）**：V1 是 **纯架构视角**，未充分审视 **数据正确性**。本轮以数据处理视角
  复审，发现 **同一条"切分-预算-重组"主干上仍潜伏一组相互关联的数据正确性缺陷**，且它们共享同一
  根因——**预算与切分的正确性未被收敛到类型层，且格式化路径绕开了统一执行引擎**。具体：

  1. **执行引擎存在第二条管线**（架构裂缝）：`format` 命令经 `chunked_llm_job._run_units →
     run_bounded_with_retries`，**绕开** `TaskExecutor` / 回退链 / `ProgressPresenter` /
     限流 acquire。V1 争取的"单一执行引擎所有者"在格式化路径上 **未兑现**。
  2. **预算防线有洞**（数据裂缝，最高 ROI）：`TokenBudget.fits()` 用 **原始** cl100k 计数判定
     整块是否放得下，**不施加** `APPROX_TOKEN_SAFETY_FACTOR`；而 `prompt_overhead` 字段虽存在
     但 **全代码库无人传非零值**。两者叠加 ⇒ DeepSeek/Qwen（主力 provider，CJK 场景）在
     "整块刚好放得下"的快路径上可 **溢出上下文窗口**。V1 的 B2 只修了一半。
  3. **frontmatter 与翻译路径无栅栏保护**（数据裂缝）：`BinarySplitter.split` 读取
     `structure.headings` 却 **忽略** `frontmatter_range`；`chunk_balance._split_by_token_budget`
     **完全不引入** `MarkdownStructure`，重写了一份段落切分。⇒ 正文格式化会改写 frontmatter，
     翻译路径仍可切断代码块（V1 的 B4 在翻译侧 **未修**）。
  4. **配置单一对象未竟、全局状态仍在热路径**：13 处 `get_config_or_none()` 调用，含
     `token_counter` 热路径；`LoadResult` 未退役；`UnifiedConfig` 仍 `extra="ignore"`
     （YAML 笔误静默丢弃）；`providers.yml` 被解析两遍；类型强转仍按字段名子串匹配。
  5. **服务契约不统一、展示层未抽取**：`FormatService` **违反契约**（无 `SessionResult`，
     三个入口全 `-> None`）；六服务共 128 处 `console.print`（`AskService` 为 0）；
     `cli/presentation.py` **不存在**（V1 P4.2 的 follow-up 未做）；状态比较一半用枚举一半用
     字符串；服务层 `getattr` 探取 `processor` 私有属性。

- **建议**：执行 **第二轮重构 R0–R4**，统一思想为 **"一条数据管线、一个执行引擎、一个配置对象、
  一种服务契约"**。R0 先做数据正确性止血（零架构破坏、最高 ROI），R1 收口第二条执行管线，
  R2 收尾配置单一对象，R3 统一服务契约并抽取展示层，R4 清理死代码与文档漂移。每阶段可独立发版、
  独立回滚。

---

## 1. 项目概览与 V1 落地复核

### 1.1 V1 P0–P4 落地复核（逐条核对当前源码）

| V1 阶段 | V1 目标 | 当前状态 | 证据 |
|---------|---------|----------|------|
| P0 | 承载性 bug 止血（B2/B3/B4/B6/B7/B8/B9…） | ✅ 多数落地 | 见 §5 对每条的"现状"列 |
| P1 | 执行引擎统一 | ✅ 主体落地 | `TaskExecutor`/`StreamCollector`/`ProgressPresenter` 真实拆分；B1 结构性消除 |
| P2 | 配置去全局 + 单一对象 | 🟡 部分 | `UnifiedConfig` 拥有 providers；但 `LoadResult` 未退役、`extra=forbid` 未落、热路径仍全局 |
| P3 | Markdown 单一管线 | ✅ 主干落地 | `MarkdownStructure`/`BinarySplitter`/`ChunkedLLMJob` 主干干净；但 frontmatter/翻译路径有缺口 |
| P4 | 服务层 / 引擎 / 导出器收尾 | 🟡 部分 | `PaperService` 去 `typer.Exit` ✓；但 `FormatService` 违约、展示层未抽取 |

**结论**：V1 的"骨架"重构成功；"最后一公里"（配置收口、服务契约统一、展示层、数据正确性收尾）
未走完，且 V1 的纯架构视角 **漏看了** 切分-预算主干上的数据正确性裂缝。这正是本文的主题。

### 1.2 模块规模热点（当前 LOC，`wc -l` 实测）

```
core/paper_explain.py            881   ← 最大文件
core/md_heading_formatter.py     692   ← 子类未充分瘦身
services/paper_service.py        653   ← explain_paper ~193 LOC 上帝函数
core/md_body_formatter.py        473
utils/batch_exporter.py          471
core/paper_explain_pipeline.py   467
config/unified_config.py         437
core/binary_splitter.py          429
core/task_executor.py            415
services/format_service.py       407   ← 契约违约
services/text_file_translator.py 391
services/translation_service.py  375
cli/commands/format_cmd.py       358   ← 上帝命令
core/batch_processor.py          347   ← V1 后已非上帝类
```

### 1.3 值得肯定的设计（V1 之后仍成立）

- **执行引擎四协作者**：`TaskExecutor.try_run_with_config` 是单次执行的唯一所有者
  （限流 acquire → adapter 查找 → `RequestProcessor` → 流式收集 → `RequestMetadata.from_execution`）。
  `GlobalBatchProcessor` 仅剩协调（escalation step + 池 sizing + 闭包接线）。
- **`llm_engine` 单模块私有依赖不变量成立**：全代码库唯一 `import llm_engine` 在
  `utils/engine_facade.py`（`grep` 复核通过）。
- **并发原语隔离**：`BoundedRetryRunner`（调度+重试堆）、`GlobalRateLimiter`（per
  `(provider,model)` 令牌桶）、`ProviderAdapterCache` 各自独立。
- **可观测性**：`bind_context`、`ExecutionReport`、`AttemptRecord`（扁平非递归，B7 结构性消除）、
  `error_keywords` 单表、provenance 分层报告。
- **原子写 + 优雅中断**：`BaseCheckpoint.save` 用 `tmp+os.replace`；`BoundedRetryRunner`
  装 SIGINT 处理器，排空在飞、返回部分结果。
- **`AskService` 是干净范本**：0 `console.print`、0 `typer`、返回 dataclass。

---

## 2. 双重视角下的设计目标重述

ask-llm 的本质是一条 **LLM 数据处理管线**：把"非结构化输入（YAML 任务 / Markdown / 论文 /
Jupyter）"经 **切分 → 预算 → 并发调用 → 重试/回退 → 重组 → 持久化 → 导出** 转化为结构化产出。

| 视角 | 核心关切 | 失败时的代价 |
|------|----------|--------------|
| **数据处理专家** | 数据保真：切分不破坏语义边界、预算不溢出上下文、重组无损、checkpoint 反映真实进度、导出 schema 一致、可复现可审计 | 静默损坏用户内容（切断代码块、改写 frontmatter）、API 失败、进度丢失 |
| **Python 架构师** | 概念完整：单一所有者、显式注入、类型即防线、关注点正交、低认知复杂度 | 维护期改一处漏 N 处、重构被锁死、库化不可行、新 provider/命令接入成本高 |

V1 偏架构视角；本文补齐数据视角。**两视角的共同根因**：正确性与所有权都未被收敛到"类型 + 唯一实现"，
而是散落在多处并行路径与约定之中。

---

## 3. 数据处理视角深度分析（本轮重点）

### 3.1 数据管线总览

```
输入源
  ├─ YAML(batch)      → BatchConfigLoader → list[BatchTask]
  ├─ .md(trans/format)→ MarkdownStructure.parse → headings / fence / frontmatter
  ├─ 目录(paper)       → paper_explain_pipeline → jobs
  └─ .ipynb(trans)     → notebook_translator → cells
        │
        ▼
   切分 / 预算        BinarySplitter(TokenBudget)  ← fit 判定 + hard split
        │               chunk_balance.rebalance（翻译侧，独立实现）
        ▼
   执行引擎           batch/trans/paper: TaskExecutor + Scheduler + Fallback
                     format:        chunked_llm_job._run_units ← 第二条管线（绕开）
        │
        ▼
   结果模型           BatchResult(attempt_history: list[AttemptRecord])
        │
        ▼
   持久化 / 导出      checkpoint(原子) + BatchResultExporter / TranslationExporter
        │
        ▼
   诊断               ExecutionReport → diagnose
```

**转换边界（每个都是数据正确性栖息点）**：源→任务、文本→结构索引、结构→切分块、块→预算判定、
块→LLM 调用、流→(response,tokens,latency)、块→重组、结果→checkpoint、结果→导出投影。
下文逐一边界审查。

### 3.2 切分-预算-重组主干：正确性裂缝（最高 ROI）

这是数据处理视角的 **核心发现**：同一条主干上有 **四个互相放大** 的正确性缺口，V1 的 B2/B4 只各修一半。

#### 3.2.1 预算 fit 判定不施安全系数（B2 半修）
- `TokenBudget.fits()`（`core/binary_splitter.py:62-64`，**已复核**）：
  ```python
  def fits(self, text: str) -> bool:
      if not text.strip():
          return True
      return self.count(text) <= self.content_max_tokens   # 原始 cl100k 计数
  ```
- 安全系数 `APPROX_TOKEN_SAFETY_FACTOR = 0.85`（`core/constants.py:45`）**只** 在
  `TokenCounter.split_hard_by_max_tokens`（`token_counter.py:254-256`）里施加。
- **后果**：`BinarySplitter.split` 的快路径"整块已放得下 ⇒ 直接返回单块"走的是 `fits()`，
  **绕过** 0.85。DeepSeek/Qwen 的 cl100k 计数对 CJK **少计 30–50%**，0.85 头寸本身就偏薄；
  叠加 fit 不施加 ⇒ "整块刚好放得下"的中文块 **真实 token 数溢出上下文窗口**，provider 端报
  `context length` 错误。这是项目主力场景（中文翻译 / 中文论文解读）的 **高频静默失败源**。

#### 3.2.2 `prompt_overhead` 是死配置（机制存在、无人接线）
- `TokenBudget.prompt_overhead`（`binary_splitter.py:52`）、
  `content_max_tokens = max(1, max_tokens - prompt_overhead)`（`:54-57`）**存在且正确**。
- 但 **唯一调用方** `MarkdownTokenSplitter.__init__(prompt_overhead_tokens=0)`
  （`markdown_token_splitter.py:33,39`）默认 0；而三个构造点 **全部不传**：
  `md_body_formatter.py:153`、`services/text_file_translator.py:106`、
  `utils/notebook_translator.py:21`。`grep prompt_overhead` 在 `prompts/`、`config/`、
  `default_config.yml` 全空——无配置旋钮，也无人测量模板 token 数。
- **后果**：预算 **只算 content、不算 prompt 模板**。大模板（如 `prompts/paper/section-*.md`、
  翻译术语表）+ 近上下文窗口的模型 ⇒ "放得下"的块实际 **prompt+content 溢出**。V1 §4.4.4 标记的
  overflow 风险 **只关了一半**：API 存在，语义未通。

#### 3.2.3 frontmatter 在正文路径无保护（B4 在正文/frontmatter 侧未修）
- `BinarySplitter.split`（`binary_splitter.py:105-124`，**已复核**）调用
  `MarkdownStructure.parse(text)` 但 **只读** `structure.headings`；`frontmatter_range` 在整个
  splitter 中 **仅出现在 docstring**（`:78`），从未作为切分边界。
- 栅栏保护 **只在** `_split_long_paragraph → _split_paragraph_with_fences`
  （`binary_splitter.py:271,337-392`）触发。frontmatter 作为"某块的前缀"被原样送给正文 LLM。
- **后果**：`format --type body` 会 **让 LLM 改写你的 YAML frontmatter**（title/date/tags），
  静默破坏文档元数据。V1 P3.1 加了"frontmatter 内的 `#` 不当标题"的标题侧保护，但 **正文侧未覆盖**。

#### 3.2.4 翻译路径 `chunk_balance` 重写切分、无栅栏保护（B4 在翻译侧未修）
- `utils/chunk_balance.py`（164 LOC）的 `_split_by_token_budget`（`:15-56`）**重新实现** 了
  段落切分 + display-math 合并 + 硬切分，与 `BinarySplitter._split_by_paragraphs_binary` /
  `_split_long_paragraph` 高度重叠；且 **完全不引入** `MarkdownStructure`
  （`grep MarkdownStructure|fence` 在该文件 **零命中**，**已复核**）。
- **后果**：翻译路径上，超大块内的 fenced 代码块仍可被 **从栅栏中间硬切**——正是 V1 B4 的同类 bug，
  只在 splitter 主路径修了，**翻译侧 rebalance 路径未修**。

#### 3.2.5 resume 走有损 legacy 重组
- 正文快路径用 position-aware 重组 `_join_chunks_position_aware`（`md_body_formatter.py:302-348`，
  消费 `start_pos/end_pos`、恢复原始块间空白）——**正确**。
- 但 `resume_from_checkpoint`（`md_body_formatter.py:449`，**已复核**）用 **legacy**
  `cls._join_chunks(final_chunks)`（`:351-372`，强制 `\n\n`），**不走** position-aware。
- **后果**：同一输入，"断点续跑"产出与"全新跑"产出 **结构不同**（连续列表项 / 紧凑表格被插入空行）。
  数据处理视角的 **可复现性** 被破坏。

> **四裂缝的共同根因**：预算与切分的正确性 **散落在** `TokenBudget.fits` / `split_hard_by_max_tokens` /
> `chunk_balance._split_by_token_budget` / resume 路径 **四个地方**，各自独立判定。V1 把"算法单份"
> 做到了 `BinarySplitter`，但 **正确性策略（安全系数、prompt 开销、栅栏/frontmatter 保护、无损重组）
> 没有单点收敛**。R0 的目标就是把它们收敛到 `TokenBudget` + `BinarySplitter` + `Reassembler` 的类型契约里。

### 3.3 token 计数近似与上下文预算

- `ENCODING_MAP`（`token_counter.py:53-65`，**已复核**）：`deepseek*` / `qwen*` 全映射到
  **`cl100k_base`**（GPT-3.5/4 分词器）。真实 DeepSeek/Qwen BPE 未加载。`gpt-4o` 正确用 `o200k_base`。
- 近似告警 `_warn_approximate_once`（`token_counter.py:100`）每模型一次 WARNING，含 85% 头寸说明——
  **可观测性到位**，但 **头寸本身不可审计**（固定常量，不随内容 CJK 密度调整）。
- 三条静默 fallback（tiktoken 缺失/`get_encoding` 返 None）落到 `count_words`（空格切分）；CJK 无空格
  ⇒ **二次少计**。
- `truncate_to_tokens` 无 tiktoken 时用 `max_tokens*4` 字符启发，与 `count_tokens` 的"词数"回退
  **不一致**——同一类"无 tiktoken 模型"两种口径。
- **热路径全局依赖**：`_default_encoding`（`token_counter.py:26-31`）在每次 `count_tokens` 链路上
  调 `get_config_or_none()` 读 `token.default_encoding`。无 `set_config` 则用 `cl100k_base` 兜底——
  库化可用，但 **每次计数触碰全局**，线程安全依赖 `set_config` 先于 worker 跑完。

### 3.4 checkpoint / resume 数据完整性

- **优雅路径正确**：`BoundedRetryRunner` 装 SIGINT（仅主线程，`concurrent.py:163-166`），首次中断
  停调度、排空在飞、返回部分结果 + `metrics.interrupted=True`；`command_runner` 仅 merge 成功结果、
  原子保存；干净全成功才 unlink。
- **缺口 1（数据丢失窗口）**：checkpoint **只在运行结束后保存一次**
  （`command_runner.py:118-120`）。无增量/周期落盘。SIGKILL / OOM / 二次中断落在 `save` 之前 ⇒
  **自上次 checkpoint 以来的全部进度丢失**。千任务慢 provider 批的真实数据丢失窗口。
- **缺口 2（两套 resume 机制）**：`batch`/`trans` 走 `run_with_checkpoint`（JSON checkpoint，
  `command_runner.py:43`）；`paper` 走 `_apply_resume_or_force`（**文件存在性** 判定，
  `paper_service.py:501`）。两套 UX、两套可审计性、两套部分失败语义——数据处理视角的 **一致性** 缺失。
- **缺口 3（off-main 静默）**：SIGINT 处理器被 `main_thread` 守卫。若 paper 在非主线程跑 runner，
  Ctrl-C 处理 **静默 no-op**，`interrupted` 永不为真。

### 3.5 导出保真

- `BatchResult.project()`（`batch_models.py:150`）**单一投影**，`BatchResultExporter` 消费
  （`batch_exporter.py:351`）。✓
- **`TranslationExporter` 不用 `project()`**：`_export_json` 手拼 `chunk_data` dict
  （`translation_exporter.py:193-216`）。两套投影，schema 漂移风险。
- **流式不一致**：`batch_exporter` 与 `translation_exporter._export_json` 用 `iterencode` 流式；
  `translation_exporter._export_text` / `_export_markdown` 仍 `write_text` **全量物化**
  （大翻译内存无界）。
- `_detect_format` 单表（`export_formats.py:12`）✓；`unwrap_translation_payload` 已移入
  `core/response_parser.py` ✓。

### 3.6 可观测性 / 可复现性 / provenance

- provenance 落地（`merge.record_leaves` + `env.py` 记 `env:<VAR>`），`config show --debug-config`
  分层报告 ✓。但 `_convert_providers_format` 在 provenance 记录 **之后** 改写 providers 形状
  （`base_url`→`api_base`），故派生字段的来源标签可能与最终 key 不对应（UI 已诚实注明）。
- **determinism 风险**：全局可变配置 + token 热路径全局 ⇒ 同输入不同进程/线程可能拿到不同
  `default_encoding`，切分结果不可复现。

---

## 4. 架构视角深度分析

### 4.1 执行引擎存在第二条管线（单一所有者裂缝）

V1 争取的"执行引擎单一所有者"在 **格式化路径** 上未兑现：

- `format` → `ChunkedLLMJob._run_units`（`chunked_llm_job.py:100`）→
  `run_bounded_with_retries`（`concurrent.py:265`）。
- 这条路径 **绕开**：`TaskExecutor`（无统一单次执行）、回退链（无 fallback）、`ProgressPresenter`
  （自建 `rich.Progress`）、`GlobalRateLimiter.acquire`（无限流 acquire）、
  `RequestMetadata.from_execution`（无统一 metadata）。
- 即 `run_bounded_with_retries` 与 `TaskExecutor + BoundedRetryRunner` 是 **两套执行语义**。
- **后果**：任何执行引擎的改进（per-provider 池、增量 checkpoint、统一 metadata、限流退避）
  **只惠及 batch/trans/paper，不惠及 format**。format 的限流/重试/进度是另一套，会独立漂移。

> 这是 V1 "单一所有者原则" 的 **复发**：V1 批评 batch/trans/paper 各复制一遍管线；重构后变成
> "batch/trans/paper 走新引擎，format 走旧引擎"。R1 的核心目标：**让 format 也走 TaskExecutor**，
> `run_bounded_with_retries` 退化为 scheduler 的一个薄入口或被删除。

### 4.2 配置：单一对象的未竟之路

- **单一源 ✓**：`UnifiedConfig`（`unified_config.py:410-437`）拥有 providers/default_provider/
  default_model；`AppConfig`（`core/models.py:161-182`）降为派生视图（`_app_config_from_unified`，
  共享 providers dict 引用）；单次 `model_validate`。
- **`LoadResult` 未退役**（`loader.py:30-45`）：仍是 service-locator 的载体；`get_config()` 返回
  `LoadResult`，调用方写 `lr.unified_config.token.default_encoding`（`token_counter.py:30`）——每调用方
  付一次间接税。
- **双叉身份未完全消除**：`AppConfig` 与 `UnifiedConfig` 仍是两个 Pydantic 类型、两套校验器
  （`AppConfig.validate_providers` 拒空，`UnifiedConfig` 无此守卫）；`ConfigManager.__init__` 仍容忍
  `unified_config: UnifiedConfig | None`（`manager.py:14-30`）——可从裸 `AppConfig` 构造，绕开
  `UnifiedConfig` 不变量。
- **`extra` 策略不一致且顶层仍 `ignore`**（**已复核**）：`UnifiedConfig` 无 `model_config`
  ⇒ 默认 `extra="ignore"`；`TranslationConfig`/`BatchConfig` 显式 `ignore`；仅 `RateLimitConfig`
  `allow`。**`extra="forbid"` 从未落地** ⇒ YAML 顶层/字段笔误（`providrs:`）**静默丢弃**，违反
  "Fail Fast"。
- **`providers.yml` 解析两遍**：`providers_catalog._load_providers_yml`（取 runtime 字段）与
  `pricing._candidate_providers_yml_paths`（取 pricing）各 `yaml.safe_load` + `resolve_env_vars`，
  候选路径逻辑重复。每次 CLI 调用还多读一次盘（`cli_session.load_pricing_with_hint`）。
- **类型强转按名字子串**（`env.py:55-78`）：`_parse_env_value` 用 `"threads" in last_key` 等判定
  int/float/bool。新数值字段名不含这些子串 ⇒ **静默存 str**。
- **env 双映射未除根**（`env.py:32,34-37`）：`ASK_LLM_TRANSLATION_THREADS` 与
  `ASK_LLM_TRANSLATION_MAX_CONCURRENT_API_CALLS` 仍都映射到同一 key；`TranslationConfig` 仍同时持有
  `threads` 与 `max_concurrent_api_calls` 两字段，靠 `before` validator 同步（`unified_config.py:109-125`）。
  P2.7 只加了冲突告警，未去重。
- **`_convert_providers_format` 仍在 `loader.py`**（`:225-313`，~88 LOC）：V1 §7 明确要求整体删除，
  未做。形态归一化（`base_url`→`api_base`、models list 归一）与 `_load_providers_yml` 各写一遍。
- **潜在 SecretStr bug**：`ConfigManager.get_provider_config`（`manager.py:79-82`）走
  `base.model_dump()` 再 `ProviderConfig.model_validate`；Pydantic v2 `SecretStr` 默认 dump 成
  `**********`，若无 api_key override，重新校验可能失败。需用 `model_dump(mode="python")`。
- **死代码**：`get_config()`（严格版，`context.py:14-26`）**src 内零调用**（`grep` 复核），仅
  `get_config_or_none` 在用。

### 4.3 服务契约不统一 + 展示层缺失

| 服务 | 结构化结果 | typer | `console.print` | 评价 |
|------|-----------|-------|-----------------|------|
| `AskService` | `AskResult`/`AskDryRunInfo` | 0 | **0** | 干净范本 |
| `PaperService` | `PaperSessionResult` | 0 | 19 | 状态泄漏已修（P4.2 ✓）；展示仍内嵌 |
| `BatchService` | `BatchRunResult`+`BatchExportResult` | 0 | 42 | 返回数据 **同时** 内嵌 print/export |
| `TranslationService` | `TranslationSessionResult` | 0 | 12 | 同上 |
| `TextFileTranslator` | `TranslationJobResult` | 0 | 23 | 展示重 |
| `FormatService` | **无** | 0 | 24 | **契约违约**：`run_sequential_format`/`run_parallel_format`/`resume_from_checkpoint` 全 `-> None`（**已复核**） |

- **`cli/presentation.py` 不存在**（`find` 复核）。展示仍在 service 内：`BatchService.print_statistics`
  （`batch_service.py:366`）、`PaperService._print_usage`（`:612`）、`TranslationService._print_*`
  （`:289,314`）、`FormatService._handle_outcome`（`:116`）。共 128 处 `console.print`（`AskService` 0）。
  ⇒ 服务 **无法 headless 复用**（测试/notebook/库），`--quiet` 仅因 `console` 自身 quiet-aware 而工作。
- **上帝函数**：`PaperService.explain_paper`（`paper_service.py:135-326`，**~193 LOC**）混 输入派发 /
  pipeline 加载 / job 构建 / dry-run / resume / prompt 装配 / 模型选择 / max-token 解析 / 回退链 /
  BatchTask 构造 / 并发计算 / 批跑 / 失败处理 / 结果写盘 / 统计报告 / 用量打印。
- **`run_batch_from_config`（`batch_service.py:142-319`，~181 LOC）是模块级自由函数**，不在
  `BatchService.run` 上。`BatchService` 在 **跑完后** 才构造，只为 print+export——**双类混淆**。
- **CLI 异常样板未统一**：`cli_errors` 上下文管理器存在（`cli/errors.py:21`），但仅少数命令包它；
  多数命令仍手写 `except FileNotFoundError/ValueError/RuntimeError/KeyboardInterrupt/Exception →
  raise typer.Exit(1)` 级联，末端才调 `raise_unexpected_cli_error`（~120 LOC 重复，**已复核**）。
- **`bootstrap_command` 被 batch/format 跳过**：`batch.py` 内联手抄 `load_cli_session`+
  `load_pricing_with_hint`；`format_cmd.py:243-263` **整段手抄** `ConfigLoader.load + set_config +
  ConfigManager + set_provider + apply_overrides + get_default_model`。与 `bootstrap_command` 必然漂移。

### 4.4 耦合与类型防线

- **`getattr` 探取 processor 私有**（**已复核**）：`text_file_translator.py:253`
  `getattr(processor, "_auth_error_logged", False)`、`:266` `getattr(processor, "last_metrics", None)`；
  `command_runner.py:125` `getattr(processor, "last_metrics", ...)` 读 `metrics.interrupted`。
  无 `Protocol`/抽象方法，processor 改名 ⇒ 运行时崩，非类型检查期暴露。违反"类型即防线"。
- **跨模块私有方法访问**：`TaskExecutor._run_translation_chunk` 调
  `processor._format_prompt(task.content, task.prompt)`（`task_executor.py:210`）。下划线说私有，执行器
  伸手。应提升为 `RequestProcessor` 公共方法或走 `processor.process(...)`（内部已 format）。
- **状态一半枚举一半字符串**（**已复核**）：`paper_service.py:283`、`command_runner.py:116`、
  `batch_exporter.py:222` 用 `TaskStatus` 枚举 ✓；但 `text_file_translator.py:246,247,296,297,343,348`
  **6 处** `r.status.value == "failed"/"success"` 字符串比较。枚举值一改名即静默错分。
  `PaperSessionResult.status: str`（`paper_service.py:91`）值仅在注释里文档化，无枚举约束 ⇒ 拼写错误
  静默错分（`"sucess"`）。
- **服务实例可变累加器**：`TranslationService._batch_results`（`:89`）、`PaperService._last_results`
  （`:281`）在 run 中填充。同一实例跑两次 ⇒ `export_report` **静默合并两次结果**。库/notebook 复用即中招。
- **`paper` 无 `--retries` 旗标**（`batch`/`trans` 有 `-r/--retries`），重试只能走 config——契约不一致。
- **`_validate_models` 用裸 Rich markup**（`batch_service.py:75-130` `[red]✗[/red]`）绕过
  `console.print_error`，破坏 quiet/log-level 映射。

### 4.5 死代码 / shim / 文档漂移

- `core/batch.py`（26 行）纯 re-export shim，自身 docstring 写"新代码应直接导入"，仍被 3 处依赖
  （`command_runner.py:23`、`global_batch_runner.py:6`、`batch_checkpoint.py:14`）。
- `BoundedRetryRunner.run`（`concurrent.py:237-262`）是 `run_with_metrics` 的薄包装且丢 metrics；
  生产路径只走 `run_bounded_with_retries`；测试路径 1 处。一个操作三入口（`run`/`run_with_metrics`/
  `run_bounded_with_retries`）。
- `retry_policy.ProviderRetryRegistry`（`:66-83`）定义了 per-provider override 机制但 **从未接线**
  （runner 收扁平 `is_retryable_error` callable）。
- **三层统计委托**：`GlobalBatchProcessor.calculate_statistics`（`:342`）→
  `calculate_statistics_by_model`（`:68`）→ `BatchStatistics.from_results`（`batch_models.py:197`），
  单一生产调用方（`batch_service.py:302`）。外加 `BatchService._export_single`（`:504-526`）
  **手算合并** `BatchStatistics`（逐字段 sum），平行求和路径，可与规范聚合器漂移。
- 格式化侧：`_load_prompt_from_file` 逐字节重复（`md_heading_formatter.py:626-645` vs
  `chunked_llm_job.py:86-98`，override-by-copy 死重）；第 4 份 `HEADING_PATTERN` 局部 re.compile
  （`md_heading_formatter.py:584`）；`_CONTEXT_BATCH_INSTRUCTION_FALLBACK` 与 prompts 文件双真源；
  `_BatchResult`/`_ChunkResult` 同构双 dataclass；两个 `resume_from_checkpoint` 各 ~100 LOC 同形。
- **per-(provider,model) 池 sizing 仍错**：`_effective_max_workers`（`batch_processor.py:117-129`）取
  **全局最小** burst；混合 provider（A burst=2、B burst=20）⇒ 整池压到 2，B 被不必要串行化。限流器
  按 `(provider,model)` 分桶正确，但 **池被全局最小 cap**——架构正确性依赖一个错误前提。V1 已标，
  **未修**。
- **文档漂移**：`AGENTS.md` 仍列已重命名/移动模块（`provider_specs.py`→`model_limits.py`、
  `provider_router.py`→`fallback_chain.py`、`config/paper_explain_pipeline.py`→`core/`）；
  `docs/implementation_status.md` 是 2024-12-24 的死文档（引用已删的 `ask_llm.py`、`utils/config_checker.py`）；
  `docs/REFACTOR_PLAN.md` V1 已判过时；`src/ask_llm/config/__pycache__/paper_explain_pipeline.cpython-313.pyc`
  是 P2.6 移动前的陈旧字节码。

---

## 5. 当前问题清单（已复核）

| # | 视角 | 严重度 | 位置 | 问题 | 现状（对照 V1） |
|---|------|--------|------|------|-----------------|
| **D1** | 数据 | 🔴 高 | `binary_splitter.py:62-64` × `token_counter.py:254` | `fits()` 不施安全系数；CJK 快路径溢出上下文 | V1 B2 半修（仅 split_hard 施 0.85） |
| **D2** | 数据 | 🔴 高 | `markdown_token_splitter.py:33,39` + 3 构造点 | `prompt_overhead` 死配置，预算只算 content | V1 §4.4.4 标记，API 在、未接线 |
| **D3** | 数据 | 🟠 中高 | `binary_splitter.py:105-124` | frontmatter 在正文路径无保护，被 LLM 改写 | V1 B4 仅修标题侧 |
| **D4** | 数据 | 🟠 中高 | `chunk_balance.py:15-56` | 翻译 rebalance 重写切分、无栅栏保护 | V1 B4 翻译侧 **未修** |
| **D5** | 数据 | 🟡 中 | `md_body_formatter.py:449` | resume 走 legacy `\n\n` 重组，与全新跑产出不一致 | V1 P3.4 未覆盖 resume |
| **D6** | 数据 | 🟡 中 | `command_runner.py:118-120` | checkpoint 仅运行结束保存；硬杀丢进度 | V1 B5 半修（优雅中断 ✓，周期落盘 ✗） |
| **D7** | 数据 | 🟡 中 | `translation_exporter.py:109,163,193` | text/md 非流式 + 不用 `project()` | V1 P4.7 部分修 |
| **A1** | 架构 | 🔴 高 | `chunked_llm_job.py:100`→`concurrent.py:265` | 第二条执行管线，format 绕开 TaskExecutor/限流/回退 | V1 单一所有者 **复发** |
| **A2** | 架构 | 🟠 中高 | `context.py` + 13 处 `get_config_or_none` | 配置全局状态仍在热路径（token_counter） | V1 P2 未竟 |
| **A3** | 架构 | 🟠 中高 | `format_service.py` 全 `-> None` | `FormatService` 违反服务契约 | V1 未覆盖 |
| **A4** | 架构 | 🟠 中高 | 6 服务 128 处 `console.print` | 展示层未抽取，服务不可 headless | V1 P4.2 follow-up 未做 |
| **A5** | 架构 | 🟡 中 | `unified_config.py:410` 等 | `extra="forbid"` 未落，YAML 笔误静默丢 | V1 P2 未竟，违反 Fail Fast |
| **A6** | 架构 | 🟡 中 | `batch_processor.py:117-129` | 全局最小 burst cap，混合 provider 串行化 | V1 已标未修 |
| **A7** | 耦合 | 🟡 中 | `text_file_translator.py:253,266`；`command_runner.py:125` | `getattr` 探取 processor 私有，无类型契约 | V1 未覆盖 |
| **A8** | 耦合 | 🟢 低中 | `text_file_translator.py:246-348` ×6 | 状态字符串比较 vs 枚举 | V1 未覆盖 |
| **A9** | 卫生 | 🟢 低 | 多处 | 死代码/shim（`core/batch.py`、`BoundedRetryRunner.run`、`ProviderRetryRegistry`、三层统计委托、重复 `_load_prompt_from_file`/第 4 份 pattern、`get_config()` 严格版零调用） | V1 部分标，未清 |
| **A10** | 卫生 | 🟢 低 | `AGENTS.md`/`implementation_status.md`/`REFACTOR_PLAN.md`/陈旧 pyc | 文档漂移 | V1 标 REFACTOR_PLAN 过时，余未理 |

---

## 6. 风险评估（排序）

| 级别 | 风险 | 触发 | 影响 |
|------|------|------|------|
| 🔴 高 | D1+D2 预算溢出 | 中文 trans/paper/format（主力场景） | 高频 `context length` 失败、成本、口碑 |
| 🔴 高 | A1 第二条执行管线 | format 命令、引擎演进 | 改一处漏 format；限流/重试/进度独立漂移 |
| 🟠 中高 | D3+D4 内容损坏 | 正文 format 含 frontmatter；翻译含代码块 | 静默破坏用户文档/代码 |
| 🟠 中高 | A2 配置全局在热路径 | 库化 / 并发 / 测试串扰 | 不可嵌入、切分不可复现、重构锁死 |
| 🟠 中高 | A3+A4 服务契约/展示 | 维护期、headless 复用 | 改一处漏 N 处、服务不可单测/库化 |
| 🟡 中 | D5 resume 不可复现 | 断点续跑 | 产出结构漂移、信任受损 |
| 🟡 中 | D6 硬杀丢进度 | 大批量 Ctrl-C/OOM | 进度丢失 |
| 🟡 中 | A5 笔误静默丢 | YAML 拼错 | 配置意图丢失、Fail Fast 违背 |
| 🟡 中 | A6 混合 provider 串行 | 多 provider 混批 | 吞吐下降 |
| 🟢 低 | A7/A8/A9/A10 | 重构期 / 新人接入 | 维护摩擦、运行时崩 |

---

## 7. 重构方案：统一主导架构

### 7.1 统一设计思想（一句话）

> **一条数据管线、一个执行引擎、一个配置对象、一种服务契约。**

四句话分别对应 R0–R3 的主导目标；R4 收尾清理。每条都把"正确性 / 所有权"从 **约定** 提升到
**类型与唯一实现**（奥卡姆剃刀 + Fail Fast + 类型即防线）。

### 7.2 目标分层

```
CLI (薄)                  cli/commands/*        参数解析 + 退出码 + 渲染
                          cli/presentation.py   ★新增：纯展示，消费 *SessionResult
                          cli_errors            统一异常→退出码（全命令包）
   ▼
UseCase Services          services/*            每 service .run()→*SessionResult；0 print / 0 typer
（无副作用）               FormatService 补齐 run()→FormatSessionResult（★修 A3/A4）
   ▼
Execution Engine          core/engine/          ★单一所有者（修 A1）
                          Scheduler(BoundedRetryRunner) + per-(provider,model) Semaphore（修 A6）
                          TaskExecutor（单次执行：限流+adapter+stream+metadata）
                          FallbackPolicy + EscalationPolicy（retry×fallback 共享预算）
                          ProgressPresenter（per-worker）
                          周期 + SIGINT 落盘（修 D6）
                          ← chunked_llm_job._run_units 不再直接调 run_bounded_with_retries，
                             改经 TaskExecutor
   ▼
Data Pipeline             core/markdown/        ★正确性单点收敛（修 D1–D5）
                          MarkdownStructure（fence+frontmatter+heading+paragraph 单次解析）
                          BudgetedSplitter（TokenBudget.fits 施安全系数 + prompt_overhead 实测接线）
                          chunk_balance 复用 BudgetedSplitter + MarkdownStructure（删并行实现）
                          Reassembler（position-aware 唯一重组，resume 也走它）
   ▼
Domain Models            core/batch_models.py   BatchResult.project() 唯一投影；AttemptRecord
                         status 全枚举（修 A8）
   ▼
Providers                core/engine/engine_facade.py（llm_engine 唯一私有依赖，不变量保持）
                         ProviderAdapterCache.get(cfg: ProviderConfig)（删 dict 分支）
   ▼
Config (注入, 无全局)     config/                ★单一对象（修 A2/A5）
                         UnifiedConfig（extra="forbid"）；LoadResult 退役；providers.yml 单解析
                         ProviderCatalog（runtime + pricing 共享）；token_counter 注入 encoding
```

### 7.3 关键类型重塑（让缺陷结构性消失）

```python
# core/markdown/budget.py —— 让 D1/D2 结构性消失
class TokenBudget:
    model: str
    max_tokens: int
    prompt_overhead: int            # 由调用方实测模板 token 数传入（不再恒 0）
    def content_max_tokens(self) -> int:
        return max(1, self.max_tokens - self.prompt_overhead)
    def fits(self, text: str) -> bool:
        if not text.strip():
            return True
        # 安全系数在 fit 判定也施加 —— 不再只留给 split_hard
        cap = self._safety_adjusted_cap()
        return TokenCounter.count_tokens(text, self.model) <= cap

# core/markdown/splitter.py —— 让 D3/D4 结构性消失
class BudgetedSplitter:
    def split(self, text: str) -> list[TextChunk]:
        struct = MarkdownStructure.parse(text)          # fence + frontmatter + heading
        protected = struct.protected_ranges()           # ★frontmatter + fence 统一保护
        ...   # 单一算法；chunk_balance 复用此类，删并行实现

# core/markdown/reassemble.py —— 让 D5 结构性消失
class Reassembler:
    @staticmethod
    def join(text, spans, replacements, hard_split_types): ...   # resume 与全新跑共用

# services/format_service.py —— 让 A3 结构性消失
class FormatService:
    def run(self, files, fmt, opts) -> FormatSessionResult: ...  # 唯一入口，返回结构化结果

# config/unified_config.py —— 让 A5 结构性消失
class UnifiedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")           # YAML 笔误即报错
```

---

## 8. 重构与改进计划 R0–R4

> 原则：每阶段独立发版、独立回滚；先数据正确性止血（R0，零架构破坏、最高 ROI），再结构统一
> （R1–R3），最后清理（R4）。每阶段测试先行。

### R0 —— 数据正确性止血（1–2 天，零行为破坏性）

> ✅ **D1–D4 已落地（v2.19.1，2026-07-19）**。安全系数 / prompt 开销 / frontmatter / 翻译栅栏保护
> 收敛到 `TokenBudget` + `BinarySplitter` 单一所有者。详见 `CHANGELOG.md` 2.19.1。
> ✅ **D5（resume 无损重组）已落地（v2.19.2，2026-07-19）**：`FormatCheckpoint` v2 存原文 + 每块 span，
> resume 改用 position-aware join（v1 回退 legacy）。详见 `CHANGELOG.md` 2.19.2。

目标：不重构架构，先把切分-预算-重组主干上的数据裂缝补齐。每条均可独立 PR。

| 任务 | 文件 | 动作 |
|------|------|------|
| **D1 fit 施安全系数** | `binary_splitter.py:62-64` | `TokenBudget.fits()` 对 `_APPROXIMATE_PREFIXES` 模型施加 `APPROX_TOKEN_SAFETY_FACTOR`（与 `split_hard_by_max_tokens` 同一常量），消除快路径溢出 |
| **D2 prompt_overhead 接线** | `md_body_formatter.py:153`、`text_file_translator.py:106`、`notebook_translator.py:21`、`markdown_token_splitter.py:33` | 用 `TokenCounter` 实测模板 token 数，作为 `prompt_overhead_tokens` 传入；`UnifiedConfig.token` 加 `prompt_overhead_tokens`（或自动测量） |
| **D3 frontmatter 保护** | `binary_splitter.py:105` | `BudgetedSplitter.split` 消费 `structure.frontmatter_range`，frontmatter 作为不可切前缀剥离（不送正文 LLM，或显式 opt-in） |
| **D4 翻译路径栅栏** | `chunk_balance.py:15-56` | `_split_by_token_budget` 改为复用 `BudgetedSplitter` + `MarkdownStructure`，删并行切分实现 |
| **D5 resume 无损重组** | `md_body_formatter.py:449` | `resume_from_checkpoint` 改用 `Reassembler`（position-aware），删 legacy `_join_chunks` 或仅留显式回退 |
| **D2 配套：安全系数可配置** | `constants.py:45`、`unified_config.py` `TokenConfig` | `APPROX_TOKEN_SAFETY_FACTOR` 提为 `UnifiedConfig.token.approx_safety_factor`（默认更保守，如 0.7） |

验收：`pytest` 全绿 + 5 个回归测试（每条一）：CJK 块不溢出、prompt+content 不溢出、frontmatter 不被
改写、翻译路径代码块不被切断、resume 与全新跑产出 byte-identical（除 LLM 随机性）。

### R1 —— 执行引擎真正统一（核心，3–5 天）

目标：消灭 A1（第二条管线）、A6（池 sizing）、D6（增量 checkpoint）；清死代码。

1. **format 走 TaskExecutor**：`ChunkedLLMJob._run_units` 不再直接调 `run_bounded_with_retries`；
   改构造 `BatchTask`（`task_kind=markdown_format`）经 `GlobalBatchProcessor.process_global_tasks`，
   复用 `TaskExecutor`/`FallbackPolicy`/`ProgressPresenter`/限流 acquire。`run_bounded_with_retries`
   退化为 scheduler 薄入口或删除。
2. **per-(provider,model) 池 sizing**：`_effective_max_workers` 全局最小 cap → per-bucket
   `Semaphore`；全局池可 `sum(bursts)`。混合 provider 不再互相串行化。
3. **checkpoint 增量化**：runner 完成回调每 K 个完成 + SIGINT 落盘（不只运行结束）。缩小硬杀丢失窗口。
4. **清死代码**：删 `core/batch.py` shim（迁移 3 处导入）、`BoundedRetryRunner.run` 薄包装、
   `_is_transient_error` shim（改 `DEFAULT_RETRY_POLICY.as_callable()`）；删或接线
   `ProviderRetryRegistry`；三层统计委托压平为 `BatchStatistics.from_results`（删
   `calculate_statistics`/`calculate_statistics_by_model`/`_export_single` 手算合并）。
5. **`TaskExecutor._run_paper_explain`/`_run_translation_chunk` 去重**：抽 `_run_attempt_skeleton`
   共享骨架（input 估算→progress→ctx→stream→metadata）。
6. **类型化 processor 契约**：`getattr(processor, "_auth_error_logged"/"last_metrics")` 提升为
   `core/protocols.py` 的 `Protocol` 方法/属性（修 A7）。

验收：批基准调用数回归（`total_calls ≤ tasks×(retries+1)`）；混合 provider 不串行化测试；
format 与 batch 走同一引擎的 grep 不变量；增量 checkpoint 中断回归。

### R2 —— 配置收尾：单一对象 + 注入（3–4 天）

目标：消灭 A2（全局热路径）、A5（extra=forbid）；完成 V1 P2 未竟项。

1. **`UnifiedConfig` 为唯一句柄**：`LoadResult` 退役；`get_config()` 返回 `UnifiedConfig`（或彻底
   注入）。`ConfigManager.unified_config` 去 `None`。删 `get_config()` 严格版（零调用）。
2. **`extra="forbid"`**：`UnifiedConfig` 及各子配置顶层 `forbid`（YAML 笔误即报错）。提供
   `ask-llm config migrate` 兼容旧字段。
3. **`providers.yml` 单解析**：新建 `ProviderCatalog`（runtime + pricing 共享模型），
   `providers_catalog` 与 `pricing` 共用，删重复候选路径与二次盘读。
4. **删 `_convert_providers_format`**：`ProviderConfig` 直接接受用户形态（`base_url` alias、
   list-of-dicts models via `field_validator`）。
5. **field-type 强转**：`_parse_env_value` 按 `UnifiedConfig` 字段注解强转，删名字子串启发。
6. **env 双映射去根**：`threads` 与 `MAX_CONCURRENT_API_CALLS` 合并为单一规范 key，删
   `_sync_threads_and_max_concurrent_api_calls`。
7. **token_counter 注入**：`TokenCounter` 构造收 `TokenConfig`（或 encoding），热路径不再触碰全局。
8. **修 SecretStr dump bug**：`ConfigManager.get_provider_config` 用 `model_dump(mode="python")`。

验收：无 `get_config()` 调用的 service 单测全绿；`extra=forbid` 下 typo 即报错测试；
`grep get_config_or_none src/` 调用点大幅下降（理想为 0）；provenance 报告含派生字段正确来源。

### R3 —— 服务契约统一 + 展示层（2–3 天）

目标：消灭 A3（FormatService 违约）、A4（展示未抽取）、A8（状态字符串）；完成 V1 P4.2 follow-up。

1. **`FormatService.run()→FormatSessionResult`**：包装 `run_sequential_format`/`run_parallel_format`，
   CLI 像 paper 一样渲染结果。新增 `FormatSessionResult`。
2. **`cli/presentation.py`**：渲染器消费 `*SessionResult` 发 `console.print_*`；剥离六服务的
   `_print_*`/`_handle_outcome`/`print_statistics`。service 全部 0 print。
3. **全命令包 `cli_errors`**：删 ~120 LOC 手写 except 级联。
4. **全命令走 `bootstrap_command`**：`batch`/`format` 接入（扩展 helper 覆盖 batch 的
   pricing-without-provider-models 场景）。
5. **status 全枚举**：`PaperSessionResult.status` 与 `text_file_translator` 的 6 处字符串比较改
   `TaskStatus`（或新 `SessionStatus`）枚举。
6. **去可变实例状态**：`_batch_results`/`_last_results` 折进返回的 `*SessionResult`，service 无实例累加。
7. **统一 resume**：paper 接入 `run_with_checkpoint`（或显式文档化文件存在性 resume 的理由）。
8. **拆上帝函数**：`PaperService.explain_paper` 拆 build/run/write/report；`run_batch_from_config`
   收为 `BatchService.run`。
9. **`paper` 加 `--retries`**，与 batch/trans 一致。

验收：service 单测无需捕获 stdout；`grep console.print src/ask_llm/services/` 仅 `AskService` 级
（理想 0）；status 字符串比较清零。

### R4 —— 清理与文档（1–2 天）

目标：消灭 A9/A10。

1. **死代码清零**：格式化侧重复 `_load_prompt_from_file`、第 4 份 `HEADING_PATTERN`、
   `_CONTEXT_BATCH_INSTRUCTION_FALLBACK` 双真源、`_BatchResult`/`_ChunkResult` 合并为泛型
   `UnitResult`、两个 `resume_from_checkpoint` 抽进 `ChunkedLLMJob` 泛型、`create_engine_adapter`/
   `EngineConfigView` 单一导出路径。
2. **命名/归属**：`ProviderManager`（core/）与 provider 族（utils/）归位；考虑 `core/engine/` 包。
3. **导出统一**：`TranslationExporter` 用 `BatchResult.project()`；text/markdown 流式。
4. **文档同步**：
   - 删/归档 `docs/implementation_status.md`（2024 死文档）、`docs/REFACTOR_PLAN.md`（V1 判过时）；
   - 刷新 `AGENTS.md`（模块名/路径全部对齐 v2.19+）、`README.md` internals 段；
   - 清 `src/ask_llm/config/__pycache__/paper_explain_pipeline.cpython-313.pyc`；
   - 本文（V2）作为当前权威；V1 标"历史保留"。

验收：`ruff`/`mypy`/`pytest` 全绿；`grep` 死代码不变量；`AGENTS.md` 模块名与 `src/` 实际一致。

---

## 9. 预期收益

| 维度 | 现状（v2.19.0） | 目标（R0–R4 后） |
|------|-----------------|------------------|
| 切分-预算正确性 | fit 不施安全系数、prompt_overhead 死、frontmatter/翻译路径无保护 | 类型层单点收敛，D1–D5 结构性消失 |
| 执行引擎 | 2 条管线（batch/trans/paper vs format） | 1 条；format 走 TaskExecutor |
| 配置全局读取点 | 13 处 `get_config_or_none`（含热路径） | 0（注入）；`extra=forbid` |
| 服务 `console.print` | 128 处（AskService 0） | 0（全进 presentation） |
| 服务契约 | FormatService 违约 | 5 服务统一 `.run()→SessionResult` |
| 死代码/shim | ~10 处 | 0 |
| 文档漂移 | AGENTS/implementation_status/REFACTOR_PLAN 过时 | 全部对齐 v2.19+ |
| 混合 provider 池 | 全局最小 cap | per-(provider,model) Semaphore |
| checkpoint | 仅运行结束保存 | 周期 + SIGINT 增量 |

粗估净删 ~800–1200 行；5 个数据正确性缺陷（D1–D5）结构性消失；服务可 headless 单测/库化；
引擎升级从"多文件 hunt + format 漏修"变"改一处"。

---

## 10. 兼容、迁移与回滚

- **配置兼容**：`extra="forbid"` 是破坏性变更。R2 发版前提供 `ask-llm config migrate` 自动迁移旧字段，
  并在 CHANGELOG 顶部置迁移指南。`threads`/`ASK_LLM_TRANSLATION_THREADS` 废弃给一个版本的 warning。
- **CLI 契约**：保留 `ask-llm`/`askllm` 双脚本名与全部子命令签名不变；CLI 行为视为契约。
- **服务签名**：`FormatService.run()` 为新增；旧 `run_sequential_format`/`run_parallel_format`
  保留为薄委托一个版本（标 deprecated），下个大版本删。
- **版本**：R0 发 patch（v2.19.x）；R1–R3 各发 minor；R2 的 `extra=forbid` 与配置收口若需破坏性，
  发 v3.0。
- **回滚单元**：每阶段独立 PR + 独立 tag；R0（止血）先行进生产；R1/R2/R3 任一可单独回退。

---

## 11. 附录

### 11.1 关键 file:line 索引（当前 v2.19.0）
- **数据主干**：`binary_splitter.py:62-64`（D1 fits）、`:105-124`（D3 frontmatter 忽略）、
  `token_counter.py:53-65`（ENCODING_MAP）+ `:254-256`（安全系数仅此一处）、
  `markdown_token_splitter.py:33,39`（D2 prompt_overhead=0）、`chunk_balance.py:15-56`（D4 翻译重切）、
  `md_body_formatter.py:449`（D5 resume 有损 join）、`constants.py:45`（0.85）。
- **执行引擎**：`chunked_llm_job.py:100`（A1 第二管线）、`batch_processor.py:117-129`（A6 池 cap）、
  `command_runner.py:118-120`（D6 checkpoint 末尾保存）、`concurrent.py:237-262`（`run` 薄包装）、
  `retry_policy.py:66-83`（`ProviderRetryRegistry` 未接线）、`batch_processor.py:68/342`（统计三层委托）。
- **配置**：`context.py:5-31`（全局）、`loader.py:225-313`（`_convert_providers_format`）、
  `unified_config.py:410`（无 model_config）、`env.py:32,34-37,55-78`（双映射 + 名字强转）、
  `manager.py:79-82`（SecretStr dump 隐患）、`token_counter.py:26-31`（热路径全局）。
- **服务**：`format_service.py:142/159/204/222/301/311`（全 None）、`paper_service.py:135-326`
  （上帝函数）、`batch_service.py:142-319`（自由函数）、`translation_service.py:89`/`paper_service.py:281`
  （可变累加）、`text_file_translator.py:246-348`（状态串比 + getattr）、`paper_service.py:501`
  （第二套 resume）。
- **耦合/卫生**：`text_file_translator.py:253,266` + `command_runner.py:125`（getattr 私有）、
  `md_heading_formatter.py:584`（第 4 份 pattern）、`md_heading_formatter.py:626-645`（重复 _load_prompt）、
  `core/batch.py`（shim）。

### 11.2 方法论
5 路并行 Explore agent（`.codegraph/` 索引，`codegraph_explore` 为主工具）覆盖执行引擎 / 服务层 /
配置 / Markdown 管线 / 提供者与 IO。本文 §5 高严重度结论（D1–D5、A1–A8）均经主线程独立 `grep`/`Read`
复核（`binary_splitter.py:62-64`、`token_counter.py:53-65`、`markdown_token_splitter.py:33,39`、
`chunk_balance.py`（无 MarkdownStructure）、`text_file_translator.py:246-348`、`format_service.py`
全 None、`md_body_formatter.py:449`、`get_config()` 零调用、`presentation.py` 不存在）。

### 11.3 相关文档
- `docs/ARCHITECTURE_REVIEW.md`：V1（2026-07-13），P0–P4 历史，**被本文取代为"当前状态"权威**。
- `AGENTS.md`：架构约定，**需刷新**（R4）。
- `CHANGELOG.md`：v2.16–v2.19 重构脉络。
- `docs/implementation_status.md`、`docs/REFACTOR_PLAN.md`：**过时，建议归档/删除**（R4）。
