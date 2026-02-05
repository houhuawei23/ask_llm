请将以下 {source_lang} 文本翻译成准确、流畅、清晰、自然的 {target_lang}，使用技术性、准确的语言风格，保持专业术语的准确性，保持原文的格式和结构：

### 翻译要求：

- **忠实传达**：完整、精确地传递原文的全部信息、核心含义与逻辑结构，不得增删、歪曲或臆测。
- **专业准确**：专业领域术语必须采用行业公认或上下文最贴切的标准译法。
- **逻辑清晰**：确保译文句内与句间的逻辑关系明确，与原文一致。
- **衔接自然**：合理使用连接词，保证段落与句子之间的过渡平滑。
- **风格一致**：保持与原文相同的语言风格，包括语气、语调、用词习惯等。
- 对于重点内容可以使用 markdown 语法加粗强调。如：这里是**重点内容**。
- 对于引用内容，请使用 markdown 语法引用。如：XXXX 内容 [1]。
  - 并将 <sup>[数字]</sup> 格式的引用 替换为 [数字] 格式的引用。
  - 如：<sup>41</sup> 替换为 [41]。
- 对于公式，请使用 markdown 语法公式。如：
  - 行内公式：$L(N, D) = \alpha N^{\beta} + \gamma D^{\delta}$。
  - 独立公式：$$L(N, D) = \alpha N^{\beta} + \gamma D^{\delta}$$。
- 标题也要采用 英文原文 + （中文翻译） 的格式。如：**下一词元预测（Next-token prediction）**。
- 列表内容要采用 markdown 语法列表。如：
  - 无序列表：
    - 列表项 1
    - 列表项 2
    - 列表项 3
  - 有序列表：
    1. 列表项 1
    2. 列表项 2
    3. 列表项 3
- 注意保留原文中的图片，并使用 markdown 语法插入图片。如：
  ```markdown
  ![图 1](path/to/image.jpg)

  > 图 1 | 图片描述。
  ```
- 注意保留原文中的表格，并使用 markdown 语法插入表格。如：
  ```markdown
  表 1 | 表格描述。
  | 列 1 | 列 2 | 列 3 |
  | :-----: | :-----: | :-----: |
  | 数据 1 | 数据 2 | 数据 3 |
  | 数据 4 | 数据 5 | 数据 6 |
  ```
- 所有专业术语、专有名词及文化特定概念，**首次在译文中出现时**，必须采用以下格式标注：`中文译名（英文原名， 缩写）` 或 `English Translation (Chinese Original)`
  - **注意**：使用全角中文括号 `（）`，括号内英文原名首字母大写，缩写按惯例大写。
  - 对于所有人名、术语、专有名词、缩写、高级词汇、生僻词，请输出：中文翻译（英语原文）。例如：迈克尔·乔丹（Michael Jordan）/熵（Entropy）/最大似然估计（Maximum Likelihood Estimation）/贝叶斯定理（Bayes' theorem）/混淆变量（Confounders）/因果推断（Causal Inference）等等。
- 对于原文中的格式错误、错别字、标点符号错误等，请修正后再进行翻译。
- 注意调整、美化译文中的格式，使其更加美观、清晰。

### 示例：

en:

```markdown
Next-token prediction has revolutionized the field of language models<sup>1</sup>, enabling breakthroughs such as ChatGPT<sup>7</sup> and sparking discussions about the early signs of artificial general intelligence<sup>8</sup>. However, its potential in multimodal learning has remained uncertain, with little evidence that this simple objective can be scaled across modalities to deliver both strong perception and high-fidelity generation. In the realm of multimodal models, vision generation has been dominated by complex diffusion models<sup>2</sup>, whereas vision-language perception has been led by compositional approaches<sup>9</sup> that combine CLIP<sup>10</sup> encoders with large language models (LLMs). Despite early attempts to unify generation and perception, such as Emu<sup>11</sup> and Chameleon<sup>12</sup>, these efforts either resort to connecting LLMs with diffusion models or fail to match the performance of task-specific methods tailored for generation and perception. This leaves open a fundamental scientific question: can a single next-token prediction framework serve as a general-purpose foundation for multimodal learning?
```

zh:

```markdown
**下一词元预测（Next-token prediction）**已经彻底改变了**语言模型（language models）**领域 [1]，实现了诸如 ChatGPT [7] 等突破，并引发了关于人工通用智能（Artificial General Intelligence, AGI）早期迹象的讨论 [8]。然而，它在多模态学习中的潜力仍然不确定，几乎没有证据表明这个简单的目标可以跨模态扩展，以提供强大的感知和高保真生成能力。在多模态模型领域，视觉生成一直由复杂的**扩散模型（Diffusion models）** [2] 主导，而视觉-语言感知则由**组合方法（Composite methods）** [9] 引领，这些方法将 CLIP [10] 编码器与大型语言模型（Large Language Models, LLMs）相结合。尽管早期尝试统一生成和感知，如 Emu [11] 和 Chameleon [12]，但这些努力要么诉诸于将 LLMs 与扩散模型连接，要么无法匹配为生成和感知量身定制的任务专用方法的性能。这留下了一个根本性的科学问题：**单一的下一词元预测框架能否作为多模态学习的通用基础？**
```

下面是待翻译的文本，请根据翻译要求进行翻译，只输出翻译后的文本，不要输出任何其他内容：

{content}
