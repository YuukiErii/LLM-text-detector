# NLG 课程项目：LLM 改写文本检测器工作总结与后续计划（更新版）

**项目名称**：LLM Text Detector / Hybrid DeBERTa-TFIDF Detector
**当前阶段**：多域训练数据成型 + 四模型改写生成中 + baseline 训练前准备
**更新时间**：2026-05-20
**本版说明**：本文件是在上一版项目总结基础上，根据今天的新进展更新而来。上一版已经总结了项目任务、模型路线、Gutenberg 小说数据、四模型改写策略、TF-IDF baseline、DeBERTa 计划等内容；本版重点补充今天完成的 academic 数据构建、Gemini 修复、ChatGPT 参数优化、四模型 academic 改写启动，以及下一阶段更具体的执行计划。

---

## 1. 项目任务回顾

本项目目标是构建一个二分类检测器，用于判断英文文本是：

```text
label = 0: human-written original text
label = 1: LLM-generated / LLM-rewritten text
```

项目核心难点不是普通 AI 作文检测，而是更具体的：

> 判断一段英文文本是否在保留原意的基础上，被 LLM 进行了改写、现代化、同义转述、风格转写或学术化重写。

根据老师给定测试集的分布，测试文本不仅包含普通英文，还包括：

```text
1. 19 世纪 / 20 世纪英语小说段落
2. 古典文学或旧式英文
3. 诗歌或强格式文本
4. NLP / computational linguistics / academic paper 段落
5. LLM 对上述文本的改写版本
```

因此，训练数据不能只依赖小说；需要至少覆盖：

```text
literature
academic
poetry / archaic English
```

当前项目已经从最初的 literature-only 数据准备，推进到 **literature + academic 多域数据构建阶段**。

---

## 2. 总体技术路线

最终系统仍然采用：

```text
Hybrid DeBERTa-TFIDF Detector
```

结构如下：

```text
Input text
   ├── DeBERTa-v3-base classifier
   │       └── P_deberta(label=1)
   │
   └── Word/Char TF-IDF + Logistic Regression
           └── P_tfidf(label=1)

Final:
P_final = α * P_deberta + (1 - α) * P_tfidf
```

### 2.1 TF-IDF baseline 的作用

TF-IDF baseline 用来捕捉：

```text
1. 词汇替换模式
2. 标点和字符 n-gram
3. 古典英语与现代英语差异
4. LLM 常见平滑表达
5. 改写后句式与原文本体的表层差异
```

它是项目的强 baseline，也会作为最终 ensemble 的一部分。

### 2.2 DeBERTa 的作用

DeBERTa-v3-base 用来捕捉：

```text
1. 上下文语义一致性
2. LLM 改写后的语义平滑化
3. 句法与语篇层面的改写痕迹
4. academic / literature 不同 domain 下的深层特征
```

### 2.3 Ensemble 的作用

最终将 TF-IDF 与 DeBERTa 融合：

```text
P_final = α * P_deberta + (1 - α) * P_tfidf
```

在 validation set 上搜索最佳：

```text
alpha
decision threshold
```

再在 internal_test 和老师测试集上评估。

---

## 3. 截至目前的总体进展

### 3.1 已完成的核心数据

当前已经完成或基本完成的数据包括：

| 数据类型 | 状态 | 数量 / 质量 |
|---|---|---|
| Literature human seed | 已完成 | 7130 条 |
| Literature rewrite prompts | 已完成 | 7130 条 |
| DeepSeek literature rewrite | 已完成 / 可用 | 质量良好 |
| Doubao literature rewrite | 已完成 / 可用 | 1789 条，1764 条通过基础质检，通过率 98.6% |
| Gemini literature rewrite | 已重新生成 / 基本可用 | 1763 条，长度指标正常，但约 131 条疑似截断 |
| ChatGPT literature rewrite | 正在运行 | 使用 GPT-5.4-mini |
| Academic human seed | 已完成 | 从 ACL-OCL 抽取 1200 条，过滤后 1098 条进入 prompt 阶段 |
| Academic rewrite prompts | 已完成 | 1098 条 |
| Academic prompts 按模型划分 | 已完成 | ChatGPT 40%，DeepSeek 40%，Gemini 10%，Doubao 10% |
| 四模型 academic rewrite | 正在运行 | 四路同时生成中 |

---

## 4. 今天的新进展总结

## 4.1 修复并完成 ACL-OCL academic 数据构建

### 4.1.1 起初遇到的问题

我们尝试使用 Hugging Face `datasets.load_dataset("WINGNUS/ACL-OCL")` 读取 ACL-OCL，但遇到两个问题：

#### 问题 A：split 名称不是 train

最初使用：

```python
load_dataset("WINGNUS/ACL-OCL", split="train", streaming=True)
```

报错：

```text
ValueError: Bad split: train. Available splits: ['test']
```

解决方式：

```python
split="test"
```

#### 问题 B：schema cast 失败

继续读取后，出现：

```text
TypeError: Couldn't cast array of type
struct<laboratory: string, institution: string, location: struct<>>
to
{}
```

原因是 ACL-OCL 中不同论文的 `authors.affiliation` 结构不一致，Hugging Face datasets 在统一 schema 时失败。

因此我们放弃 `datasets.load_dataset()`，改为：

```text
huggingface_hub.snapshot_download()
+
逐文件 json.loads()
```

### 4.1.2 Hugging Face 下载限流问题

下载 ACL-OCL 时遇到大量：

```text
HTTP Error 429
Rate limited
```

说明未登录 Hugging Face 时被限流。

解决方式包括：

```text
1. 登录 Hugging Face
2. 减少下载范围
3. 使用 allow_patterns 只下载相关 JSON
4. 使用本地缓存继续跑
```

最终通过限制下载和本地缓存，成功获得可用数据文件。

### 4.1.3 发现并修复 JSON 读取问题

一开始脚本虽然下载了数据，但抽取结果是：

```text
Saved 0 samples
```

我们用 `debug_acl_file.py` 检查了单个 ACL-OCL JSON 文件，确认文件结构为：

```json
{
  "paper_id": "...",
  "title": "...",
  "abstract": "...",
  "pdf_parse": {
    "body_text": [
      {
        "text": "...",
        "section": "Introduction"
      }
    ]
  }
}
```

也就是说：

```text
一个 .json 文件 = 一篇论文 = 一个 JSON dict
```

之前的 `iter_json_records()` 主要支持：

```text
JSONL
JSON array
```

没有正确支持“单个 JSON dict 文件”，所以每个文件扫描都是：

```text
0it
```

已修复为支持：

```text
1. single JSON dict
2. JSON array
3. JSONL
```

修复后成功抽取 academic seed。

### 4.1.4 Academic seed 构建结果

最终成功生成：

```text
data/processed/academic_seed.jsonl
```

数量：

```text
1200 条
```

样本字段格式：

```json
{
  "id": "human_academic_000001",
  "text": "...",
  "label": 0,
  "domain": "academic",
  "source": "acl_ocl:2020",
  "pair_id": "pair_academic_000001",
  "generation": "human",
  "metadata": {
    "paper_id": "2020",
    "title": "...",
    "year": "",
    "venue": null,
    "section": "Introduction"
  }
}
```

抽取内容主要来自 NLP / NLG / system paper 段落，例如 Xiaomingbot、automated news generation、natural language generation 等相关论文段落。

### 4.1.5 Academic seed 小问题

发现部分段落结尾可能不完整，例如：

```text
The system also uses a pretrained Generated News
```

因此在后续 prompt 准备阶段加入了结尾完整性过滤。

---

## 4.2 完成 academic rewrite prompts 构建

新增脚本：

```text
src/data/prepare_academic_rewrite_prompts.py
```

输入：

```text
data/processed/academic_seed.jsonl
```

输出：

```text
data/processed/rewrite_prompts_academic.jsonl
```

原始 academic seed：

```text
1200 条
```

经过过滤后：

```text
1098 条 academic rewrite prompts
```

过滤原因主要包括：

```text
1. 结尾不完整
2. 过短
3. 过长
4. 基础字段异常
```

当前 1098 条数量已经足够，不需要强行放宽过滤条件补满 1200。保持质量优先。

### 4.2.1 Academic prompt 类型

设计了三种 academic prompt：

```text
academic_paraphrase
academic_modernize
academic_style_transfer
```

对应目标：

```text
1. 保留技术含义和术语
2. 不总结、不省略
3. 维持大致长度
4. 改写为清晰、自然、论文风格英文
5. 不输出解释、标题、项目符号
```

示例目标风格：

```text
Paraphrase the following academic paragraph in clear, polished research English.
Preserve all technical meaning, claims, terminology, examples, and approximate length.
Do not summarize, omit details, or add new information.
Return only the rewritten paragraph.
```

---

## 4.3 完成 academic prompts 按模型划分

新增脚本：

```text
src/data/split_academic_rewrite_prompts_by_model.py
```

划分比例采用：

```text
ChatGPT: 40%
DeepSeek: 40%
Gemini: 10%
Doubao: 10%
```

对 1098 条 academic prompts，理论分配约为：

```text
ChatGPT: 439–440 条
DeepSeek: 439–440 条
Gemini: 109–110 条
Doubao: 109–110 条
```

之所以不均分，是因为 academic 段落更依赖：

```text
1. 术语保持
2. 论点稳定
3. 长度控制
4. 不总结、不遗漏
```

因此更信任：

```text
ChatGPT / GPT-5.4-mini
DeepSeek
```

Gemini 之前存在截断问题，豆包速度较慢，所以只保留 10% 各自增加生成器多样性。

生成文件：

```text
data/processed/rewrite_prompts_academic_chatgpt.jsonl
data/processed/rewrite_prompts_academic_deepseek.jsonl
data/processed/rewrite_prompts_academic_gemini.jsonl
data/processed/rewrite_prompts_academic_doubao.jsonl
data/processed/rewrite_prompts_academic_with_generator.jsonl
```

当前状态：

```text
四个模型的 academic 改写正在同时运行中。
```

---

## 4.4 修复 Gemini literature 改写脚本并重新生成数据

### 4.4.1 之前 Gemini 的问题

旧版 Gemini 输出质量极差：

```text
Total samples: 50
Passed: 4 / 50
Pass rate: 0.0800
Length ratio mean ≈ 0.28
too_short_relative: 46
too_short_absolute: 16
```

典型输出是半句话，例如：

```text
Mr. Lorry accepted the small, uncertain hand that was trustingly offered to him, pressing it to his lips with a degree
```

判断原因：

```text
1. max_tokens 默认只有 650，可能太低
2. 没有检查 finish_reason
3. load_finished_task_ids 会把旧的坏输出也当作完成
4. 中转站对 Gemini 模型的 max_tokens 映射可能不稳定
```

### 4.4.2 已完成脚本修复

已更新：

```text
src/data/generate_gemini_rewrites.py
```

主要修改：

```text
1. 默认 max_tokens 改为 2000
2. 记录 finish_reason
3. 如果 finish_reason == "length"，默认写入 failed
4. load_finished_task_ids 只认非空且质检通过样本
5. 增加 looks_truncated 检测
6. 增加 possibly_truncated 质量问题
7. 质量失败样本不写入成功文件
8. 增加 allow_low_quality 和 allow_length_finish 参数
```

### 4.4.3 新 Gemini literature 输出质量

重新跑出的 Gemini literature 文件：

```text
data/processed/llm_rewrite_gemini.jsonl
```

当前检查结果：

```text
总样本数: 1763
基础 quality check: 1763 / 1763 通过
平均 source word count: 约 105
平均 rewrite word count: 约 104
平均 length ratio: 约 1.01
平均 lexical Jaccard: 约 0.275
```

说明此前“整体过短”的问题已经基本解决。

### 4.4.4 仍存在的问题：约 131 条疑似静默截断

虽然基础 quality check 通过率为 100%，但额外检查发现：

```text
约 131 / 1763 条结尾不像完整句子
约占 7.4%
```

典型问题：

```text
... and I told Lucie as much. What
... while the moon and the scudding clouds drifted in the
... show me how I may become
... gripped the child’s fingers and, with a curt word to Eliza
```

这些样本不应直接进入最终训练集。

当前判断：

```text
Gemini literature 数据整体可用，但需要过滤或重跑约 131 条疑似截断样本。
```

后续需要新增脚本：

```text
src/data/filter_or_collect_truncated_rewrites.py
```

目标：

```text
1. 找出结尾不完整的 Gemini 样本
2. 导出对应 task_id
3. 可选：生成待重跑 prompts
4. 或在 build_full_dataset 阶段自动过滤这些样本
```

---

## 4.5 优化 ChatGPT literature 调用脚本

今天更新了：

```text
src/data/generate_chatgpt_rewrites.py
```

### 4.5.1 默认模型调整

原计划使用：

```text
gpt-5.4
```

后来根据任务性质，调整为：

```text
gpt-5.4-mini
```

原因：

```text
1. 本任务是批量改写，不是复杂推理
2. 重点是保留含义、改写措辞、控制长度
3. GPT-5.4-mini 性价比和速度更好
4. 质量预计足够作为 ChatGPT 生成器分支
```

不推荐使用：

```text
gpt-5.4-nano
```

作为主力改写模型，因为 nano 更适合分类、抽取、路由、摘要等轻任务，文学和学术改写容易出现过短、过平、总结化。

### 4.5.2 参数优化

ChatGPT 脚本默认参数更新为：

```text
temperature = 0.7
top_p = 0.9
max_tokens = 1000
sleep = 0.3
model = gpt-5.4-mini
```

对于 academic rewrite，建议：

```text
max_tokens = 1200
```

### 4.5.3 质量控制优化

脚本现在已经实现：

```text
1. 空输出写入 failed
2. 质量失败写入 failed
3. load_finished_task_ids 只认质检通过样本
4. 不合格样本不污染成功文件
```

当前状态：

```text
ChatGPT literature rewrite 正在运行中。
```

---

## 4.6 代码偏好更新

今天明确了后续代码协作偏好：

```text
以后当需要修改脚本时，直接提供修改后的完整可替换脚本。
不要只告诉“应该改哪里”。
```

之后所有脚本更新都会直接以完整文件形式给出。

---

## 5. 当前项目状态表

| 模块 | 当前状态 | 备注 |
|---|---|---|
| 项目结构 | 已完成 | PyCharm + venv + GitHub |
| 老师测试集 inspect | 已完成 | 仅用于理解分布，不用于训练 |
| Literature human seed | 已完成 | 7130 条 |
| Literature rewrite prompts | 已完成 | 7130 条 |
| Literature prompts 四模型划分 | 已完成 | 原始为 25% / 25% / 25% / 25% |
| DeepSeek literature rewrite | 已完成 / 可用 | 质量好 |
| Doubao literature rewrite | 已完成 / 可用 | 1789 条，98.6% 通过 |
| Gemini literature rewrite | 已完成 / 基本可用 | 1763 条，约 131 条疑似截断需处理 |
| ChatGPT literature rewrite | 正在运行 | GPT-5.4-mini |
| Academic seed | 已完成 | ACL-OCL 抽取 1200 条 |
| Academic prompts | 已完成 | 过滤后 1098 条 |
| Academic prompts 四模型划分 | 已完成 | ChatGPT 40%，DeepSeek 40%，Gemini 10%，Doubao 10% |
| Academic 四模型 rewrite | 正在运行 | 四路同时进行 |
| Poetry seed | 尚未开始 | 可作为后续增强 |
| build_full_dataset.py | 已完成 | 后续需增强截断过滤 |
| split_dataset_by_pair.py | 已完成 | 防止 pair leakage |
| TF-IDF baseline | 脚本已完成，待最终数据 | 可先用 literature-only 跑 |
| DeBERTa | 尚未开始 | 等 full dataset 成型 |
| Ensemble | 尚未开始 | DeBERTa 后进行 |
| Teacher test evaluation | 尚未开始 | 最终阶段再运行 |

---

## 6. 当前正在运行的任务

你现在正在同时运行：

```text
1. ChatGPT literature rewrite
2. ChatGPT academic rewrite
3. DeepSeek academic rewrite
4. Gemini academic rewrite
5. Doubao academic rewrite
```

因此明天第一件事应该不是继续写新模型，而是先检查这些生成结果质量。

---

## 7. 明天优先检查项

明天建议按以下顺序检查。

### 7.1 检查 ChatGPT literature

```powershell
python src/data/inspect_llm_rewrite.py --input data/processed/llm_rewrite_chatgpt.jsonl
```

重点看：

```text
Total samples
Pass rate
Length ratio mean / median
Lexical Jaccard mean / median
Quality issues
Failed examples
```

理想目标：

```text
Pass rate > 95%
Length ratio mean: 0.8–1.3
No large-scale too_short_relative
No large-scale contains_meta_text
```

### 7.2 检查 academic ChatGPT

```powershell
python src/data/inspect_llm_rewrite.py --input data/processed/llm_rewrite_academic_chatgpt.jsonl
```

理想目标：

```text
Pass rate > 95%
Length ratio: 0.8–1.3
术语保留较好
没有大量总结化
```

### 7.3 检查 academic DeepSeek

```powershell
python src/data/inspect_llm_rewrite.py --input data/processed/llm_rewrite_academic_deepseek.jsonl
```

理想目标：

```text
Pass rate > 95%
Length ratio 接近 1
Jaccard 不过高
没有复制原文
```

### 7.4 检查 academic Gemini

```powershell
python src/data/inspect_llm_rewrite.py --input data/processed/llm_rewrite_academic_gemini.jsonl
```

重点关注：

```text
1. 是否仍然有截断
2. 是否 too_short_relative
3. 是否 finish_reason=length
4. 是否 possibly_truncated
```

如果 Gemini academic 质量不好，直接考虑：

```text
1. 丢弃 Gemini academic 分支
2. 或将 Gemini academic prompts 转给 ChatGPT / DeepSeek 补跑
```

### 7.5 检查 academic Doubao

```powershell
python src/data/inspect_llm_rewrite.py --input data/processed/llm_rewrite_academic_doubao.jsonl
```

重点关注：

```text
1. 是否过长
2. 是否 contains_meta_text
3. 是否解释化 / 扩写化
```

---

## 8. 下一步必须补的脚本

## 8.1 截断样本过滤 / 收集脚本

需要新增：

```text
src/data/filter_or_collect_truncated_rewrites.py
```

用途：

```text
1. 输入 llm_rewrite_gemini.jsonl
2. 检测结尾不完整样本
3. 输出 clean 文件
4. 输出 truncated task_id 列表
5. 可选输出待重跑 prompts
```

建议输出：

```text
data/processed/llm_rewrite_gemini_clean.jsonl
data/processed/llm_rewrite_gemini_truncated.jsonl
data/processed/rewrite_prompts_gemini_truncated_rerun.jsonl
```

这个脚本也可以用于 academic Gemini。

---

## 8.2 合并 human seeds 脚本

需要新增：

```text
src/data/merge_human_seeds.py
```

输入：

```text
data/processed/human_seed.jsonl
data/processed/academic_seed.jsonl
```

输出：

```text
data/processed/human_seed_combined.jsonl
```

未来若增加 poetry：

```text
data/processed/poetry_seed.jsonl
```

也加入合并。

---

## 8.3 升级 build_full_dataset.py

当前 `build_full_dataset.py` 已经能过滤：

```text
empty text
quality false
label error
missing pair_id
```

但需要增强：

```text
1. 可选过滤 possibly_truncated
2. 可选过滤结尾不完整
3. 支持多个 human seed 文件
4. 支持按 domain 报告统计
5. 支持指定 output report
```

这样最终 full dataset 会更干净。

---

## 8.4 inspect_human_seed.py 支持命令行参数

如果当前还不支持，建议升级为：

```powershell
python src/data/inspect_human_seed.py --input data/processed/academic_seed.jsonl
```

这样 literature / academic / poetry 都能复用。

---

## 9. 数据集构建计划更新

### 9.1 Literature-only 第一版

当 ChatGPT literature 完成后，可以先构建 literature-only 数据集。

输入：

```text
human_seed.jsonl
llm_rewrite_deepseek.jsonl
llm_rewrite_doubao.jsonl
llm_rewrite_chatgpt.jsonl
llm_rewrite_gemini_clean.jsonl
```

输出：

```text
full_dataset_literature.jsonl
```

用途：

```text
先跑完整 pipeline，获得第一版 TF-IDF baseline 结果。
```

### 9.2 Literature + Academic 第二版

当 academic 四模型 rewrite 完成并检查后，构建：

```text
human_seed_combined.jsonl
full_dataset_lit_academic.jsonl
```

包含：

```text
literature human
academic human
literature LLM rewrite
academic LLM rewrite
```

用途：

```text
训练更贴近老师测试集的最终模型。
```

### 9.3 Poetry 第三版，可选

若时间允许，补充：

```text
poetry_seed.jsonl: 300–500 条
poetry LLM rewrite: 300–500 条
```

构建：

```text
full_dataset_lit_academic_poetry.jsonl
```

用于最终模型和 ablation。

---

## 10. 训练计划更新

## 10.1 先跑 TF-IDF baseline

一旦 literature-only 或 lit+academic 数据集建好，先跑：

```powershell
python src/data/split_dataset_by_pair.py `
  --input data/processed/full_dataset_literature.jsonl `
  --prefix literature_
```

然后：

```powershell
python src/models/train_tfidf_baseline.py `
  --train data/processed/literature_train.jsonl `
  --valid data/processed/literature_valid.jsonl `
  --test data/processed/literature_internal_test.jsonl `
  --output_dir outputs/models/tfidf_baseline_literature
```

之后对 lit+academic 重复：

```powershell
python src/data/split_dataset_by_pair.py `
  --input data/processed/full_dataset_lit_academic.jsonl `
  --prefix lit_academic_
```

```powershell
python src/models/train_tfidf_baseline.py `
  --train data/processed/lit_academic_train.jsonl `
  --valid data/processed/lit_academic_valid.jsonl `
  --test data/processed/lit_academic_internal_test.jsonl `
  --output_dir outputs/models/tfidf_baseline_lit_academic
```

### 10.2 TF-IDF baseline 预期作用

它会快速告诉我们：

```text
1. 数据是否有明显泄漏
2. label 是否可分
3. 是否存在某个 generator 特征过强
4. domain 加入后是否导致性能下降
5. 是否需要更平衡的 generator / domain 比例
```

### 10.3 再训练 DeBERTa

TF-IDF 跑通后再写：

```text
src/models/train_deberta.py
```

推荐配置：

```text
model_name = microsoft/deberta-v3-base
max_length = 512
learning_rate = 2e-5
batch_size = 8 或 16
epochs = 3
weight_decay = 0.01
warmup_ratio = 0.1
metric_for_best_model = f1
```

如果显存不足：

```text
gradient_accumulation_steps = 2 或 4
fp16 = True
batch_size = 4 或 8
```

---

## 11. Ablation 设计更新

后续报告可以做如下 ablation：

| 实验编号 | 数据设置 | 模型 | 目的 |
|---|---|---|---|
| E1 | Literature-only | TF-IDF | 最基础 baseline |
| E2 | Literature-only | DeBERTa | 深度模型 baseline |
| E3 | Literature + Academic | TF-IDF | 检查 academic domain 是否提升泛化 |
| E4 | Literature + Academic | DeBERTa | 主模型 |
| E5 | Literature + Academic + Poetry | DeBERTa | 可选最终增强 |
| E6 | Full Data | TF-IDF + DeBERTa Ensemble | 最终提交模型 |

还可以做 generator ablation：

| 实验 | 说明 |
|---|---|
| remove Gemini | 检查 Gemini 噪声是否影响 |
| remove Doubao | 检查 Doubao 扩写倾向是否影响 |
| DeepSeek + ChatGPT only | 高质量生成器训练集 |
| all generators | 多样性最大版本 |

---

## 12. 当前主要风险与应对

### 12.1 Gemini 静默截断风险

风险：

```text
部分 Gemini 输出不是 too_short，但结尾明显不完整。
```

应对：

```text
1. 使用 finish_reason
2. 使用结尾完整性检测
3. 过滤 clean 文件
4. 截断样本可重跑或丢弃
```

### 12.2 Academic 原文 PDF parse 噪声

风险：

```text
ACL-OCL body_text 来自 PDF parse，可能有断句、引用、表格、figure 文字。
```

应对：

```text
1. prepare_academic_rewrite_prompts 已过滤结尾不完整
2. 后续 inspect academic seed
3. full dataset 里保留 metadata，方便错误分析
```

### 12.3 Generator imbalance

风险：

```text
某个 generator 数量或风格过强，模型可能学 generator 而不是 LLM 改写本质。
```

应对：

```text
1. report generator distribution
2. 尝试 DeepSeek+ChatGPT-only 与 all-generator 对比
3. 控制各 generator 数量
```

### 12.4 Domain imbalance

风险：

```text
literature 7130，academic 1098，academic 比例较小。
```

应对：

```text
1. academic 仍然足以补充分布
2. 如果老师测试集 academic 比重大，可提高 academic 权重
3. 训练时可使用 class/domain-aware sampling
```

### 12.5 Test leakage

原则保持不变：

```text
老师测试集绝不用于训练、调参、选择 threshold、选择 prompt。
```

---

## 13. 明天建议执行顺序

明天建议按这个顺序做：

### Step 1：检查所有正在跑的输出

```powershell
python src/data/inspect_llm_rewrite.py --input data/processed/llm_rewrite_chatgpt.jsonl
python src/data/inspect_llm_rewrite.py --input data/processed/llm_rewrite_academic_chatgpt.jsonl
python src/data/inspect_llm_rewrite.py --input data/processed/llm_rewrite_academic_deepseek.jsonl
python src/data/inspect_llm_rewrite.py --input data/processed/llm_rewrite_academic_gemini.jsonl
python src/data/inspect_llm_rewrite.py --input data/processed/llm_rewrite_academic_doubao.jsonl
```

### Step 2：写截断过滤脚本

```text
src/data/filter_or_collect_truncated_rewrites.py
```

优先处理：

```text
llm_rewrite_gemini.jsonl
llm_rewrite_academic_gemini.jsonl
```

### Step 3：生成 clean Gemini 文件

```text
llm_rewrite_gemini_clean.jsonl
llm_rewrite_academic_gemini_clean.jsonl
```

### Step 4：合并 human seeds

```text
human_seed.jsonl + academic_seed.jsonl
→ human_seed_combined.jsonl
```

### Step 5：构建 full dataset

先构建：

```text
full_dataset_literature.jsonl
```

再构建：

```text
full_dataset_lit_academic.jsonl
```

### Step 6：split by pair_id

```text
literature_train / valid / internal_test
lit_academic_train / valid / internal_test
```

### Step 7：训练 TF-IDF baseline

先拿 literature-only 跑一次，再拿 lit+academic 跑一次。

---

## 14. 已完成脚本清单（更新）

当前已经完成或正在使用的脚本：

```text
src/data/inspect_teacher_test.py
src/data/build_human_seed_from_txt.py
src/data/inspect_human_seed.py
src/data/prepare_rewrite_prompts.py
src/data/split_rewrite_prompts_by_model.py
src/data/generate_deepseek_rewrites.py
src/data/generate_doubao_rewrites.py
src/data/generate_doubao_rewrites_parallel.py
src/data/generate_chatgpt_rewrites.py
src/data/generate_gemini_rewrites.py
src/data/inspect_llm_rewrite.py
src/data/build_academic_seed_from_acl_ocl.py
src/data/debug_acl_file.py
src/data/prepare_academic_rewrite_prompts.py
src/data/split_academic_rewrite_prompts_by_model.py
src/data/build_full_dataset.py
src/data/split_dataset_by_pair.py
src/models/train_tfidf_baseline.py
```

后续优先新增：

```text
src/data/filter_or_collect_truncated_rewrites.py
src/data/merge_human_seeds.py
src/data/build_full_dataset_v2.py 或升级 build_full_dataset.py
src/models/train_deberta.py
src/models/ensemble.py
src/evaluation/evaluate_teacher_test.py
```

---

## 15. 当前结论

今天的关键进展非常大：

```text
1. ACL-OCL academic 数据路线完全跑通
2. 成功生成 1200 条 academic human seed
3. 过滤后得到 1098 条高质量 academic rewrite prompts
4. 完成 academic prompts 四模型分配
5. 四个模型 academic rewrite 已经开始同时运行
6. Gemini literature 截断问题基本修复
7. Gemini literature 生成了 1763 条长度正常样本
8. 发现并定位 Gemini 静默截断风险
9. ChatGPT 脚本改为 GPT-5.4-mini 并优化参数
10. 明确以后代码修改直接提供完整脚本
```

项目现在已经进入：

```text
最终训练数据成型前夜
```

下一步只要完成：

```text
1. 检查四模型 academic rewrite
2. 过滤 Gemini 截断样本
3. 合并 literature + academic 数据
4. split
5. 训练 TF-IDF baseline
```

就可以得到第一版完整可报告实验结果。

---

## 16. 2026-05-20 Full Dataset and Training-Code Update

Current status after the full poetry/data-prep pass:

```text
Poetry human seed:
  data/processed/poetry_seed.jsonl
  500 rows from verified Project Gutenberg poetry books.

Poetry rewrite prompts:
  data/processed/rewrite_prompts_poetry.jsonl
  split into ChatGPT 200 / DeepSeek 200 / Gemini 50 / Doubao 50.

Poetry LLM rewrites:
  ChatGPT: 200 kept
  DeepSeek: 199 kept, 1 quality-filtered
  Gemini: 46 kept after retry, 4 final failures
  Doubao: 50 kept
```

Final merged datasets:

```text
data/processed/human_seed_combined.jsonl
  8830 human rows
  literature 7130 / academic 1200 / poetry 500

data/processed/full_dataset_lit_academic_poetry.jsonl
  17295 total rows
  label 0 human: 8830
  label 1 LLM: 8465
  literature: 14008
  academic: 2292
  poetry: 995
```

Main pair-safe split:

```text
data/processed/lit_academic_poetry_train.jsonl
  13836 rows, 7064 human / 6772 LLM

data/processed/lit_academic_poetry_valid.jsonl
  1728 rows, 882 human / 846 LLM

data/processed/lit_academic_poetry_internal_test.jsonl
  1731 rows, 884 human / 847 LLM

Leakage check:
  train-valid pair_id overlap: 0
  train-test pair_id overlap: 0
  valid-test pair_id overlap: 0
```

Training code now ready but not run:

```text
src/models/train_tfidf_baseline.py
  Word + char TF-IDF Logistic Regression.
  Defaults point to lit_academic_poetry split.
  Saves validation/test prediction JSONL with metadata.

src/models/train_deberta.py
  DeBERTa-v3 sequence classifier training entrypoint.
  Defaults point to lit_academic_poetry split.
  Uses lazy torch/transformers imports and saves prediction JSONL.

src/models/ensemble.py
  Tunes alpha and threshold on validation predictions.
  Evaluates the selected fusion on internal_test.
```

Non-training verification completed:

```text
.venv/Scripts/python.exe -m py_compile ...
src/models/train_tfidf_baseline.py --help
src/models/train_deberta.py --help
src/models/ensemble.py --help
```

Next step:

```text
1. Install neural dependencies if missing:
   python -m pip install -r requirements.txt

2. Train TF-IDF baseline.

3. Fine-tune DeBERTa.

4. Run ensemble tuning after both prediction files exist.

5. Only after internal metrics are stable, add final teacher-test inference/API code.
```

---

## 17. 2026-05-20 TF-IDF Baseline Result

Command:

```text
.venv/Scripts/python.exe src/models/train_tfidf_baseline.py
  --train data/processed/lit_academic_poetry_train.jsonl
  --valid data/processed/lit_academic_poetry_valid.jsonl
  --test data/processed/lit_academic_poetry_internal_test.jsonl
  --output_dir outputs/models/tfidf_lit_academic_poetry
```

Overall metrics:

```text
Validation:
  accuracy  0.9155
  precision 0.9464
  recall    0.8771
  f1        0.9104
  roc_auc   0.9699

Internal test:
  accuracy  0.9087
  precision 0.9389
  recall    0.8701
  f1        0.9032
  roc_auc   0.9659
```

Internal-test domain breakdown:

```text
literature: n=1393, acc=0.9182, f1=0.9139
academic:   n=258,  acc=0.8837, f1=0.8729
poetry:     n=80,   acc=0.8250, f1=0.8056
```

Internal-test generator breakdown:

```text
ChatGPT:  n=241, acc=0.6515, recall=0.6515
DeepSeek: n=233, acc=0.9742, recall=0.9742
Gemini:   n=187, acc=0.9144, recall=0.9144
Doubao:   n=186, acc=0.9785, recall=0.9785
Human:    n=884, false positives=48
```

Takeaway:

```text
TF-IDF baseline is strong enough for a first reportable baseline.
The main weakness is ChatGPT-style rewrites, followed by poetry.
This is exactly where DeBERTa should help if it captures deeper semantic/discourse cues.
```

---

## 18. 2026-05-20 DeBERTa Training Result

Important environment note:

```text
Initial Transformers 5.8.1 runs produced NaN gradients/loss for DeBERTa.
The stable setup was:
  torch 2.11.0+cu128
  transformers 4.57.6
  CUDA 12.8
  RTX 5080 Laptop GPU

requirements.txt was pinned to transformers>=4.40,<5.
```

Final training command:

```text
.venv/Scripts/python.exe src/models/train_deberta.py
  --train data/processed/lit_academic_poetry_train.jsonl
  --valid data/processed/lit_academic_poetry_valid.jsonl
  --test data/processed/lit_academic_poetry_internal_test.jsonl
  --output_dir outputs/models/deberta_lit_academic_poetry
  --model_name microsoft/deberta-v3-base
  --max_length 512
  --batch_size 4
  --eval_batch_size 8
  --gradient_accumulation_steps 2
  --learning_rate 1e-5
  --max_grad_norm 1.0
  --optim adamw_torch
  --epochs 3
```

Validation curve:

```text
epoch 1:
  f1 0.9505, accuracy 0.9514, roc_auc 0.9889

epoch 2:
  f1 0.9278, accuracy 0.9253, roc_auc 0.9916

epoch 3:
  f1 0.9441, accuracy 0.9433, roc_auc 0.9931

Best checkpoint selected by valid F1: epoch 1 / checkpoint-1730.
```

Final saved prediction metrics using the best checkpoint:

```text
Validation:
  accuracy  0.9514
  precision 0.9472
  recall    0.9539
  f1        0.9505
  roc_auc   0.9889

Internal test:
  accuracy  0.9284
  precision 0.9208
  recall    0.9339
  f1        0.9273
  roc_auc   0.9830
```

Internal-test domain breakdown:

```text
literature: n=1393, acc=0.9340, f1=0.9328
academic:   n=258,  acc=0.9186, f1=0.9202
poetry:     n=80,   acc=0.8625, f1=0.8493
```

Internal-test generator breakdown:

```text
ChatGPT:  n=241, acc/recall=0.7925
DeepSeek: n=233, acc/recall=0.9785
Gemini:   n=187, acc/recall=0.9947
Doubao:   n=186, acc/recall=1.0000
Human:    n=884, false positives=68
```

Comparison against TF-IDF baseline:

```text
Internal-test F1:
  TF-IDF:   0.9032
  DeBERTa:  0.9273

ChatGPT internal-test recall:
  TF-IDF:   0.6515
  DeBERTa:  0.7925

Poetry internal-test F1:
  TF-IDF:   0.8056
  DeBERTa:  0.8493
```

Next step:

```text
Run src/models/ensemble.py using the saved TF-IDF and DeBERTa prediction files.
```

---

## 19. 2026-05-20 Ensemble Result

Inputs:

```text
TF-IDF predictions:
  outputs/models/tfidf_lit_academic_poetry/predictions/tfidf_valid_predictions.jsonl
  outputs/models/tfidf_lit_academic_poetry/predictions/tfidf_internal_test_predictions.jsonl

DeBERTa predictions:
  outputs/models/deberta_lit_academic_poetry/predictions/deberta_valid_predictions.jsonl
  outputs/models/deberta_lit_academic_poetry/predictions/deberta_internal_test_predictions.jsonl
```

Coarse search result:

```text
output_dir: outputs/models/ensemble_lit_academic_poetry
best alpha: 0.4
best threshold: 0.5
valid f1: 0.9565
internal-test f1: 0.9353
```

Fine search result:

```text
output_dir: outputs/models/ensemble_lit_academic_poetry_fine
best alpha: 0.33
best threshold: 0.48
valid f1: 0.9583
internal-test f1: 0.9388
```

Final current-best ensemble metrics:

```text
Validation:
  accuracy  0.9595
  precision 0.9652
  recall    0.9515
  f1        0.9583
  roc_auc   0.9870

Internal test:
  accuracy  0.9405
  precision 0.9450
  recall    0.9327
  f1        0.9388
  roc_auc   0.9812
```

Internal-test domain breakdown:

```text
literature: n=1393, acc=0.9462, f1=0.9446
academic:   n=258,  acc=0.9264, f1=0.9261
poetry:     n=80,   acc=0.8875, f1=0.8767
```

Internal-test generator breakdown:

```text
ChatGPT:  n=241, acc/recall=0.7884
DeepSeek: n=233, acc/recall=0.9828
Gemini:   n=187, acc/recall=0.9947
Doubao:   n=186, acc/recall=0.9946
Human:    n=884, false positives=46
```

Comparison:

```text
Internal-test F1:
  TF-IDF:       0.9032
  DeBERTa:      0.9273
  Ensemble fine 0.9388

Internal-test human false positives:
  DeBERTa:      68
  Ensemble fine 46

Internal-test poetry F1:
  TF-IDF:       0.8056
  DeBERTa:      0.8493
  Ensemble fine 0.8767
```

Current next step:

```text
Build final inference/evaluation script:
  1. Load DeBERTa best_model and tokenizer.
  2. Load TF-IDF model/vectorizers.
  3. Fuse with alpha=0.33 and threshold=0.48.
  4. Run on teacher test only once the inference path is checked.
```

---

## 20. 2026-05-20 Final Inference and Teacher-Test Result

Implemented final inference script:

```text
src/evaluation/predict_ensemble.py
```

The script:

```text
1. Reads JSON list or JSONL input records.
2. Loads TF-IDF vectorizers and logistic regression from:
   outputs/models/tfidf_lit_academic_poetry
3. Loads DeBERTa best_model and tokenizer from:
   outputs/models/deberta_lit_academic_poetry
4. Loads ensemble alpha/threshold from:
   outputs/models/ensemble_lit_academic_poetry_fine/fusion_config.json
5. Writes detailed JSONL predictions, optional metrics, and a submission JSON.
```

Sanity check on internal test:

```text
command:
  .venv/Scripts/python.exe src/evaluation/predict_ensemble.py
    --input data/processed/lit_academic_poetry_internal_test.jsonl
    --output outputs/predictions/internal_test_final_ensemble_predictions.jsonl
    --submission outputs/predictions/internal_test_final_ensemble_submission.json
    --metrics outputs/predictions/internal_test_final_ensemble_metrics.json
    --batch_size 16

result:
  accuracy  0.9405
  precision 0.9450
  recall    0.9327
  f1        0.9388
  roc_auc   0.9812
  confusion [[838, 46], [57, 790]]
```

This reproduces the tuned ensemble's internal-test score, so the independent
inference path is consistent with the ensemble tuning output.

Teacher test command:

```text
.venv/Scripts/python.exe src/evaluation/predict_ensemble.py
  --input data/raw/teacher_test.json
  --output outputs/predictions/teacher_test_final_ensemble_predictions.jsonl
  --submission outputs/predictions/teacher_test_submission_minimal.json
  --metrics outputs/predictions/teacher_test_final_ensemble_metrics.json
  --batch_size 16
  --minimal_submission
```

Teacher test result:

```text
samples:   300
accuracy:  0.9033
precision: 0.8712
recall:    0.9467
f1:        0.9073
roc_auc:   0.9663
confusion: [[129, 21], [8, 142]]
```

Submission output:

```text
outputs/predictions/teacher_test_submission_minimal.json
```

Submission format:

```json
[
  {"label": 1, "probability": 0.8177978235674512},
  {"label": 1, "probability": 0.8981660004603468},
  {"label": 0, "probability": 0.3320058690406726}
]
```

Prediction distribution on teacher test:

```text
predicted LLM:   163
predicted human: 137
```

Teacher-test error analysis:

```text
script:
  src/evaluation/analyze_predictions.py

command:
  .venv/Scripts/python.exe src/evaluation/analyze_predictions.py
    --predictions outputs/predictions/teacher_test_final_ensemble_predictions.jsonl
    --input data/raw/teacher_test.json
    --output outputs/predictions/teacher_test_error_analysis.md
    --threshold 0.48
    --examples 8

output:
  outputs/predictions/teacher_test_error_analysis.md
```

Error summary:

```text
false positives: 21
false negatives: 8

branch agreement patterns:
  tfidf=1, deberta=1, final=1: 140
  tfidf=0, deberta=0, final=0: 123
  tfidf=0, deberta=1, final=1: 20
  tfidf=0, deberta=1, final=0: 8
  tfidf=1, deberta=0, final=0: 6
  tfidf=1, deberta=0, final=1: 3

branch accuracy using threshold 0.48:
  tfidf:   0.8900
  deberta: 0.8867
  final:   0.9033
```

Interpretation for report:

```text
The final ensemble improves over either branch alone on teacher test.
Most high-confidence false positives are human literary/poetic or academic
passages that DeBERTa scores as strongly LLM-like. False negatives tend to be
LLM rewrites whose surface form remains close to human prose or archaic poetry.
This supports the conservative claim that the ensemble helps but does not
fully solve distribution shift in stylistically unusual human text.
```

Report-ready tables:

```text
outputs/predictions/final_report_tables.md

Contents:
  1. Main metrics for TF-IDF, DeBERTa, ensemble, and teacher test.
  2. Ensemble internal-test domain breakdown.
  3. Ensemble internal-test generator breakdown.
  4. Teacher-test error summary.
```

Current next step:

```text
Prepare the final report artifacts:
  1. Move the tables and error-analysis observations into the written report.
  2. Decide whether an API wrapper is needed for submission; the core CLI
     inference path is already ready.
```
