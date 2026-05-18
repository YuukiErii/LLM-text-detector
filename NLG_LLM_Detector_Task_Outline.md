# NLG 课程项目任务说明与完整实施大纲

## 项目标题

**基于 DeBERTa 与多粒度 TF-IDF 特征融合的 LLM 生成文本检测系统**

英文标题建议：

**Detecting LLM-Rewritten Text with DeBERTa and Multi-level TF-IDF Features**

---

## 1. 项目背景

随着大语言模型的发展，LLM 可以生成高度流畅、语法正确、语义连贯的文本。传统的文本分类或抄袭检测方法很难直接判断一段文本是否由人类写作，还是由 LLM 生成、改写或润色。

本课程项目关注一个二分类任务：给定一段英文文本，判断其来源是：

- **Human-written text**：人类原创文本，记为 `label = 0`
- **LLM-generated / LLM-rewritten text**：由大语言模型生成、改写、现代化重述或润色的文本，记为 `label = 1`

老师给定的 JSON 测试集显示，该任务并不是普通的“AI 作文检测”，而更接近于“**文学与学术文本的 LLM 改写检测**”。测试集中包含大量英文文学原文、旧式英文、诗歌、叙事小说段落、NLP/语言学学术段落，以及对应的 LLM 改写或仿写文本。因此，本项目的核心目标是构建一个能够识别“改写痕迹”和“风格现代化痕迹”的文本检测系统。

---

## 2. 任务定义

### 2.1 输入

输入是一段英文文本：

```json
{
  "text": "The chamber within was illuminated solely by the amber radiance..."
}
```

### 2.2 输出

输出是二分类标签：

```json
{
  "label": 1,
  "probability": 0.83
}
```

其中：

| 标签 | 含义 |
|---|---|
| `0` | Human-written |
| `1` | LLM-generated / LLM-rewritten |

### 2.3 任务目标

模型需要判断文本是否具有 LLM 改写或生成特征，包括但不限于：

1. 原始文学风格是否被现代化；
2. 旧式英文、作者个人风格、方言或诗歌形式是否被平滑化；
3. 学术段落是否被改写得更加解释性、模板化、流畅化；
4. 词汇是否被替换为更现代、更抽象、更“高级”的表达；
5. 句法结构是否变得过度规整；
6. 文本是否缺少原文中的自然不规则性。

---

## 3. 项目总体方案

本项目采用一个混合式检测系统：

```text
输入文本
   │
   ├── 分支 A：DeBERTa-v3-base 神经分类器
   │       └── 输出 P_deberta(label=1)
   │
   ├── 分支 B：Word/Char TF-IDF + Logistic Regression
   │       └── 输出 P_tfidf(label=1)
   │
   └── 概率融合层
           └── P_final = α × P_deberta + (1 - α) × P_tfidf
                    │
                    └── threshold 判断，输出最终 label
```

最终模型名称建议：

**Hybrid DeBERTa-TFIDF Detector**

中文名称：

**DeBERTa 与多粒度 n-gram 特征融合的 LLM 文本检测器**

---

## 4. 为什么采用混合模型

### 4.1 单独使用 DeBERTa 的优势

DeBERTa-v3-base 能够建模上下文语义、句间关系和深层风格特征，适合识别：

- 文本是否过于平滑；
- 句式是否过于标准化；
- 语义表达是否像经过模型重述；
- 是否存在 LLM 常见的解释性扩写；
- 学术表达是否被改写为更模板化的形式。

### 4.2 单独使用 TF-IDF 的优势

老师测试集中的文本含有大量细粒度表层差异，例如：

- 古体拼写；
- 旧式标点；
- 诗歌换行；
- 破折号、分号、引号习惯；
- 特定作者的固定表达；
- LLM 改写后的现代同义词。

这些特征对字符级和词级 n-gram 非常友好。因此，Word/Char TF-IDF + 线性分类器可以作为强 baseline，也可以作为最终模型的辅助分支。

### 4.3 融合模型的动机

DeBERTa 负责深层语义和上下文风格判断，TF-IDF 负责表层词汇、拼写、标点和格式判断。两者互补，因此采用概率融合可以提高鲁棒性。

---

## 5. 数据集构建方案

### 5.1 基本原则

老师提供的 JSON 文件应被视为最终评测集，不应直接用于训练或调参。外部训练集需要自行构造，并尽量复刻评测集分布，但不能包含评测集中的原文或近似文本。

核心原则：

```text
训练集应接近测试集分布，但不能泄漏测试集内容。
```

### 5.2 标签定义

| 标签 | 数据来源 | 构造方式 |
|---|---|---|
| `0` | Human-written | 公开领域文学文本、诗歌、NLP 学术论文原文 |
| `1` | LLM-rewritten | 对 human 文本进行 paraphrase、modernize、rewrite、polish |

### 5.3 数据来源

#### 5.3.1 文学小说段落

建议来源：

- Project Gutenberg 公版英文小说；
- 19 世纪至 20 世纪早期英文文学；
- 与测试集风格相近但不重复的作者和作品。

推荐作者：

- Charles Dickens
- Thomas Hardy
- Jack London
- O. Henry
- Mark Twain
- Mary Shelley
- H. G. Wells
- Bret Harte
- Nathaniel Hawthorne

构造方式：

1. 下载公版小说文本；
2. 去除版权声明、目录、页眉页脚；
3. 按自然段切分；
4. 保留 80–400 words 的段落；
5. 每条原文作为 `label = 0`；
6. 对每条原文生成一条或多条 LLM 改写，作为 `label = 1`。

#### 5.3.2 诗歌与古体英文

建议来源：

- Project Gutenberg 诗歌集；
- public-domain poems；
- 古体英文或押韵文本。

构造方式：

1. 按 stanza 或 4–12 行切分；
2. human 原文标记为 `label = 0`；
3. 让 LLM 改写为现代诗、同风格诗或现代英文释义，标记为 `label = 1`。

建议比例：10%–15%。

#### 5.3.3 NLP / 语言学学术段落

建议来源：

- ACL Anthology；
- ACL OCL Corpus；
- 传统 NLP 论文段落。

优先主题：

- parsing；
- machine translation；
- word sense disambiguation；
- discourse；
- coreference；
- grammar induction；
- alignment；
- semantics；
- scope ambiguity。

构造方式：

1. 抽取 abstract、introduction、method 中的自然段；
2. 去除公式、表格、参考文献；
3. 保留 80–350 words 的段落；
4. 原文作为 `label = 0`；
5. 使用 LLM 进行 academic paraphrase 或 clarity rewrite，作为 `label = 1`。

#### 5.3.4 通用 AI 检测数据

可以少量加入公开 AI 检测语料，例如 HC3，但比例不应过高。

建议比例：不超过 10%–20%。

原因：老师测试集主要是文学和学术改写，不是普通问答文本。过多使用通用 AI 检测数据可能导致模型学到错误分布。

---

## 6. 推荐数据比例

建议构造一个约 4000–6000 条的外部训练语料。

| 子域 | Human 样本 | LLM 样本 | 总比例 |
|---|---:|---:|---:|
| 文学小说 | 1400 | 1400 | 50% |
| 学术 NLP 段落 | 840 | 840 | 30% |
| 诗歌 / 古体英文 | 280 | 280 | 10% |
| 通用 AI 检测数据 | 280 | 280 | 10% |
| 合计 | 2800 | 2800 | 100% |

如果时间紧，可以采用最小可行版本：

| 子域 | Human | LLM | 合计 |
|---|---:|---:|---:|
| 文学小说 | 600 | 600 | 1200 |
| 学术 NLP 段落 | 300 | 300 | 600 |
| 诗歌 / 古体英文 | 100 | 100 | 200 |
| 合计 | 1000 | 1000 | 2000 |

---

## 7. LLM 样本生成 Prompt 设计

### 7.1 文学段落改写 Prompt

```text
Rewrite the following passage in fluent contemporary English while preserving the meaning, scene, and narrative perspective. Do not summarize. Keep approximately the same length.

[PASSAGE]
```

```text
Paraphrase the following literary passage. Preserve the plot and imagery, but replace the original wording and sentence structure with natural modern prose.

[PASSAGE]
```

```text
Rewrite the passage as if it were a polished literary imitation. Keep the same events and mood, but use different names, wording, and sentence structures.

[PASSAGE]
```

### 7.2 古体英文现代化 Prompt

```text
Modernize the following archaic English passage while preserving its meaning and tone. Keep the output approximately the same length.

[PASSAGE]
```

### 7.3 诗歌改写 Prompt

```text
Rewrite the following poem in a similar poetic style with different wording. Preserve the imagery and emotional tone.

[POEM]
```

```text
Paraphrase the following poem into contemporary poetic English. Do not explain it.

[POEM]
```

### 7.4 学术段落改写 Prompt

```text
Paraphrase the following academic paragraph in a clear, polished research style. Preserve all technical meaning and keep approximately the same length.

[PARAGRAPH]
```

```text
Rewrite the following NLP research paragraph to improve clarity and fluency while preserving the original claims and terminology.

[PARAGRAPH]
```

---

## 8. 数据清洗与去重

### 8.1 文本清洗

保留：

- 原始大小写；
- 标点；
- 破折号；
- 引号；
- 分号；
- 古体拼写；
- 诗歌换行。

删除：

- 版权声明；
- 目录；
- 页眉页脚；
- 参考文献；
- 过短段落；
- 乱码严重段落；
- 空文本。

不要做过度 normalization，因为拼写、标点和格式本身就是重要特征。

### 8.2 与老师测试集去重

必须避免训练集和老师测试集重叠。

建议三种去重方式：

1. Exact match 去重；
2. Character 5-gram Jaccard similarity 去重；
3. Sentence embedding cosine similarity 去重。

建议阈值：

```text
char 5-gram Jaccard similarity > 0.35：删除
embedding cosine similarity > 0.85：删除
```

### 8.3 Pair-level split

同一条 human 原文及其所有 LLM 改写必须进入同一个 split。

错误划分示例：

```text
human 原文进入 train
LLM 改写进入 validation
```

这会造成泄漏。

正确划分示例：

```text
pair_id = gut_dickens_0001
human 与所有 rewrite 全部进入 train 或全部进入 validation
```

### 8.4 Source-level split

更严格的方式是按作品或论文划分：

```text
同一本小说的段落尽量只进入同一个 split
同一篇论文的段落尽量只进入同一个 split
```

这样可以避免模型记住具体作者或论文风格。

---

## 9. 数据格式

建议使用 JSONL 格式，每行一个样本：

```json
{"id":"gut_dickens_0001_human","text":"...","label":0,"domain":"literature","source":"gutenberg","pair_id":"gut_dickens_0001","generation":"human","prompt_type":"none"}
{"id":"gut_dickens_0001_llm","text":"...","label":1,"domain":"literature","source":"gpt-4.1","pair_id":"gut_dickens_0001","generation":"llm_rewrite","prompt_type":"modernize"}
```

字段说明：

| 字段 | 含义 |
|---|---|
| `id` | 样本唯一编号 |
| `text` | 文本内容 |
| `label` | 0 或 1 |
| `domain` | literature / poetry / academic / general |
| `source` | 数据来源 |
| `pair_id` | 原文与改写对应编号 |
| `generation` | human / llm_rewrite |
| `prompt_type` | 使用的改写 prompt 类型 |

---

## 10. 数据划分

外部构造数据划分为：

```text
train: 70%
validation: 15%
internal test: 15%
```

老师 JSON 保留为 final evaluation set。

完整流程：

```text
外部数据
   ├── train：训练模型
   ├── validation：调参、选 threshold、选 ensemble 权重
   └── internal test：内部稳定性评估

老师 JSON
   └── final test：最终只评估一次
```

---

## 11. Baseline 设计

### 11.1 Baseline 0：Majority Class

如果类别平衡，则准确率约为 50%。

```text
Always predict label = 0 or label = 1
```

用途：最弱基线。

### 11.2 Baseline 1：Handcrafted Features + Logistic Regression

人工特征包括：

- 文本长度；
- 句子数量；
- 平均句长；
- 平均词长；
- type-token ratio；
- 标点比例；
- 逗号数量；
- 分号数量；
- 破折号数量；
- 引号数量；
- 大写词比例；
- 重复 bigram/trigram 比例。

分类器：

```text
Logistic Regression
```

用途：可解释 baseline。

### 11.3 Baseline 2：Word TF-IDF + Logistic Regression

配置：

```python
TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    lowercase=True,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

分类器：

```python
LogisticRegression(
    C=1.0,
    max_iter=3000,
    class_weight="balanced"
)
```

### 11.4 Baseline 3：Word + Char TF-IDF + Logistic Regression

Word TF-IDF：

```python
TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    lowercase=True,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

Char TF-IDF：

```python
TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    lowercase=True,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

拼接后输入 Logistic Regression。

这是本项目的主 baseline。

---

## 12. 最终模型架构

### 12.1 总体结构

```text
Input Text
   │
   ├── Branch A: DeBERTa-v3-base
   │       ├── Tokenizer
   │       ├── Transformer Encoder
   │       ├── Dropout
   │       ├── Linear Classification Head
   │       └── Softmax → P_deberta(label=1)
   │
   ├── Branch B: TF-IDF Linear Classifier
   │       ├── Word-level TF-IDF
   │       ├── Character-level TF-IDF
   │       ├── Feature Concatenation
   │       ├── Logistic Regression
   │       └── Predict Proba → P_tfidf(label=1)
   │
   └── Probability Fusion
           ├── P_final = α × P_deberta + (1 - α) × P_tfidf
           └── if P_final ≥ threshold: label = 1 else label = 0
```

### 12.2 DeBERTa 分支

模型：

```text
microsoft/deberta-v3-base
```

输入：

```text
text
```

Tokenizer 参数：

```python
max_length = 512
padding = "max_length"
truncation = True
```

分类头：

```text
DeBERTa encoder output
   ↓
Dropout
   ↓
Linear(hidden_size, 2)
   ↓
Softmax
```

输出：

```text
P_deberta(label=1)
```

### 12.3 TF-IDF 分支

Word n-gram：

```text
1-gram and 2-gram
```

Character n-gram：

```text
3-gram to 5-gram
```

分类器：

```text
Logistic Regression
```

输出：

```text
P_tfidf(label=1)
```

### 12.4 融合层

融合公式：

```text
P_final = α × P_deberta + (1 - α) × P_tfidf
```

推荐初始值：

```text
α = 0.7
threshold = 0.5
```

最终值通过 validation set 搜索确定。

---

## 13. 训练流程

### 13.1 Step 1：准备外部训练数据

1. 收集 Gutenberg 文学段落；
2. 收集 public-domain 诗歌；
3. 收集 ACL/NLP 学术段落；
4. 清洗文本；
5. 过滤长度不合适文本；
6. 使用多种 prompt 生成 LLM 改写；
7. 保存为 JSONL；
8. 与老师测试集去重；
9. 按 pair_id 或 source 划分 train/validation/internal test。

### 13.2 Step 2：训练 TF-IDF 分支

1. 读取 train 文本和 label；
2. 使用 word-level TF-IDF 提取词级特征；
3. 使用 char-level TF-IDF 提取字符级特征；
4. 拼接两种稀疏矩阵；
5. 训练 Logistic Regression；
6. 在 validation set 上输出 `P_tfidf`；
7. 保存 vectorizer 和 classifier。

保存文件：

```text
artifacts/tfidf_word.pkl
artifacts/tfidf_char.pkl
artifacts/tfidf_lr.pkl
```

### 13.3 Step 3：训练 DeBERTa 分支

推荐参数：

```text
model_name = microsoft/deberta-v3-base
max_length = 512
learning_rate = 2e-5
batch_size = 8 或 16
epochs = 3 到 5
weight_decay = 0.01
warmup_ratio = 0.1
metric_for_best_model = f1
```

如果显存不足：

```text
batch_size = 4
gradient_accumulation_steps = 2 或 4
fp16 = True
```

训练过程：

1. 加载 tokenizer；
2. tokenize train 和 validation 文本；
3. 初始化 DeBERTaForSequenceClassification；
4. 设置 CrossEntropyLoss；
5. 使用 AdamW 优化器；
6. 每个 epoch 在 validation set 上评估；
7. 保存 validation F1 最高的 checkpoint。

保存文件：

```text
artifacts/deberta_model/
artifacts/deberta_tokenizer/
```

### 13.4 Step 4：融合权重调参

在 validation set 上获得：

```text
P_deberta
P_tfidf
true_label
```

搜索：

```python
alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
thresholds = [0.45, 0.50, 0.55, 0.60]
```

选择 F1 最高的组合。

保存：

```json
{
  "alpha": 0.7,
  "threshold": 0.5
}
```

### 13.5 Step 5：内部测试

在 internal test set 上评估：

- Accuracy；
- Precision；
- Recall；
- F1；
- ROC-AUC；
- Confusion Matrix。

### 13.6 Step 6：最终测试

在老师 JSON 上只运行一次最终模型。

禁止：

- 使用老师 JSON 训练；
- 使用老师 JSON 调 threshold；
- 根据老师 JSON 结果反复改模型；
- 将老师 JSON 拆分为训练集和测试集，除非老师明确允许。

---

## 14. 评价指标

主要指标：

```text
Accuracy
Precision
Recall
F1-score
```

建议主指标：

```text
F1-score
```

因为二分类检测中，precision 和 recall 同样重要。

混淆矩阵：

|  | Pred Human | Pred LLM |
|---|---:|---:|
| True Human | TN | FP |
| True LLM | FN | TP |

错误类型：

- FP：人类文本被误判为 LLM；
- FN：LLM 改写文本被误判为人类。

---

## 15. 错误分析计划

### 15.1 False Positive 分析

Human 文本被误判为 LLM 的可能原因：

1. 文本本身过于正式；
2. 学术段落结构规整；
3. 文学文本语言高度流畅；
4. 原作者风格与 LLM 改写风格相似；
5. 段落过短，缺乏足够风格信号。

### 15.2 False Negative 分析

LLM 文本被误判为 Human 的可能原因：

1. LLM 很好地模仿了旧式英文；
2. 改写幅度较小；
3. 文本保留大量原文词汇；
4. 诗歌或短文本信息不足；
5. 生成文本中包含非标准标点或古体词。

### 15.3 按子域分析

分别统计：

- literature；
- poetry；
- archaic English；
- academic NLP；
- general AI text。

查看模型在哪类文本上表现最好或最差。

---

## 16. 模型部署方案

### 16.1 部署目标

部署一个简单 API，输入文本，返回：

```json
{
  "label": 1,
  "label_name": "LLM-generated",
  "probability": 0.83,
  "p_deberta": 0.87,
  "p_tfidf": 0.74
}
```

### 16.2 推荐部署方式

使用 FastAPI：

```text
FastAPI backend
   ├── load DeBERTa model
   ├── load tokenizer
   ├── load TF-IDF vectorizers
   ├── load Logistic Regression classifier
   ├── load fusion config
   └── expose /predict endpoint
```

### 16.3 API 设计

#### POST `/predict`

请求：

```json
{
  "text": "Input passage here..."
}
```

返回：

```json
{
  "label": 1,
  "label_name": "LLM-generated",
  "probability": 0.83,
  "p_deberta": 0.87,
  "p_tfidf": 0.74
}
```

### 16.4 部署目录结构

```text
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── final/
├── scripts/
│   ├── collect_gutenberg.py
│   ├── collect_acl.py
│   ├── generate_llm_rewrites.py
│   ├── clean_data.py
│   ├── deduplicate.py
│   └── split_data.py
├── training/
│   ├── train_tfidf.py
│   ├── train_deberta.py
│   ├── tune_ensemble.py
│   └── evaluate.py
├── app/
│   ├── main.py
│   └── predictor.py
├── artifacts/
│   ├── tfidf_word.pkl
│   ├── tfidf_char.pkl
│   ├── tfidf_lr.pkl
│   ├── deberta_model/
│   ├── deberta_tokenizer/
│   └── fusion_config.json
├── requirements.txt
└── README.md
```

---

## 17. 完整实现步骤

### 阶段一：理解任务与测试集

1. 阅读老师提供的任务要求；
2. 阅读老师给定 JSON；
3. 确认 `label = 0` 和 `label = 1` 的含义；
4. 分析文本类型；
5. 统计 label 分布；
6. 统计文本长度分布；
7. 人工查看若干 human 与 LLM 样本；
8. 得出测试集主要是文学/学术改写检测任务。

### 阶段二：构造外部训练集

1. 选择 Project Gutenberg 作品；
2. 下载文本；
3. 清洗版权和目录；
4. 按自然段切分；
5. 过滤长度；
6. 收集诗歌段落；
7. 收集 NLP 学术段落；
8. 对每条 human 文本分配 `pair_id`；
9. 使用多种 prompt 生成 LLM 改写；
10. 保存所有样本为 JSONL；
11. 与老师 JSON 做 exact match 去重；
12. 做近似去重；
13. 按 pair_id 或 source 划分 train/validation/internal test。

### 阶段三：实现 baseline

1. 实现 Majority baseline；
2. 实现 handcrafted feature extraction；
3. 训练 Logistic Regression；
4. 实现 Word TF-IDF；
5. 训练 Word TF-IDF + Logistic Regression；
6. 实现 Word + Char TF-IDF；
7. 训练 Word + Char TF-IDF + Logistic Regression；
8. 在 validation 和 internal test 上记录指标。

### 阶段四：实现 DeBERTa

1. 安装 transformers、datasets、evaluate；
2. 加载 `microsoft/deberta-v3-base`；
3. 编写 Dataset 类；
4. tokenize 数据；
5. 设置训练参数；
6. 微调 DeBERTa；
7. 保存最佳 checkpoint；
8. 在 validation/internal test 上评估。

### 阶段五：实现融合模型

1. 在 validation set 上生成 `P_deberta`；
2. 在 validation set 上生成 `P_tfidf`；
3. 搜索 alpha 和 threshold；
4. 保存最佳融合配置；
5. 在 internal test 上评估融合模型；
6. 与单模型结果比较。

### 阶段六：最终评估

1. 加载老师 JSON；
2. 不训练、不调参；
3. 使用最终融合模型预测；
4. 计算最终指标；
5. 输出混淆矩阵；
6. 挑选错误案例；
7. 分析模型失误原因。

### 阶段七：部署

1. 编写 `predictor.py`；
2. 加载 DeBERTa、tokenizer、TF-IDF、LR、fusion config；
3. 编写 `predict(text)` 函数；
4. 使用 FastAPI 封装 `/predict`；
5. 本地启动服务；
6. 用样本文本测试 API；
7. 可选：制作简单网页输入框。

### 阶段八：报告与展示

1. 写任务背景；
2. 描述测试集特点；
3. 描述外部训练集构造；
4. 描述 baseline；
5. 描述 DeBERTa；
6. 描述融合方法；
7. 展示实验表格；
8. 展示混淆矩阵；
9. 展示错误案例；
10. 总结模型优缺点。

---

## 18. 推荐实验表格

| Model | Accuracy | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---:|---|
| Majority Class |  |  |  |  | Weak baseline |
| Handcrafted + LR |  |  |  |  | Interpretable baseline |
| Word TF-IDF + LR |  |  |  |  | Lexical baseline |
| Word + Char TF-IDF + LR |  |  |  |  | Main traditional baseline |
| DeBERTa-v3-base |  |  |  |  | Neural model |
| DeBERTa + TF-IDF Ensemble |  |  |  |  | Final model |

---

## 19. 预期结论

预期结果可能是：

1. Majority baseline 只能达到约 50%；
2. Handcrafted features 有一定效果，但不够稳定；
3. Word TF-IDF 可以捕捉 LLM 常见词汇替换；
4. Char TF-IDF 对古体英文、诗歌和标点非常有效；
5. DeBERTa 在深层语义和风格判断上表现较好；
6. DeBERTa + TF-IDF 融合模型整体最稳健；
7. 模型最容易混淆的是高度正式的人类学术文本和高质量仿写的 LLM 文本。

---

## 20. 项目风险与解决方案

### 风险 1：训练集和测试集分布不一致

解决：训练集重点构造文学、诗歌、旧式英文和 NLP 学术改写数据。

### 风险 2：测试集泄漏

解决：老师 JSON 只用于最终评估；所有调参都在 validation set 完成。

### 风险 3：LLM 改写 prompt 单一

解决：使用多种 prompt 和多个生成温度，避免模型只学到单一模板。

### 风险 4：DeBERTa 过拟合

解决：使用 validation early stopping、weight decay、dropout、source-level split。

### 风险 5：显存不足

解决：使用 `batch_size=4`、`gradient_accumulation_steps=4`、`fp16=True`。

---

## 21. 最终交付物

项目最终应包含：

1. 数据构造脚本；
2. 清洗和去重脚本；
3. 训练集、验证集、内部测试集；
4. TF-IDF baseline 代码；
5. DeBERTa 训练代码；
6. 融合模型代码；
7. 最终评估脚本；
8. 模型部署 API；
9. 实验结果表格；
10. 错误分析；
11. 项目报告；
12. 展示 PPT。

---

## 22. 一句话总结

本项目的核心不是简单训练一个 AI 文本检测器，而是构建一个专门针对“英文文学与学术文本 LLM 改写”的检测系统。最终模型采用 DeBERTa 捕捉深层语义与风格变化，同时使用 Word/Char TF-IDF 捕捉拼写、标点、古体英文和 n-gram 表层线索，并通过概率融合完成最终判断。
