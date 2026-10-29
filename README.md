# DrawOffer-Eval

DrawOffer 长期求职记忆引擎的独立测评。引擎根据一位求职者的简历、岗位、模拟面试与真实面试记录，推断他在每个知识点上能答到多深、简历说法是否夸大、经历讲不讲得清、离目标岗位还差什么，并预测下一道题能否答好。

本仓库只放测评代码、方法、剧本与结果；引擎本身在 DrawOffer 主仓库（`drawoffer.memory`，私有），以可编辑依赖引入。没有主仓库时可以阅读方法、剧本与结果，但不能重跑。

结果见 [reports/RESULTS.md](reports/RESULTS.md)。

## 测了什么

| 编号 | 测评 | 命令 | 是否调用模型 |
| --- | --- | --- | --- |
| 1 | 与通用记忆正面对比：DrawOffer 引擎、Mem0 + 大模型、整段记录交给大模型 | `python -m dreval.benchmarks.comparison` | 是 |
| 3 | 稳定性与可追溯性：重复推断一致、删除来源后无残留、原话真在原文里 | `python -m dreval.benchmarks.stability` | 否（读取缓存的观察器输出） |
| 6 | 性能：几百到几千次作答的长历史，全量推断与检查点续算的耗时 | `python -m dreval.benchmarks.performance` | 否 |

第 2 项（公开知识追踪数据集 ASSISTments、EdNet、Junyi 上对比 BKT、IRT、DKT）、第 4 项（剧本扩到 100 份以上）、第 5 项（真实数据校准）尚未完成，见文末“后续”。

## 数据

- **虚构求职者剧本**：DrawOffer 金标集的 25 份剧本，公开副本在 [data/personas](data/personas)，每份写明各知识点随时间的真实水平、经历的真实分量、简历说法的真伪，以及 5–10 个检查点（某天系统应该得出什么结论）。全部虚构，金标集的 `gold check` 会扫描联系方式等个人信息。
- **模拟来源**（留在 DrawOffer 仓库，不公开）：每份剧本按产品的真实流程生成的记录（简历、岗位、模拟面试逐题问答与评分、真实面试复盘），面试官与评分使用产品自己的提示词，求职者由另一家厂商的模型按剧本当天的真实水平扮演。三种做法读到的是同一批记录。
- **划分**：开发集 16 份（p01–p16，其中 p01、p02、p03、p05、p07、p08、p09、p12、p13 原属测试集，调参时看过结果，已转入开发集），测试集 9 份封存，只在最终评测时用 `--split test --final` 打开一次。

## 对比方法（第 1 项）

三种做法读同一批记录、回答同一批检查点，用金标集同一个判分函数（`drawoffer.memory.gold.personas.evaluate`）判分：

- **DrawOffer 引擎**：观察器（模型）把每条记录读成结构化观察并逐字摘录原话，推断（不调用模型）据此给出全部结论。检查点只从结论里读，引擎事先不知道会被问什么。观察器模型 `deepseek-v4.1-flash`。
- **整段记录交给大模型**：到检查点那天为止的全部记录按时间排好，连同检查点的问题一起交给 `gpt-6.1-terra`（Codex CLI，推理强度 medium），要求结构化作答并逐字摘录原话。
- **Mem0 + 大模型**：开源 `mem0ai` 2.2.1，逐条写入记录（Mem0 用同一个 `gpt-6.1-terra` 抽取记忆），删除的来源到期时删除对应记忆；作答时每个问题检索 20 条记忆，模型只看检索结果作答。嵌入在本机运行（FastEmbed `BAAI/bge-small-zh-v1.5`），向量库为本地 Qdrant。

四个指标：

- **预测下一题 AUC**：引擎回放时，在吸收每道知识题之前预测“能通过第 L 层”的概率；两个基线对同样的题、同样的层级给概率，只看这道题之前的记录。标签是产品评分器的判定。AUC 的区间按题重抽样（同一题的几个层级一起抽）。
- **能力判断准确率**：`ability_level` 检查点，允许差一级；证据不足时必须答“证据不足”。另报全部检查点、按类型的通过率，带 95% Wilson 区间。
- **结论有原话依据**：基线的每个回答都要附原话，原话（按省略号分段）全部能在原始记录里找到才算。引擎的每条结论，所依据的观察带有原话且全部在原文中，或结论只依据结构化记录（简历行、岗位、评分器给的深度）且指向的来源存在，才算。
- **一致性**：同一输入跑 5 次，每次都是独立的模型采样（不读缓存）。引擎这边是观察器重新采样、端到端重算；基线这边是作答重新采样，Mem0 连记忆抽取也重新采样。报告“5 次结论（通过与否）相同”和“5 次答案逐字相同”。

### 对比中偏向基线的地方

- **基线被直接问到考点**：基线被问“X 现在什么水平”“是否存在 Y 问题”，引擎则要自己发现所有问题。
- **基线用的模型更强**：`gpt-6.1-terra` 强于引擎观察器的 `deepseek-v4.1-flash`。
- **基线只回答被问的检查点**：没有“误报别的问题”的机会，引擎的误报会在 `issue_absent` 检查点上被扣分。

所以基线的分数是偏高的估计。

### 其他限制

- 剧本仍是开发方起草的初稿（`status=draft`），还没有经过两位标注人核对。
- 样本量小：16 份开发集剧本、约 100 个检查点，区间较宽，所以每个数都附了区间。
- AUC 只取第 1 次运行（每个基线每道题调用一次模型），一致性只在检查点上测。
- Mem0 的实体链接依赖 spaCy 英文模型，对中文没有作用，未启用；它的记忆抽取与向量检索照常工作。Mem0 2.2.1 的写入流程只做新增，旧记忆不会被更新或合并，这是该版本的设计。
- 预测标签来自产品评分器，不是人工判定；评分器本身的误差在 DrawOffer 主仓库另有测评。

## 稳定性与可追溯性（第 3 项）

观察器输出固定（读取缓存），测的是推断本身：

- **确定性**：每个检查点日期推断 5 次，比较后验均值与协方差、全部问题及其状态历史、全部结论及其依据的哈希；再在 2 个新进程里用不同的 `PYTHONHASHSEED` 推断，比较同样的哈希。
- **删除无残留**：逐一删除每个来源（一份简历、一场面试……），按产品的做法让它的片段从此不再参与推断，检查：只依赖它的结论是否全部消失、是否还有结论引用它的片段或观察、是否还有来自它的观察。
- **原话**：观察里的每条原话，到它所指的片段原文中查找（忽略空白、标点、全半角）。模拟面试评分器会用省略号把几段摘录拼成一条，按段核对，也报告整句直接匹配的条数。

## 性能（第 6 项）

用引擎自带的生成模型（`drawoffer.memory.sim`）造出模拟学生，在三个目标技术的知识点上随机答题，每场 6 题、每两天一场，历史长度从 10 场到 500 场（3000 次作答、6000 条观察）。每档 5 个学生，每人计时 3 次：

- 引擎 1（默认）全量推断；
- 引擎 2 全量推断，以及从上一场结束时的检查点续算（产品保存检查点，新一场面试只需算这一场）；续算结果先与全量结果逐位比对，一致才计时。

## 运行

需要：Python 3.12+、[uv](https://docs.astral.sh/uv/)、DrawOffer 主仓库（私有）检出在本仓库的同级目录 `../DrawOffer`；第 1 项另需 Codex CLI（已登录）和主仓库开发环境里配置的观察器模型路由。

本次结果对应 DrawOffer 提交 `cd08cfe4e`：

```bash
git -C ../DrawOffer checkout cd08cfe4e
```

```bash
uv sync
```

测评读取的是 DrawOffer 金标集里的剧本与模拟来源；本仓库 `data/personas/` 是剧本的公开副本，供阅读。两者是否一致：

```bash
uv run python -m dreval.datacheck
```

`.env`（不提交）写明主仓库读取哪份配置：

```bash
DRAWOFFER_ENV_FILE=/path/to/drawoffer/.env.development
```

```bash
set -a; . ./.env; set +a
uv run python -m dreval.benchmarks.stability --split dev --repeats 5 --out runs/stability-dev.json
uv run python -m dreval.benchmarks.performance --sessions 10,50,100,200,500 --learners 5 --out runs/performance.json
uv run python -m dreval.benchmarks.comparison --split dev --runs 5 --jobs 6 --out runs/comparison-dev.json
uv run python -m dreval.report --comparison runs/comparison-dev.json --stability runs/stability-dev.json --performance runs/performance.json
```

模型输出缓存在 `cache/`（按请求哈希，JSON Lines），重跑同样的命令不再调用模型；`DREVAL_CACHE_ONLY=1` 时缺缓存直接报错，可证明一次重跑没有花钱。缓存含模拟记录原文，不提交。

可调的环境变量：`DREVAL_BASELINE_MODEL`（默认 `gpt-6.1-terra`）、`DREVAL_BASELINE_EFFORT`（默认 `medium`）、`DREVAL_ENGINE_MODEL`（默认 `deepseek-v4.1-flash`）、`DREVAL_GOLD_ROOT`（金标集目录）。

## 后续

- **第 2 项**：ASSISTments 2009、EdNet-KT1、Junyi 上跑推断算法，与 BKT（pyBKT）、IRT、DKT 比 AUC。需要把数据集的“题目—知识点—对错”接成引擎的观察，并新增依赖。
- **第 4 项**：剧本扩到 100 份以上，按专业方向与水平分层报告。新剧本要先写真实情况与检查点，再按产品流程模拟来源（每份剧本数十次模型调用）。
- **第 5 项**：DrawOffer 已实现面试前预测存档、面试后勾选与校准报表（`MEMORY_CHECK_ENABLED`，默认关闭），打开线上开关后即可开始积累；本仓库届时读取匿名化的汇总指标。
