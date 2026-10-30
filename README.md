# DrawOffer-Eval

DrawOffer 长期求职记忆引擎的独立测评。引擎根据一位求职者的简历、岗位、模拟面试与真实面试记录，推断他在每个知识点上能答到多深、简历说法是否夸大、经历讲不讲得清、离目标岗位还差什么，并预测下一道题能否答好。

本仓库只放测评代码、方法、剧本与结果；引擎本身在 DrawOffer 主仓库（`drawoffer.memory`，私有），以可编辑依赖引入。没有主仓库时可以阅读方法、剧本与结果，但不能重跑。

结果见 [reports/RESULTS.md](reports/RESULTS.md)，原始结果 JSON 在 [results/](results)。

## 测了什么

| 编号 | 测评 | 命令 | 是否调用模型 |
| --- | --- | --- | --- |
| 1 | 与通用记忆正面对比：DrawOffer 引擎、Mem0 + 大模型、整段记录交给大模型 | `python -m dreval.benchmarks.comparison` | 是 |
| 2 | 公开知识追踪数据集 ASSISTments 2009、EdNet-KT1、Junyi：引擎与 BKT、IRT、PFA、DKT 比预测下一题，后两个带时间戳，测遗忘模型 | `scripts/kt-parallel.sh` | 否 |
| 3 | 稳定性与可追溯性：重复推断一致、删除来源后无残留、原话真在原文里 | `python -m dreval.benchmarks.stability` | 否（读取缓存的观察器输出） |
| 6 | 性能：几百到几千次作答的长历史，全量推断与检查点续算的耗时 | `python -m dreval.benchmarks.performance` | 否 |

第 4 项（剧本扩到 100 份以上）、第 5 项（真实数据校准）尚未完成，见文末“后续”。

## 数据

- **虚构求职者剧本**：DrawOffer 金标集的 25 份剧本，公开副本在 [data/personas](data/personas)，每份写明各知识点随时间的真实水平、经历的真实分量、简历说法的真伪，以及 5–10 个检查点（某天系统应该得出什么结论）。全部虚构，金标集的 `gold check` 会扫描联系方式等个人信息。
- **模拟来源**（留在 DrawOffer 仓库，不公开）：每份剧本按产品的真实流程生成的记录（简历、岗位、模拟面试逐题问答与评分、真实面试复盘），面试官与评分使用产品自己的提示词，求职者由另一家厂商的模型按剧本当天的真实水平扮演。三种做法读到的是同一批记录。
- **划分**：开发集 16 份（p01–p16，其中 p01、p02、p03、p05、p07、p08、p09、p12、p13 原属测试集，调参时看过结果，已转入开发集），测试集 9 份（p17、p18、p19、p20、p22、p25、p27、p29、p30）在开发期间封存，2026-10-30 用 `--final` 打开一次，结果见报告；开封后没有再调整引擎、提示词或评测代码。

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
- 样本量小：全部 25 份剧本、129 个检查点，测试集只有 9 份、45 个检查点，区间较宽，所以每个数都附了区间。
- AUC 只取第 1 次运行（每个基线每道题调用一次模型），一致性只在检查点上测。
- Mem0 的实体链接依赖 spaCy 英文模型，对中文没有作用，未启用；它的记忆抽取与向量检索照常工作。Mem0 2.2.1 的写入流程只做新增，旧记忆不会被更新或合并，这是该版本的设计。
- 预测标签来自产品评分器，不是人工判定；评分器本身的误差在 DrawOffer 主仓库另有测评。

## 公开知识追踪数据集（第 2 项）

数据（`src/dreval/kt/data.py`，都从 USTC 镜像下载并校验 sha256，不放进本仓库）：

- **ASSISTments 2009–2010 skill builder**（corrected 版）：去掉没有知识点的行，同一 `order_id` 挂多个知识点的合成一次作答，至少 2 次作答的学生全部保留。没有时间戳。
- **EdNet-KT1**：每道题的知识点取 `questions.csv` 的标签（可以有多个），`user_answer` 等于正确答案算答对，时间取 `timestamp`。
- **Junyi Academy 数学练习日志**：按常用的做法，知识点与题目都取练习（`exercise`），首次作答且没看提示算答对，时间取 `time_done`。

EdNet 与 Junyi 的学生太多，按种子 0 从至少 20 次作答的学生中随机抽 5000 名，每人只取最早的 500 次作答。

按学生 5 折交叉验证，每个模型只在训练折的学生上拟合，对测试折学生的每次作答只用他之前的作答预测，报告 AUC、RMSE、log loss：

- **引擎**：知识点挂在一个虚构的技术节点下，每次作答是一条 `ability` 观察（区分度 1，多知识点平分），带上作答的真实时间，预测取引擎吸收这条作答之前给出的通过概率。四个版本：产品参数且所有难度为 0（不用训练集）；从训练集估每个知识点的难度；同上但关闭遗忘（稳定度设为无穷大）；对照组用 IRT 估出的每道题难度。
- **IRT**：Rasch 模型，训练集上 L2 惩罚的极大似然估题目难度，测试学生的能力从总体先验开始、每答一题按拉普拉斯近似更新。训练集没见过的题用其知识点的平均难度。
- **PFA**：每个知识点的截距与此前成功、失败次数的系数（次数取 log(1 + n)），scikit-learn 逻辑回归。
- **BKT**：pyBKT 1.4.3 标准模型，每个知识点一个，5 次随机初始化的 EM。多知识点的作答取各知识点预测的平均。
- **DKT**：LSTM（隐层 100），输入为知识点 × 对错的多热编码，训练集留 10% 学生按 AUC 早停。

限制与说明：

- ASSISTments 没有时间戳，作答按 `order_id` 排序、每题间隔 1 分钟，引擎的遗忘模型在这里不起作用；EdNet 与 Junyi 用真实时间。遗忘的作用看“隔天后”一列：所涉知识点距上次练习都已至少 1 天的作答，以及同一引擎打开与关闭遗忘的逐折配对差。
- 遗忘与稳定度用的是产品参数，没有在这些数据上调。
- 引擎在产品里不会从其他用户身上学题目难度，“题目难度”一行是对照，用来说明与 IRT、DKT 的差距主要来自每道题的难度，不是产品能力。
- pyBKT 1.4.3 与 NumPy 2.x 不兼容（把 (1, 1) 数组写进标量元素），引擎依赖 NumPy 2.5，所以运行时把这个值转成标量（`baselines._patch_pybkt`），拟合本身不变；它的 `predict` 会重排数据框，预测按每行的键取回。
- pyBKT 在没有 C++ 扩展时用纯 Python 做 EM 的 E 步，EdNet 一折要几个小时。`src/dreval/kt/fast_bkt.py` 是同一段前向后向计算的 numba 版本，逐句对应、运算顺序不变，运行时替换 pyBKT 的 `inner`。`tests/test_kt.py` 检查两者的预测逐位相同；ASSISTments 第 4 折用两种版本各跑一次，AUC、RMSE、log loss 逐位相同（1226 秒对 28 秒）。
- 需要可选依赖：`uv sync --extra kt`（pyBKT、numba、scikit-learn 1.5、PyTorch、pandas）。

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
uv run python -m dreval.benchmarks.comparison --split all --final --runs 5 --jobs 6 --out runs/comparison-all.json
uv run python -m dreval.benchmarks.stability --split all --repeats 5 --out runs/stability-all.json
JOBS=6 scripts/kt-parallel.sh   # 第 2 项：每个（折, 模型）一个进程，结果写 runs/kt-cells-assist09/，再合并为 runs/kt-assist09.json
DATASET=ednet JOBS=4 scripts/kt-parallel.sh
DATASET=junyi JOBS=4 scripts/kt-parallel.sh
uv run python -m dreval.report --kt runs/kt-assist09.json --kt runs/kt-ednet.json --kt runs/kt-junyi.json --comparison runs/comparison-test.json --comparison runs/comparison-all.json --comparison runs/comparison-dev.json --stability runs/stability-test.json --stability runs/stability-all.json --performance runs/performance.json
```

模型输出缓存在 `cache/`（按请求哈希，JSON Lines），重跑同样的命令不再调用模型；`DREVAL_CACHE_ONLY=1` 时缺缓存直接报错，可证明一次重跑没有花钱。缓存含模拟记录原文，不提交。

可调的环境变量：`DREVAL_BASELINE_MODEL`（默认 `gpt-6.1-terra`）、`DREVAL_BASELINE_EFFORT`（默认 `medium`）、`DREVAL_ENGINE_MODEL`（默认 `deepseek-v4.1-flash`）、`DREVAL_GOLD_ROOT`（金标集目录）。

## 后续

- **第 4 项**：剧本扩到 100 份以上，按专业方向与水平分层报告。新剧本要先写真实情况与检查点，再按产品流程模拟来源（每份剧本数十次模型调用）。
- **第 5 项**：DrawOffer 已实现面试前预测存档、面试后勾选与校准报表（`MEMORY_CHECK_ENABLED`，默认关闭），打开线上开关后即可开始积累；本仓库届时读取匿名化的汇总指标。
