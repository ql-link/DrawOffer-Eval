# 虚构求职者剧本

25 份剧本，复制自 DrawOffer 金标集（`apps/backend/tests/memory_gold/personas/<ID>/persona.json`，DrawOffer 提交 `cd08cfe4e`）。全部虚构：公司与学校是明显虚构的名字，不含联系方式；金标集的 `gold check` 会检查这些。

每份剧本包含：

- `truth`：答案。各知识点随时间的真实水平、经历的真实角色与分量、简历说法的真伪；
- `timeline`：事件与规格（简历全文、岗位描述、模拟面试的考察点、真实面试的问题、用户更正、删除）；
- `checkpoints`：5–10 个检查点，某天结束时系统应得出的结论；
- `split`：`dev` 或 `test`（无此字段时按编号哈希划分）。

测评实际读取的是按剧本模拟出的记录（`sources.json`）。它们由产品的面试官与评分提示词生成，留在 DrawOffer 仓库，不在这里。
