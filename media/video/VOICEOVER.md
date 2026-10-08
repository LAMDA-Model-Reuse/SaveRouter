# SAVERouter explainer voiceover

The generated videos use seven scenes and run for approximately one minute.
Quantitative wording follows `docs/PUBLIC_CLAIMS.md`.

## English

1. LLM routing can cut serving costs. But before a router saves anything, it
   must first pay for supervision.
2. Training data is not free. Every candidate model run on every historical
   query adds an upfront bill that conventional evaluations often leave out.
3. SAVERouter changes the acquisition process. For each training query, it
   selects exactly K informative model outcomes instead of filling the entire
   query-by-model matrix.
4. It then shares capability evidence across related query groups, while
   learning query-level residual corrections for fine-grained routing
   decisions.
5. We evaluate the full economics with two metrics. SA-BEP asks when
   serving-time savings repay supervision. SA-CR measures amortized cost at a
   deployment horizon.
6. Across four routing benchmarks, SAVERouter uses about thirty-three to
   forty-one percent of available training feedback, while reducing break-even
   volume by approximately one point nine to nine point five times versus the
   fastest conventional router.
7. Read the paper, explore payback interactively, or install SAVERouter from
   PyPI. Routing should pay for itself.

## 中文

1. 大模型路由可以降低推理成本。但在省下第一分钱之前，路由器必须先为监督数据买单。
2. 训练数据并不免费。让每个候选模型回答每条历史查询，会产生一笔可观的前期成本，而传统评测往往忽略了它。
3. SAVERouter 重新设计了反馈获取过程。对每条训练查询，它只选择 K 个最有信息量的模型结果，不再填满整个查询模型矩阵。
4. 随后，它在相关查询组之间共享能力信息，同时学习查询级残差，保留细粒度的路由决策。
5. 我们用两个指标衡量完整经济性。SA-BEP 回答节省何时覆盖监督成本；SA-CR 衡量指定部署规模下的摊销成本。
6. 在四个路由基准上，SAVERouter 只使用约百分之三十三到四十一的训练反馈。相比最快的传统路由器，回本部署量降低约一点九到九点五倍。
7. 阅读论文，在线探索回本曲线，或者从 PyPI 安装 SAVERouter。让路由真正为自己买单。
