# SAVERouter Public Claims

This file is the source of truth for quantitative and methodological claims
used in the README, project page, release notes, talks, and social posts. Keep
the wording qualified as written below and update all public surfaces together
when the paper changes.

## Canonical identity

- **Name:** SAVERouter
- **Paper title:** *Routing Should Pay for Itself: Sparse Supervision for
  Economical LLM Routing*
- **Paper:** <https://arxiv.org/abs/2609.37402>
- **Repository:** <https://github.com/LAMDA-Model-Reuse/SaveRouter>
- **Project page:** <https://lamda-model-reuse.github.io/SaveRouter/>
- **Colab:** <https://colab.research.google.com/github/LAMDA-Model-Reuse/SaveRouter/blob/main/examples/saverouter_colab.ipynb>
- **One-line description:** SAVERouter learns an economical LLM router from
  sparse query-model feedback while accounting for the upfront cost of
  acquiring that supervision.

## Approved headline claims

1. **Feedback use.** In the main setting, SAVERouter uses approximately
   **33--41% of the available training feedback** across four routing
   benchmarks while maintaining competitive or better routing quality.
2. **Earlier payback.** At the target quality, SAVERouter reduces the
   break-even deployment volume by approximately **1.9--9.5x** compared with
   the fastest conventional fully supervised router.
3. **Evaluation scope.** The four benchmarks are **LLMRouterBench,
   Mixinstruct, MMR-Bench, and RouterBench**. The main experiments use an
   in-domain 20%/80% train/test split with seed 42 and `K = 4` acquired model
   outcomes per training query.

### Evidence

- Claims 1 and 2: paper abstract and introduction.
- Benchmark-level results: Section 4.2, Table 1 (`tab:main_results`).
- Feedback percentages for `K = 4`: appendix table "Effect of supervision
  budget K across four routing benchmarks" (`tab:k_scaling`): 33.33%, 33.33%,
  41.17%, and 36.36%, respectively.
- Break-even comparison underlying the 1.9--9.5x range: Table 1. The rounded
  ratios compare SAVERouter with the fastest conventional router that reaches
  the target quality on each benchmark.

## Metric definitions

Let `C0` be upfront supervision expenditure, `Cb` the per-query serving cost
of the best single model, and `Cr` the routed per-query serving cost at the
best-single-model quality target.

```text
SA-BEP = ceil(C0 / (Cb - Cr))
SA-CR@H = (C0 + H * Cr) / (H * Cb)
```

- **SA-BEP** is the deployment volume required for serving-time savings to
  recover the upfront supervision expenditure. Lower is better.
- **SA-CR@H** is the cost ratio after amortizing supervision expenditure over
  `H` deployment queries. Lower is better; a value below 1 means the upfront
  supervision expenditure has been recovered by horizon `H`.
- The paper reports **SA-CR@1M**, using `H = 1,000,000`.
- A metric is reported as infinity when the target quality is unreachable or
  the per-query saving is not positive.

## Wording constraints

- Say **"approximately"** or **"about"** with the headline ranges.
- Attribute the numbers to the four evaluated routing benchmarks; do not
  present them as universal guarantees.
- Say **"model-execution cost used to acquire routing supervision"**, not total
  training cost or complete system cost.
- Do not compare absolute monetary costs across benchmarks. SA-BEP and SA-CR
  are evaluated within each benchmark.
- Use **"maintaining competitive or better routing quality"** for the aggregate
  feedback claim.
- Use **"fastest conventional fully supervised router"** for the break-even
  comparison.
