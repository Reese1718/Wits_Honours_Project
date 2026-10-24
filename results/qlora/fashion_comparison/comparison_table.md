# LLM Recommender Experiment Comparison

Primary ranking metric: **NDCG@10**

Models are ranked separately within each dataset. Results from different datasets are not directly compared.

## Fashion

| Rank | Experiment | Type | Users | Hit@10 | Recall@10 | NDCG@10 | MRR@10 | CatalogueCoverage@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Qwen QLoRA | llm | 3,135 | 0.011802 | 0.011802 | 0.007224 | 0.005839 | 0.240215 |
| 2 | Llama QLoRA | llm | 3,135 | 0.008612 | 0.008612 | 0.004857 | 0.003702 | 0.106300 |

Best result by NDCG@10: **Qwen QLoRA** (0.007224).

### Paired comparisons

The difference is calculated as Experiment A minus Experiment B on the same test users.

| Experiment A | Experiment B | Metric | Difference | 95% CI | Conclusion |
| --- | --- | --- | ---: | --- | --- |
| Llama QLoRA | Qwen QLoRA | NDCG@10 | -0.002368 | [-0.004417, -0.000319] | Qwen QLoRA is higher |

