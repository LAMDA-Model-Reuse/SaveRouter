"""Template for real model feedback acquisition."""

from saverouter import collect_fixed_k_supervision


def reveal(query_index, model_index):
    # response, billed_cost = call_model(queries[query_index], models[model_index])
    # reward = judge(response, references[query_index])
    # return reward, billed_cost
    raise NotImplementedError("Connect this callback to your model and evaluator.")


# feedback = collect_fixed_k_supervision(
#     n_queries=len(queries),
#     n_models=len(models),
#     group_ids=group_ids,
#     reveal=reveal,
#     k=4,
#     seed=42,
# )
