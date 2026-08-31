from evaluation import evaluate_policy, summarize_results
from policies.tabular_q_learning import run_policy



results = evaluate_policy(
    policy=run_policy,
    student_config=student_config,
    seeds=validation_seeds,
    scenario_modes=validation_scenarios,
    domain_randomization=True,
)

print(results)
print(summarize_results(results))