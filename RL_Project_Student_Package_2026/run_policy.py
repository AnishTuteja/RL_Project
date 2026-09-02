from _evaluation import evaluate_policy, summarize_results
from RL_Project_Student_Package_2026.nn_sarsa import run_policy, ROLL_NUMBER
from industrial_inventory_env import generate_student_config

student_config = generate_student_config(ROLL_NUMBER)

validation_seeds = range(100, 110)
validation_scenarios = ("random", "seasonal", "trend", "shock")



results = evaluate_policy(
    policy=run_policy,
    student_config=student_config,
    seeds=validation_seeds,
    scenario_modes=validation_scenarios,
    domain_randomization=True,
)

print(results)
print(summarize_results(results))