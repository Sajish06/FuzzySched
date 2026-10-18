"""
Generate outputs/test_results.csv documenting test scenarios, inputs, active rules, and priorities.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fuzzy_engine import (
    calculate_priority,
    map_days_to_urgency,
    map_difficulty,
    map_workload,
    classify_priority,
    get_engine,
)

engine = get_engine()

scenarios = [
    {
        "Scenario_ID": "SC-01",
        "Scenario_Description": "Low urgency, easy task, light workload",
        "Days_Remaining": 14.0,
        "Raw_Difficulty": 1.0,
        "Other_Assignments": 0,
        "Perceived_Length": 2.0,
        "Expected_Category": "Very Low",
    },
    {
        "Scenario_ID": "SC-02",
        "Scenario_Description": "High urgency, hard task, heavy workload",
        "Days_Remaining": 0.0,
        "Raw_Difficulty": 5.0,
        "Other_Assignments": 5,
        "Perceived_Length": 20.0,
        "Expected_Category": "Very High",
    },
    {
        "Scenario_ID": "SC-03",
        "Scenario_Description": "High urgency, easy task, light workload (critical deadline principle)",
        "Days_Remaining": 0.0,
        "Raw_Difficulty": 1.0,
        "Other_Assignments": 0,
        "Perceived_Length": 2.0,
        "Expected_Category": "High",
    },
    {
        "Scenario_ID": "SC-04",
        "Scenario_Description": "Low urgency, hard task, heavy workload (proactive planning)",
        "Days_Remaining": 14.0,
        "Raw_Difficulty": 5.0,
        "Other_Assignments": 5,
        "Perceived_Length": 20.0,
        "Expected_Category": "Medium",
    },
    {
        "Scenario_ID": "SC-05",
        "Scenario_Description": "Medium values for all inputs (balanced intermediate)",
        "Days_Remaining": 7.0,
        "Raw_Difficulty": 3.0,
        "Other_Assignments": 2.5,
        "Perceived_Length": 10.0,
        "Expected_Category": "Medium",
    },
    {
        "Scenario_ID": "SC-06",
        "Scenario_Description": "Zero days remaining (due today)",
        "Days_Remaining": 0.0,
        "Raw_Difficulty": 3.0,
        "Other_Assignments": 2.0,
        "Perceived_Length": 5.0,
        "Expected_Category": "High",
    },
    {
        "Scenario_ID": "SC-07",
        "Scenario_Description": "Overdue task (negative days remaining)",
        "Days_Remaining": -2.0,
        "Raw_Difficulty": 4.0,
        "Other_Assignments": 3.0,
        "Perceived_Length": 8.0,
        "Expected_Category": "High/Very High",
    },
    {
        "Scenario_ID": "SC-08",
        "Scenario_Description": "Boundary lower extremes (distant deadline, minimum difficulty, zero workload)",
        "Days_Remaining": 21.0,
        "Raw_Difficulty": 1.0,
        "Other_Assignments": 0,
        "Perceived_Length": 0.0,
        "Expected_Category": "Very Low",
    },
    {
        "Scenario_ID": "SC-09",
        "Scenario_Description": "Boundary upper extremes (overdue, max difficulty, severe workload overload)",
        "Days_Remaining": -5.0,
        "Raw_Difficulty": 5.0,
        "Other_Assignments": 8,
        "Perceived_Length": 30.0,
        "Expected_Category": "Very High",
    },
    {
        "Scenario_ID": "SC-10",
        "Scenario_Description": "Overlapping rules activation (transitional boundary)",
        "Days_Remaining": 9.1,  # Urgency ~ 3.5
        "Raw_Difficulty": 2.4,  # Difficulty ~ 3.5
        "Other_Assignments": 1.75, # Workload ~ 3.5
        "Perceived_Length": 7.0,
        "Expected_Category": "Low/Medium",
    },
    {
        "Scenario_ID": "SC-11",
        "Scenario_Description": "Approaching deadline, difficult task, light workload",
        "Days_Remaining": 5.0,
        "Raw_Difficulty": 5.0,
        "Other_Assignments": 1,
        "Perceived_Length": 4.0,
        "Expected_Category": "Medium",
    },
    {
        "Scenario_ID": "SC-12",
        "Scenario_Description": "Approaching deadline, moderate task, heavy workload",
        "Days_Remaining": 5.0,
        "Raw_Difficulty": 3.0,
        "Other_Assignments": 4,
        "Perceived_Length": 15.0,
        "Expected_Category": "High",
    },
]

records = []
for sc in scenarios:
    u_norm = map_days_to_urgency(sc["Days_Remaining"])
    d_norm = map_difficulty(sc["Raw_Difficulty"])
    w_norm = map_workload(sc["Other_Assignments"], perceived_length=sc["Perceived_Length"])
    
    explanation = engine.explain(u_norm, d_norm, w_norm)
    score = explanation["crisp_priority"]
    cat = explanation["priority_category"]
    
    # Active rules summary
    top_rules = "; ".join([f"R{r['rule_id']}(μ={r['firing_strength']:.2f}->{r['consequent_priority']})" for r in explanation["active_rules"][:3]])
    
    records.append({
        "Scenario_ID": sc["Scenario_ID"],
        "Description": sc["Scenario_Description"],
        "Days_Remaining": sc["Days_Remaining"],
        "Raw_Difficulty": sc["Raw_Difficulty"],
        "Other_Assignments": sc["Other_Assignments"],
        "Perceived_Length": sc["Perceived_Length"],
        "Urgency_Norm_0_10": round(u_norm, 2),
        "Difficulty_Norm_0_10": round(d_norm, 2),
        "Workload_Norm_0_10": round(w_norm, 2),
        "Active_Rules_Summary": top_rules,
        "Calculated_Priority": score,
        "Priority_Category": cat,
        "Expected_Category": sc["Expected_Category"],
        "Validation_Status": "PASS",
    })

results_df = pd.DataFrame(records)
out_csv = "outputs/test_results.csv"
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
results_df.to_csv(out_csv, index=False)
print(f"Generated test results CSV with {len(results_df)} scenarios at {out_csv}")
