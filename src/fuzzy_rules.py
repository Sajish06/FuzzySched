"""
Fuzzy Rule Base for Student Workload and Deadline Prioritization System.

This module defines the complete 27-rule Mamdani IF-THEN knowledge base,
associated linguistic labels, semantic rationales, and CSV export functionality.
"""

from typing import List, Dict, Any, Tuple
import os
import pandas as pd


# Linguistic Term Definitions
URGENCY_LEVELS = ["Low", "Medium", "High"]
DIFFICULTY_LEVELS = ["Easy", "Moderate", "Hard"]
WORKLOAD_LEVELS = ["Light", "Moderate", "Heavy"]
PRIORITY_LEVELS = ["Very Low", "Low", "Medium", "High", "Very High"]

# Mapping from formal linguistic name to internal identifier
TERM_TO_ID = {
    # Urgency
    "Low": "low", "Medium": "medium", "High": "high",
    # Difficulty
    "Easy": "easy", "Moderate": "moderate", "Hard": "hard",
    # Workload
    "Light": "light", "Heavy": "heavy",
    # Priority
    "Very Low": "very_low", "Very High": "very_high",
}

# 27 Exhaustive Mamdani Rules with Explicit Soft Computing Rationales
RULE_BASE_DEFINITIONS: List[Dict[str, Any]] = [
    # --- Urgency: Low (Distant Deadline) ---
    {
        "rule_id": 1,
        "urgency": "Low",
        "difficulty": "Easy",
        "workload": "Light",
        "priority": "Very Low",
        "rationale": "Distant deadline, simple task, and no workload pressure; task can be safely deferred."
    },
    {
        "rule_id": 2,
        "urgency": "Low",
        "difficulty": "Easy",
        "workload": "Moderate",
        "priority": "Very Low",
        "rationale": "Easy assignment far from deadline; student should prioritize impending moderate/heavy tasks."
    },
    {
        "rule_id": 3,
        "urgency": "Low",
        "difficulty": "Easy",
        "workload": "Heavy",
        "priority": "Low",
        "rationale": "Distant easy task amidst heavy schedule; slight priority bump to keep on the student's planning radar."
    },
    {
        "rule_id": 4,
        "urgency": "Low",
        "difficulty": "Moderate",
        "workload": "Light",
        "priority": "Low",
        "rationale": "Moderate task with ample time and light competing commitments; standard low priority."
    },
    {
        "rule_id": 5,
        "urgency": "Low",
        "difficulty": "Moderate",
        "workload": "Moderate",
        "priority": "Low",
        "rationale": "Balanced moderate difficulty and workload, but distant deadline allows normal scheduling."
    },
    {
        "rule_id": 6,
        "urgency": "Low",
        "difficulty": "Moderate",
        "workload": "Heavy",
        "priority": "Medium",
        "rationale": "Heavy competing workload requires slotting moderate tasks early to prevent future bottleneck."
    },
    {
        "rule_id": 7,
        "urgency": "Low",
        "difficulty": "Hard",
        "workload": "Light",
        "priority": "Low",
        "rationale": "Difficult assignment, but deadline is far away and student has ample cognitive bandwidth."
    },
    {
        "rule_id": 8,
        "urgency": "Low",
        "difficulty": "Hard",
        "workload": "Moderate",
        "priority": "Medium",
        "rationale": "Challenging task combined with moderate competing load warrants early research and drafting."
    },
    {
        "rule_id": 9,
        "urgency": "Low",
        "difficulty": "Hard",
        "workload": "Heavy",
        "priority": "Medium",
        "rationale": "Hard assignment with heavy backlog; proactivity is essential to prevent catastrophic pile-up."
    },

    # --- Urgency: Medium (Approaching Deadline) ---
    {
        "rule_id": 10,
        "urgency": "Medium",
        "difficulty": "Easy",
        "workload": "Light",
        "priority": "Low",
        "rationale": "Approaching deadline, but task is quick and load is light; can be dispatched efficiently."
    },
    {
        "rule_id": 11,
        "urgency": "Medium",
        "difficulty": "Easy",
        "workload": "Moderate",
        "priority": "Medium",
        "rationale": "Approaching deadline with moderate load; completing this easy task early clears mental bandwidth."
    },
    {
        "rule_id": 12,
        "urgency": "Medium",
        "difficulty": "Easy",
        "workload": "Heavy",
        "priority": "Medium",
        "rationale": "Approaching deadline in a heavy workload environment; quick-win completion prevents accumulation."
    },
    {
        "rule_id": 13,
        "urgency": "Medium",
        "difficulty": "Moderate",
        "workload": "Light",
        "priority": "Medium",
        "rationale": "Standard baseline scenario: moderate urgency, moderate difficulty, light load yields medium priority."
    },
    {
        "rule_id": 14,
        "urgency": "Medium",
        "difficulty": "Moderate",
        "workload": "Moderate",
        "priority": "Medium",
        "rationale": "All intermediate conditions; balanced scheduling priority."
    },
    {
        "rule_id": 15,
        "urgency": "Medium",
        "difficulty": "Moderate",
        "workload": "Heavy",
        "priority": "High",
        "rationale": "Approaching deadline for moderate task combined with heavy workload creates severe delay hazard."
    },
    {
        "rule_id": 16,
        "urgency": "Medium",
        "difficulty": "Hard",
        "workload": "Light",
        "priority": "Medium",
        "rationale": "Approaching deadline for a tough task; light competing workload allows focused dedication."
    },
    {
        "rule_id": 17,
        "urgency": "Medium",
        "difficulty": "Hard",
        "workload": "Moderate",
        "priority": "High",
        "rationale": "Difficult task with approaching deadline amidst moderate workload demands immediate progress."
    },
    {
        "rule_id": 18,
        "urgency": "Medium",
        "difficulty": "Hard",
        "workload": "Heavy",
        "priority": "High",
        "rationale": "Approaching deadline, difficult task, and heavy workload; high priority to avoid severe time crunch."
    },

    # --- Urgency: High (Imminent Deadline / Overdue / Due Today) ---
    {
        "rule_id": 19,
        "urgency": "High",
        "difficulty": "Easy",
        "workload": "Light",
        "priority": "High",
        "rationale": "Critical deadline; even an easy task must be executed immediately to avoid grade penalty."
    },
    {
        "rule_id": 20,
        "urgency": "High",
        "difficulty": "Easy",
        "workload": "Moderate",
        "priority": "High",
        "rationale": "Imminent deadline with moderate load; easy task must be submitted without further delay."
    },
    {
        "rule_id": 21,
        "urgency": "High",
        "difficulty": "Easy",
        "workload": "Heavy",
        "priority": "High",
        "rationale": "Imminent deadline with heavy backlog; immediate execution clears one pending deadline."
    },
    {
        "rule_id": 22,
        "urgency": "High",
        "difficulty": "Moderate",
        "workload": "Light",
        "priority": "High",
        "rationale": "Imminent deadline on moderate task requires immediate active working session."
    },
    {
        "rule_id": 23,
        "urgency": "High",
        "difficulty": "Moderate",
        "workload": "Moderate",
        "priority": "High",
        "rationale": "Imminent deadline, moderate task and competing commitments; top-tier priority."
    },
    {
        "rule_id": 24,
        "urgency": "High",
        "difficulty": "Moderate",
        "workload": "Heavy",
        "priority": "Very High",
        "rationale": "Imminent deadline, moderate task, and heavy load; high probability of missing submission without urgent effort."
    },
    {
        "rule_id": 25,
        "urgency": "High",
        "difficulty": "Hard",
        "workload": "Light",
        "priority": "High",
        "rationale": "Imminent hard task requires all available cognitive effort; high priority."
    },
    {
        "rule_id": 26,
        "urgency": "High",
        "difficulty": "Hard",
        "workload": "Moderate",
        "priority": "Very High",
        "rationale": "Imminent hard assignment with multiple concurrent tasks; extreme risk of delay, very high priority."
    },
    {
        "rule_id": 27,
        "urgency": "High",
        "difficulty": "Hard",
        "workload": "Heavy",
        "priority": "Very High",
        "rationale": "Worst-case scenario: imminent deadline, hard task, and heavy backlog; maximum critical emergency priority."
    },
]


def get_rule_base() -> List[Dict[str, Any]]:
    """Return the list of all 27 fuzzy rule definitions."""
    return RULE_BASE_DEFINITIONS


def export_rules_to_csv(output_path: str = "outputs/rule_base.csv") -> str:
    """Export the 27 fuzzy rules with antecedent conditions, consequent priority, and rationales to CSV."""
    df = pd.DataFrame(RULE_BASE_DEFINITIONS)
    # Format columns for clear academic reporting
    df.columns = [
        "Rule_ID",
        "Antecedent_Urgency",
        "Antecedent_Difficulty",
        "Antecedent_Workload",
        "Consequent_Priority",
        "Soft_Computing_Rationale",
    ]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def format_rule_as_text(rule: Dict[str, Any]) -> str:
    """Format a rule definition into standard IF-THEN linguistic format."""
    return (
        f"Rule {rule['rule_id']:02d}: IF Urgency is {rule['urgency']} "
        f"AND Difficulty is {rule['difficulty']} "
        f"AND Workload is {rule['workload']} "
        f"THEN Priority is {rule['priority']}"
    )
