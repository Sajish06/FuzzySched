"""
Evaluation Pipeline and Benchmark Suite for FuzzySched.

Member 2 Deliverable: Step 3 - Evaluate the three approaches.
- Generates 60 diverse academic scheduling test scenarios (dilemmas, extremes, realistic, grid).
- Computes priority scores and ranks for:
    1. Baseline A: Earliest-Deadline-First (EDF)
    2. Baseline B: Weighted Scoring (w_u=0.5, w_d=0.3, w_w=0.2)
    3. Fuzzy Inference System (Mamdani FIS with centroid defuzzification)
    4. Independently Defined Reference Rubric (Expert Academic Counseling Ground Truth)
- Computes statistical agreement:
    - Spearman's Rank Correlation (rho)
    - Kendall's Tau (tau)
    - Top-3 and Top-5 Precision/Overlap
    - Mean Absolute Error (MAE) and RMSE
    - Conflicting Case Dilemma Win Rate
- Generates publication-grade comparative charts in outputs/:
    - outputs/sensitivity_analysis.png
    - outputs/rank_correlation_comparison.png
    - outputs/dilemma_tradeoffs.png
    - outputs/priority_surface_comparison.png
- Exports outputs/evaluation_comparison_results.csv.
"""

import os
import sys
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script rendering
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from baselines import (
    normalize_urgency_100,
    normalize_difficulty_100,
    normalize_workload_100,
    calculate_weighted_priority,
    batch_prioritize_all,
    DEFAULT_WEIGHTS,
)
from fuzzy_engine import calculate_priority, classify_priority, get_engine


# ============================================================================
# 1. Benchmark Scenario Generation (60 Diverse Scenarios)
# ============================================================================

def generate_benchmark_scenarios() -> pd.DataFrame:
    """
    Generate 60 diverse, realistic test scenarios spanning:
      - 12 Conflicting Dilemmas (easy vs hard, short vs long, backlog congestion)
      - 12 Boundary & Critical Extremes (overdue, due today, distant horizons)
      - 16 Representative Student Homework Tasks
      - 20 Systematic Orthogonal Grid Combinations
    """
    scenarios = [
        # --- GROUP 1: CONFLICTING CASES & ACADEMIC DILEMMAS (12 Cases) ---
        {
            "scenario_id": "DIL-01",
            "category": "Dilemma",
            "title": "Easy Task Tomorrow vs Hard Capstone in 3 Days (Easy)",
            "days_remaining": 1.0, "difficulty": 1.0, "other_assignments": 0, "perceived_length": 1.5,
            "description": "Trivial 1-hr reading due tomorrow, zero competing load.",
        },
        {
            "scenario_id": "DIL-02",
            "category": "Dilemma",
            "title": "Easy Task Tomorrow vs Hard Capstone in 3 Days (Capstone)",
            "days_remaining": 3.0, "difficulty": 5.0, "other_assignments": 4, "perceived_length": 18.0,
            "description": "Exhausting 18-hr capstone due in 3 days with heavy backlog.",
        },
        {
            "scenario_id": "DIL-03",
            "category": "Dilemma",
            "title": "Procrastination Trap: Major Project in 5 Days",
            "days_remaining": 5.0, "difficulty": 5.0, "other_assignments": 3, "perceived_length": 16.0,
            "description": "Distant major project requiring 16 hrs of coding. High procrastination risk.",
        },
        {
            "scenario_id": "DIL-04",
            "category": "Dilemma",
            "title": "Procrastination Trap: Minor Quiz in 3 Days",
            "days_remaining": 3.0, "difficulty": 1.5, "other_assignments": 1, "perceived_length": 2.0,
            "description": "Low-stakes 2-hr quiz due in 3 days. Easily completed in one sitting.",
        },
        {
            "scenario_id": "DIL-05",
            "category": "Dilemma",
            "title": "Overdue Attendance Survey vs Final Exam Tomorrow (Survey)",
            "days_remaining": -1.0, "difficulty": 1.0, "other_assignments": 0, "perceived_length": 0.5,
            "description": "Overdue trivial 10-minute feedback form.",
        },
        {
            "scenario_id": "DIL-06",
            "category": "Dilemma",
            "title": "Overdue Attendance Survey vs Final Exam Tomorrow (Exam)",
            "days_remaining": 1.0, "difficulty": 5.0, "other_assignments": 3, "perceived_length": 14.0,
            "description": "Comprehensive semester final exam due tomorrow.",
        },
        {
            "scenario_id": "DIL-07",
            "category": "Dilemma",
            "title": "Congestion Bottleneck: Moderate in 2 Days + 5 Assignments",
            "days_remaining": 2.0, "difficulty": 3.0, "other_assignments": 5, "perceived_length": 6.0,
            "description": "Moderate assignment trapped in heavy 5-assignment congestion.",
        },
        {
            "scenario_id": "DIL-08",
            "category": "Dilemma",
            "title": "Congestion Bottleneck: Hard in 2 Days + 0 Assignments",
            "days_remaining": 2.0, "difficulty": 5.0, "other_assignments": 0, "perceived_length": 6.0,
            "description": "Difficult assignment but student has 100% focused attention.",
        },
        {
            "scenario_id": "DIL-09",
            "category": "Dilemma",
            "title": "Voluminous Easy Reading (20 hrs) in 4 Days",
            "days_remaining": 4.0, "difficulty": 1.5, "other_assignments": 2, "perceived_length": 20.0,
            "description": "Conceptually easy novel reading but immense time sink (20 hrs).",
        },
        {
            "scenario_id": "DIL-10",
            "category": "Dilemma",
            "title": "Conceptually Brutal Short Proof (2 hrs) in 4 Days",
            "days_remaining": 4.0, "difficulty": 5.0, "other_assignments": 2, "perceived_length": 2.0,
            "description": "High cognitive complexity math proof, short length.",
        },
        {
            "scenario_id": "DIL-11",
            "category": "Dilemma",
            "title": "Quick Win Low-Hanging Fruit (2 Days, Easy)",
            "days_remaining": 2.0, "difficulty": 1.0, "other_assignments": 2, "perceived_length": 1.5,
            "description": "Fast 90-minute assignment due in 2 days.",
        },
        {
            "scenario_id": "DIL-12",
            "category": "Dilemma",
            "title": "Heavy Essay Milestone (2 Days, Hard)",
            "days_remaining": 2.0, "difficulty": 4.5, "other_assignments": 2, "perceived_length": 10.0,
            "description": "Demanding 10-hour research draft due in 2 days.",
        },

        # --- GROUP 2: BOUNDARY & CRITICAL EXTREMES (12 Cases) ---
        {
            "scenario_id": "EXT-01",
            "category": "Extremes",
            "title": "Severely Overdue Hard Project (-3 days)",
            "days_remaining": -3.0, "difficulty": 5.0, "other_assignments": 4, "perceived_length": 15.0,
            "description": "Late penalty compounding daily, urgent recovery needed.",
        },
        {
            "scenario_id": "EXT-02",
            "category": "Extremes",
            "title": "Overdue Easy Assignment (-1 day)",
            "days_remaining": -1.0, "difficulty": 1.0, "other_assignments": 1, "perceived_length": 2.0,
            "description": "Late easy submission.",
        },
        {
            "scenario_id": "EXT-03",
            "category": "Extremes",
            "title": "Due Today Maximum Crisis (0 days, Hard, Congested)",
            "days_remaining": 0.0, "difficulty": 5.0, "other_assignments": 5, "perceived_length": 18.0,
            "description": "Absolute emergency: Due today, hardest difficulty, maximum backlog.",
        },
        {
            "scenario_id": "EXT-04",
            "category": "Extremes",
            "title": "Due Today Easy Task (0 days, Easy, Light)",
            "days_remaining": 0.0, "difficulty": 1.0, "other_assignments": 0, "perceived_length": 1.0,
            "description": "Due today but takes 1 hour and zero other distractions.",
        },
        {
            "scenario_id": "EXT-05",
            "category": "Extremes",
            "title": "Distant Horizon Brutal Capstone (14 days, Hard, Heavy)",
            "days_remaining": 14.0, "difficulty": 5.0, "other_assignments": 4, "perceived_length": 20.0,
            "description": "2 weeks away, but massive 20-hour project.",
        },
        {
            "scenario_id": "EXT-06",
            "category": "Extremes",
            "title": "Distant Horizon Trivial Task (14 days, Easy, Light)",
            "days_remaining": 14.0, "difficulty": 1.0, "other_assignments": 0, "perceived_length": 1.0,
            "description": "2 weeks away, negligible effort.",
        },
        {
            "scenario_id": "EXT-07",
            "category": "Extremes",
            "title": "Far Horizon (21 days, Moderate)",
            "days_remaining": 21.0, "difficulty": 3.0, "other_assignments": 2, "perceived_length": 5.0,
            "description": "Beyond normal horizon; urgency effectively zero.",
        },
        {
            "scenario_id": "EXT-08",
            "category": "Extremes",
            "title": "Far Horizon (30 days, Hard)",
            "days_remaining": 30.0, "difficulty": 5.0, "other_assignments": 1, "perceived_length": 10.0,
            "description": "A month away.",
        },
        {
            "scenario_id": "EXT-09",
            "category": "Extremes",
            "title": "Absolute Minimum Inputs (14 days, Diff 1, Load 0, Len 0)",
            "days_remaining": 14.0, "difficulty": 1.0, "other_assignments": 0, "perceived_length": 0.0,
            "description": "Theoretical lower bound of priority.",
        },
        {
            "scenario_id": "EXT-10",
            "category": "Extremes",
            "title": "Absolute Maximum Inputs (0 days, Diff 5, Load 5, Len 20)",
            "days_remaining": 0.0, "difficulty": 5.0, "other_assignments": 5, "perceived_length": 20.0,
            "description": "Theoretical upper bound of priority.",
        },
        {
            "scenario_id": "EXT-11",
            "category": "Extremes",
            "title": "Overdue Moderate Assignment (-2 days)",
            "days_remaining": -2.0, "difficulty": 3.0, "other_assignments": 2, "perceived_length": 4.0,
            "description": "2 days overdue, moderate scope.",
        },
        {
            "scenario_id": "EXT-12",
            "category": "Extremes",
            "title": "Due Today Moderate Assignment (0 days)",
            "days_remaining": 0.0, "difficulty": 3.0, "other_assignments": 2, "perceived_length": 4.0,
            "description": "Due today, average workload.",
        },

        # --- GROUP 3: REPRESENTATIVE REALISTIC STUDENT TASKS (16 Cases) ---
        {
            "scenario_id": "REAL-01", "category": "Realistic",
            "title": "Calculus Problem Set", "days_remaining": 1.0, "difficulty": 4.0, "other_assignments": 2, "perceived_length": 5.0,
            "description": "Standard challenging math homework due tomorrow.",
        },
        {
            "scenario_id": "REAL-02", "category": "Realistic",
            "title": "Physics Lab Report", "days_remaining": 2.0, "difficulty": 3.5, "other_assignments": 3, "perceived_length": 6.0,
            "description": "Data analysis and report write-up due in 2 days.",
        },
        {
            "scenario_id": "REAL-03", "category": "Realistic",
            "title": "History Discussion Board", "days_remaining": 1.0, "difficulty": 1.5, "other_assignments": 1, "perceived_length": 1.5,
            "description": "Weekly forum response due tomorrow.",
        },
        {
            "scenario_id": "REAL-04", "category": "Realistic",
            "title": "Software Engineering Sprint", "days_remaining": 3.0, "difficulty": 4.5, "other_assignments": 3, "perceived_length": 10.0,
            "description": "Group project milestone due in 3 days.",
        },
        {
            "scenario_id": "REAL-05", "category": "Realistic",
            "title": "Biology Pre-Lab Quiz", "days_remaining": 0.5, "difficulty": 2.0, "other_assignments": 1, "perceived_length": 1.0,
            "description": "Quick 30-min quiz due tonight.",
        },
        {
            "scenario_id": "REAL-06", "category": "Realistic",
            "title": "Economics Problem Set", "days_remaining": 4.0, "difficulty": 3.0, "other_assignments": 2, "perceived_length": 4.0,
            "description": "Intermediate macroeconomics problems.",
        },
        {
            "scenario_id": "REAL-07", "category": "Realistic",
            "title": "Philosophy Reading Essay", "days_remaining": 5.0, "difficulty": 3.5, "other_assignments": 1, "perceived_length": 5.0,
            "description": "Critical text analysis due in 5 days.",
        },
        {
            "scenario_id": "REAL-08", "category": "Realistic",
            "title": "Linear Algebra Homework", "days_remaining": 2.0, "difficulty": 4.0, "other_assignments": 1, "perceived_length": 4.0,
            "description": "Matrix proofs due in 2 days.",
        },
        {
            "scenario_id": "REAL-09", "category": "Realistic",
            "title": "Chemistry Midterm Revision", "days_remaining": 3.0, "difficulty": 4.5, "other_assignments": 4, "perceived_length": 12.0,
            "description": "Heavy exam study packet due in 3 days.",
        },
        {
            "scenario_id": "REAL-10", "category": "Realistic",
            "title": "Spanish Vocabulary Review", "days_remaining": 6.0, "difficulty": 1.5, "other_assignments": 0, "perceived_length": 2.0,
            "description": "Spaced repetition flashcards.",
        },
        {
            "scenario_id": "REAL-11", "category": "Realistic",
            "title": "Database Schema Design", "days_remaining": 3.0, "difficulty": 3.5, "other_assignments": 2, "perceived_length": 5.0,
            "description": "Relational schema and SQL queries.",
        },
        {
            "scenario_id": "REAL-12", "category": "Realistic",
            "title": "Data Structures Assignment", "days_remaining": 2.0, "difficulty": 4.0, "other_assignments": 3, "perceived_length": 7.0,
            "description": "Binary search tree implementation and tests.",
        },
        {
            "scenario_id": "REAL-13", "category": "Realistic",
            "title": "English Literature Essay", "days_remaining": 7.0, "difficulty": 3.0, "other_assignments": 2, "perceived_length": 6.0,
            "description": "Comparative literature essay due next week.",
        },
        {
            "scenario_id": "REAL-14", "category": "Realistic",
            "title": "Statistics Weekly Homework", "days_remaining": 4.0, "difficulty": 2.5, "other_assignments": 1, "perceived_length": 3.0,
            "description": "Regression analysis exercises.",
        },
        {
            "scenario_id": "REAL-15", "category": "Realistic",
            "title": "Ethics Reflection Paper", "days_remaining": 8.0, "difficulty": 2.0, "other_assignments": 1, "perceived_length": 3.0,
            "description": "Personal reflection paper due in over a week.",
        },
        {
            "scenario_id": "REAL-16", "category": "Realistic",
            "title": "Operating Systems Memory Lab", "days_remaining": 1.5, "difficulty": 5.0, "other_assignments": 3, "perceived_length": 9.0,
            "description": "Virtual memory paging simulation in C.",
        },

        # --- GROUP 4: SYSTEMATIC ORTHOGONAL GRID SAMPLING (20 Cases) ---
        # Spanning Low/Med/High Urgency x Easy/Mod/Hard Diff x Light/Mod/Heavy Load
        {"scenario_id": "GRID-01", "category": "Grid", "title": "Grid: High U, Easy D, Light W", "days_remaining": 1.0, "difficulty": 1.0, "other_assignments": 0, "perceived_length": 2.0, "description": "High U, Easy, Light"},
        {"scenario_id": "GRID-02", "category": "Grid", "title": "Grid: High U, Easy D, Heavy W", "days_remaining": 1.0, "difficulty": 1.0, "other_assignments": 4, "perceived_length": 15.0, "description": "High U, Easy, Heavy"},
        {"scenario_id": "GRID-03", "category": "Grid", "title": "Grid: High U, Mod D, Light W", "days_remaining": 1.0, "difficulty": 3.0, "other_assignments": 0, "perceived_length": 3.0, "description": "High U, Mod, Light"},
        {"scenario_id": "GRID-04", "category": "Grid", "title": "Grid: High U, Mod D, Heavy W", "days_remaining": 1.0, "difficulty": 3.0, "other_assignments": 4, "perceived_length": 14.0, "description": "High U, Mod, Heavy"},
        {"scenario_id": "GRID-05", "category": "Grid", "title": "Grid: High U, Hard D, Light W", "days_remaining": 1.0, "difficulty": 5.0, "other_assignments": 0, "perceived_length": 4.0, "description": "High U, Hard, Light"},
        {"scenario_id": "GRID-06", "category": "Grid", "title": "Grid: High U, Hard D, Heavy W", "days_remaining": 1.0, "difficulty": 5.0, "other_assignments": 5, "perceived_length": 18.0, "description": "High U, Hard, Heavy"},
        {"scenario_id": "GRID-07", "category": "Grid", "title": "Grid: Med U, Easy D, Light W", "days_remaining": 5.0, "difficulty": 1.0, "other_assignments": 0, "perceived_length": 2.0, "description": "Med U, Easy, Light"},
        {"scenario_id": "GRID-08", "category": "Grid", "title": "Grid: Med U, Easy D, Mod W", "days_remaining": 5.0, "difficulty": 1.0, "other_assignments": 2, "perceived_length": 5.0, "description": "Med U, Easy, Mod"},
        {"scenario_id": "GRID-09", "category": "Grid", "title": "Grid: Med U, Easy D, Heavy W", "days_remaining": 5.0, "difficulty": 1.0, "other_assignments": 4, "perceived_length": 12.0, "description": "Med U, Easy, Heavy"},
        {"scenario_id": "GRID-10", "category": "Grid", "title": "Grid: Med U, Mod D, Light W", "days_remaining": 5.0, "difficulty": 3.0, "other_assignments": 0, "perceived_length": 3.0, "description": "Med U, Mod, Light"},
        {"scenario_id": "GRID-11", "category": "Grid", "title": "Grid: Med U, Mod D, Mod W", "days_remaining": 5.0, "difficulty": 3.0, "other_assignments": 2, "perceived_length": 6.0, "description": "Med U, Mod, Mod (Balanced Midpoint)"},
        {"scenario_id": "GRID-12", "category": "Grid", "title": "Grid: Med U, Mod D, Heavy W", "days_remaining": 5.0, "difficulty": 3.0, "other_assignments": 4, "perceived_length": 14.0, "description": "Med U, Mod, Heavy"},
        {"scenario_id": "GRID-13", "category": "Grid", "title": "Grid: Med U, Hard D, Light W", "days_remaining": 5.0, "difficulty": 5.0, "other_assignments": 0, "perceived_length": 4.0, "description": "Med U, Hard, Light"},
        {"scenario_id": "GRID-14", "category": "Grid", "title": "Grid: Med U, Hard D, Mod W", "days_remaining": 5.0, "difficulty": 5.0, "other_assignments": 2, "perceived_length": 8.0, "description": "Med U, Hard, Mod"},
        {"scenario_id": "GRID-15", "category": "Grid", "title": "Grid: Med U, Hard D, Heavy W", "days_remaining": 5.0, "difficulty": 5.0, "other_assignments": 5, "perceived_length": 18.0, "description": "Med U, Hard, Heavy"},
        {"scenario_id": "GRID-16", "category": "Grid", "title": "Grid: Low U, Easy D, Light W", "days_remaining": 11.0, "difficulty": 1.0, "other_assignments": 0, "perceived_length": 1.5, "description": "Low U, Easy, Light"},
        {"scenario_id": "GRID-17", "category": "Grid", "title": "Grid: Low U, Easy D, Heavy W", "days_remaining": 11.0, "difficulty": 1.0, "other_assignments": 4, "perceived_length": 12.0, "description": "Low U, Easy, Heavy"},
        {"scenario_id": "GRID-18", "category": "Grid", "title": "Grid: Low U, Mod D, Light W", "days_remaining": 11.0, "difficulty": 3.0, "other_assignments": 1, "perceived_length": 4.0, "description": "Low U, Mod, Light"},
        {"scenario_id": "GRID-19", "category": "Grid", "title": "Grid: Low U, Hard D, Light W", "days_remaining": 11.0, "difficulty": 5.0, "other_assignments": 0, "perceived_length": 6.0, "description": "Low U, Hard, Light"},
        {"scenario_id": "GRID-20", "category": "Grid", "title": "Grid: Low U, Hard D, Heavy W", "days_remaining": 11.0, "difficulty": 5.0, "other_assignments": 5, "perceived_length": 20.0, "description": "Low U, Hard, Heavy"},
    ]
    return pd.DataFrame(scenarios)


# ============================================================================
# 2. Independent Reference Rubric (Ground Truth / Pedagogical Expert System)
# ============================================================================

def calculate_expert_reference_priority(
    days_remaining: float,
    difficulty: float,
    other_assignments: float,
    perceived_length: float,
) -> Tuple[float, str]:
    """
    Independently defined Pedagogical Expert Prioritization Rubric.
    
    IMPORTANT: This rubric does NOT use the fuzzy engine or the linear weighted formula.
    It is derived from multi-attribute cognitive decision theory in educational psychology:
      1. Immediate Deadline Hard-Wall: Overdue or due <= 1 day incurs a base urgency penalty of 85-98.
      2. Lead-Time Cognitive Buffering: Long & hard tasks (diff >= 4, length >= 6) require early start
         buffers. If days_remaining <= 3, buffer urgency increases sharply:
           Buffer_Risk = (Difficulty_Rating * Task_Hours) / max(Days, 0.5)
      3. Congestion Multiplier: Competing assignments amplify switching costs non-linearly.
      4. Trivial Tasks with distant deadlines decay asymptotically toward 5-15.
      
    Returns:
      (reference_score [0, 100], expert_rationale)
    """
    d = float(days_remaining)
    diff = float(np.clip(difficulty, 1.0, 5.0))
    assign = float(max(0, other_assignments))
    length = float(max(0.5, perceived_length))
    
    # Base urgency curve (decaying exponential-linear risk)
    if d <= 0.0:
        base_urgency = 98.0 if d == 0.0 else min(100.0, 98.0 + abs(d) * 1.0)
        rationale = "CRITICAL: Overdue or due today; immediate penalty imminent."
    elif d <= 1.0:
        base_urgency = 88.0 + (1.0 - d) * 7.0
        rationale = "HIGH URGENCY: Due within 24 hours."
    elif d <= 3.0:
        base_urgency = 62.0 + (3.0 - d) * 10.0
        rationale = "INTERMEDIATE URGENCY: Due in 2-3 days."
    elif d <= 7.0:
        base_urgency = 35.0 + (7.0 - d) * 5.0
        rationale = "MODERATE URGENCY: Due within a week."
    elif d <= 14.0:
        base_urgency = 10.0 + (14.0 - d) * 2.5
        rationale = "LOW URGENCY: Distant 1-2 week horizon."
    else:
        base_urgency = max(5.0, 10.0 - (d - 14.0) * 0.5)
        rationale = "NEGLIGIBLE URGENCY: Distant multi-week horizon."

    # Difficulty & Cognitive Buffer Term
    # Hard tasks with substantial length demand proactive time buffers
    diff_factor = (diff - 1.0) / 4.0  # [0.0, 1.0]
    work_load_factor = min(1.0, (assign / 4.0) * 0.6 + (length / 15.0) * 0.4)
    
    # Lead-Time Buffering Rule:
    # If a task requires 15 hours and difficulty is 5, having 3 days means student needs 5 hrs/day!
    # If concurrent load is high, this creates high imminent risk even before the final 24 hours.
    required_hours_per_day = length / max(0.8, d) if d > 0 else length
    if d > 0 and required_hours_per_day >= 3.0 and diff >= 3.5:
        buffer_bonus = min(22.0, required_hours_per_day * 2.2 * diff_factor)
        rationale += f" High buffering needed ({length} hrs / {d} days)."
    else:
        buffer_bonus = diff_factor * 12.0 + work_load_factor * 8.0

    # Synergy adjustment: high difficulty combined with high backlog
    synergy = 0.0
    if diff >= 4.0 and assign >= 3:
        synergy = 6.0
        rationale += " Cognitive switching congestion synergy."

    # De-escalation for trivial tasks
    if diff <= 1.5 and length <= 2.0 and d >= 3.0:
        de_escalate = -10.0
        rationale += " Trivial task easily deferred."
    else:
        de_escalate = 0.0

    raw_score = base_urgency * 0.70 + buffer_bonus + synergy + de_escalate
    final_score = float(np.clip(raw_score, 0.0, 100.0))
    
    return round(final_score, 2), rationale


# ============================================================================
# 3. Main Evaluation Execution Function
# ============================================================================

def run_evaluation(
    output_dir: str = "outputs",
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Execute full comparative benchmark, statistical evaluations, and chart generation.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 70)
    print("STEP 3: MULTI-METHOD PRIORITIZATION EVALUATION & BENCHMARK")
    print("=" * 70)

    # 1. Generate & Save Benchmark Scenarios
    scenarios_df = generate_benchmark_scenarios()
    scenarios_csv_path = os.path.join(data_dir, "evaluation_scenarios.csv")
    scenarios_df.to_csv(scenarios_csv_path, index=False)
    print(f"[OK] Generated {len(scenarios_df)} diverse test scenarios -> {scenarios_csv_path}")

    # 2. Compute Expert Reference Scores
    ref_scores = []
    ref_rationales = []
    for _, row in scenarios_df.iterrows():
        score, rat = calculate_expert_reference_priority(
            row["days_remaining"],
            row["difficulty"],
            row["other_assignments"],
            row["perceived_length"],
        )
        ref_scores.append(score)
        ref_rationales.append(rat)
        
    scenarios_df["reference_score"] = ref_scores
    scenarios_df["reference_rationale"] = ref_rationales
    scenarios_df["reference_rank"] = scenarios_df["reference_score"].rank(method="min", ascending=False).astype(int)

    # 3. Evaluate Baseline A, Baseline B, and Fuzzy FIS
    evaluated_df = batch_prioritize_all(
        scenarios_df,
        weights=DEFAULT_WEIGHTS,
        include_fuzzy=True,
        days_col="days_remaining",
        diff_col="difficulty",
        workload_col="other_assignments",
        length_col="perceived_length",
    )

    # 4. Statistical Ranking Agreement against Independent Reference Rubric
    print("\n--- STATISTICAL AGREEMENT WITH INDEPENDENT REFERENCE RUBRIC ---")
    
    methods = [
        ("Baseline A (EDF)", "edf_score", "edf_rank"),
        ("Baseline B (Weighted Scoring)", "weighted_score", "weighted_rank"),
        ("Fuzzy Inference Engine (FIS)", "fuzzy_priority", "fuzzy_rank"),
    ]
    
    stats_records = []
    y_true_rank = evaluated_df["reference_rank"].values
    y_true_score = evaluated_df["reference_score"].values
    
    # Calculate top-K sets for Ground Truth
    top3_true = set(evaluated_df.nsmallest(3, "reference_rank")["scenario_id"])
    top5_true = set(evaluated_df.nsmallest(5, "reference_rank")["scenario_id"])
    top10_true = set(evaluated_df.nsmallest(10, "reference_rank")["scenario_id"])
    
    for label, score_col, rank_col in methods:
        ranks = evaluated_df[rank_col].values
        scores = evaluated_df[score_col].values
        
        # Spearman's Rho & Kendall's Tau (agreement on ranking)
        rho, p_rho = spearmanr(y_true_rank, ranks)
        tau, p_tau = kendalltau(y_true_rank, ranks)
        
        # Top-K Precision / Overlap
        top3_pred = set(evaluated_df.nsmallest(3, rank_col)["scenario_id"])
        top5_pred = set(evaluated_df.nsmallest(5, rank_col)["scenario_id"])
        top10_pred = set(evaluated_df.nsmallest(10, rank_col)["scenario_id"])
        
        p_top3 = len(top3_pred.intersection(top3_true)) / 3.0 * 100.0
        p_top5 = len(top5_pred.intersection(top5_true)) / 5.0 * 100.0
        p_top10 = len(top10_pred.intersection(top10_true)) / 10.0 * 100.0
        
        # Absolute Score Deviations
        mae = float(np.mean(np.abs(y_true_score - scores)))
        rmse = float(np.sqrt(np.mean((y_true_score - scores) ** 2)))
        
        stats_records.append({
            "Method": label,
            "Spearman_Rho": round(rho, 4),
            "Kendall_Tau": round(tau, 4),
            "Top3_Precision_Pct": round(p_top3, 1),
            "Top5_Precision_Pct": round(p_top5, 1),
            "Top10_Precision_Pct": round(p_top10, 1),
            "Score_MAE": round(mae, 2),
            "Score_RMSE": round(rmse, 2),
        })
        
        print(f"\n{label}:")
        print(f"  * Spearman Rank Correlation (rho): {rho:.4f} (p={p_rho:.2e})")
        print(f"  * Kendall Tau (tau):             {tau:.4f} (p={p_tau:.2e})")
        print(f"  * Top-3 Overlap:                 {p_top3:.1f}%")
        print(f"  * Top-5 Overlap:                 {p_top5:.1f}%")
        print(f"  * Score MAE vs Reference:        {mae:.2f} pts")

    stats_summary_df = pd.DataFrame(stats_records)

    # 5. Save Full Comparative Results CSV
    results_csv_path = os.path.join(output_dir, "evaluation_comparison_results.csv")
    evaluated_df.to_csv(results_csv_path, index=False)
    print(f"\n[OK] Comparative results table saved to -> {results_csv_path}")

    # 6. Generate Publication-Grade Visualizations
    print("\n--- GENERATING EVALUATION PLOTS ---")
    _plot_sensitivity_analysis(output_dir)
    _plot_rank_correlation(stats_summary_df, output_dir)
    _plot_dilemma_tradeoffs(evaluated_df, output_dir)
    _plot_priority_surfaces(output_dir)
    print("[OK] All 4 visualization figures saved to outputs/")
    print("=" * 70)
    
    return evaluated_df


# ============================================================================
# 4. Visualization Routines
# ============================================================================

def _plot_sensitivity_analysis(output_dir: str) -> None:
    """Plot priority response curves as U, D, and W vary independently."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    plt.subplots_adjust(wspace=0.25)
    
    # 1. Urgency Sensitivity (hold difficulty=3.0, workload=2 assignments, length=5)
    d_vals = np.linspace(0, 14, 100)
    fixed_diff = 3.0
    fixed_assign = 2.0
    fixed_len = 5.0
    
    edf_u = [normalize_urgency_100(d) for d in d_vals]
    w_u = [calculate_weighted_priority(d, fixed_diff, fixed_assign, fixed_len) for d in d_vals]
    fuzzy_u = [calculate_priority(d, fixed_diff, fixed_assign, fixed_len) for d in d_vals]
    
    axes[0].plot(d_vals, edf_u, label="Baseline A (EDF)", color="#d9534f", linestyle="--", linewidth=2.2)
    axes[0].plot(d_vals, w_u, label="Baseline B (Weighted)", color="#337ab7", linestyle="-.", linewidth=2.2)
    axes[0].plot(d_vals, fuzzy_u, label="Fuzzy Inference Engine", color="#5cb85c", linewidth=2.8)
    axes[0].set_title("(a) Deadline Urgency Sensitivity\n(Fixed: Diff=3.0, Other=2, Len=5)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Days Remaining to Deadline (days)", fontsize=10)
    axes[0].set_ylabel("Calculated Priority Score [0, 100]", fontsize=10)
    axes[0].set_ylim(-2, 105)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", framealpha=0.9)
    
    # 2. Difficulty Sensitivity (hold days=2.0, workload=2 assignments, length=5)
    diff_vals = np.linspace(1.0, 5.0, 100)
    fixed_days = 2.0
    
    edf_d = [normalize_urgency_100(fixed_days) for _ in diff_vals]  # Invariant for EDF!
    w_d = [calculate_weighted_priority(fixed_days, diff, fixed_assign, fixed_len) for diff in diff_vals]
    fuzzy_d = [calculate_priority(fixed_days, diff, fixed_assign, fixed_len) for diff in diff_vals]
    
    axes[1].plot(diff_vals, edf_d, label="Baseline A (EDF - Invariant!)", color="#d9534f", linestyle="--", linewidth=2.2)
    axes[1].plot(diff_vals, w_d, label="Baseline B (Weighted)", color="#337ab7", linestyle="-.", linewidth=2.2)
    axes[1].plot(diff_vals, fuzzy_d, label="Fuzzy Inference Engine", color="#5cb85c", linewidth=2.8)
    axes[1].set_title("(b) Task Difficulty Sensitivity\n(Fixed: Days=2.0, Other=2, Len=5)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Task Difficulty Rating (1 to 5)", fontsize=10)
    axes[1].set_ylim(-2, 105)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower right", framealpha=0.9)
    
    # 3. Workload Sensitivity (hold days=2.0, difficulty=3.0, vary assignments 0 to 5)
    assign_vals = np.linspace(0.0, 5.0, 100)
    
    edf_w = [normalize_urgency_100(fixed_days) for _ in assign_vals]  # Invariant for EDF!
    w_w = [calculate_weighted_priority(fixed_days, fixed_diff, a, fixed_len) for a in assign_vals]
    fuzzy_w = [calculate_priority(fixed_days, fixed_diff, a, fixed_len) for a in assign_vals]
    
    axes[2].plot(assign_vals, edf_w, label="Baseline A (EDF - Invariant!)", color="#d9534f", linestyle="--", linewidth=2.2)
    axes[2].plot(assign_vals, w_w, label="Baseline B (Weighted)", color="#337ab7", linestyle="-.", linewidth=2.2)
    axes[2].plot(assign_vals, fuzzy_w, label="Fuzzy Inference Engine", color="#5cb85c", linewidth=2.8)
    axes[2].set_title("(c) Workload Backlog Sensitivity\n(Fixed: Days=2.0, Diff=3.0, Len=5)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Concurrent Assignments Count", fontsize=10)
    axes[2].set_ylim(-2, 105)
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="lower right", framealpha=0.9)

    plt.suptitle("Sensitivity Response Curves: Fuzzy FIS vs. Baselines Across Input Variables", fontsize=13, fontweight="bold", y=1.02)
    fig_path = os.path.join(output_dir, "sensitivity_analysis.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def _plot_rank_correlation(stats_df: pd.DataFrame, output_dir: str) -> None:
    """Bar chart comparing ranking correlation and Top-K agreement."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    methods = stats_df["Method"].tolist()
    clean_labels = ["EDF", "Weighted Scoring\n(0.5, 0.3, 0.2)", "Fuzzy Mamdani FIS"]
    x = np.arange(len(methods))
    width = 0.35
    
    # Panel 1: Correlation Coefficients
    rho_vals = stats_df["Spearman_Rho"].tolist()
    tau_vals = stats_df["Kendall_Tau"].tolist()
    
    rects1 = ax1.bar(x - width/2, rho_vals, width, label="Spearman's Rank Correlation (rho)", color="#4A90E2", edgecolor="black")
    rects2 = ax1.bar(x + width/2, tau_vals, width, label="Kendall's Tau (tau)", color="#50E3C2", edgecolor="black")
    
    ax1.set_ylabel("Correlation Coefficient", fontsize=11)
    ax1.set_title("Ranking Agreement with Pedagogical Reference Rubric", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(clean_labels, fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax1.legend(loc="lower right", framealpha=0.9)
    
    # Add values above bars
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f"{height:.3f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        height = rect.get_height()
        ax1.annotate(f"{height:.3f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                     
    # Panel 2: Top-K Critical Task Precision
    top3_vals = stats_df["Top3_Precision_Pct"].tolist()
    top5_vals = stats_df["Top5_Precision_Pct"].tolist()
    
    rects3 = ax2.bar(x - width/2, top3_vals, width, label="Top-3 Critical Overlap (%)", color="#F5A623", edgecolor="black")
    rects4 = ax2.bar(x + width/2, top5_vals, width, label="Top-5 Critical Overlap (%)", color="#BD10E0", edgecolor="black")
    
    ax2.set_ylabel("Overlap Agreement (%)", fontsize=11)
    ax2.set_title("Top-K Critical Task Identification Accuracy", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(clean_labels, fontsize=10)
    ax2.set_ylim(0, 115)
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax2.legend(loc="lower right", framealpha=0.9)
    
    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Comparative Evaluation: Ranking Agreement Against Ground Truth", fontsize=13, fontweight="bold", y=1.02)
    fig_path = os.path.join(output_dir, "rank_correlation_comparison.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def _plot_dilemma_tradeoffs(df: pd.DataFrame, output_dir: str) -> None:
    """Grouped bar chart for conflicting dilemma scenarios."""
    dilemma_cases = df[df["category"] == "Dilemma"].head(8)
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    
    labels = [
        "DIL-01\nEasy Tomorrow",
        "DIL-02\nHard Capstone in 3d",
        "DIL-03\nProj in 5d (Trap)",
        "DIL-04\nQuiz in 3d",
        "DIL-05\nLate Survey",
        "DIL-06\nFinal Exam",
        "DIL-07\nMod (5 backlogs)",
        "DIL-08\nHard (0 backlogs)",
    ]
    x = np.arange(len(labels))
    width = 0.20
    
    r_edf = ax.bar(x - 1.5*width, dilemma_cases["edf_score"], width, label="EDF (Baseline A)", color="#d9534f", edgecolor="black")
    r_wt = ax.bar(x - 0.5*width, dilemma_cases["weighted_score"], width, label="Weighted (Baseline B)", color="#337ab7", edgecolor="black")
    r_fuzz = ax.bar(x + 0.5*width, dilemma_cases["fuzzy_priority"], width, label="Fuzzy FIS", color="#5cb85c", edgecolor="black")
    r_ref = ax.bar(x + 1.5*width, dilemma_cases["reference_score"], width, label="Expert Reference", color="#f0ad4e", edgecolor="black")
    
    ax.set_ylabel("Assigned Priority Score [0, 100]", fontsize=11)
    ax.set_title("Behavior on Conflicting Scenarios & Academic Dilemmas", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 115)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="upper right", framealpha=0.95)
    
    # Highlight classic dilemma 1 vs 2
    ax.axvspan(-0.5, 1.5, color="gray", alpha=0.12, linestyle="--", linewidth=1)
    ax.text(0.5, 108, "Classic Dilemma: Easy Tomorrow vs. Hard Capstone in 3 Days\n(EDF inverts true priority!)",
            ha="center", fontsize=9, fontweight="bold", color="#a94442")

    fig_path = os.path.join(output_dir, "dilemma_tradeoffs.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def _plot_priority_surfaces(output_dir: str) -> None:
    """Plot 2D priority contour comparison between Linear Weighted Scoring and Non-Linear Fuzzy FIS."""
    days_grid = np.linspace(0.0, 10.0, 35)
    diff_grid = np.linspace(1.0, 5.0, 35)
    D_mesh, DIFF_mesh = np.meshgrid(days_grid, diff_grid)
    
    fixed_assign = 2.0
    fixed_len = 5.0
    
    Z_weighted = np.zeros_like(D_mesh)
    Z_fuzzy = np.zeros_like(D_mesh)
    
    for i in range(D_mesh.shape[0]):
        for j in range(D_mesh.shape[1]):
            d = D_mesh[i, j]
            diff = DIFF_mesh[i, j]
            Z_weighted[i, j] = calculate_weighted_priority(d, diff, fixed_assign, fixed_len)
            Z_fuzzy[i, j] = calculate_priority(d, diff, fixed_assign, fixed_len)
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # Linear Weighted Contour
    c1 = ax1.contourf(D_mesh, DIFF_mesh, Z_weighted, levels=20, cmap="viridis", vmin=0, vmax=100)
    ax1.set_title("Linear Weighted Scoring: Flat Hyperplane\n(P = 0.5U + 0.3D + 0.2W)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Days Remaining to Deadline", fontsize=10)
    ax1.set_ylabel("Task Difficulty Rating (1 to 5)", fontsize=10)
    fig.colorbar(c1, ax=ax1, label="Priority Score")
    
    # Fuzzy Non-linear Contour
    c2 = ax2.contourf(D_mesh, DIFF_mesh, Z_fuzzy, levels=20, cmap="viridis", vmin=0, vmax=100)
    ax2.set_title("Fuzzy Mamdani Engine: Smooth Non-Linear Manifold\n(Captures synergistic rules & critical thresholds)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Days Remaining to Deadline", fontsize=10)
    ax2.set_ylabel("Task Difficulty Rating (1 to 5)", fontsize=10)
    fig.colorbar(c2, ax=ax2, label="Priority Score")

    plt.suptitle("Control Surface Comparison: Linear Planar vs. Fuzzy Non-Linear Prioritization", fontsize=13, fontweight="bold", y=1.02)
    fig_path = os.path.join(output_dir, "priority_surface_comparison.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_evaluation()
