"""
Comprehensive Automated Test Suite for Fuzzy Logic Prioritization Engine.

Tests all 10 required domain scenarios, sensitivity curves, input validation,
boundary conditions, overlapping activations, and batch processing compatibility.
"""

import sys
import os
import math
import pytest
import numpy as np
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fuzzy_engine import (
    calculate_priority,
    batch_calculate_priority,
    map_days_to_urgency,
    map_difficulty,
    map_workload,
    classify_priority,
    get_engine,
    FuzzyPriorityEngine,
)
from membership_functions import (
    eval_trimf,
    eval_trapmf,
    fuzzify_urgency,
    fuzzify_difficulty,
    fuzzify_workload,
    fuzzify_priority,
)


# ============================================================================
# 1. Ten Core Scenario Tests
# ============================================================================

class TestCoreScenarios:
    """Test suite covering the 10 mandated fuzzy inference scenarios."""
    
    def test_scenario_1_low_urgency_easy_light(self):
        """Scenario 1: Low urgency, easy task, light workload -> Very Low priority."""
        score = calculate_priority(
            days_remaining=14.0,  # Urgency = 0.0 (Low)
            difficulty=1.0,       # Difficulty = 0.0 (Easy)
            other_assignments=0,  # Workload = 0.0 (Light)
        )
        assert 0.0 <= score <= 20.0, f"Expected Very Low priority (<20), got {score}"
        assert classify_priority(score) == "Very Low"

    def test_scenario_2_high_urgency_hard_heavy(self):
        """Scenario 2: High urgency, hard task, heavy workload -> Very High priority."""
        score = calculate_priority(
            days_remaining=0.0,   # Urgency = 10.0 (High)
            difficulty=5.0,       # Difficulty = 10.0 (Hard)
            other_assignments=5,  # Workload = 10.0 (Heavy)
        )
        assert 80.0 <= score <= 100.0, f"Expected Very High priority (>80), got {score}"
        assert classify_priority(score) == "Very High"

    def test_scenario_3_high_urgency_easy_light(self):
        """
        Scenario 3: High urgency, easy task, light workload.
        Critical deadline principle: must NOT be ignored merely because it is easy.
        Expected priority: High.
        """
        score = calculate_priority(
            days_remaining=0.0,   # Urgency = 10.0 (High)
            difficulty=1.0,       # Difficulty = 0.0 (Easy)
            other_assignments=0,  # Workload = 0.0 (Light)
        )
        assert 65.0 <= score <= 85.0, f"Expected High priority [65, 85], got {score}"
        assert classify_priority(score) == "High"

    def test_scenario_4_low_urgency_hard_heavy(self):
        """
        Scenario 4: Low urgency, hard task, heavy workload.
        Distant deadline, but hard task + heavy backlog requires proactive planning.
        Expected priority: Medium.
        """
        score = calculate_priority(
            days_remaining=14.0,  # Urgency = 0.0 (Low)
            difficulty=5.0,       # Difficulty = 10.0 (Hard)
            other_assignments=5,  # Workload = 10.0 (Heavy)
        )
        assert 40.0 <= score <= 60.0, f"Expected Medium priority [40, 60], got {score}"
        assert classify_priority(score) == "Medium"

    def test_scenario_5_medium_values_all_inputs(self):
        """Scenario 5: Medium values for all inputs -> Medium priority (balanced midpoint)."""
        score = calculate_priority(
            days_remaining=7.0,   # Urgency = 5.0 (Medium)
            difficulty=3.0,       # Difficulty = 5.0 (Moderate)
            other_assignments=2.5 # Workload = 5.0 (Moderate)
        )
        assert math.isclose(score, 50.0, abs_tol=1.0), f"Expected ~50.0, got {score}"
        assert classify_priority(score) == "Medium"

    def test_scenario_6_zero_days_remaining(self):
        """Scenario 6: Tasks with zero days remaining (due today) receive maximum urgency."""
        urgency = map_days_to_urgency(0.0)
        assert urgency == 10.0
        score = calculate_priority(days_remaining=0.0, difficulty=3.0, other_assignments=2.5)
        assert score >= 70.0, f"Expected high priority for task due today, got {score}"

    def test_scenario_7_overdue_tasks(self):
        """Scenario 7: Overdue tasks (negative days) receive maximum urgency (10.0) consistently."""
        urg_neg1 = map_days_to_urgency(-1.0)
        urg_neg5 = map_days_to_urgency(-5.0)
        urg_zero = map_days_to_urgency(0.0)
        
        assert urg_neg1 == 10.0
        assert urg_neg5 == 10.0
        assert urg_zero == 10.0
        
        score_neg = calculate_priority(days_remaining=-2.0, difficulty=4.0, other_assignments=3.0)
        score_zero = calculate_priority(days_remaining=0.0, difficulty=4.0, other_assignments=3.0)
        assert math.isclose(score_neg, score_zero, abs_tol=1e-5), "Overdue score must equal zero-day score"

    def test_scenario_8_missing_or_invalid_inputs(self):
        """Scenario 8: Missing or invalid input values must raise explicit exceptions."""
        # None inputs
        with pytest.raises(ValueError, match="cannot be None"):
            calculate_priority(days_remaining=None, difficulty=3.0, other_assignments=2)
            
        with pytest.raises(ValueError, match="cannot be None"):
            calculate_priority(days_remaining=5.0, difficulty=None, other_assignments=2)
            
        with pytest.raises(ValueError, match="cannot be None"):
            calculate_priority(days_remaining=5.0, difficulty=3.0, other_assignments=None)
            
        # Non-numeric string inputs
        with pytest.raises(TypeError, match="must be a numeric value"):
            calculate_priority(days_remaining="urgent", difficulty=3.0, other_assignments=2)
            
        # NaN and Inf inputs
        with pytest.raises(ValueError, match="cannot be NaN"):
            calculate_priority(days_remaining=float('nan'), difficulty=3.0, other_assignments=2)
            
        with pytest.raises(ValueError, match="cannot be infinite"):
            calculate_priority(days_remaining=float('inf'), difficulty=3.0, other_assignments=2)
            
        # Negative workload
        with pytest.raises(ValueError, match="cannot be negative"):
            calculate_priority(days_remaining=5.0, difficulty=3.0, other_assignments=-1)

    def test_scenario_9_boundary_values(self):
        """Scenario 9: Boundary values for every input produce finite scores in [0, 100]."""
        boundaries = [
            # (days, diff, other, len)
            (0.0, 1.0, 0.0, 0.0),
            (14.0, 1.0, 0.0, 0.0),
            (0.0, 5.0, 0.0, 0.0),
            (0.0, 1.0, 5.0, 0.0),
            (14.0, 5.0, 5.0, 20.0),
            (100.0, 1.0, 0.0, 0.0),   # Far distant deadline
            (-10.0, 5.0, 10.0, 50.0),  # Heavily overdue, heavy backlog
        ]
        
        for days, diff, other, plen in boundaries:
            score = calculate_priority(days, diff, other, perceived_length=plen)
            assert np.isfinite(score), f"Score for boundary ({days}, {diff}, {other}, {plen}) is not finite: {score}"
            assert 0.0 <= score <= 100.0, f"Score out of range [0, 100]: {score}"

    def test_scenario_10_overlapping_rules_activation(self):
        """Scenario 10: Input combinations activating overlapping rules execute smoothly."""
        engine = get_engine()
        # In overlap region: urgency=3.5 (in Low and Medium), diff=3.5, workload=3.5
        explanation = engine.explain(urgency_score=3.5, difficulty_score=3.5, workload_score=3.5)
        
        # Multiple rules must be activated
        assert explanation["active_rules_count"] >= 2, "Expected multiple active rules in overlap region"
        assert 0.0 <= explanation["crisp_priority"] <= 100.0
        assert explanation["priority_category"] in ["Low", "Medium"]
        
        # Verify firing strengths sum to a meaningful aggregate
        strengths = [r["firing_strength"] for r in explanation["active_rules"]]
        assert all(0.0 < s <= 1.0 for s in strengths)


# ============================================================================
# 2. Monotonicity & Sensitivity Tests
# ============================================================================

class TestSensitivityAndMonotonicity:
    """Test directional monotonicity of fuzzy engine response."""
    
    def test_urgency_monotonicity(self):
        """Increasing deadline urgency (diff & work fixed) must not decrease priority."""
        engine = get_engine()
        fixed_diff = 5.0
        fixed_work = 5.0
        
        u_vals = np.linspace(0.0, 10.0, 21)
        scores = [engine.compute(u, fixed_diff, fixed_work) for u in u_vals]
        
        deltas = np.diff(scores)
        min_delta = deltas.min()
        assert min_delta >= -1e-4, f"Priority decreased when urgency increased! Min delta: {min_delta}"
        assert scores[-1] > scores[0], "Max urgency must produce higher priority than min urgency"

    def test_difficulty_monotonicity(self):
        """Increasing difficulty (urgency & work fixed) must not decrease priority."""
        engine = get_engine()
        fixed_urg = 5.0
        fixed_work = 5.0
        
        d_vals = np.linspace(0.0, 10.0, 21)
        scores = [engine.compute(fixed_urg, d, fixed_work) for d in d_vals]
        
        deltas = np.diff(scores)
        min_delta = deltas.min()
        assert min_delta >= -1e-4, f"Priority decreased when difficulty increased! Min delta: {min_delta}"
        assert scores[-1] > scores[0], "Max difficulty must produce higher priority than min difficulty"

    def test_workload_monotonicity(self):
        """Increasing workload (urgency & diff fixed) must not decrease priority."""
        engine = get_engine()
        fixed_urg = 5.0
        fixed_diff = 5.0
        
        w_vals = np.linspace(0.0, 10.0, 21)
        scores = [engine.compute(fixed_urg, fixed_diff, w) for w in w_vals]
        
        deltas = np.diff(scores)
        min_delta = deltas.min()
        assert min_delta >= -1e-4, f"Priority decreased when workload increased! Min delta: {min_delta}"
        assert scores[-1] > scores[0], "Max workload must produce higher priority than min workload"


# ============================================================================
# 3. Batch Processing & Dataset Compatibility Tests
# ============================================================================

class TestBatchProcessing:
    """Test batch evaluation on both sample tasks and actual dataset."""
    
    def test_batch_sample_tasks(self):
        """Batch evaluate sample_tasks.csv and check output structure."""
        csv_path = "data/sample_tasks.csv"
        assert os.path.exists(csv_path), "sample_tasks.csv does not exist"
        
        df = pd.read_csv(csv_path)
        res = batch_calculate_priority(df)
        
        # Verify required columns are added
        expected_cols = [
            "urgency_normalized",
            "difficulty_normalized",
            "workload_normalized",
            "fuzzy_priority",
            "priority_category",
        ]
        for col in expected_cols:
            assert col in res.columns, f"Missing output column: {col}"
            
        assert len(res) == len(df), "Row count changed after batch processing"
        assert res["fuzzy_priority"].isnull().sum() == 0, "Null priority values found"
        assert (res["fuzzy_priority"] >= 0.0).all() and (res["fuzzy_priority"] <= 100.0).all()
        
        # Verify original df was not mutated
        for col in expected_cols:
            assert col not in df.columns

    def test_batch_actual_dataset(self):
        """Batch evaluate actual dataset/studentsessions.csv."""
        csv_path = "dataset/studentsessions.csv"
        if not os.path.exists(csv_path):
            pytest.skip("dataset/studentsessions.csv not found")
            
        df = pd.read_csv(csv_path)
        res = batch_calculate_priority(df)
        
        assert len(res) == 26
        assert res["fuzzy_priority"].isnull().sum() == 0
        assert (res["fuzzy_priority"] >= 0.0).all() and (res["fuzzy_priority"] <= 100.0).all()
        
        # Check assignment N has highest workload and very high priority
        n_row = res[res["assignment_id"].str.strip() == "N"].iloc[0]
        assert n_row["priority_category"] == "Very High"


# ============================================================================
# 4. Mathematical Membership Functions Tests
# ============================================================================

class TestMembershipFunctions:
    """Validate mathematical formulation of trimf and trapmf."""
    
    def test_trimf_mathematics(self):
        """Test triangular MF evaluation."""
        # Peak at 5.0, base [2.5, 7.5]
        assert eval_trimf(1.0, 2.5, 5.0, 7.5) == 0.0
        assert eval_trimf(2.5, 2.5, 5.0, 7.5) == 0.0
        assert eval_trimf(3.75, 2.5, 5.0, 7.5) == 0.5
        assert eval_trimf(5.0, 2.5, 5.0, 7.5) == 1.0
        assert eval_trimf(6.25, 2.5, 5.0, 7.5) == 0.5
        assert eval_trimf(7.5, 2.5, 5.0, 7.5) == 0.0
        assert eval_trimf(9.0, 2.5, 5.0, 7.5) == 0.0

    def test_trapmf_mathematics(self):
        """Test trapezoidal MF evaluation."""
        # Plateau [8.0, 10.0], left ramp [5.0, 8.0]
        assert eval_trapmf(4.0, 5.0, 8.0, 10.0, 10.0) == 0.0
        assert eval_trapmf(5.0, 5.0, 8.0, 10.0, 10.0) == 0.0
        assert eval_trapmf(6.5, 5.0, 8.0, 10.0, 10.0) == 0.5
        assert eval_trapmf(8.0, 5.0, 8.0, 10.0, 10.0) == 1.0
        assert eval_trapmf(9.0, 5.0, 8.0, 10.0, 10.0) == 1.0
        assert eval_trapmf(10.0, 5.0, 8.0, 10.0, 10.0) == 1.0
