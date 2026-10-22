"""
Automated Pytest Suite for Baseline Prioritization Algorithms (src/baselines.py).

Tests:
  - Normalization helpers (urgency, difficulty, workload) bounds and clipping.
  - Overdue deadline handling in EDF.
  - Baseline A Earliest-Deadline-First ranking and scoring.
  - Baseline B Weighted Scoring computation, weights normalization, and formula verification.
  - Weight sensitivity and edge cases.
  - Batch ranking and multi-method comparisons.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from baselines import (
    normalize_urgency_100,
    normalize_difficulty_100,
    normalize_workload_100,
    edf_priority_score,
    rank_by_edf,
    compute_weighted_score,
    calculate_weighted_priority,
    rank_by_weighted_score,
    batch_prioritize_all,
    DEFAULT_WEIGHTS,
)


class TestNormalizations:
    """Verify input normalization bounds [0, 100]."""
    
    def test_urgency_bounds_and_overdue(self):
        # Overdue should receive 100
        assert normalize_urgency_100(-3.0) == 100.0
        assert normalize_urgency_100(-0.5) == 100.0
        # Due today should receive 100
        assert normalize_urgency_100(0.0) == 100.0
        # Distant deadline should receive 0.0
        assert normalize_urgency_100(14.0) == 0.0
        assert normalize_urgency_100(20.0) == 0.0
        # Midpoint (7 days out of 14) -> 50.0
        assert pytest.approx(normalize_urgency_100(7.0), 0.01) == 50.0

    def test_difficulty_bounds(self):
        # Likert 1 -> 0.0
        assert normalize_difficulty_100(1.0) == 0.0
        # Likert 3 -> 50.0
        assert pytest.approx(normalize_difficulty_100(3.0), 0.01) == 50.0
        # Likert 5 -> 100.0
        assert normalize_difficulty_100(5.0) == 100.0
        # Out of bounds clamping
        assert normalize_difficulty_100(0.5) == 0.0
        assert normalize_difficulty_100(6.0) == 100.0

    def test_workload_bounds(self):
        # Zero assignments, no length -> 0.0
        assert normalize_workload_100(0.0) == 0.0
        # Max assignments (5), no length -> 100.0
        assert normalize_workload_100(5.0) == 100.0
        # Composite with length
        # 0 assignments, max length (20) -> 0.35 * 100 = 35.0
        assert pytest.approx(normalize_workload_100(0.0, perceived_length=20.0), 0.01) == 35.0
        # Max assignments (5), max length (20) -> 100.0
        assert pytest.approx(normalize_workload_100(5.0, perceived_length=20.0), 0.01) == 100.0

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            normalize_urgency_100(None)
        with pytest.raises(TypeError):
            normalize_difficulty_100("invalid")
        with pytest.raises(ValueError):
            normalize_workload_100(-1.0)


class TestBaselineA_EDF:
    """Verify Baseline A: Earliest-Deadline-First behavior."""
    
    def test_edf_overdue_ranked_first(self):
        df = pd.DataFrame({
            "task_id": ["T1", "T2", "T3", "T4"],
            "days_remaining": [3.0, -1.0, 0.0, 7.0],
            "difficulty": [2.0, 3.0, 4.0, 5.0],
        })
        ranked = rank_by_edf(df, days_col="days_remaining")
        # -1.0 must be first, then 0.0, then 3.0, then 7.0
        assert ranked.iloc[0]["task_id"] == "T2"
        assert ranked.iloc[1]["task_id"] == "T3"
        assert ranked.iloc[2]["task_id"] == "T1"
        assert ranked.iloc[3]["task_id"] == "T4"
        assert list(ranked["edf_rank"]) == [1, 2, 3, 4]

    def test_edf_priority_score_monotonicity(self):
        s_overdue = edf_priority_score(-2.0)
        s_today = edf_priority_score(0.0)
        s_near = edf_priority_score(2.0)
        s_far = edf_priority_score(10.0)
        
        assert s_overdue == 100.0
        assert s_today == 100.0
        assert s_today > s_near > s_far >= 0.0


class TestBaselineB_WeightedScoring:
    """Verify Baseline B: Weighted Scoring behavior."""
    
    def test_default_weights_calculation(self):
        # P = 0.5*U + 0.3*D + 0.2*W
        u = 80.0
        d = 60.0
        w = 40.0
        expected = 0.5 * 80.0 + 0.3 * 60.0 + 0.2 * 40.0  # 40 + 18 + 8 = 66.0
        res = compute_weighted_score(u, d, w, weights=(0.5, 0.3, 0.2))
        assert pytest.approx(res, 0.01) == expected

    def test_automatic_weight_normalization(self):
        # Weights (5, 3, 2) should normalize to (0.5, 0.3, 0.2)
        score1 = compute_weighted_score(100.0, 50.0, 0.0, weights=(0.5, 0.3, 0.2))
        score2 = compute_weighted_score(100.0, 50.0, 0.0, weights=(5.0, 3.0, 2.0))
        assert pytest.approx(score1, 0.01) == score2

    def test_extreme_inputs(self):
        # Minimum inputs -> 0.0
        assert compute_weighted_score(0.0, 0.0, 0.0) == 0.0
        # Maximum inputs -> 100.0
        assert compute_weighted_score(100.0, 100.0, 100.0) == 100.0


class TestMultiMethodBatchIntegration:
    """Verify batch_prioritize_all integrates EDF, Weighted, and Fuzzy correctly."""
    
    def test_batch_prioritize_all_runs(self):
        sample_df = pd.DataFrame({
            "task_id": ["T1", "T2", "T3"],
            "days_remaining": [1.0, 5.0, 0.0],
            "difficulty": [2.0, 5.0, 4.0],
            "other_assignments": [0, 4, 2],
            "perceived_length": [2, 15, 6],
        })
        enriched = batch_prioritize_all(sample_df)
        
        # Verify required columns exist
        for col in [
            "urgency_100", "difficulty_100", "workload_100",
            "edf_score", "edf_rank",
            "weighted_score", "weighted_rank",
            "fuzzy_priority", "fuzzy_rank",
        ]:
            assert col in enriched.columns, f"Missing column: {col}"
            
        # Due today (T3) must have rank 1 in EDF
        assert enriched.loc[enriched["task_id"] == "T3", "edf_rank"].values[0] == 1
