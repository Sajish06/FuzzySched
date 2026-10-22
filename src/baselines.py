"""
Baseline Prioritization Algorithms for Student Task Scheduling.

Member 2 Deliverable: Step 2 - Build two baseline prioritization methods.
Implements:
  - Baseline A: Earliest-Deadline-First (EDF) sorting and priority scoring.
  - Baseline B: Weighted Scoring (linear multi-criteria model).
  - Batch ranking and comparison utilities integrating the Fuzzy Priority Engine.
"""

from typing import Dict, Any, Optional, Union, Tuple, List
import math
import numpy as np
import pandas as pd

# Default Normalization Constants
DEFAULT_MAX_DEADLINE_DAYS: float = 14.0   # Horizon threshold for deadline urgency
DEFAULT_MIN_DIFFICULTY: float = 1.0       # Minimum rating on Likert scale
DEFAULT_MAX_DIFFICULTY: float = 5.0       # Maximum rating on Likert scale
DEFAULT_MAX_ASSIGNMENTS: float = 5.0      # Benchmark for heavy concurrent assignment count
DEFAULT_MAX_LENGTH: float = 20.0          # Max length benchmark based on dataset
DEFAULT_LENGTH_WEIGHT: float = 0.35       # Blend weight of perceived length in workload

# Default Baseline B Weights (Normalized sum = 1.0)
# Justification:
#   - Urgency (0.50): Non-negotiable hard deadline constraints. Missing deadlines causes late penalties or zero credit.
#   - Difficulty (0.30): Complex assignments demand larger cognitive preparation buffers and error margin.
#   - Workload (0.20): Concurrent assignment congestion amplifies context switching costs.
DEFAULT_WEIGHTS: Tuple[float, float, float] = (0.50, 0.30, 0.20)


# ============================================================================
# Input Validation & Normalization Helpers (Scaled to [0.0, 100.0])
# ============================================================================

def _validate_numeric(val: Any, name: str, allow_none: bool = False) -> Optional[float]:
    """Validate that an input value is a finite number."""
    if val is None:
        if allow_none:
            return None
        raise ValueError(f"Input '{name}' cannot be None.")
    
    try:
        f_val = float(val)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"Input '{name}' must be a numeric value, got {type(val).__name__} ({val!r}).") from exc
        
    if math.isnan(f_val):
        raise ValueError(f"Input '{name}' cannot be NaN.")
    if math.isinf(f_val):
        raise ValueError(f"Input '{name}' cannot be infinite.")
        
    return f_val


def normalize_urgency_100(
    days_remaining: float,
    max_days: float = DEFAULT_MAX_DEADLINE_DAYS,
) -> float:
    """
    Normalize raw days remaining to an urgency score in [0.0, 100.0].
    
    Logic:
      - Overdue tasks (days_remaining < 0) or due today (== 0) receive maximum urgency: 100.0.
      - Tasks beyond max_days receive minimum urgency: 0.0.
      - Linear interpolation between 0 and max_days:
          U = 100.0 * (1.0 - days_remaining / max_days)
    """
    d = _validate_numeric(days_remaining, "days_remaining")
    if d <= 0.0:
        return 100.0
    if d >= max_days:
        return 0.0
    return float(np.clip(100.0 * (1.0 - (d / max_days)), 0.0, 100.0))


def normalize_difficulty_100(
    raw_difficulty: float,
    min_val: float = DEFAULT_MIN_DIFFICULTY,
    max_val: float = DEFAULT_MAX_DIFFICULTY,
) -> float:
    """
    Normalize task difficulty from [min_val, max_val] to [0.0, 100.0].
    
    Formula:
      D = 100.0 * (raw - min_val) / (max_val - min_val)
    """
    diff = _validate_numeric(raw_difficulty, "raw_difficulty")
    if max_val <= min_val:
        raise ValueError(f"max_val ({max_val}) must be strictly greater than min_val ({min_val}).")
        
    norm = 100.0 * (diff - min_val) / (max_val - min_val)
    return float(np.clip(norm, 0.0, 100.0))


def normalize_workload_100(
    other_assignments: float,
    perceived_length: Optional[float] = None,
    max_assignments: float = DEFAULT_MAX_ASSIGNMENTS,
    max_length: float = DEFAULT_MAX_LENGTH,
    length_weight: float = DEFAULT_LENGTH_WEIGHT,
) -> float:
    """
    Normalize concurrent assignments and optional perceived length to [0.0, 100.0].
    
    Synthesizes assignment count with task length when length is available.
    """
    assign = _validate_numeric(other_assignments, "other_assignments")
    if assign < 0.0:
        raise ValueError(f"other_assignments cannot be negative, got {assign}.")
        
    norm_assign = float(np.clip((assign / max_assignments) * 100.0, 0.0, 100.0))
    
    len_val = _validate_numeric(perceived_length, "perceived_length", allow_none=True)
    if len_val is None:
        return norm_assign
        
    if len_val < 0.0:
        raise ValueError(f"perceived_length cannot be negative, got {len_val}.")
        
    norm_len = float(np.clip((len_val / max_length) * 100.0, 0.0, 100.0))
    w = float(np.clip(length_weight, 0.0, 1.0))
    composite = (1.0 - w) * norm_assign + w * norm_len
    return float(np.clip(composite, 0.0, 100.0))


# ============================================================================
# Baseline A: Earliest-Deadline-First (EDF)
# ============================================================================

def edf_priority_score(
    days_remaining: float,
    max_days: float = DEFAULT_MAX_DEADLINE_DAYS,
) -> float:
    """
    Compute continuous EDF priority score in [0.0, 100.0].
    
    In EDF, priority is solely determined by proximity to deadline.
    Overdue tasks (days_remaining < 0) and due-today tasks (== 0) get 100.0.
    Distant tasks (>= max_days) get 0.0.
    """
    return normalize_urgency_100(days_remaining, max_days=max_days)


def rank_by_edf(
    df: pd.DataFrame,
    days_col: Optional[str] = None,
    tie_breaker_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Sort tasks by Earliest-Deadline-First.
    
    Overdue tasks (days < 0) and imminent tasks (days == 0) are ranked first.
    Optionally breaks ties using a secondary column (e.g., difficulty or workload).
    
    Adds:
      - 'edf_score': Priority score based on deadline urgency [0, 100].
      - 'edf_rank': Ordinal rank (1 = highest priority).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("DataFrame is empty.")
        
    res = df.copy()
    
    # Resolve days column
    candidates = ["days_till_deadline_started", "days_remaining", "days_till_deadline", "days_left"]
    actual_days_col = days_col
    if not actual_days_col:
        for c in candidates:
            if c in res.columns:
                actual_days_col = c
                break
    if not actual_days_col or actual_days_col not in res.columns:
        raise KeyError(f"Could not resolve deadline days column. Provided: {days_col}")
        
    # Calculate EDF priority score
    res["edf_score"] = res[actual_days_col].apply(lambda d: edf_priority_score(d))
    
    # Sort criteria: days_remaining ASC (overdue first)
    sort_cols = [actual_days_col]
    sort_ascending = [True]
    
    if tie_breaker_col and tie_breaker_col in res.columns:
        sort_cols.append(tie_breaker_col)
        sort_ascending.append(False)  # Higher difficulty/workload breaks tie
        
    res = res.sort_values(by=sort_cols, ascending=sort_ascending).reset_index(drop=True)
    res["edf_rank"] = res.index + 1
    
    return res


# ============================================================================
# Baseline B: Weighted Scoring (Linear Multi-Criteria Evaluation)
# ============================================================================

def compute_weighted_score(
    urgency_100: float,
    difficulty_100: float,
    workload_100: float,
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> float:
    """
    Compute linear weighted prioritization score:
      P = w_u * U + w_d * D + w_w * W
      
    Args:
      urgency_100: Normalized urgency score in [0.0, 100.0]
      difficulty_100: Normalized difficulty score in [0.0, 100.0]
      workload_100: Normalized workload score in [0.0, 100.0]
      weights: 3-tuple of (w_u, w_d, w_w). Normalized automatically if sum != 1.0.
      
    Returns:
      Linear weighted priority score in [0.0, 100.0].
    """
    u = float(np.clip(_validate_numeric(urgency_100, "urgency_100"), 0.0, 100.0))
    d = float(np.clip(_validate_numeric(difficulty_100, "difficulty_100"), 0.0, 100.0))
    w = float(np.clip(_validate_numeric(workload_100, "workload_100"), 0.0, 100.0))
    
    if len(weights) != 3:
        raise ValueError(f"weights must be a 3-tuple of (w_urgency, w_diff, w_workload), got {weights}.")
        
    w_u, w_d, w_w = weights
    w_sum = w_u + w_d + w_w
    if w_sum <= 0:
        raise ValueError("Sum of weights must be strictly positive.")
        
    # Normalize weights so sum is 1.0
    w_u_norm = w_u / w_sum
    w_d_norm = w_d / w_sum
    w_w_norm = w_w / w_sum
    
    score = w_u_norm * u + w_d_norm * d + w_w_norm * w
    return float(np.clip(score, 0.0, 100.0))


def calculate_weighted_priority(
    days_remaining: float,
    difficulty: float,
    other_assignments: float,
    perceived_length: Optional[float] = None,
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
    max_days: float = DEFAULT_MAX_DEADLINE_DAYS,
    min_difficulty: float = DEFAULT_MIN_DIFFICULTY,
    max_difficulty: float = DEFAULT_MAX_DIFFICULTY,
    max_assignments: float = DEFAULT_MAX_ASSIGNMENTS,
    max_length: float = DEFAULT_MAX_LENGTH,
    length_weight: float = DEFAULT_LENGTH_WEIGHT,
) -> float:
    """
    Public single-task API to calculate Baseline B weighted priority score.
    """
    u100 = normalize_urgency_100(days_remaining, max_days=max_days)
    d100 = normalize_difficulty_100(difficulty, min_val=min_difficulty, max_val=max_difficulty)
    w100 = normalize_workload_100(
        other_assignments,
        perceived_length=perceived_length,
        max_assignments=max_assignments,
        max_length=max_length,
        length_weight=length_weight,
    )
    return compute_weighted_score(u100, d100, w100, weights=weights)


def rank_by_weighted_score(
    df: pd.DataFrame,
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
    days_col: Optional[str] = None,
    diff_col: Optional[str] = None,
    workload_col: Optional[str] = None,
    length_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Batch process and rank tasks by Baseline B Weighted Scoring.
    
    Adds:
      - 'weighted_score': Calculated score [0, 100].
      - 'weighted_rank': Ordinal rank (1 = highest priority).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("DataFrame is empty.")
        
    res = df.copy()
    
    # Auto-resolve columns
    def _find_col(cands: List[str], provided: Optional[str]) -> str:
        if provided and provided in res.columns:
            return provided
        for c in cands:
            if c in res.columns:
                return c
        raise KeyError(f"Could not find required column from {cands}")

    c_days = _find_col(["days_till_deadline_started", "days_remaining", "days_till_deadline"], days_col)
    c_diff = _find_col(["perceived_difficulty", "difficulty"], diff_col)
    c_work = _find_col(["num_other_assignments", "other_assignments", "workload"], workload_col)
    c_len = None
    for cand in ["perceived_length", "length", "duration"]:
        if cand in res.columns:
            c_len = cand
            break

    scores = []
    for _, row in res.iterrows():
        len_val = row[c_len] if (c_len and pd.notnull(row[c_len])) else None
        p = calculate_weighted_priority(
            row[c_days],
            row[c_diff],
            row[c_work],
            perceived_length=len_val,
            weights=weights,
        )
        scores.append(round(p, 2))
        
    res["weighted_score"] = scores
    res = res.sort_values(by="weighted_score", ascending=False).reset_index(drop=True)
    res["weighted_rank"] = res.index + 1
    
    return res


# ============================================================================
# Integrated Multi-Method Batch Prioritization
# ============================================================================

def batch_prioritize_all(
    df: pd.DataFrame,
    weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
    include_fuzzy: bool = True,
    days_col: Optional[str] = None,
    diff_col: Optional[str] = None,
    workload_col: Optional[str] = None,
    length_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Unified batch prioritization that executes all three methods:
      1. Baseline A: Earliest-Deadline-First (EDF)
      2. Baseline B: Weighted Linear Scoring
      3. Fuzzy Inference System (if include_fuzzy=True)
      
    Returns a unified DataFrame enriched with scores, ranks, and rank comparison diffs.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("DataFrame is empty.")
        
    res = df.copy()
    
    # Resolve columns
    def _find_col(cands: List[str], provided: Optional[str]) -> str:
        if provided and provided in res.columns:
            return provided
        for c in cands:
            if c in res.columns:
                return c
        raise KeyError(f"Could not find required column from {cands}")

    c_days = _find_col(["days_till_deadline_started", "days_remaining", "days_till_deadline"], days_col)
    c_diff = _find_col(["perceived_difficulty", "difficulty"], diff_col)
    c_work = _find_col(["num_other_assignments", "other_assignments", "workload"], workload_col)
    c_len = None
    for cand in ["perceived_length", "length", "duration"]:
        if cand in res.columns:
            c_len = cand
            break

    # Normalize inputs
    urgency_norm = []
    diff_norm = []
    work_norm = []
    edf_scores = []
    weighted_scores = []
    
    for _, row in res.iterrows():
        len_val = row[c_len] if (c_len and pd.notnull(row[c_len])) else None
        u = normalize_urgency_100(row[c_days])
        d = normalize_difficulty_100(row[c_diff])
        w = normalize_workload_100(row[c_work], perceived_length=len_val)
        
        urgency_norm.append(round(u, 2))
        diff_norm.append(round(d, 2))
        work_norm.append(round(w, 2))
        
        # EDF
        edf_scores.append(round(u, 2))
        
        # Weighted Scoring
        w_score = compute_weighted_score(u, d, w, weights=weights)
        weighted_scores.append(round(w_score, 2))

    res["urgency_100"] = urgency_norm
    res["difficulty_100"] = diff_norm
    res["workload_100"] = work_norm
    res["edf_score"] = edf_scores
    res["weighted_score"] = weighted_scores
    
    # Assign EDF rank (sort by days ASC, break ties with difficulty DESC)
    # Using pandas rank to preserve original order
    # For EDF, lower days = better rank (1 = highest priority).
    res["edf_rank"] = res[c_days].rank(method="min", ascending=True).astype(int)
    
    # For Weighted, higher score = better rank (1 = highest priority).
    res["weighted_rank"] = res["weighted_score"].rank(method="min", ascending=False).astype(int)
    
    # Fuzzy Engine Integration
    if include_fuzzy:
        try:
            from fuzzy_engine import calculate_priority, classify_priority
            fuzzy_scores = []
            fuzzy_cats = []
            for _, row in res.iterrows():
                len_val = row[c_len] if (c_len and pd.notnull(row[c_len])) else None
                f_prio = calculate_priority(
                    row[c_days],
                    row[c_diff],
                    row[c_work],
                    perceived_length=len_val,
                )
                fuzzy_scores.append(round(f_prio, 2))
                fuzzy_cats.append(classify_priority(f_prio))
                
            res["fuzzy_priority"] = fuzzy_scores
            res["fuzzy_category"] = fuzzy_cats
            res["fuzzy_rank"] = res["fuzzy_priority"].rank(method="min", ascending=False).astype(int)
            
            # Rank Shifts (Positive = Fuzzy prioritizes it higher than baseline)
            res["shift_fuzzy_vs_edf"] = res["edf_rank"] - res["fuzzy_rank"]
            res["shift_fuzzy_vs_weighted"] = res["weighted_rank"] - res["fuzzy_rank"]
            
        except ImportError as e:
            print(f"Warning: Could not import fuzzy_engine ({e}). Skipping fuzzy scores.")
            
    return res
