"""
Fuzzy Inference Engine for Student Workload and Deadline Prioritization System.

This module implements a complete Mamdani Fuzzy Inference System (FIS) using scikit-fuzzy.
It transforms raw student task parameters (days until deadline, difficulty rating,
concurrent assignments, and optional perceived length) into normalized fuzzy universes,
evaluates a 27-rule knowledge base, and performs centroid defuzzification to generate
a calibrated priority score between 0 and 100.
"""

from typing import Dict, Any, Optional, Union, Tuple, List
import math
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from membership_functions import (
    INPUT_UNIVERSE,
    OUTPUT_UNIVERSE,
    DEFAULT_URGENCY_PARAMS,
    DEFAULT_DIFFICULTY_PARAMS,
    DEFAULT_WORKLOAD_PARAMS,
    DEFAULT_PRIORITY_PARAMS,
    fuzzify_urgency,
    fuzzify_difficulty,
    fuzzify_workload,
    fuzzify_priority,
    eval_trimf,
    eval_trapmf,
)
from fuzzy_rules import get_rule_base


# ============================================================================
# Domain Configuration & Normalization Defaults
# ============================================================================
DEFAULT_MAX_DEADLINE_DAYS: float = 14.0   # Beyond 14 days, urgency is 0.0 (Low)
DEFAULT_MIN_DIFFICULTY: float = 1.0       # Minimum rating on Likert scale
DEFAULT_MAX_DIFFICULTY: float = 5.0       # Maximum rating on Likert scale
DEFAULT_MAX_ASSIGNMENTS: float = 5.0      # Benchmark for heavy assignment backlog
DEFAULT_MAX_LENGTH: float = 20.0          # Max length benchmark based on dataset
DEFAULT_LENGTH_WEIGHT: float = 0.35       # Weight of length in composite workload


# ============================================================================
# Input Validation & Preprocessing
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


def map_days_to_urgency(days_remaining: float, max_days: float = DEFAULT_MAX_DEADLINE_DAYS) -> float:
    """
    Map raw days remaining until deadline into a normalized urgency score [0, 10].
    
    Logic:
      - Overdue tasks (days_remaining < 0) or due today (days_remaining == 0)
        receive maximum urgency: 10.0.
      - Tasks with deadlines >= max_days receive minimum urgency: 0.0.
      - Between 0 and max_days, urgency decays linearly:
          urgency = 10.0 * (1.0 - days_remaining / max_days)
    """
    d = _validate_numeric(days_remaining, "days_remaining")
    if d <= 0.0:
        return 10.0
    if d >= max_days:
        return 0.0
    return float(10.0 * (1.0 - (d / max_days)))


def map_difficulty(
    raw_difficulty: float,
    min_val: float = DEFAULT_MIN_DIFFICULTY,
    max_val: float = DEFAULT_MAX_DIFFICULTY,
) -> float:
    """
    Map raw task difficulty (default 1 to 5 scale) to normalized scale [0, 10].
    
    Formula:
      normalized = 10.0 * (raw - min_val) / (max_val - min_val)
      Clamped strictly to [0.0, 10.0].
    """
    diff = _validate_numeric(raw_difficulty, "raw_difficulty")
    if max_val <= min_val:
        raise ValueError(f"max_val ({max_val}) must be strictly greater than min_val ({min_val}).")
    
    norm = 10.0 * (diff - min_val) / (max_val - min_val)
    return float(np.clip(norm, 0.0, 10.0))


def map_workload(
    other_assignments: float,
    perceived_length: Optional[float] = None,
    max_assignments: float = DEFAULT_MAX_ASSIGNMENTS,
    max_length: float = DEFAULT_MAX_LENGTH,
    length_weight: float = DEFAULT_LENGTH_WEIGHT,
) -> float:
    """
    Map concurrent assignments and optional perceived length to normalized workload [0, 10].
    
    Rationale:
      - Assignment count measures concurrent context-switching overhead.
      - If perceived_length is provided, workload is synthesized as a weighted composite:
          workload = (1 - w_len) * norm_assignments + w_len * norm_length
      - If perceived_length is None, workload is solely derived from assignment count.
      - Negative assignment counts or lengths are rejected.
    """
    assign = _validate_numeric(other_assignments, "other_assignments")
    if assign < 0.0:
        raise ValueError(f"other_assignments cannot be negative, got {assign}.")
        
    norm_assign = float(np.clip((assign / max_assignments) * 10.0, 0.0, 10.0))
    
    len_val = _validate_numeric(perceived_length, "perceived_length", allow_none=True)
    if len_val is None:
        return norm_assign
        
    if len_val < 0.0:
        raise ValueError(f"perceived_length cannot be negative, got {len_val}.")
        
    norm_len = float(np.clip((len_val / max_length) * 10.0, 0.0, 10.0))
    w = np.clip(length_weight, 0.0, 1.0)
    composite = (1.0 - w) * norm_assign + w * norm_len
    return float(np.clip(composite, 0.0, 10.0))


def classify_priority(score: float) -> str:
    """
    Convert a crisp priority score [0, 100] into a categorical linguistic label.
    
    Thresholds correspond to the standard partition centers:
      - [0.0, 17.5)   -> Very Low
      - [17.5, 37.5)  -> Low
      - [37.5, 62.5)  -> Medium
      - [62.5, 82.5)  -> High
      - [82.5, 100.0] -> Very High
    """
    if score < 17.5:
        return "Very Low"
    elif score < 37.5:
        return "Low"
    elif score < 62.5:
        return "Medium"
    elif score < 82.5:
        return "High"
    else:
        return "Very High"


# ============================================================================
# Core Fuzzy Engine Class
# ============================================================================

class FuzzyPriorityEngine:
    """
    Reusable Mamdani Fuzzy Inference Engine for Student Task Prioritization.
    
    Builds and manages scikit-fuzzy Antecedents, Consequent, and the complete 27-rule base.
    Uses centroid defuzzification for smooth, continuous priority output in [0, 100].
    """
    
    def __init__(
        self,
        urgency_params: Optional[Dict[str, Any]] = None,
        difficulty_params: Optional[Dict[str, Any]] = None,
        workload_params: Optional[Dict[str, Any]] = None,
        priority_params: Optional[Dict[str, Any]] = None,
    ):
        self.urgency_params = urgency_params or DEFAULT_URGENCY_PARAMS
        self.difficulty_params = difficulty_params or DEFAULT_DIFFICULTY_PARAMS
        self.workload_params = workload_params or DEFAULT_WORKLOAD_PARAMS
        self.priority_params = priority_params or DEFAULT_PRIORITY_PARAMS
        
        self._build_control_system()
        
    def _build_control_system(self) -> None:
        """Instantiate scikit-fuzzy control system with antecedents, consequent, and rules."""
        # Universes
        self.urgency_var = ctrl.Antecedent(INPUT_UNIVERSE, 'urgency')
        self.difficulty_var = ctrl.Antecedent(INPUT_UNIVERSE, 'difficulty')
        self.workload_var = ctrl.Antecedent(INPUT_UNIVERSE, 'workload')
        self.priority_var = ctrl.Consequent(OUTPUT_UNIVERSE, 'priority', defuzzify_method='centroid')
        
        # Populate Antecedent MFs
        for label, p in self.urgency_params.items():
            if len(p) == 3:
                self.urgency_var[label] = fuzz.trimf(self.urgency_var.universe, p)
            else:
                self.urgency_var[label] = fuzz.trapmf(self.urgency_var.universe, p)
                
        for label, p in self.difficulty_params.items():
            if len(p) == 3:
                self.difficulty_var[label] = fuzz.trimf(self.difficulty_var.universe, p)
            else:
                self.difficulty_var[label] = fuzz.trapmf(self.difficulty_var.universe, p)
                
        for label, p in self.workload_params.items():
            if len(p) == 3:
                self.workload_var[label] = fuzz.trimf(self.workload_var.universe, p)
            else:
                self.workload_var[label] = fuzz.trapmf(self.workload_var.universe, p)
                
        # Populate Consequent MFs
        for label, p in self.priority_params.items():
            if len(p) == 3:
                self.priority_var[label] = fuzz.trimf(self.priority_var.universe, p)
            else:
                self.priority_var[label] = fuzz.trapmf(self.priority_var.universe, p)
                
        # Construct 27 Mamdani Rules from knowledge base
        rule_defs = get_rule_base()
        skf_rules = []
        for r in rule_defs:
            u_key = r["urgency"].lower().replace(' ', '_')
            d_key = r["difficulty"].lower().replace(' ', '_')
            w_key = r["workload"].lower().replace(' ', '_')
            p_key = r["priority"].lower().replace(' ', '_')
            
            antecedent_clause = (
                self.urgency_var[u_key] & 
                self.difficulty_var[d_key] & 
                self.workload_var[w_key]
            )
            consequent_clause = self.priority_var[p_key]
            skf_rules.append(ctrl.Rule(antecedent_clause, consequent_clause))
            
        self.control_system = ctrl.ControlSystem(skf_rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
        
    def compute(self, urgency_score: float, difficulty_score: float, workload_score: float) -> float:
        """
        Execute fuzzy inference and centroid defuzzification.
        
        Args:
            urgency_score: Urgency in [0.0, 10.0]
            difficulty_score: Difficulty in [0.0, 10.0]
            workload_score: Workload in [0.0, 10.0]
            
        Returns:
            Defuzzified priority score in [0.0, 100.0]
        """
        u = float(np.clip(_validate_numeric(urgency_score, "urgency_score"), 0.0, 10.0))
        d = float(np.clip(_validate_numeric(difficulty_score, "difficulty_score"), 0.0, 10.0))
        w = float(np.clip(_validate_numeric(workload_score, "workload_score"), 0.0, 10.0))
        
        self.simulation.input['urgency'] = u
        self.simulation.input['difficulty'] = d
        self.simulation.input['workload'] = w
        
        self.simulation.compute()
        score = float(self.simulation.output['priority'])
        return float(np.clip(score, 0.0, 100.0))

    def explain(self, urgency_score: float, difficulty_score: float, workload_score: float) -> Dict[str, Any]:
        """
        Explain the fuzzy inference process in detail.
        Returns membership degrees, firing strengths of active rules, and crisp output.
        """
        u = float(np.clip(_validate_numeric(urgency_score, "urgency_score"), 0.0, 10.0))
        d = float(np.clip(_validate_numeric(difficulty_score, "difficulty_score"), 0.0, 10.0))
        w = float(np.clip(_validate_numeric(workload_score, "workload_score"), 0.0, 10.0))
        
        score = self.compute(u, d, w)
        cat = classify_priority(score)
        
        u_mem = fuzzify_urgency(u, self.urgency_params)
        d_mem = fuzzify_difficulty(d, self.difficulty_params)
        w_mem = fuzzify_workload(w, self.workload_params)
        
        active_rules = []
        for r in get_rule_base():
            u_key = r["urgency"].lower().replace(' ', '_')
            d_key = r["difficulty"].lower().replace(' ', '_')
            w_key = r["workload"].lower().replace(' ', '_')
            
            mu_u = u_mem.get(u_key, 0.0)
            mu_d = d_mem.get(d_key, 0.0)
            mu_w = w_mem.get(w_key, 0.0)
            
            firing_strength = min(mu_u, mu_d, mu_w)
            if firing_strength > 1e-4:
                active_rules.append({
                    "rule_id": r["rule_id"],
                    "urgency": r["urgency"],
                    "difficulty": r["difficulty"],
                    "workload": r["workload"],
                    "consequent_priority": r["priority"],
                    "firing_strength": round(firing_strength, 4),
                    "rationale": r["rationale"],
                })
                
        # Sort active rules by firing strength descending
        active_rules.sort(key=lambda x: x["firing_strength"], reverse=True)
        
        return {
            "inputs": {
                "urgency_score": round(u, 2),
                "difficulty_score": round(d, 2),
                "workload_score": round(w, 2),
            },
            "membership_degrees": {
                "urgency": {k: round(v, 4) for k, v in u_mem.items()},
                "difficulty": {k: round(v, 4) for k, v in d_mem.items()},
                "workload": {k: round(v, 4) for k, v in w_mem.items()},
            },
            "active_rules_count": len(active_rules),
            "active_rules": active_rules,
            "crisp_priority": round(score, 2),
            "priority_category": cat,
        }


# Global singleton instance for high performance
_DEFAULT_ENGINE: Optional[FuzzyPriorityEngine] = None

def get_engine() -> FuzzyPriorityEngine:
    """Return or lazily initialize the default FuzzyPriorityEngine instance."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = FuzzyPriorityEngine()
    return _DEFAULT_ENGINE


# ============================================================================
# Public API Functions
# ============================================================================

def calculate_priority(
    days_remaining: Union[int, float],
    difficulty: Union[int, float],
    other_assignments: Union[int, float],
    perceived_length: Optional[Union[int, float]] = None,
    max_days: float = DEFAULT_MAX_DEADLINE_DAYS,
    min_difficulty: float = DEFAULT_MIN_DIFFICULTY,
    max_difficulty: float = DEFAULT_MAX_DIFFICULTY,
    max_assignments: float = DEFAULT_MAX_ASSIGNMENTS,
    max_length: float = DEFAULT_MAX_LENGTH,
    length_weight: float = DEFAULT_LENGTH_WEIGHT,
    engine: Optional[FuzzyPriorityEngine] = None,
) -> float:
    """
    Public API to calculate crisp priority score (0 to 100) for a student task.
    
    Parameters:
      days_remaining: Number of days until deadline (<=0 indicates overdue/due today).
      difficulty: Task difficulty rating (e.g. 1 to 5 on Likert scale).
      other_assignments: Count of concurrent competing assignments.
      perceived_length: Optional estimated length/duration of the assignment.
      max_days: Deadline threshold where urgency becomes 0.0 (default 14.0).
      min_difficulty: Min value of raw difficulty scale (default 1.0).
      max_difficulty: Max value of raw difficulty scale (default 5.0).
      max_assignments: Upper count benchmark for workload (default 5.0).
      max_length: Upper length benchmark for workload (default 20.0).
      length_weight: Blend weight of length in workload (default 0.35).
      engine: Optional pre-configured FuzzyPriorityEngine instance.
      
    Returns:
      Crisp priority score in [0.0, 100.0].
    """
    urgency_norm = map_days_to_urgency(days_remaining, max_days=max_days)
    difficulty_norm = map_difficulty(difficulty, min_val=min_difficulty, max_val=max_difficulty)
    workload_norm = map_workload(
        other_assignments,
        perceived_length=perceived_length,
        max_assignments=max_assignments,
        max_length=max_length,
        length_weight=length_weight,
    )
    
    active_engine = engine or get_engine()
    return active_engine.compute(urgency_norm, difficulty_norm, workload_norm)


def batch_calculate_priority(
    df: pd.DataFrame,
    days_col: Optional[str] = None,
    diff_col: Optional[str] = None,
    workload_col: Optional[str] = None,
    length_col: Optional[str] = None,
    max_days: float = DEFAULT_MAX_DEADLINE_DAYS,
    min_difficulty: float = DEFAULT_MIN_DIFFICULTY,
    max_difficulty: float = DEFAULT_MAX_DIFFICULTY,
    max_assignments: float = DEFAULT_MAX_ASSIGNMENTS,
    max_length: float = DEFAULT_MAX_LENGTH,
    length_weight: float = DEFAULT_LENGTH_WEIGHT,
    engine: Optional[FuzzyPriorityEngine] = None,
) -> pd.DataFrame:
    """
    Batch process a pandas DataFrame of student tasks.
    
    Adds normalized input columns and fuzzy priority scores without mutating
    the original DataFrame. Supports flexible column resolution matching dataset conventions.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")
        
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
        
    result_df = df.copy()
    
    # Case-insensitive column matcher
    cols_lower = {str(c).lower().strip(): c for c in df.columns}
    
    def resolve_col(candidates: List[str], provided: Optional[str]) -> Optional[str]:
        if provided:
            if provided in df.columns:
                return provided
            low = provided.lower().strip()
            if low in cols_lower:
                return cols_lower[low]
            raise KeyError(f"Specified column '{provided}' not found in DataFrame columns: {list(df.columns)}")
        for cand in candidates:
            if cand in df.columns:
                return cand
            low = cand.lower().strip()
            if low in cols_lower:
                return cols_lower[low]
        return None

    actual_days_col = resolve_col(
        ["days_till_deadline_started", "days_till_deadline", "days_remaining", "days_left"],
        days_col
    )
    actual_diff_col = resolve_col(
        ["perceived_difficulty", "difficulty", "task_difficulty"],
        diff_col
    )
    actual_work_col = resolve_col(
        ["num_other_assignments", "other_assignments", "workload"],
        workload_col
    )
    actual_len_col = resolve_col(
        ["perceived_length", "length", "duration"],
        length_col
    )
    
    if actual_days_col is None:
        raise KeyError("Could not identify deadline days column. Please specify 'days_col'.")
    if actual_diff_col is None:
        raise KeyError("Could not identify difficulty column. Please specify 'diff_col'.")
    if actual_work_col is None:
        raise KeyError("Could not identify workload/assignments column. Please specify 'workload_col'.")
        
    active_engine = engine or get_engine()
    
    urgency_list = []
    diff_list = []
    workload_list = []
    priority_list = []
    category_list = []
    
    for _, row in result_df.iterrows():
        d_val = row[actual_days_col]
        diff_val = row[actual_diff_col]
        work_val = row[actual_work_col]
        len_val = row[actual_len_col] if (actual_len_col and actual_len_col in row and pd.notnull(row[actual_len_col])) else None
        
        u_norm = map_days_to_urgency(d_val, max_days=max_days)
        d_norm = map_difficulty(diff_val, min_val=min_difficulty, max_val=max_difficulty)
        w_norm = map_workload(
            work_val,
            perceived_length=len_val,
            max_assignments=max_assignments,
            max_length=max_length,
            length_weight=length_weight,
        )
        prio = active_engine.compute(u_norm, d_norm, w_norm)
        cat = classify_priority(prio)
        
        urgency_list.append(round(u_norm, 2))
        diff_list.append(round(d_norm, 2))
        workload_list.append(round(w_norm, 2))
        priority_list.append(round(prio, 2))
        category_list.append(cat)
        
    result_df["urgency_normalized"] = urgency_list
    result_df["difficulty_normalized"] = diff_list
    result_df["workload_normalized"] = workload_list
    result_df["fuzzy_priority"] = priority_list
    result_df["priority_category"] = category_list
    
    return result_df
