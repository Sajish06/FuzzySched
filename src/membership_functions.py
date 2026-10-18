"""
Membership Functions for Fuzzy Logic-Based Student Workload and Deadline Prioritization System.

This module defines the universes of discourse, membership function parameters,
mathematical evaluation functions (triangular and trapezoidal), and visualization utilities.
All parameters are configurable to allow easy calibration and sensitivity testing.
"""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

# ============================================================================
# Universes of Discourse
# ============================================================================
# Inputs: 0 to 10 scale (resolution 0.1 for high precision)
UNIVERSE_INPUT_MIN = 0.0
UNIVERSE_INPUT_MAX = 10.0
INPUT_STEP = 0.1
INPUT_UNIVERSE = np.arange(UNIVERSE_INPUT_MIN, UNIVERSE_INPUT_MAX + INPUT_STEP / 2, INPUT_STEP)

# Output: 0 to 100 scale (resolution 0.5 for centroid integration)
UNIVERSE_OUTPUT_MIN = 0.0
UNIVERSE_OUTPUT_MAX = 100.0
OUTPUT_STEP = 0.5
OUTPUT_UNIVERSE = np.arange(UNIVERSE_OUTPUT_MIN, UNIVERSE_OUTPUT_MAX + OUTPUT_STEP / 2, OUTPUT_STEP)

# ============================================================================
# Configurable Default Parameters
# ============================================================================
# Input 1: Urgency (Low, Medium, High)
# Low: Left-shoulder trapezoid [0, 0, 2.0, 5.0]
# Medium: Symmetric triangle [2.5, 5.0, 7.5]
# High: Right-shoulder trapezoid [5.0, 8.0, 10.0, 10.0]
DEFAULT_URGENCY_PARAMS: Dict[str, List[float]] = {
    'low': [0.0, 0.0, 2.0, 5.0],
    'medium': [2.5, 5.0, 7.5],
    'high': [5.0, 8.0, 10.0, 10.0],
}

# Input 2: Task Difficulty (Easy, Moderate, Hard)
# Easy: Left-shoulder trapezoid [0, 0, 2.0, 5.0]
# Moderate: Symmetric triangle [2.5, 5.0, 7.5]
# Hard: Right-shoulder trapezoid [5.0, 8.0, 10.0, 10.0]
DEFAULT_DIFFICULTY_PARAMS: Dict[str, List[float]] = {
    'easy': [0.0, 0.0, 2.0, 5.0],
    'moderate': [2.5, 5.0, 7.5],
    'hard': [5.0, 8.0, 10.0, 10.0],
}

# Input 3: Workload (Light, Moderate, Heavy)
# Light: Left-shoulder trapezoid [0, 0, 2.0, 5.0]
# Moderate: Symmetric triangle [2.5, 5.0, 7.5]
# Heavy: Right-shoulder trapezoid [5.0, 8.0, 10.0, 10.0]
DEFAULT_WORKLOAD_PARAMS: Dict[str, List[float]] = {
    'light': [0.0, 0.0, 2.0, 5.0],
    'moderate': [2.5, 5.0, 7.5],
    'heavy': [5.0, 8.0, 10.0, 10.0],
}

# Output: Priority (Very Low, Low, Medium, High, Very High)
# Symmetrically balanced partition across [0, 100] guaranteeing monotonicity
DEFAULT_PRIORITY_PARAMS: Dict[str, List[float]] = {
    'very_low': [0.0, 0.0, 10.0, 25.0],
    'low': [0.0, 25.0, 50.0],
    'medium': [25.0, 50.0, 75.0],
    'high': [50.0, 75.0, 100.0],
    'very_high': [75.0, 90.0, 100.0, 100.0],
}


# ============================================================================
# Mathematical Membership Function Evaluators
# ============================================================================

def eval_trimf(x: Union[float, np.ndarray], a: float, b: float, c: float) -> Union[float, np.ndarray]:
    """
    Evaluate triangular membership function at point(s) x.
    
    Formula:
      mu(x) = 0,                         if x <= a or x >= c
      mu(x) = (x - a) / (b - a),         if a < x <= b
      mu(x) = (c - x) / (c - b),         if b < x < c
      mu(x) = 1,                         if x == b
    """
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr)
    
    # Left slope
    if a != b:
        left = (x_arr > a) & (x_arr <= b)
        y[left] = (x_arr[left] - a) / (b - a)
    else:
        y[x_arr == a] = 1.0
        
    # Right slope
    if b != c:
        right = (x_arr > b) & (x_arr < c)
        y[right] = (c - x_arr[right]) / (c - b)
        
    y[x_arr == b] = 1.0
    
    if np.isscalar(x):
        return float(y.item())
    return y


def eval_trapmf(x: Union[float, np.ndarray], a: float, b: float, c: float, d: float) -> Union[float, np.ndarray]:
    """
    Evaluate trapezoidal membership function at point(s) x.
    
    Formula:
      mu(x) = 0,                         if x <= a or x >= d
      mu(x) = (x - a) / (b - a),         if a < x < b
      mu(x) = 1,                         if b <= x <= c
      mu(x) = (d - x) / (d - c),         if c < x < d
    """
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr)
    
    # Plateau
    plateau = (x_arr >= b) & (x_arr <= c)
    y[plateau] = 1.0
    
    # Left slope
    if a != b:
        left = (x_arr > a) & (x_arr < b)
        y[left] = (x_arr[left] - a) / (b - a)
        
    # Right slope
    if c != d:
        right = (x_arr > c) & (x_arr < d)
        y[right] = (d - x_arr[right]) / (d - c)
        
    if np.isscalar(x):
        return float(y.item())
    return y


def compute_membership(x: float, params: List[float]) -> float:
    """Helper to evaluate either triangular (len 3) or trapezoidal (len 4) MF."""
    if len(params) == 3:
        return float(eval_trimf(x, params[0], params[1], params[2]))
    elif len(params) == 4:
        return float(eval_trapmf(x, params[0], params[1], params[2], params[3]))
    else:
        raise ValueError(f"Invalid membership parameter length: {len(params)}. Expected 3 or 4.")


def fuzzify_variable(val: float, params_dict: Dict[str, List[float]]) -> Dict[str, float]:
    """Compute membership degrees for all linguistic sets of a variable."""
    return {label: compute_membership(val, p) for label, p in params_dict.items()}


def fuzzify_urgency(val: float, params: Optional[Dict[str, List[float]]] = None) -> Dict[str, float]:
    """Calculate membership degrees of an urgency score in [Low, Medium, High]."""
    p = params or DEFAULT_URGENCY_PARAMS
    return fuzzify_variable(val, p)


def fuzzify_difficulty(val: float, params: Optional[Dict[str, List[float]]] = None) -> Dict[str, float]:
    """Calculate membership degrees of a difficulty score in [Easy, Moderate, Hard]."""
    p = params or DEFAULT_DIFFICULTY_PARAMS
    return fuzzify_variable(val, p)


def fuzzify_workload(val: float, params: Optional[Dict[str, List[float]]] = None) -> Dict[str, float]:
    """Calculate membership degrees of a workload score in [Light, Moderate, Heavy]."""
    p = params or DEFAULT_WORKLOAD_PARAMS
    return fuzzify_variable(val, p)


def fuzzify_priority(val: float, params: Optional[Dict[str, List[float]]] = None) -> Dict[str, float]:
    """Calculate membership degrees of a priority score in [Very Low, Low, Medium, High, Very High]."""
    p = params or DEFAULT_PRIORITY_PARAMS
    return fuzzify_variable(val, p)


# ============================================================================
# Visualization & Plotting Utility
# ============================================================================

def plot_all_membership_functions(
    output_path: str = "outputs/membership_functions.png",
    urgency_params: Optional[Dict[str, List[float]]] = None,
    difficulty_params: Optional[Dict[str, List[float]]] = None,
    workload_params: Optional[Dict[str, List[float]]] = None,
    priority_params: Optional[Dict[str, List[float]]] = None,
    dpi: int = 300,
) -> str:
    """
    Plot high-resolution publication-quality membership functions for all 3 inputs and the output.
    Saves the figure to output_path.
    """
    urg_p = urgency_params or DEFAULT_URGENCY_PARAMS
    diff_p = difficulty_params or DEFAULT_DIFFICULTY_PARAMS
    work_p = workload_params or DEFAULT_WORKLOAD_PARAMS
    prio_p = priority_params or DEFAULT_PRIORITY_PARAMS

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Fuzzy Membership Functions: Student Workload & Deadline Prioritization Engine",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    palette = {
        'low': '#2563EB',      # Royal Blue
        'easy': '#2563EB',
        'light': '#2563EB',
        'very_low': '#1E3A8A', # Navy Blue
        
        'medium': '#F59E0B',   # Amber
        'moderate': '#F59E0B',
        
        'high': '#DC2626',     # Crimson
        'hard': '#DC2626',
        'heavy': '#DC2626',
        'very_high': '#991B1B',# Dark Red
    }

    # 1. Deadline Urgency
    ax_urg = axes[0, 0]
    for label, p in urg_p.items():
        if len(p) == 3:
            y = eval_trimf(INPUT_UNIVERSE, *p)
        else:
            y = eval_trapmf(INPUT_UNIVERSE, *p)
        color = palette.get(label, '#4B5563')
        ax_urg.plot(INPUT_UNIVERSE, y, label=label.capitalize(), color=color, linewidth=2.5)
        ax_urg.fill_between(INPUT_UNIVERSE, y, alpha=0.15, color=color)
    ax_urg.set_title("Input 1: Deadline Urgency", fontsize=12, fontweight="bold")
    ax_urg.set_xlabel("Urgency Score (0 - 10)", fontsize=10)
    ax_urg.set_ylabel("Degree of Membership (μ)", fontsize=10)
    ax_urg.set_xlim(0, 10)
    ax_urg.set_ylim(-0.05, 1.05)
    ax_urg.grid(True, linestyle="--", alpha=0.5)
    ax_urg.legend(loc="upper right", framealpha=0.9)

    # 2. Task Difficulty
    ax_diff = axes[0, 1]
    for label, p in diff_p.items():
        if len(p) == 3:
            y = eval_trimf(INPUT_UNIVERSE, *p)
        else:
            y = eval_trapmf(INPUT_UNIVERSE, *p)
        color = palette.get(label, '#4B5563')
        ax_diff.plot(INPUT_UNIVERSE, y, label=label.capitalize(), color=color, linewidth=2.5)
        ax_diff.fill_between(INPUT_UNIVERSE, y, alpha=0.15, color=color)
    ax_diff.set_title("Input 2: Task Difficulty", fontsize=12, fontweight="bold")
    ax_diff.set_xlabel("Normalized Difficulty (0 - 10)", fontsize=10)
    ax_diff.set_ylabel("Degree of Membership (μ)", fontsize=10)
    ax_diff.set_xlim(0, 10)
    ax_diff.set_ylim(-0.05, 1.05)
    ax_diff.grid(True, linestyle="--", alpha=0.5)
    ax_diff.legend(loc="upper right", framealpha=0.9)

    # 3. Current Workload
    ax_work = axes[1, 0]
    for label, p in work_p.items():
        if len(p) == 3:
            y = eval_trimf(INPUT_UNIVERSE, *p)
        else:
            y = eval_trapmf(INPUT_UNIVERSE, *p)
        color = palette.get(label, '#4B5563')
        ax_work.plot(INPUT_UNIVERSE, y, label=label.capitalize(), color=color, linewidth=2.5)
        ax_work.fill_between(INPUT_UNIVERSE, y, alpha=0.15, color=color)
    ax_work.set_title("Input 3: Current Workload", fontsize=12, fontweight="bold")
    ax_work.set_xlabel("Normalized Workload (0 - 10)", fontsize=10)
    ax_work.set_ylabel("Degree of Membership (μ)", fontsize=10)
    ax_work.set_xlim(0, 10)
    ax_work.set_ylim(-0.05, 1.05)
    ax_work.grid(True, linestyle="--", alpha=0.5)
    ax_work.legend(loc="upper right", framealpha=0.9)

    # 4. Output Priority
    ax_prio = axes[1, 1]
    prio_colors = {
        'very_low': '#1E3A8A', # Deep Blue
        'low': '#0284C7',      # Light Blue
        'medium': '#EAB308',   # Gold
        'high': '#EA580C',     # Orange
        'very_high': '#DC2626',# Crimson
    }
    for label, p in prio_p.items():
        if len(p) == 3:
            y = eval_trimf(OUTPUT_UNIVERSE, *p)
        else:
            y = eval_trapmf(OUTPUT_UNIVERSE, *p)
        color = prio_colors.get(label, '#4B5563')
        display_label = label.replace('_', ' ').title()
        ax_prio.plot(OUTPUT_UNIVERSE, y, label=display_label, color=color, linewidth=2.5)
        ax_prio.fill_between(OUTPUT_UNIVERSE, y, alpha=0.15, color=color)
    ax_prio.set_title("Output: Task Priority", fontsize=12, fontweight="bold")
    ax_prio.set_xlabel("Priority Score (0 - 100)", fontsize=10)
    ax_prio.set_ylabel("Degree of Membership (μ)", fontsize=10)
    ax_prio.set_xlim(0, 100)
    ax_prio.set_ylim(-0.05, 1.05)
    ax_prio.grid(True, linestyle="--", alpha=0.5)
    ax_prio.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return output_path
