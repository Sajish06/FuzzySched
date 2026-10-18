# FuzzySched: Fuzzy Logic-Based Student Workload and Deadline Prioritization System

A college Soft Computing project implementing a complete Mamdani Fuzzy Inference System (FIS) for intelligent student task prioritization.

## Project Overview

Students frequently struggle with prioritizing academic tasks when deadlines, task difficulty, and concurrent workloads clash. Traditional linear rules (like FIFO or earliest-deadline-first) ignore task complexity and cognitive overload. This engine uses **Fuzzy Set Theory** to model human-like prioritization, outputting a calibrated priority score from **0.0 (negligible priority)** to **100.0 (maximum critical priority)**.

### Features
- **3 Linguistic Inputs**: Deadline Urgency (Low, Medium, High), Task Difficulty (Easy, Moderate, Hard), and Current Workload (Light, Moderate, Heavy).
- **Composite Workload**: Synthesizes concurrent assignment count with estimated task length.
- **27 Mamdani Rules**: Fully documented knowledge base adhering to pedagogical principles (critical deadline dominance, backlog risk prevention).
- **Centroid Defuzzification**: Monotonic, smooth continuous output with symmetric Ruspini partitions (0 directional inversions).
- **Batch Processing**: Enriches pandas DataFrames of tasks without mutating original data.
- **Independent & Modular**: Pure Python engine without Streamlit dependencies, ready for dashboard integration.

---

## Directory Structure

```
FuzzySched/
├── src/
│   ├── fuzzy_engine.py           # Core Mamdani inference engine & public API
│   ├── membership_functions.py   # Mathematical MFs (trimf, trapmf) & plotting
│   └── fuzzy_rules.py            # 27-rule knowledge base & rationales
├── tests/
│   └── test_fuzzy_engine.py      # Automated pytest suite (17/17 tests passing)
├── data/
│   └── sample_tasks.csv          # Sample dataset spanning all edge cases
├── dataset/
│   └── studentsessions.csv       # Original student session dataset
├── outputs/
│   ├── membership_functions.png  # Publication-grade membership function plot
│   ├── rule_base.csv             # 27 rules table with soft computing rationales
│   └── test_results.csv          # Evaluated test cases and active rule logs
├── scripts/
│   └── generate_test_results.py  # Script to regenerate outputs/test_results.csv
├── docs/
│   └── fuzzy_design.md           # Full mathematical specification & design doc
├── requirements.txt              # Project dependencies
├── pytest.ini                    # Pytest configuration
└── README.md                     # Project overview and quick start guide
```

---

## Quick Start

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Automated Tests
```bash
python -m pytest tests/test_fuzzy_engine.py -v
```

### 3. Usage Example (Single Task)
```python
from src.fuzzy_engine import calculate_priority

# Example: Task due in 1 day, difficulty 4/5, with 3 other assignments
score = calculate_priority(
    days_remaining=1.0,
    difficulty=4.0,
    other_assignments=3,
    perceived_length=6.0
)
print(f"Priority Score: {score:.2f}/100")
```

### 4. Batch Process Tasks DataFrame
```python
import pandas as pd
from src.fuzzy_engine import batch_calculate_priority

df = pd.read_csv("data/sample_tasks.csv")
enriched_df = batch_calculate_priority(df)
print(enriched_df[["title", "fuzzy_priority", "priority_category"]].head())
```
