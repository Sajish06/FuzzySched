"""CSV-backed task storage, input validation and the three rankings the dashboard shows."""
import math
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
STORE = DATA / "my_tasks.csv"
SAMPLE = DATA / "sample_tasks.csv"
COLS = ["title", "days_remaining", "difficulty", "other_assignments", "perceived_length"]


def _clean(title, days, difficulty, others, length=None):
    """Validate one task; return (title, days, difficulty, others, length). Raises ValueError."""
    title = "" if title is None or pd.isna(title) else str(title).strip()
    if not title:
        raise ValueError("Task name is required.")
    try:
        days, difficulty, others = float(days), float(difficulty), float(others)
        length = None if length is None or pd.isna(length) else float(length)
    except (TypeError, ValueError):
        raise ValueError("Days, difficulty, assignments and length must be numbers.") from None
    if not all(math.isfinite(v) for v in (days, difficulty, others) + ((length,) if length is not None else ())):
        raise ValueError("Values must be finite numbers.")
    if not 1 <= difficulty <= 5:
        raise ValueError("Difficulty must be between 1 and 5.")
    if others < 0 or (length is not None and length < 0):
        raise ValueError("Assignments and length cannot be negative.")
    return title, days, difficulty, others, length


def load_tasks(path=STORE):
    """Return (tasks, dropped). Invalid rows are skipped, never deleted from the file.
    The index is the row's position in the CSV, so delete_task can target it."""
    try:
        raw = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=COLS), 0
    missing = [c for c in COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"{Path(path).name} is missing columns: {missing}")
    rows = {}
    for i, r in raw.iterrows():
        try:
            rows[i] = _clean(*(r[c] for c in COLS))
        except ValueError:
            pass
    return pd.DataFrame.from_dict(rows, orient="index", columns=COLS), len(raw) - len(rows)


def add_task(title, days, difficulty, others, length=None, path=STORE):
    row = _clean(title, days, difficulty, others, length)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row], columns=COLS)
    try:
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pass
    new.to_csv(path, index=False)


def delete_task(index, path=STORE):
    raw = pd.read_csv(path)
    raw.drop(index=index).to_csv(path, index=False)


def load_sample(path=STORE):
    """Copy the repo's sample tasks into the store so the dashboard has something to show."""
    s = pd.read_csv(SAMPLE).rename(columns={
        "days_till_deadline": "days_remaining",
        "perceived_difficulty": "difficulty",
        "num_other_assignments": "other_assignments",
    })
    s["title"] = s["title"].astype(str)
    s[COLS].to_csv(path, index=False)


def rank_tasks(tasks):
    """Score with all three methods (Member 2's baselines + the fuzzy engine); 1 = do first."""
    import fuzzy_engine  # noqa: F401 - fail loudly; batch_prioritize_all silently skips fuzzy on ImportError
    from baselines import batch_prioritize_all

    out = batch_prioritize_all(
        tasks, days_col="days_remaining", diff_col="difficulty",
        workload_col="other_assignments", length_col="perceived_length",
    )
    return out.sort_values(["fuzzy_rank", "days_remaining"])
