import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import task_store as ts


def test_add_load_delete_roundtrip(tmp_path):
    p = tmp_path / "t.csv"
    ts.add_task("A", 2, 4, 3, None, path=p)
    ts.add_task("B", -1, 1, 0, 5, path=p)
    df, dropped = ts.load_tasks(p)
    assert dropped == 0 and list(df.title) == ["A", "B"]
    ts.delete_task(df.index[0], path=p)
    assert list(ts.load_tasks(p)[0].title) == ["B"]


@pytest.mark.parametrize("args", [("", 1, 3, 0), ("x", "abc", 3, 0), ("x", 1, 6, 0),
                                  ("x", 1, 3, -1), ("x", float("nan"), 3, 0), ("x", 1, 3, 0, -2)])
def test_invalid_input_rejected(tmp_path, args):
    with pytest.raises(ValueError):
        ts.add_task(*args, path=tmp_path / "t.csv")
    assert not (tmp_path / "t.csv").exists()


def test_bad_rows_skipped_not_deleted(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("title,days_remaining,difficulty,other_assignments,perceived_length\nok,1,3,0,\nbad,x,9,0,\n")
    df, dropped = ts.load_tasks(p)
    assert len(df) == 1 and dropped == 1
    assert "bad" in p.read_text()


def test_missing_file_is_empty(tmp_path):
    assert ts.load_tasks(tmp_path / "none.csv")[0].empty


def test_rank_tasks_columns_and_deadline_order(tmp_path):
    pytest.importorskip("skfuzzy")
    p = tmp_path / "t.csv"
    ts.add_task("easy tomorrow", 1, 1, 0, path=p)
    ts.add_task("hard in 3 days", 3, 5, 5, path=p)
    ts.add_task("overdue", -2, 3, 1, path=p)
    out = ts.rank_tasks(ts.load_tasks(p)[0])
    assert out.set_index("title").edf_rank.to_dict() == {"overdue": 1, "easy tomorrow": 2, "hard in 3 days": 3}
    assert out.fuzzy_priority.between(0, 100).all() and out.weighted_score.between(0, 100).all()
