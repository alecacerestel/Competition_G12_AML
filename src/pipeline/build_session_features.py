import ast
from pathlib import Path
import sys
import pandas as pd

# Ajuste o root do projeto se necessário
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # <-- era [1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_processor import DataProcessor  # se você já usa isso


# ---------- Helpers ----------
def parse_list_column(value):
    """convertion strings to list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return ast.literal_eval(value)
    return []

# Count how many apply actions exist in the session.
def count_apply(actions_list):
    return actions_list.count("apply")


# Get last item of a list
def last_item(lst):
    return lst[-1]


# Get previous item of a list
def prev_item(lst):
    return lst[-2]


# Check if last two actions are different
def changed_last_two(actions):
    return int(actions[-1] != actions[-2])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: DataFrame with columns [session_id, job_ids, actions]
    Output: one row per session with numeric/categorical session-level features.
    """
    out = df.copy()

    # 1) Parse string -> list
    out["jobs_list"] = out["job_ids"].apply(parse_list_column)
    out["actions_list"] = out["actions"].apply(parse_list_column)

    # Just check if jobs_list and actions_list have same length
    ok = out["jobs_list"].apply(len) == out["actions_list"].apply(len)
    if not ok.all():
        bad = out.loc[~ok, ["session_id", "job_ids", "actions"]].head(5)
        raise ValueError(f"Mismatch between jobs_list and actions_list. Examples:\n{bad}")

    # 2) Basic behavior -> length, counts, ratios
    out["seq_length"] = out["jobs_list"].apply(len)
    out["num_apply"] = out["actions_list"].apply(count_apply)
    out["num_view"] = out["seq_length"] - out["num_apply"]
    out["apply_ratio"] = out["num_apply"] / out["seq_length"]

    # 3) Recency -> last and previous actions and jobs
    out["last_job_id"] = out["jobs_list"].apply(last_item)
    out["prev_job_id"] = out["jobs_list"].apply(prev_item)
    out["last_action"] = out["actions_list"].apply(last_item)
    out["prev_action"] = out["actions_list"].apply(prev_item)
    out["action_changed"] = out["actions_list"].apply(changed_last_two)
    # Turn actions into binary --> it is not on our notebook but may be useful
    action_map = {"view": 0, "apply": 1}
    out["last_action_bin"] = out["last_action"].map(action_map).astype("int8")
    out["prev_action_bin"] = out["prev_action"].map(action_map).astype("int8")


    # 4) Final table with selected features
    features = out[[
        "session_id",
        "seq_length",
        "num_view",
        "num_apply",
        "apply_ratio",
        "last_job_id",
        "last_action",
        "last_action_bin",
        "prev_job_id",
        "prev_action",
        "prev_action_bin",
        "action_changed",
    ]].copy()

    return features


def main():
    # load raw data
    processor = DataProcessor()
    jobs, x_train, y_train, x_test = processor.load_data()

    # Build features for train/test sessions
    train_features = build_features(x_train)
    test_features = build_features(x_test)

    # saving features
    output_dir = PROJECT_ROOT / "Data" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)


    train_path = output_dir / "session_features_train.csv"
    test_path = output_dir / "session_features_test.csv"

    train_features.to_csv(train_path, index=False)
    test_features.to_csv(test_path, index=False)

    print(f"Saved {train_path}  shape={train_features.shape}")
    print(f"Saved {test_path}   shape={test_features.shape}")


if __name__ == "__main__":
    main()

