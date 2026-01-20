import ast
from pathlib import Path
import sys
import pandas as pd

# Set up project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]  
sys.path.insert(0, str(PROJECT_ROOT))


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

# Calculate action change ratio
def action_change_ratio(actions):
    n = len(actions)
    if n <= 1:
        return 0.0

    changes = 0
    prev = actions[0]
    for a in actions[1:]:
        if a != prev:
            changes += 1
        prev = a

    return changes / (n - 1)



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
    out["apply_ratio"] = out["num_apply"] / out["seq_length"] if out["seq_length"].all() > 0 else 0
    out["view_ratio"] = out["num_view"] / out["seq_length"] if out["seq_length"].all() > 0 else 0

    #Convert actions to binary sequence 
    action_map = {"view": False, "apply": True}

    def to_binary_actions(actions):
        return [action_map[a] for a in actions]

    out["actions_bin"] = out["actions_list"].apply(to_binary_actions)

    # 3) Recency -> last and previous actions and jobs
    out["last_job_id"] = out["jobs_list"].apply(last_item)
    out["prev_job_id"] = out["jobs_list"].apply(prev_item)
    out["last_action"] = out["actions_list"].apply(last_item)
    out["prev_action"] = out["actions_list"].apply(prev_item)
    out["is_last_action_changed"] = out["actions_list"].apply(changed_last_two)
    # Turn actions into binary --> it is not on our notebook but may be useful
    action_map = {"view": 0, "apply": 1}
    out["last_action_bin"] = out["last_action"].map(action_map).astype("int8")
    out["prev_action_bin"] = out["prev_action"].map(action_map).astype("int8")

    # Binary feature: last action is apply
    out["last_action_is_apply"] = (out["actions_list"].str.get(-1).eq("apply").fillna(False).astype("int8"))

    # action change ratio
    out["action_change_ratio"] = out["actions_list"].apply(action_change_ratio)

    # consecutive behavior
    def max_consecutive(actions, target):
        best = 0
        current = 0
        for a in actions:
            if a == target:
                current += 1
                if current > best:
                    best = current
            else:
                current = 0
        return best

    def tail_consecutive(actions, target):
        count = 0
        for a in reversed(actions):
            if a == target:
                count += 1
            else:
                break
        return count

    def switch_count(actions):
        if len(actions) < 2:
            return 0
        changes = 0
        prev = actions[0]
        for a in actions[1:]:
            if a != prev:
                changes += 1
            prev = a
        return changes

    # consecutive actions features
    def compute_max_view_run(actions):
        return max_consecutive(actions, "view")

    def compute_max_apply_run(actions):
        return max_consecutive(actions, "apply")

    def compute_last_view_run(actions):
        return tail_consecutive(actions, "view")

    def compute_last_apply_run(actions):
        return tail_consecutive(actions, "apply")

    out["max_view_run"] = out["actions_list"].apply(compute_max_view_run)
    out["max_apply_run"] = out["actions_list"].apply(compute_max_apply_run)
    out["last_view_run"] = out["actions_list"].apply(compute_last_view_run)
    out["last_apply_run"] = out["actions_list"].apply(compute_last_apply_run)
    out["switch_count"] = out["actions_list"].apply(switch_count)

    # session state based on runs and apply ratio
    def assign_state_from_runs(apply_ratio, max_view_run, max_apply_run):
        # thresholds: simples para começar; ajuste depois se quiser
        if apply_ratio <= 0.2 or max_view_run >= 4:
            return "exploratory"
        if apply_ratio >= 0.8 or max_apply_run >= 3:
            return "decisive"
        return "mixed"

    states = []
    for _, row in out.iterrows():
        st = assign_state_from_runs(row["apply_ratio"], row["max_view_run"], row["max_apply_run"])
        states.append(st)
    out["session_state"] = states

    out["is_exploratory"] = (out["session_state"] == "exploratory").astype("int8")
    out["is_mixed"]       = (out["session_state"] == "mixed").astype("int8")
    out["is_decisive"]    = (out["session_state"] == "decisive").astype("int8")


  



    # 4) Final table with selected features
    features = out[[
        #"session_id",
        "seq_length",
        #"num_view",
        #"num_apply",
        "apply_ratio",
        "view_ratio",
        #"actions_bin",
        # "last_job_id",
        #"prev_action",
        #"last_action",
        "last_action_is_apply",
        # "last_action_bin",
        # "prev_job_id",
        # "prev_action_bin",
        #"is_last_action_changed",
        "action_change_ratio",
        #"max_view_run",
        # "max_apply_run",
        # "last_view_run",
        # "last_apply_run",
        # "switch_count",
        # "is_exploratory",
        # "is_mixed",
        # "is_decisive",
    ]].copy()

    return features


def main():
    from .data_processor import DataProcessor  # Import here to avoid circular import
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

