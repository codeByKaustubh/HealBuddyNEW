import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.data import get_feature_columns, get_target_column


def _prepare_dataframe(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> pd.DataFrame:
    clean = df.copy()
    clean[target_col] = clean[target_col].astype(str).str.strip()
    for col in feature_cols:
        clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0)
        clean[col] = clean[col].clip(lower=0, upper=1).astype(int)
    return clean


def _add_pattern_key(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    out["_pattern_key"] = out[feature_cols].astype(str).agg("|".join, axis=1)
    return out


def _conflict_table(df_with_key: pd.DataFrame, target_col: str) -> pd.DataFrame:
    grouped = (
        df_with_key.groupby("_pattern_key")[target_col]
        .agg(
            unique_diseases="nunique",
            diseases=lambda x: ", ".join(sorted(set(x))),
        )
        .reset_index()
    )
    return grouped[grouped["unique_diseases"] > 1].copy()


def _resolve_by_majority_vote(df_with_key: pd.DataFrame, target_col: str) -> pd.DataFrame:
    vote_table = (
        df_with_key.groupby(["_pattern_key", target_col]).size().reset_index(name="votes")
    )
    vote_table = vote_table.sort_values(
        by=["_pattern_key", "votes", target_col], ascending=[True, False, True]
    )
    winners = vote_table.drop_duplicates(subset=["_pattern_key"], keep="first")[
        ["_pattern_key", target_col]
    ]
    feature_per_pattern = df_with_key.drop_duplicates(subset=["_pattern_key"], keep="first").drop(
        columns=[target_col]
    )
    merged = feature_per_pattern.merge(winners, on="_pattern_key", how="left")
    cols = [c for c in merged.columns if c != "_pattern_key"]
    return merged[cols].copy()


def clean_dataset(
    input_path: str,
    output_path: str,
    report_path: str,
    conflicts_path: str,
    strategy: str,
) -> Dict[str, object]:
    raw_df = pd.read_csv(input_path)
    target_col = get_target_column(raw_df)
    feature_cols = get_feature_columns(raw_df)

    prepared = _prepare_dataframe(raw_df, feature_cols, target_col)
    prepared_with_key = _add_pattern_key(prepared, feature_cols)

    input_rows = len(prepared_with_key)
    exact_duplicate_rows = int(prepared_with_key.duplicated().sum())
    dedup_df = prepared_with_key.drop_duplicates().copy()

    conflicts_df = _conflict_table(dedup_df, target_col)
    conflicting_keys = set(conflicts_df["_pattern_key"].tolist())
    conflict_rows = dedup_df[dedup_df["_pattern_key"].isin(conflicting_keys)].copy()

    if strategy == "drop_conflicts":
        cleaned = dedup_df[~dedup_df["_pattern_key"].isin(conflicting_keys)].copy()
        cleaned = cleaned.drop(columns=["_pattern_key"])
    elif strategy == "majority_vote":
        cleaned = _resolve_by_majority_vote(prepared_with_key, target_col)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(conflicts_path).parent.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(output_path, index=False)
    conflict_rows.to_csv(conflicts_path, index=False)

    report: Dict[str, object] = {
        "input_path": input_path,
        "strategy": strategy,
        "target_column": target_col,
        "feature_count": len(feature_cols),
        "input_rows": input_rows,
        "input_classes": int(prepared[target_col].nunique()),
        "exact_duplicate_rows": exact_duplicate_rows,
        "unique_rows_after_exact_dedup": int(len(dedup_df)),
        "conflicting_pattern_count": int(len(conflicts_df)),
        "conflicting_rows_after_exact_dedup": int(len(conflict_rows)),
        "output_path": output_path,
        "output_rows": int(len(cleaned)),
        "output_classes": int(cleaned[target_col].nunique()),
        "duplicate_ratio_input": round(exact_duplicate_rows / max(input_rows, 1), 6),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate and conflict-audit symptom-disease dataset."
    )
    parser.add_argument(
        "--input",
        default="symptom_disease_dataset_realistic_duplicated.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="cleaned_dataset.csv",
        help="Output cleaned CSV path.",
    )
    parser.add_argument(
        "--report",
        default="quality_report.json",
        help="Output quality report JSON path.",
    )
    parser.add_argument(
        "--conflicts",
        default="conflict_patterns.csv",
        help="Output CSV containing conflicting patterns.",
    )
    parser.add_argument(
        "--strategy",
        choices=["drop_conflicts", "majority_vote"],
        default="drop_conflicts",
        help="How to resolve symptom-pattern label conflicts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = clean_dataset(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        conflicts_path=args.conflicts,
        strategy=args.strategy,
    )
    print("Data cleaning complete.")
    print(f"Input rows: {report['input_rows']}")
    print(f"Output rows: {report['output_rows']}")
    print(f"Conflicting patterns: {report['conflicting_pattern_count']}")
    print(f"Report: {args.report}")
    print(f"Cleaned dataset: {args.output}")
    print(f"Conflicts: {args.conflicts}")


if __name__ == "__main__":
    main()
