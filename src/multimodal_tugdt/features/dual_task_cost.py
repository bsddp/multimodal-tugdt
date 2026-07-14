"""Pair-level dual-task cost features with explicit metric direction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from multimodal_tugdt.config import DualTaskCostConfig


@dataclass(frozen=True)
class DualTaskCostDataset:
    """Pair-level values, normalized costs, and pairing counts."""

    frame: pd.DataFrame
    paired_group_count: int
    skipped_group_count: int


def _metric_stem(metric: str) -> str:
    return metric.replace("__", "_")


def _cost_column(metric: str) -> str:
    stem = _metric_stem(metric).removesuffix("_pct")
    return f"dtc__{stem}_pct"


def _output_columns(config: DualTaskCostConfig) -> list[str]:
    columns = [*config.group_columns, "single_trial_id", "dual_trial_id"]
    for metric in config.metric_directions:
        stem = _metric_stem(metric)
        columns.extend(
            [
                f"single__{stem}",
                f"dual__{stem}",
                _cost_column(metric),
            ]
        )
    columns.append("dtc_valid_metric_count")
    return columns


def calculate_dual_task_costs(
    fused: pd.DataFrame,
    config: DualTaskCostConfig,
) -> DualTaskCostDataset:
    """Calculate positive-as-deterioration cost for complete single/dual pairs.

    Pairing is strict: each configured group may contain at most one row for each
    configured condition. Groups missing either condition are counted as skipped.
    """
    required = {
        *config.group_columns,
        "trial_id",
        "condition",
        *config.metric_directions,
    }
    missing = sorted(required - set(fused.columns))
    if missing:
        raise ValueError("Dual-task cost input is missing columns: " + ", ".join(missing))

    relevant = fused.loc[
        fused["condition"].isin([config.single_condition, config.dual_condition])
    ].copy()
    duplicate_keys = [*config.group_columns, "condition"]
    duplicated = relevant.duplicated(duplicate_keys, keep=False)
    if duplicated.any():
        examples = relevant.loc[duplicated, duplicate_keys].drop_duplicates().head(3)
        labels = ["/".join(map(str, row)) for row in examples.itertuples(index=False, name=None)]
        raise ValueError(
            "Dual-task cost pairing requires one trial per group and condition; duplicates: "
            + ", ".join(labels)
        )

    rows: list[dict[str, object]] = []
    skipped = 0
    groupby_key: str | list[str] = (
        config.group_columns[0] if len(config.group_columns) == 1 else list(config.group_columns)
    )
    for group_value, group in relevant.groupby(groupby_key, dropna=False, sort=True):
        single = group.loc[group["condition"] == config.single_condition]
        dual = group.loc[group["condition"] == config.dual_condition]
        if len(single) != 1 or len(dual) != 1:
            skipped += 1
            continue
        single_row = single.iloc[0]
        dual_row = dual.iloc[0]
        group_values = group_value if isinstance(group_value, tuple) else (group_value,)
        row: dict[str, object] = dict(zip(config.group_columns, group_values, strict=True))
        row["single_trial_id"] = single_row["trial_id"]
        row["dual_trial_id"] = dual_row["trial_id"]
        valid_metric_count = 0
        for metric, direction in config.metric_directions.items():
            stem = _metric_stem(metric)
            single_value = pd.to_numeric(pd.Series([single_row[metric]]), errors="coerce").iloc[0]
            dual_value = pd.to_numeric(pd.Series([dual_row[metric]]), errors="coerce").iloc[0]
            row[f"single__{stem}"] = single_value
            row[f"dual__{stem}"] = dual_value
            cost = float("nan")
            if (
                pd.notna(single_value)
                and pd.notna(dual_value)
                and abs(float(single_value)) > np.finfo(float).eps
            ):
                numerator = (
                    float(dual_value) - float(single_value)
                    if direction == "higher_is_worse"
                    else float(single_value) - float(dual_value)
                )
                cost = numerator / abs(float(single_value)) * 100
                valid_metric_count += 1
            row[_cost_column(metric)] = cost
        row["dtc_valid_metric_count"] = valid_metric_count
        rows.append(row)

    output = pd.DataFrame(rows, columns=_output_columns(config))
    return DualTaskCostDataset(output, len(output), skipped)
