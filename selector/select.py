from __future__ import annotations

"""Coverage-constrained CVaR selector with diversity and speaker controls."""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .diversity import cosine_to_selected
from .quotas import SliceQuota, compute_slice_keys, load_quotas
from .utils import cvar, normalize, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a curated subset satisfying slice quotas and diversity constraints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scores", type=Path, required=True, help="scores.parquet path")
    parser.add_argument("--quotas", type=Path, required=True, help="Quota YAML path")
    parser.add_argument("--alpha", type=float, required=True, help="CVaR tail fraction")
    parser.add_argument(
        "--K-hours",
        dest="k_hours",
        type=float,
        required=True,
        help="Total selection budget in hours",
    )
    parser.add_argument(
        "--diversity-min-cos",
        dest="diversity_min_cos",
        type=float,
        default=0.0,
        help="Reject candidates whose cosine similarity to any selected item is >= 1 - margin",
    )
    parser.add_argument(
        "--uncert-beta",
        dest="uncert_beta",
        type=float,
        default=0.0,
        help="Weight applied to proxy uncertainty (sigma) in the loss proxy",
    )
    parser.add_argument(
        "--weights",
        nargs=3,
        type=float,
        metavar=("W_DNSMOS", "W_NISQA", "W_SQUIM"),
        default=[1.0, 1.0, 1.0],
        help="Weights for DNMSOS/NISQA/SQUIM components when forming the loss proxy",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output curated CSV path")
    parser.add_argument(
        "--alpha-sweep",
        dest="alpha_sweep",
        type=Path,
        help="Optional CSV path to log CVaR across multiple alphas",
    )
    parser.add_argument(
        "--slice-stats",
        dest="slice_stats",
        type=Path,
        help="Optional JSON path with per-slice coverage statistics",
    )
    parser.add_argument(
        "--allow-speaker-repeat",
        action="store_true",
        help="Allow repeated speaker_ids in the curated subset",
    )
    parser.add_argument(
        "--denylist",
        type=Path,
        help="Optional newline-delimited list of speaker_ids to exclude",
    )
    return parser.parse_args()


class SelectionState:
    """Mutable selection state used by the greedy selector."""

    def __init__(
        self,
        budget_seconds: float,
        diversity_margin: float,
        allow_speaker_repeat: bool,
        deny_speakers: Set[str],
        embeddings: np.ndarray,
        durations: np.ndarray,
        speakers: Sequence[str],
        slice_keys: Sequence[str],
    ) -> None:
        self.budget_seconds = budget_seconds
        self.diversity_margin = diversity_margin
        self.allow_speaker_repeat = allow_speaker_repeat
        self.deny_speakers = deny_speakers
        self.embeddings = embeddings
        self.durations = durations
        self.speakers = list(speakers)
        self.slice_keys = list(slice_keys)

        self.selected_indices: List[int] = []
        self.selected_set: Set[int] = set()
        self.selected_speakers: Set[str] = set()
        self.slice_counts: Counter[str] = Counter()
        self.rejections: Counter[str] = Counter()
        self.total_duration = 0.0
        self.nearest_cos: Dict[int, float] = {}

    def can_add(self, idx: int) -> Tuple[bool, Optional[str], float]:
        if idx in self.selected_set:
            return False, "duplicate", 0.0
        speaker = self.speakers[idx]
        if speaker in self.deny_speakers:
            return False, "denylist", 0.0
        if not self.allow_speaker_repeat and speaker in self.selected_speakers:
            return False, "speaker", 0.0
        duration = float(self.durations[idx])
        if self.total_duration + duration > self.budget_seconds:
            return False, "budget", 0.0
        if not self.selected_indices:
            return True, None, 0.0
        cosine = cosine_to_selected(self.embeddings, self.selected_indices, idx)
        if cosine >= 1.0 - self.diversity_margin:
            return False, "diversity", cosine
        return True, None, cosine

    def add(self, idx: int, cosine: float) -> None:
        self.selected_indices.append(idx)
        self.selected_set.add(idx)
        self.selected_speakers.add(self.speakers[idx])
        self.total_duration += float(self.durations[idx])
        self.nearest_cos[idx] = cosine
        self.slice_counts[self.slice_keys[idx]] += 1

    def reject(self, reason: str) -> None:
        self.rejections[reason] += 1


def load_speaker_denylist(path: Optional[Path]) -> Set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"Speaker denylist not found: {path}")
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "utt_id",
        "wav_path",
        "speaker_id",
        "lang",
        "distortion",
        "sr",
        "snr_db",
        "device",
        "room",
        "duration_s",
        "dnsmos",
        "nisqa",
        "squim_sdr_proxy",
        "ssl_content_vec",
        "ssl_speaker_vec",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"scores parquet missing required columns: {sorted(missing)}")
    df = df.copy().reset_index(drop=True)
    df["slice_key"] = compute_slice_keys(df)
    return df


def compute_loss_proxy(
    df: pd.DataFrame,
    weights: Sequence[float],
    uncert_beta: float,
) -> pd.Series:
    w_dns, w_nis, w_squim = weights
    dn_norm = normalize(df["dnsmos"].to_numpy())
    nis_norm = normalize(df["nisqa"].to_numpy())
    squim_norm = normalize(df["squim_sdr_proxy"].to_numpy())

    df["dnsmos_norm"] = dn_norm
    df["nisqa_norm"] = nis_norm
    df["squim_norm"] = squim_norm

    sigma = df.get("proxy_sigma", pd.Series(np.zeros(len(df)), dtype=np.float32)).fillna(0.0)
    loss = (
        w_dns * (1.0 - dn_norm)
        + w_nis * (1.0 - nis_norm)
        + w_squim * (1.0 - squim_norm)
        + float(uncert_beta) * sigma.to_numpy(dtype=np.float32)
    )
    df["loss_proxy"] = loss.astype(np.float32)
    return df["loss_proxy"]


def group_indices_by_slice(df: pd.DataFrame) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for slice_key, group in df.groupby("slice_key"):
        order = group.sort_values("loss_proxy", ascending=False).index.tolist()
        groups[slice_key] = order
    return groups


def coverage_fill(
    quotas: Dict[str, SliceQuota],
    slice_groups: Dict[str, List[int]],
    state: SelectionState,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    deficits: Dict[str, int] = {}
    pointers: Dict[str, int] = {key: 0 for key in slice_groups}
    for key, quota in quotas.items():
        if key not in slice_groups:
            deficits[key] = quota.minimum
            continue
        order = slice_groups[key]
        pointer = pointers[key]
        while pointer < len(order) and state.slice_counts[key] < quota.minimum:
            idx = order[pointer]
            pointer += 1
            ok, reason, cosine = state.can_add(idx)
            if ok:
                state.add(idx, cosine)
            else:
                state.reject(reason or "unknown")
        pointers[key] = pointer
        if state.slice_counts[key] < quota.minimum:
            deficits[key] = quota.minimum - state.slice_counts[key]
    return deficits, pointers


def budget_fill(
    slice_groups: Dict[str, List[int]],
    state: SelectionState,
    pointers: Dict[str, int],
) -> None:
    slice_order = sorted(slice_groups.keys())
    made_progress = True
    while made_progress and state.total_duration < state.budget_seconds:
        made_progress = False
        for key in slice_order:
            order = slice_groups[key]
            pointer = pointers.get(key, 0)
            while pointer < len(order) and state.total_duration < state.budget_seconds:
                idx = order[pointer]
                pointer += 1
                if idx in state.selected_set:
                    continue
                ok, reason, cosine = state.can_add(idx)
                if ok:
                    state.add(idx, cosine)
                    made_progress = True
                    break
                state.reject(reason or "unknown")
            pointers[key] = pointer
    if state.total_duration >= state.budget_seconds:
        LOGGER.info("Budget reached: %.2f hours", state.total_duration / 3600.0)


def alpha_sweep(losses: Sequence[float], path: Optional[Path]) -> None:
    if path is None:
        return
    alphas = [0.05, 0.10, 0.20]
    rows = [{"alpha": alpha, "cvar_loss": cvar(losses, alpha, higher_is_worse=True)} for alpha in alphas]
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Wrote alpha sweep diagnostics to %s", path)


def write_slice_stats(
    df: pd.DataFrame,
    quotas: Dict[str, SliceQuota],
    path: Optional[Path],
) -> None:
    if path is None:
        return
    stats: Dict[str, Dict[str, object]] = {}
    for slice_key, group in df.groupby("slice_key"):
        target = quotas.get(slice_key)
        target_min = target.minimum if target else 0
        stats[slice_key] = {
            "target": target_min,
            "selected": int(len(group)),
            "deficit": max(0, target_min - len(group)),
            "coverage_pass": len(group) >= target_min,
            "avg_loss": float(group["loss_proxy"].mean()) if not group.empty else None,
            "total_duration_s": float(group["duration_s"].sum()),
        }
    for key, quota in quotas.items():
        if key not in stats:
            stats[key] = {
                "target": quota.minimum,
                "selected": 0,
                "deficit": quota.minimum,
                "coverage_pass": False,
                "avg_loss": None,
                "total_duration_s": 0.0,
            }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))
    LOGGER.info("Wrote slice stats to %s", path)


def curated_output(df: pd.DataFrame, state: SelectionState) -> pd.DataFrame:
    selected_df = df.iloc[state.selected_indices].copy()
    selected_df["selected_rank"] = np.arange(1, len(selected_df) + 1)
    nn_cos = [state.nearest_cos.get(idx, 0.0) for idx in state.selected_indices]
    selected_df["nearest_neighbor_cos"] = nn_cos
    selected_df = selected_df.reset_index(drop=True)

    drop_cols = ["ssl_content_vec", "ssl_speaker_vec"]
    for col in drop_cols:
        if col in selected_df.columns:
            selected_df = selected_df.drop(columns=[col])

    column_order = [
        "utt_id",
        "wav_path",
        "speaker_id",
        "lang",
        "distortion",
        "sr",
        "snr_db",
        "device",
        "room",
        "duration_s",
        "dnsmos",
        "nisqa",
        "utmos",
        "squim_sdr_proxy",
        "loss_proxy",
        "slice_key",
        "selected_rank",
        "nearest_neighbor_cos",
    ]
    existing = [col for col in column_order if col in selected_df.columns]
    selected_df = selected_df[existing + [col for col in selected_df.columns if col not in existing]]
    return selected_df


def main() -> None:
    args = parse_args()
    setup_logging()

    quotas = load_quotas(args.quotas)
    scores_df = pd.read_parquet(args.scores)
    scores_df = prepare_dataframe(scores_df)
    compute_loss_proxy(scores_df, args.weights, args.uncert_beta)

    embeddings = np.asarray(scores_df["ssl_content_vec"].tolist(), dtype=np.float32)
    durations = scores_df["duration_s"].to_numpy(dtype=np.float32)
    speakers = scores_df["speaker_id"].astype(str).tolist()
    slice_keys = scores_df["slice_key"].tolist()

    budget_seconds = float(args.k_hours) * 3600.0
    denylist = load_speaker_denylist(args.denylist)
    state = SelectionState(
        budget_seconds=budget_seconds,
        diversity_margin=args.diversity_min_cos,
        allow_speaker_repeat=args.allow_speaker_repeat,
        deny_speakers=denylist,
        embeddings=embeddings,
        durations=durations,
        speakers=speakers,
        slice_keys=slice_keys,
    )

    slice_groups = group_indices_by_slice(scores_df)
    coverage_deficits, pointers = coverage_fill(quotas, slice_groups, state)
    LOGGER.info("Coverage deficits: %s", coverage_deficits or "none")

    budget_fill(slice_groups, state, pointers)

    curated_df = curated_output(scores_df, state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    curated_df.to_csv(args.out, index=False)
    LOGGER.info(
        "Selected %d items (%.2f hours) written to %s",
        len(curated_df),
        state.total_duration / 3600.0,
        args.out,
    )

    alpha_sweep(curated_df["loss_proxy"].to_numpy(dtype=np.float32), args.alpha_sweep)
    write_slice_stats(curated_df, quotas, args.slice_stats)

    LOGGER.info("Rejections: %s", dict(state.rejections))


if __name__ == "__main__":
    main()
