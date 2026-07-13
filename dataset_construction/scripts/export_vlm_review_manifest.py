#!/usr/bin/env python3
"""Export a review manifest from v2 quality.jsonl for manual inspection.

Generates:
  - review_manifest.jsonl : filtered & prioritized queue entries
  - review_summary.json   : per-class / per-backend / per-flag statistics
  - review_manifest.csv   : spreadsheet-friendly summary (optional)

Usage:
  python3 dataset_construction/scripts/export_vlm_review_manifest.py \
    --pseudo-root dataset_construction/derived/visdrone_pseudo_obb_v2
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from typing import Any

# Priority ordering for queue entries (matching protocol doc §12)
PRIORITY_ORDER = [
    "class_conflict",            # P0: VLM class vs HBB class conflict
    "intra_class_confusion",     # P1: tricycle/awning_tricycle, motor/bicycle
    "occlusion_truncation",      # P2: heavy occlusion or truncation
    "geometry_low_semantic_high",# P3: low geometry but high semantic score
    "random_sample",             # P4: random sample
]


def get_class_name(record: dict) -> str:
    return (
        record.get("class_name")
        or record.get("target_class_name")
        or record.get("source_class_name")
        or "unknown"
    )


def read_jsonl(path: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def classify_priority(record: dict) -> str:
    """Assign priority tier to a review queue entry."""
    quality = record.get("quality", {})
    flags = quality.get("flags", [])

    # P0: class conflict
    if any("vlm_class_conflict" in f for f in flags):
        return "class_conflict"

    # P1: intra-class confusion (tricycle vs awning_tricycle, motor vs bicycle)
    class_name = get_class_name(record)
    vlm_class = record.get("vlm_class_name", "") or quality.get("vlm_class_name", "")
    confusion_pairs = [
        {"tricycle", "awning_tricycle"},
        {"motor", "bicycle"},
    ]
    if vlm_class and class_name != vlm_class:
        for pair in confusion_pairs:
            if class_name in pair and vlm_class in pair:
                return "intra_class_confusion"

    # P2: occlusion / truncation
    if any(f in flags for f in ("heavy_occlusion", "heavy_truncation")):
        return "occlusion_truncation"

    # P3: geometry low but semantics high
    gs = quality.get("geometry_score", 0)
    ss = quality.get("semantic_score", 0)
    if gs < 0.50 and ss >= 0.60:
        return "geometry_low_semantic_high"

    return "random_sample"


def export(args: argparse.Namespace) -> None:
    pseudo_root = args.pseudo_root
    quality_path = f"{pseudo_root}/quality.jsonl"
    review_queue_path = f"{pseudo_root}/review_queue.jsonl"
    output_dir = args.output_dir or pseudo_root

    records = read_jsonl(quality_path)
    review_queue = read_jsonl(review_queue_path)

    if not records and not review_queue:
        print(f"No records found at {pseudo_root}")
        print("Run the pipeline first, or check the path.")
        return

    # ---- Filter queue entries ----
    queue = review_queue if review_queue else [
        r for r in records
        if r.get("quality", {}).get("quality_status") != "auto_accept"
    ]

    # ---- Assign priorities ----
    for entry in queue:
        entry["_priority"] = classify_priority(entry)

    # Sort by priority
    priority_map = {p: i for i, p in enumerate(PRIORITY_ORDER)}
    queue.sort(key=lambda e: (priority_map.get(e.get("_priority", "random_sample"), 99),
                               -(e.get("quality", {}).get("final_score", 0))))

    # ---- Filter by args ----
    if args.class_name:
        queue = [e for e in queue if get_class_name(e) == args.class_name]
    if args.min_score is not None:
        queue = [e for e in queue if e.get("quality", {}).get("final_score", 0) >= args.min_score]
    if args.max_entries:
        queue = queue[:args.max_entries]

    # ---- Write manifest JSONL ----
    manifest_path = f"{output_dir}/review_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in queue:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ---- Write CSV summary ----
    if args.csv:
        csv_path = f"{output_dir}/review_manifest.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "sample_id", "split", "class_name", "vlm_class_name",
                "final_score", "geometry_score", "semantic_score",
                "area_ratio", "center_shift", "foreground_ratio",
                "priority", "flags", "queue_reason",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for entry in queue:
                q = entry.get("quality", {})
                row = {
                    "sample_id": entry.get("sample_id", ""),
                    "split": entry.get("split", ""),
                    "class_name": get_class_name(entry),
                    "vlm_class_name": entry.get("vlm_class_name", "") or q.get("vlm_class_name", ""),
                    "final_score": q.get("final_score", ""),
                    "geometry_score": q.get("geometry_score", ""),
                    "semantic_score": q.get("semantic_score", ""),
                    "area_ratio": q.get("area_ratio", ""),
                    "center_shift": q.get("center_shift", ""),
                    "foreground_ratio": q.get("foreground_ratio", ""),
                    "priority": entry.get("_priority", ""),
                    "flags": ", ".join(q.get("flags", [])),
                    "queue_reason": entry.get("queue_reason", ""),
                }
                writer.writerow(row)

    # ---- Write summary JSON ----
    summary: dict[str, Any] = {
        "total_records": len(records),
        "review_queue_total": len(queue),
        "by_class": {},
        "by_backend": dict(Counter(e.get("vlm_backend", "unknown") for e in queue)),
        "by_priority": dict(Counter(e.get("_priority", "unknown") for e in queue)),
        "by_flag": {},
        "score_distribution": {
            "auto_accept": len([r for r in records if r.get("quality", {}).get("quality_status") == "auto_accept"]),
            "needs_review": len([r for r in records if r.get("quality", {}).get("quality_status") == "needs_review"]),
        },
    }

    # Per-class breakdown
    class_groups = defaultdict(list)
    for e in queue:
        class_groups[get_class_name(e)].append(e)
    for cls, entries in sorted(class_groups.items()):
        scores = [e.get("quality", {}).get("final_score", 0) for e in entries]
        summary["by_class"][cls] = {
            "count": len(entries),
            "avg_final_score": round(sum(scores) / max(len(scores), 1), 4) if scores else 0,
            "priorities": dict(Counter(e.get("_priority", "") for e in entries)),
        }

    # Flag frequency
    all_flags: Counter = Counter()
    for e in queue:
        for flag in e.get("quality", {}).get("flags", []):
            all_flags[flag] += 1
    summary["by_flag"] = dict(all_flags.most_common())

    summary_path = f"{output_dir}/review_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- Print report ----
    print(f"Exported {len(queue)} entries to {manifest_path}")
    print(f"Summary written to {summary_path}")
    if args.csv:
        print(f"CSV written to {csv_path}")
    print()
    print("Priority breakdown:")
    for pri, count in summary["by_priority"].items():
        print(f"  {pri}: {count}")
    print()
    print("Per-class breakdown:")
    for cls, info in summary["by_class"].items():
        print(f"  {cls}: {info['count']} samples, avg score {info['avg_final_score']}")
    print()
    print("Top flags:")
    for flag, count in list(summary["by_flag"].items())[:10]:
        print(f"  {flag}: {count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pseudo-root",
        default="dataset_construction/derived/visdrone_pseudo_obb_v2",
        help="Path to pseudo OBB output directory",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for manifest files (default: same as pseudo-root)",
    )
    parser.add_argument("--class-name", default=None, help="Filter by class name")
    parser.add_argument("--min-score", type=float, default=None, help="Minimum final score")
    parser.add_argument("--max-entries", type=int, default=None, help="Max entries to export")
    parser.add_argument("--csv", action="store_true", help="Also export CSV")
    return parser


def main() -> None:
    export(build_parser().parse_args())


if __name__ == "__main__":
    main()
