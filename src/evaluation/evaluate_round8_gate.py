import argparse
import json
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INTERNAL = PROJECT_ROOT / "outputs" / "predictions" / "round8_oneshot_internal_test_metrics.json"
DEFAULT_RESIDUAL_DEV = PROJECT_ROOT / "outputs" / "predictions" / "round8_oneshot_residual_dev_metrics.json"
DEFAULT_RESIDUAL_PROBE = PROJECT_ROOT / "outputs" / "predictions" / "round8_oneshot_residual_probe_metrics.json"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "outputs" / "evaluation" / "round8_oneshot_gate_report.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "docs" / "ROUND8_ONESHOT_GATE_REPORT.md"


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_delta(block: Dict) -> Dict:
    step7 = block["step7"]
    final = block["round8_oneshot"]
    delta = block["delta_vs_step7"]
    return {
        "step7_f1": step7["f1"],
        "round8_f1": final["f1"],
        "f1_delta": final["f1"] - step7["f1"],
        "step7_fp": step7["false_positives"],
        "round8_fp": final["false_positives"],
        "fp_delta": final["false_positives"] - step7["false_positives"],
        "step7_fn": step7["false_negatives"],
        "round8_fn": final["false_negatives"],
        "fn_delta": final["false_negatives"] - step7["false_negatives"],
        **delta,
    }


def gate_checks(metrics: Dict, args) -> Dict:
    internal = metric_delta(metrics["internal_test"])
    residual_dev = metric_delta(metrics["residual_dev"])
    residual_probe = metric_delta(metrics["residual_probe"])
    checks = [
        {
            "gate": "internal_test_f1_retention",
            "required": f">= Step7 - {args.internal_f1_tolerance:.4f}",
            "observed": f"{internal['round8_f1']:.4f} vs Step7 {internal['step7_f1']:.4f}",
            "pass": internal["round8_f1"] >= internal["step7_f1"] - args.internal_f1_tolerance,
        },
        {
            "gate": "residual_dev_f1_gain",
            "required": f">= Step7 + {args.residual_dev_min_gain:.4f}",
            "observed": f"delta {residual_dev['f1_delta']:.4f}",
            "pass": residual_dev["f1_delta"] >= args.residual_dev_min_gain,
        },
        {
            "gate": "residual_probe_f1_gain",
            "required": f">= Step7 + {args.residual_probe_min_gain:.4f}",
            "observed": f"delta {residual_probe['f1_delta']:.4f}",
            "pass": residual_probe["f1_delta"] >= args.residual_probe_min_gain,
        },
        {
            "gate": "internal_test_new_fp",
            "required": f"<= {args.max_internal_new_fp}",
            "observed": str(internal["new_false_positives"]),
            "pass": internal["new_false_positives"] <= args.max_internal_new_fp,
        },
        {
            "gate": "residual_probe_new_fp",
            "required": f"<= {args.max_residual_probe_new_fp}",
            "observed": str(residual_probe["new_false_positives"]),
            "pass": residual_probe["new_false_positives"] <= args.max_residual_probe_new_fp,
        },
        {
            "gate": "residual_probe_net_correct",
            "required": ">= 0",
            "observed": str(residual_probe["net_correct_gain"]),
            "pass": residual_probe["net_correct_gain"] >= 0,
        },
    ]
    return {
        "deltas": {
            "internal_test": internal,
            "residual_dev": residual_dev,
            "residual_probe": residual_probe,
        },
        "checks": checks,
        "all_passed": all(item["pass"] for item in checks),
    }


def write_markdown(report: Dict, path: Path) -> None:
    lines = [
        "# Round8 One-Shot Gate Report",
        "",
        "This report uses non-teacher surfaces only. Teacher-test diagnostics remain blocked unless all gates pass.",
        "",
        "## Metric Deltas",
        "",
        "| Split | Step7 F1 | Round8 F1 | F1 Delta | Net Correct | New FP | Fixed FN | Used |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, block in report["deltas"].items():
        lines.append(
            f"| {split} | {block['step7_f1']:.4f} | {block['round8_f1']:.4f} | {block['f1_delta']:.4f} | "
            f"{block['net_correct_gain']} | {block['new_false_positives']} | {block['fixed_false_negatives']} | "
            f"{block['selector_used_rows']} |"
        )
    lines.extend(["", "## Gates", "", "| Gate | Required | Observed | Pass |", "| --- | --- | --- | --- |"])
    for item in report["checks"]:
        lines.append(f"| {item['gate']} | {item['required']} | {item['observed']} | {item['pass']} |")
    lines.extend(["", "## Decision", "", "```text", report["decision"], "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Round8 one-shot selector promotion gate.")
    parser.add_argument("--internal_test_metrics", default=str(DEFAULT_INTERNAL))
    parser.add_argument("--residual_dev_metrics", default=str(DEFAULT_RESIDUAL_DEV))
    parser.add_argument("--residual_probe_metrics", default=str(DEFAULT_RESIDUAL_PROBE))
    parser.add_argument("--output_json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output_md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--internal_f1_tolerance", type=float, default=0.003)
    parser.add_argument("--residual_dev_min_gain", type=float, default=0.04)
    parser.add_argument("--residual_probe_min_gain", type=float, default=0.03)
    parser.add_argument("--max_internal_new_fp", type=int, default=1)
    parser.add_argument("--max_residual_probe_new_fp", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = {
        "internal_test": load_json(Path(args.internal_test_metrics)),
        "residual_dev": load_json(Path(args.residual_dev_metrics)),
        "residual_probe": load_json(Path(args.residual_probe_metrics)),
    }
    report = gate_checks(metrics, args)
    report["inputs"] = {
        "internal_test": args.internal_test_metrics,
        "residual_dev": args.residual_dev_metrics,
        "residual_probe": args.residual_probe_metrics,
    }
    report["decision"] = (
        "ROUND8_TEACHER_TEST_DIAGNOSTIC_ALLOWED = yes; freeze scripts/configs before running teacher-test once."
        if report["all_passed"]
        else "ROUND8_TEACHER_TEST_DIAGNOSTIC_ALLOWED = no; keep Step7 as final baseline and treat the selector as diagnostic/reusable only."
    )
    write_json(report, Path(args.output_json))
    write_markdown(report, Path(args.output_md))
    print("=" * 70)
    print("Round8 gate evaluated")
    print("=" * 70)
    print(f"All gates passed: {report['all_passed']}")
    print(report["decision"])
    print(f"Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
