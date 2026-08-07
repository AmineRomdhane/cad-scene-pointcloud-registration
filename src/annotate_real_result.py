import argparse
import csv
from pathlib import Path


EXTRA_COLUMNS = [
    "manual_quality",
    "usable_for_report",
    "usable_for_training",
    "failure_mode",
    "notes",
]


def main():
    parser = argparse.ArgumentParser(description="Annotate a row in real registration results CSV.")
    parser.add_argument("--csv", default="real_results/tables/real_registration_results.csv")
    parser.add_argument("--match", required=True, help="Text to match inside run_name or output_folder")
    parser.add_argument("--manual_quality", default="")
    parser.add_argument("--usable_for_report", default="")
    parser.add_argument("--usable_for_training", default="")
    parser.add_argument("--failure_mode", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise RuntimeError(f"CSV not found: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    for col in EXTRA_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)

    matched = 0

    for row in rows:
        run_name = row.get("run_name", "")
        output_folder = row.get("output_folder", "")

        if args.match in run_name or args.match in output_folder:
            row["manual_quality"] = args.manual_quality
            row["usable_for_report"] = args.usable_for_report
            row["usable_for_training"] = args.usable_for_training
            row["failure_mode"] = args.failure_mode
            row["notes"] = args.notes
            matched += 1

    if matched == 0:
        print(f"[WARNING] No row matched: {args.match}")
    else:
        print(f"[OK] Annotated {matched} row(s).")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
