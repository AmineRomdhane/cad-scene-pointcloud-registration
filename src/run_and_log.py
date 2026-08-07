import argparse
import subprocess
import sys
from pathlib import Path

from real_results_logger import log_result_folder


def project_root():
    return Path(__file__).resolve().parents[1]


def find_output_dir(command):
    for i, token in enumerate(command):
        if token == "--output" and i + 1 < len(command):
            return command[i + 1]

    raise RuntimeError("Could not find --output in command. Cannot log result automatically.")


def main():
    parser = argparse.ArgumentParser(
        description="Run a registration/refinement command, then log its result into real_results."
    )

    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run. Use: python src/run_and_log.py -- python src/02_register..."
    )

    args = parser.parse_args()

    command = args.command

    if len(command) > 0 and command[0] == "--":
        command = command[1:]

    if len(command) == 0:
        raise RuntimeError("No command provided.")

    output_dir = find_output_dir(command)

    print("[INFO] Running command:")
    print(" ".join(command))

    result = subprocess.run(command, cwd=project_root())

    if result.returncode != 0:
        print("[ERROR] Command failed. Result will not be logged.")
        sys.exit(result.returncode)

    print("[INFO] Command finished successfully.")
    print("[INFO] Logging result...")

    log_result_folder(output_dir)


if __name__ == "__main__":
    main()
