"""Install ThermoTwin editable dependencies and execute the first-time preparation workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Install and prepare ThermoTwin")
    parser.add_argument("--profile", choices=["smoke", "research"], default="smoke")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(root)], check=True)
    command = [sys.executable, "-m", "thermotwin", "prepare", "--profile", args.profile]
    if args.force:
        command.append("--force")
    subprocess.run(command, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
