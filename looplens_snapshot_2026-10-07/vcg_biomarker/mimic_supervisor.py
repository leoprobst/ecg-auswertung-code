from __future__ import annotations

import argparse
from multiprocessing import freeze_support

from .mimic.config import load_or_discover_config
from .mimic.supervisor import LocalJobSupervisor


def main() -> int:
    parser = argparse.ArgumentParser(prog="vcg-mimic-supervisor")
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    config = load_or_discover_config(args.project)
    config.ensure_directories()
    LocalJobSupervisor(config).run_forever()
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
