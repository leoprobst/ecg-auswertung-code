from __future__ import annotations

import sys
from multiprocessing import freeze_support


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "mimic":
        from .mimic_cli import main as mimic_main

        return mimic_main(sys.argv[2:])

    from .main import main as gui_main

    gui_main()
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
