import sys

from lentic.cli import main

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        if getattr(sys, "frozen", False):
            input("Press Enter to close.")
        raise SystemExit(1)
