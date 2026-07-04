"""Backward-compatible wrapper for the approach 2 CLI."""

from .cli.run_pipeline import main, parse_args, run_pipeline


if __name__ == "__main__":
    main()
