"""Independent HBPLUS-format geometry; numerical HBPLUS parity is unvalidated.

All geometry and legacy argument parsing run in Rust. The console script makes
the compatibility executable available in installed Python wheels.
"""
from protein_interface._core import hbplus_format

__all__ = ["hbplus_format"]


def main():
    import sys
    from protein_interface._core import hbplus_cli

    try:
        hbplus_cli(sys.argv[1:])
    except ValueError as exc:
        print(f"protein-interface-hbplus: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
