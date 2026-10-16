"""CLI: python -m symspot.synth --n 20 --size small --seed 0 --out data/synth"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from symspot.synth.export import generate


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m symspot.synth", description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="число чертежей")
    ap.add_argument("--size", choices=["small", "large"], default="small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/synth"))
    ap.add_argument("--no-raster", action="store_true", help="не создавать PNG и грязный PNG")
    args = ap.parse_args(argv)
    t0 = time.time()
    names = generate(args.n, args.size, args.seed, args.out, raster=not args.no_raster)
    print(f"{len(names)} чертежей в {args.out} за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
