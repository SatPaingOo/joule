"""build_native.py — compile all native kernels with zig (nostdlib).

Usage:  python src/jouleai/native/build_native.py
Requires: pip install ziglang
Produces: quant_gemv.dll, expert_ffn.dll, decode_kernel.dll (next to sources)
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
KERNELS = [
    ("quant_gemv.c", "quant_gemv.dll", []),
    ("expert_ffn.c", "expert_ffn.dll", []),
    ("decode_kernel.c", "decode_kernel.dll", ["-lkernel32"]),
]


def build(src: str, out: str, extra: list[str]) -> bool:
    cmd = ["python", "-m", "ziglang", "cc", "-O3", "-mcpu=native",
           "-fno-stack-check", "-fno-builtin", "-I.", "-nostdlib", "-shared",
           src, "-o", out, *extra]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    ok = (HERE / out).exists()
    print(f"{'OK  ' if ok else 'FAIL'} {src} -> {out}")
    if not ok:
        print(r.stderr[-400:])
    return ok


def main() -> int:
    ok = all(build(s, o, x) for s, o, x in KERNELS)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
