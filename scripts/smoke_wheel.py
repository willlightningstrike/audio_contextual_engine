from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    root = Path(__file__).parents[1]
    wheels = sorted((root / "dist").glob("contextual_engine-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one built wheel, found {len(wheels)}")

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        target = temporary_path / "site-packages"
        subprocess.run(
            ["uv", "pip", "install", "--target", str(target), "--no-deps", str(wheels[0])],
            check=True,
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(target)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; from engine.standards import load_taxonomy; import engine.web.app as app; rows = load_taxonomy(); static = Path(app.__file__).with_name('static'); assert len(rows) == 703; assert (static / 'index.html').is_file(); assert (Path(__import__('engine').__file__).parent / 'data' / 'standards' / 'manifest.json').is_file(); print(f'Installed wheel taxonomy rows: {len(rows)}; static assets present')",
            ],
            check=True,
            cwd=temporary_path,
            env=environment,
        )
    print("Wheel installation, imports, static assets, and standards data passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
