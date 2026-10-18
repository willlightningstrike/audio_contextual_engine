from __future__ import annotations

import re
import subprocess
import sys


FORBIDDEN_DISTRIBUTIONS = {
    "torch",
    "torchvision",
    "torchaudio",
    "sentence-transformers",
    "tensorflow",
    "tensorflow-macos",
    "tensorflow-cpu",
    "mlx-cpu",
    "mlx-cuda",
}
_REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?(?=\s|[<>=!~;@]|$)")


def normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def find_forbidden_distributions(requirements_text: str) -> set[str]:
    forbidden = {normalize_distribution_name(name) for name in FORBIDDEN_DISTRIBUTIONS}
    offenders: set[str] = set()
    for line in requirements_text.splitlines():
        match = _REQUIREMENT_NAME.match(line.strip())
        if match and normalize_distribution_name(match.group(1)) in forbidden:
            offenders.add(normalize_distribution_name(match.group(1)))
    return offenders


def exported_requirements() -> str:
    result = subprocess.run(
        ["uv", "export", "--format", "requirements-txt", "--no-dev"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def main() -> int:
    offenders = find_forbidden_distributions(exported_requirements())
    if offenders:
        print("Forbidden distributions in resolved dependencies: " + ", ".join(sorted(offenders)))
        return 1
    print("Dependency audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
