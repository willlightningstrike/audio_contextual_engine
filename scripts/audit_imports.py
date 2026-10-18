from __future__ import annotations

import ast
from pathlib import Path


MODEL_IMPORTS = {"mlx", "mlx_audio", "mlx_vlm", "mlx_embeddings", "transformers"}
VISUAL_IMPORTS = {"cv2", "imageio", "PIL"}


def _is_model_adapter(path: Path) -> bool:
    return path.parts[:2] == ("engine", "models")


def find_forbidden_imports(tree: ast.AST, path: Path) -> list[str]:
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".", 1)[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots = [node.module.split(".", 1)[0]]
        else:
            continue
        for root in roots:
            if root in VISUAL_IMPORTS:
                offenders.append(root)
            elif root in MODEL_IMPORTS and not _is_model_adapter(path):
                offenders.append(root)
    return offenders


def main() -> int:
    root = Path(__file__).parents[1]
    offenders: list[str] = []
    for source_path in (root / "engine").rglob("*.py"):
        relative_path = source_path.relative_to(root)
        found = find_forbidden_imports(ast.parse(source_path.read_text()), relative_path)
        offenders.extend(f"{relative_path}: {name}" for name in found)
    if offenders:
        print("Forbidden application imports:\n" + "\n".join(offenders))
        return 1
    print("Import audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
