from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"


def _load_audit_module(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dependency_audit_rejects_normalized_forbidden_distribution():
    audit_dependencies = _load_audit_module("audit_dependencies")

    offenders = audit_dependencies.find_forbidden_distributions(
        "mlx==0.32.0\nToRcHvIsIoN==0.1.0\n"
    )

    assert offenders == {"torchvision"}


@pytest.mark.parametrize("distribution", ["tensorflow", "tensorflow-macos", "tensorflow-cpu"])
def test_dependency_audit_rejects_every_tensorflow_runtime(distribution: str):
    audit_dependencies = _load_audit_module("audit_dependencies")

    offenders = audit_dependencies.find_forbidden_distributions(f"{distribution}==2.19.1\n")

    assert offenders == {distribution}


def test_import_audit_allows_model_adapters_but_rejects_visual_imports():
    audit_imports = _load_audit_module("audit_imports")
    model_tree = ast.parse("import mlx\nfrom transformers import AutoTokenizer\n")
    visual_tree = ast.parse("from PIL import Image\n")

    assert audit_imports.find_forbidden_imports(
        model_tree, Path("engine/models/asr.py")
    ) == []
    assert audit_imports.find_forbidden_imports(
        visual_tree, Path("engine/audio.py")
    ) == ["PIL"]


def test_import_audit_rejects_non_adapter_model_and_all_visual_imports():
    audit_imports = _load_audit_module("audit_imports")
    non_adapter_tree = ast.parse("import mlx\nfrom transformers import AutoTokenizer\n")
    visual_tree = ast.parse("import cv2\nimport imageio.v3\nfrom PIL import Image\n")

    assert audit_imports.find_forbidden_imports(
        non_adapter_tree, Path("engine/audio.py")
    ) == ["mlx", "transformers"]
    assert audit_imports.find_forbidden_imports(
        visual_tree, Path("engine/audio.py")
    ) == ["cv2", "imageio", "PIL"]
