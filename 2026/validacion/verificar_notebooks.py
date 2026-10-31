"""Ejecutar notebooks docentes desde kernels nuevos en una copia temporal."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = (
    "2026/actividades/lab_0_warmup_rl.ipynb",
    "2026/docentes/q_learning_referencia_ejecutada.ipynb",
    "2026/clase2/mini_rlhf.ipynb",
    "2026/clase2/llm_policy_gradients.ipynb",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/notebooks")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="rl2026-notebooks-") as tmp:
        workspace = Path(tmp) / "workspace"
        for path in (ROOT / "2026").rglob("*"):
            if path.suffix in (".py", ".ipynb") and "outputs" not in path.parts:
                target = workspace / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        kernels = Path(tmp) / "kernels"
        kernel = kernels / "rl2026"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "RL 2026 verification", "language": "python"}))
        for relative in NOTEBOOKS:
            print("Ejecutando:", relative, flush=True)
            notebook = nbformat.read(workspace / relative, as_version=4)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.outputs = []
                    cell.execution_count = None
            manager = KernelManager(kernel_name="rl2026", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
            client = NotebookClient(notebook, km=manager, timeout=180,
                                    resources={"metadata": {"path": str(workspace)}})
            # El kernel seleccionado utiliza exactamente este entorno, no el del usuario.
            try:
                client.execute(env=dict(os.environ, MPLBACKEND="Agg"))
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
            errors = [o for c in notebook.cells if c.cell_type == "code"
                      for o in c.get("outputs", []) if o.output_type == "error"]
            assert not errors, relative
            nbformat.write(notebook, args.output / Path(relative).name)
            results.append({"path": relative, "status": "ok",
                            "code_cells": sum(c.cell_type == "code" for c in notebook.cells)})
    (args.output / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print("OK: notebooks ejecutados sin cambiar los originales ni sus outputs locales.")


if __name__ == "__main__":
    main()
