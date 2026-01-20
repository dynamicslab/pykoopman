from __future__ import annotations

import nbformat

nb_path = r"docs\tutorial_koopman_eigenfunction_model_slow_manifold.ipynb"
print(f"Reading {nb_path}...")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

count = 0
for cell in nb.cells:
    if cell.cell_type == "code":
        original_source = cell.source
        new_source = original_source.replace('accelerator="gpu"', 'accelerator="auto"')
        if original_source != new_source:
            cell.source = new_source
            count += 1

print(f"Patched {count} cells.")
with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print("Done.")
