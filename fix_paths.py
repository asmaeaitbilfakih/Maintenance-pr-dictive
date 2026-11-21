"""
fix_paths.py : remplace les chemins en dur (D:\\...) des notebooks par une détection automatique
de la racine du projet, pour que les notebooks marchent sur n'importe quel PC.

Usage (depuis la racine du projet) :  python fix_paths.py
"""
import json
import re
from pathlib import Path

NEW_ROOT = ('ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] '
            'if (p / "src" / "features.py").exists())')

# ROOT = Path(r"D:\...")  ->  détection automatique
RE_ROOT = re.compile(r'ROOT = Path\(r?["\'][A-Za-z]:[^"\']*["\']\)')
# ROOT / "D:\...\data\processed/xxx"  ->  ROOT / "data/processed/xxx"
RE_ABS = re.compile(r'ROOT / ["\'][A-Za-z]:[^"\']*?[\\/]data[\\/]processed[\\/]')

changed = 0
for nb_path in sorted(Path("notebooks").glob("*.ipynb")):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    touched = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        new = RE_ROOT.sub(lambda m: NEW_ROOT, src)
        new = RE_ABS.sub('ROOT / "data/processed/', new)
        if new != src:
            cell["source"] = new.splitlines(keepends=True)
            touched = True
    if touched:
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("corrigé :", nb_path)
        changed += 1

print(f"\n{changed} notebook(s) corrigé(s).")
# vérification : il ne doit rester aucun chemin Windows en dur
left = [p.name for p in Path("notebooks").glob("*.ipynb")
        if re.search(r'[A-Z]:\\\\', p.read_text(encoding="utf-8"))]
print("Chemins en dur restants dans :", left if left else "aucun")
