# -*- coding: utf-8 -*-
"""
SANKETA — Model Backup Script
backend/backup_models.py
Backs up every existing model file and class mapping to backend/models_backup/.
Never overwrites existing backup files if already present.
"""

import shutil
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
MODELS_DIR = BACKEND_DIR / "app" / "models"
BACKUP_DIR = BACKEND_DIR / "models_backup"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def backup_models():
    print("=" * 80)
    print("PART B: BACKING UP CURRENT MODELS")
    print("=" * 80)
    print(f"Source dir : {MODELS_DIR}")
    print(f"Backup dir : {BACKUP_DIR}")

    model_files = list(MODELS_DIR.glob("*.*"))
    print(f"Found {len(model_files)} files in models directory.\n")

    backed_up = []
    skipped = []

    for f in model_files:
        dest = BACKUP_DIR / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
            backed_up.append(f.name)
            print(f"  [COPIED]  {f.name} -> models_backup/{f.name}")
        else:
            skipped.append(f.name)
            print(f"  [EXISTS]  {f.name} already preserved in models_backup")

    # Also create explicitly named standard backup aliases if not present
    aliases = [
        ("static_model_v2.keras", "isl_static_backup.keras"),
        ("dynamic_model_v2.keras", "isl_dynamic_backup.keras"),
        ("static_model.keras", "static_model_original_backup.keras"),
        ("dynamic_model.keras", "dynamic_model_original_backup.keras"),
    ]

    for src_name, alias_name in aliases:
        src = MODELS_DIR / src_name
        dest = BACKUP_DIR / alias_name
        if src.exists() and not dest.exists():
            shutil.copy2(src, dest)
            print(f"  [ALIAS]   {src_name} -> models_backup/{alias_name}")

    print("\n" + "=" * 80)
    print(f"BACKUP SUMMARY: {len(backed_up)} newly backed up, {len(skipped)} already preserved.")
    print("=" * 80)

if __name__ == "__main__":
    backup_models()
