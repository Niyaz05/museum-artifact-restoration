# scripts/check_one_file.py
from pathlib import Path
obj = "33092"
dam_train = Path("data/damaged/train")
print("Checking:", dam_train)
# list 20 files in folder
print("Some files in damaged/train:", list(dam_train.iterdir())[:20])
# check common extensions
found = []
for ext in ("jpg","jpeg","png","JPG","PNG"):
    p = dam_train / f"{obj}.{ext}"
    if p.exists():
        found.append(str(p))
# try glob any file starting with id
glob_found = list(dam_train.glob(f"{obj}.*"))
print("Exact found:", found)
print("Glob found:", glob_found)
