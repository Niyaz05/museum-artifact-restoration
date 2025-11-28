# scripts/find_missing_files.py
from pathlib import Path
import sys

# read missing lists created earlier
miss_train = Path("small_train_ids_missing.txt")
miss_val   = Path("small_val_ids_missing.txt")
missing = []
if miss_train.exists():
    missing += [x.strip() for x in miss_train.read_text().splitlines() if x.strip()]
if miss_val.exists():
    missing += [x.strip() for x in miss_val.read_text().splitlines() if x.strip()]

if not missing:
    print("No missing list files found. Run the scanner first.")
    sys.exit(0)

search_root = Path(".")
exts = ['.jpg','.jpeg','.png','.JPG','.PNG']

found_map = {}
for mid in missing:
    found_map[mid] = []
    for p in search_root.rglob(f"{mid}*"):
        # skip the missing-lists and other non-file matches
        if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png'):
            found_map[mid].append(str(p))

# print summary
total_missing = len(missing)
total_found = sum(1 for k,v in found_map.items() if v)
print(f"Missing ids: {total_missing}, Found on disk: {total_found}")
for k,v in list(found_map.items())[:100]:
    if v:
        print(f"{k} -> {v[:3]}")
    # optionally show those not found
not_found = [k for k,v in found_map.items() if not v]
if not_found:
    print("\nNot found at all (sample 40):", not_found[:40])
    Path("not_found_overall.txt").write_text("\n".join(not_found))
    print("Wrote not_found_overall.txt")
else:
    print("All missing ids located somewhere.")
