from pathlib import Path

base = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset")
print("Directories:")
dirs = [d.name for d in base.iterdir() if d.is_dir()]
for d in dirs:
    print(f"  {d}")

print("\nSearching for ZIP files...")
zips = list(base.rglob('*.zip'))
print(f"Total ZIP files: {len(zips)}")

if zips:
    print("\nZIP file locations:")
    for z in zips[:5]:
        print(f"  {z.parent.name}/{z.name}")




