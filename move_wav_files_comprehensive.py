"""
Comprehensive script to find and move WAV files to MOSI-VIDS
"""
from pathlib import Path
import shutil

# Destination directory
dest_dir = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset\MOSI-VIDS")
dest_dir.mkdir(parents=True, exist_ok=True)

# Search all possible source locations
base_dir = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset")
possible_sources = [
    base_dir / "MOSI-AUDIO",
    base_dir / "MOSI-Audios",
    base_dir / "MOSI-Videos",
    base_dir / "MOSI-VIDS",
    base_dir
]

print(f"Searching for WAV files in CMU-MOSI Dataset...")
print(f"Destination: {dest_dir}\n")

# Find all WAV files in the entire dataset
all_wav_files = list(base_dir.rglob('*.wav'))
print(f"Found {len(all_wav_files)} WAV files total in dataset\n")

if not all_wav_files:
    print("No WAV files found anywhere in the dataset.")
    print("\nChecking directory structure:")
    for source in possible_sources:
        if source.exists():
            items = list(source.iterdir())[:3]
            print(f"  {source.name}: {len(items)} items")
    exit(0)

# Group by location
from collections import defaultdict
by_location = defaultdict(list)
for wav in all_wav_files:
    by_location[wav.parent].append(wav)

print("WAV files found in:")
for loc, files in by_location.items():
    print(f"  {loc}: {len(files)} files")

# Move all WAV files to destination
print(f"\nMoving files to {dest_dir}...")
moved = 0
errors = 0

for wav_file in all_wav_files:
    # Skip if already in destination
    if wav_file.parent == dest_dir:
        continue
    
    dest_file = dest_dir / wav_file.name
    
    # Handle name conflicts
    counter = 1
    original_dest = dest_file
    while dest_file.exists():
        dest_file = dest_dir / f"{original_dest.stem}_{counter}{original_dest.suffix}"
        counter += 1
    
    try:
        shutil.move(str(wav_file), str(dest_file))
        moved += 1
        if moved <= 10:
            print(f"  ✓ Moved: {wav_file.name}")
    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f"  ✗ Error moving {wav_file.name}: {e}")

print(f"\nSummary:")
print(f"  Moved: {moved} files")
print(f"  Errors: {errors} files")
print(f"\nWAV files are now in: {dest_dir}")




