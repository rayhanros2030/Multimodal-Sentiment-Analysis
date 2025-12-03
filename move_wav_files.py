"""
Move WAV files from MOSI-AUDIO to MOSI-VIDS
"""
from pathlib import Path
import shutil

# Source and destination
source_dir = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset\MOSI-AUDIO")
dest_dir = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset\MOSI-VIDS")

# Create destination directory
dest_dir.mkdir(parents=True, exist_ok=True)
print(f"Source: {source_dir}")
print(f"Destination: {dest_dir}\n")

# Find all WAV files (recursively)
wav_files = list(source_dir.rglob('*.wav'))
print(f"Found {len(wav_files)} WAV files")

if not wav_files:
    print("\nNo WAV files found! Checking directory structure...")
    if source_dir.exists():
        items = list(source_dir.iterdir())[:10]
        print(f"\nContents of {source_dir}:")
        for item in items:
            print(f"  - {item.name} ({'DIR' if item.is_dir() else 'FILE'})")
    exit(0)

# Move them
moved = 0
skipped = 0
errors = 0

for wav_file in wav_files:
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
            print(f"  ✓ Moved: {wav_file.name} -> {dest_file.name}")
    except Exception as e:
        errors += 1
        print(f"  ✗ Error moving {wav_file.name}: {e}")

print(f"\nSummary:")
print(f"  Moved: {moved} files")
print(f"  Skipped: {skipped} files (name conflicts)")
print(f"  Errors: {errors} files")
print(f"\nDone! Files are now in: {dest_dir}")
