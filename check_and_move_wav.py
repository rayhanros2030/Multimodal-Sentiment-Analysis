"""
Check MOSI-Videos and move WAV files to MOSI-AUDIO
"""
from pathlib import Path
import shutil

# Directories
source_dir = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset\MOSI-Videos")
dest_dir = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset\MOSI-AUDIO")

# Create destination
dest_dir.mkdir(parents=True, exist_ok=True)

print(f"Checking: {source_dir}")
print(f"Moving to: {dest_dir}\n")

# Check what's in the directory
if not source_dir.exists():
    print(f"ERROR: {source_dir} does not exist!")
    exit(1)

items = list(source_dir.iterdir())
print(f"Items in MOSI-Videos: {len(items)}")
for item in items[:10]:
    print(f"  - {item.name} ({'DIR' if item.is_dir() else 'FILE'})")

# Search for files
mp4s = list(source_dir.rglob('*.mp4'))
wavs = list(source_dir.rglob('*.wav'))
zips = list(source_dir.rglob('*.zip'))

print(f"\nFile counts:")
print(f"  MP4 files: {len(mp4s)}")
print(f"  WAV files: {len(wavs)}")
print(f"  ZIP files: {len(zips)}")

if wavs:
    print(f"\nMoving {len(wavs)} WAV files...")
    moved = 0
    for wav_file in wavs:
        dest_file = dest_dir / wav_file.name
        try:
            if not dest_file.exists():
                shutil.move(str(wav_file), str(dest_file))
                moved += 1
                if moved <= 5:
                    print(f"  ✓ Moved: {wav_file.name}")
            else:
                print(f"  ⊗ Skipped (exists): {wav_file.name}")
        except Exception as e:
            print(f"  ✗ Error: {wav_file.name} - {e}")
    print(f"\nDone! Moved {moved} files to {dest_dir}")
else:
    print("\nNo WAV files found in MOSI-Videos.")
    if zips:
        print(f"\nNote: Found {len(zips)} ZIP files. WAV files might be inside ZIP archives.")
        print("You may need to extract the ZIP files first.")




