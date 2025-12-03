"""
Move all WAV files from MOSI-VIDS to MOSI-AUDIO
"""
from pathlib import Path
import shutil

# Source and destination
source_dir = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset\MOSI-VIDS")
dest_dir = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset\MOSI-AUDIO")

# Create destination if it doesn't exist
dest_dir.mkdir(parents=True, exist_ok=True)

print(f"Source: {source_dir}")
print(f"Destination: {dest_dir}\n")

# Find all WAV files
wav_files = list(source_dir.glob('*.wav'))
print(f"Found {len(wav_files)} WAV files in {source_dir}")

if not wav_files:
    print("No WAV files found to move.")
    exit(0)

# Move files
moved = 0
skipped = 0
errors = 0

for wav_file in wav_files:
    dest_file = dest_dir / wav_file.name
    
    try:
        if not dest_file.exists():
            shutil.move(str(wav_file), str(dest_file))
            moved += 1
            if moved <= 10:
                print(f"  [OK] Moved: {wav_file.name}")
        else:
            skipped += 1
            if skipped <= 3:
                print(f"  [SKIP] Skipped (already exists): {wav_file.name}")
    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f"  [ERROR] Error moving {wav_file.name}: {e}")

print(f"\nSummary:")
print(f"  Moved: {moved} files")
print(f"  Skipped: {skipped} files")
print(f"  Errors: {errors} files")
print(f"\nDone! WAV files are now in: {dest_dir}")

# Verify
final_wavs = list(dest_dir.glob('*.wav'))
print(f"\nTotal WAV files in {dest_dir}: {len(final_wavs)}")

