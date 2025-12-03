"""
Extract and move WAV files from CMU-MOSI dataset
"""
from pathlib import Path
import zipfile
import shutil

# Source and destination directories
source_dir = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset\MOSI-Videos")
dest_dir = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset\MOSI-VIDS")

# Create destination if it doesn't exist
dest_dir.mkdir(parents=True, exist_ok=True)
print(f"Destination directory: {dest_dir}")

# Find all WAV files (already extracted)
wav_files = list(source_dir.rglob('*.wav'))
print(f"\nFound {len(wav_files)} WAV files in {source_dir}")

# Find all ZIP files that might contain WAV files
zip_files = list(source_dir.rglob('*.zip'))
print(f"Found {len(zip_files)} ZIP files in {source_dir}")

# Move existing WAV files
moved_count = 0
for wav_file in wav_files:
    dest_file = dest_dir / wav_file.name
    if not dest_file.exists():
        try:
            shutil.move(str(wav_file), str(dest_file))
            moved_count += 1
            if moved_count <= 5:
                print(f"  Moved: {wav_file.name}")
        except Exception as e:
            print(f"  Error moving {wav_file.name}: {e}")
    else:
        print(f"  Skipping {wav_file.name} (already exists in destination)")

print(f"\nMoved {moved_count} WAV files")

# Extract WAV files from ZIP files if needed
if zip_files and moved_count == 0:
    print(f"\nNo WAV files found, but found ZIP files. Extracting WAV files from ZIP...")
    extracted_count = 0
    for zip_file in zip_files[:10]:  # Limit to first 10 for safety
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    if member.lower().endswith('.wav'):
                        # Extract to destination
                        zip_ref.extract(member, dest_dir)
                        extracted_count += 1
                        if extracted_count <= 5:
                            print(f"  Extracted: {member}")
        except Exception as e:
            print(f"  Error extracting {zip_file.name}: {e}")
    
    print(f"\nExtracted {extracted_count} WAV files from ZIP archives")

print(f"\nDone! Check {dest_dir} for WAV files.")




