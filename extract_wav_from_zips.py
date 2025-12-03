"""
Extract WAV files from ZIP archives in MOSI-Videos and move them to MOSI-AUDIO
"""
from pathlib import Path
import zipfile
import shutil

# Directories
source_dir = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset\MOSI-Videos")
dest_dir = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset\MOSI-AUDIO")

# Create destination
dest_dir.mkdir(parents=True, exist_ok=True)

print(f"Source: {source_dir}")
print(f"Destination: {dest_dir}\n")

# Find all ZIP files
zip_files = list(source_dir.glob('*.zip'))
print(f"Found {len(zip_files)} ZIP files\n")

if not zip_files:
    print("No ZIP files found!")
    exit(0)

# Extract WAV files from ZIPs
extracted = 0
errors = 0
skipped = 0

for zip_file in zip_files:
    try:
        with zipfile.ZipFile(zip_file, 'r') as zf:
            # Find WAV files in this ZIP
            wav_files_in_zip = [f for f in zf.namelist() if f.lower().endswith('.wav')]
            
            for wav_name in wav_files_in_zip:
                # Get just the filename (handle paths inside ZIP)
                wav_filename = Path(wav_name).name
                dest_file = dest_dir / wav_filename
                
                # Handle name conflicts
                counter = 1
                original_dest = dest_file
                while dest_file.exists():
                    dest_file = dest_dir / f"{original_dest.stem}_{counter}{original_dest.suffix}"
                    counter += 1
                
                try:
                    # Extract to destination
                    zf.extract(wav_name, dest_dir)
                    # Rename if needed (in case ZIP had subdirectories)
                    extracted_file = dest_dir / wav_name
                    if extracted_file.exists() and extracted_file != dest_file:
                        if dest_file.exists():
                            extracted_file.unlink()  # Delete if dest already exists
                        else:
                            extracted_file.rename(dest_file)
                    else:
                        # File might already be at dest_file location
                        pass
                    
                    extracted += 1
                    if extracted <= 10:
                        print(f"  ✓ Extracted: {wav_filename} from {zip_file.name}")
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"  ✗ Error extracting {wav_name} from {zip_file.name}: {e}")
    
    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f"  ✗ Error reading ZIP {zip_file.name}: {e}")

print(f"\nSummary:")
print(f"  Extracted: {extracted} WAV files")
print(f"  Errors: {errors}")
print(f"\nDone! WAV files are now in: {dest_dir}")

# Check final count
final_wavs = list(dest_dir.glob('*.wav'))
print(f"\nTotal WAV files in {dest_dir}: {len(final_wavs)}")




