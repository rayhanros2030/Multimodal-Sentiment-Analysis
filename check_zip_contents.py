"""
Check what's inside the ZIP files
"""
import zipfile
from pathlib import Path

source_dir = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset\MOSI-Videos")
zip_files = list(source_dir.glob('*.zip'))[:3]  # Check first 3

print(f"Checking contents of ZIP files in {source_dir}\n")

for zip_file in zip_files:
    print(f"{zip_file.name}:")
    try:
        with zipfile.ZipFile(zip_file, 'r') as zf:
            files = zf.namelist()
            print(f"  Contains {len(files)} files:")
            for f in files[:5]:
                print(f"    - {f}")
            # Check for WAV files
            wavs = [f for f in files if f.lower().endswith('.wav')]
            mp4s = [f for f in files if f.lower().endswith('.mp4')]
            print(f"  WAV files: {len(wavs)}")
            print(f"  MP4 files: {len(mp4s)}")
    except Exception as e:
        print(f"  Error: {e}")
    print()




