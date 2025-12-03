"""
Verify CMU-MOSI Dataset completeness
"""
from pathlib import Path

base_dir = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset")

print("=" * 60)
print("CMU-MOSI Dataset Verification")
print("=" * 60)

# Check directory structure
print("\n1. Directory Structure:")
dirs = [d for d in base_dir.iterdir() if d.is_dir()]
for d in sorted(dirs):
    print(f"   [OK] {d.name}/")

# Check audio files
print("\n2. Audio Files:")
audio_dir = base_dir / 'MOSI-AUDIO'
if audio_dir.exists():
    wav_files = list(audio_dir.glob('*.wav'))
    print(f"   Location: MOSI-AUDIO/")
    print(f"   Count: {len(wav_files)} WAV files")
    if wav_files:
        print(f"   Sample: {wav_files[0].name}")
else:
    print("   [MISSING] MOSI-AUDIO directory not found")

# Check transcript files
print("\n3. Transcript Files:")
transcript_dir = base_dir / 'MOSI-Transcript'
if transcript_dir.exists():
    textonly_files = list(transcript_dir.rglob('*.textonly'))
    txt_files = list(transcript_dir.rglob('*.txt'))
    print(f"   Location: MOSI-Transcript/")
    print(f"   Count: {len(textonly_files)} .textonly files, {len(txt_files)} .txt files")
    if textonly_files:
        print(f"   Sample: {textonly_files[0].name}")
    elif txt_files:
        print(f"   Sample: {txt_files[0].name}")
else:
    print("   [MISSING] MOSI-Transcript directory not found")

# Check video files
print("\n4. Video Files:")
video_dir = base_dir / 'MOSI-Videos'
if video_dir.exists():
    zip_files = list(video_dir.glob('*.zip'))
    mp4_files = list(video_dir.glob('*.mp4'))
    print(f"   Location: MOSI-Videos/")
    print(f"   Count: {len(zip_files)} ZIP files, {len(mp4_files)} MP4 files")
    if zip_files:
        print(f"   Status: Videos are in ZIP format (need extraction)")
        print(f"   Sample: {zip_files[0].name}")
    elif mp4_files:
        print(f"   Status: Videos are extracted")
        print(f"   Sample: {mp4_files[0].name}")
else:
    print("   [MISSING] MOSI-Videos directory not found")

# Check labels
print("\n5. Labels File:")
labels_file = base_dir / 'labels.json'
if labels_file.exists():
    print(f"   Location: labels.json")
    print(f"   Status: [OK] Found")
    try:
        import json
        with open(labels_file, 'r') as f:
            labels = json.load(f)
            print(f"   Count: {len(labels)} label entries")
    except Exception as e:
        print(f"   Error reading labels: {e}")
else:
    print("   [MISSING] labels.json not found")

# Summary
print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)

required = {
    'Audio files': len(list((base_dir / 'MOSI-AUDIO').glob('*.wav'))) if (base_dir / 'MOSI-AUDIO').exists() else 0,
    'Transcript files': len(list((base_dir / 'MOSI-Transcript').rglob('*.textonly'))) + len(list((base_dir / 'MOSI-Transcript').rglob('*.txt'))) if (base_dir / 'MOSI-Transcript').exists() else 0,
    'Video files (ZIP)': len(list((base_dir / 'MOSI-Videos').glob('*.zip'))) if (base_dir / 'MOSI-Videos').exists() else 0,
    'Labels file': 1 if (base_dir / 'labels.json').exists() else 0
}

all_present = all(count > 0 for count in required.values())

for item, count in required.items():
    status = "[OK]" if count > 0 else "[MISSING]"
    print(f"  {status} {item}: {count}")

print("\n" + "=" * 60)
if all_present:
    print("[OK] Dataset is COMPLETE for transfer learning!")
    print("\nNote: Video files are in ZIP format.")
    print("The script can work with Audio + Transcript only.")
    print("If you want to extract videos, extract the ZIP files in MOSI-Videos/")
else:
    print("[INCOMPLETE] Dataset is missing some components.")
    print("Missing components are marked above.")
print("=" * 60)

