"""
Check for video files in CMU-MOSI Dataset
"""
from pathlib import Path

base = Path(r"C:\Users\PC\Downloads\CMU-MOSI Dataset")

print("=" * 60)
print("Checking for Video Files")
print("=" * 60)

# Check all directories
print("\nAll directories in dataset:")
dirs = [d for d in base.iterdir() if d.is_dir()]
for d in sorted(dirs):
    print(f"  - {d.name}/")

# Check for MOSI-Videos (various possible names)
possible_video_dirs = [
    base / 'MOSI-Videos',
    base / 'MOSI-Videos',
    base / 'videos',
    base / 'video',
    base / 'MOSI-VIDS',
]

print("\nChecking for video directories...")
video_dir_found = None
for vd in possible_video_dirs:
    if vd.exists():
        print(f"\nFound: {vd.name}/")
        video_dir_found = vd
        
        # Check contents
        items = list(vd.iterdir())
        print(f"  Items: {len(items)}")
        
        # Check file types
        mp4_files = list(vd.rglob('*.mp4'))
        avi_files = list(vd.rglob('*.avi'))
        mov_files = list(vd.rglob('*.mov'))
        mkv_files = list(vd.rglob('*.mkv'))
        zip_files = list(vd.rglob('*.zip'))
        
        print(f"  MP4 files: {len(mp4_files)}")
        print(f"  AVI files: {len(avi_files)}")
        print(f"  MOV files: {len(mov_files)}")
        print(f"  MKV files: {len(mkv_files)}")
        print(f"  ZIP files: {len(zip_files)}")
        
        if items:
            print(f"\n  Sample items (first 5):")
            for item in items[:5]:
                file_type = "DIR" if item.is_dir() else "FILE"
                print(f"    - {item.name} ({file_type})")
        break

# Search entire dataset recursively for video files
print("\n" + "=" * 60)
print("Recursive search for video files:")
print("=" * 60)

all_mp4 = list(base.rglob('*.mp4'))
all_avi = list(base.rglob('*.avi'))
all_mov = list(base.rglob('*.mov'))
all_mkv = list(base.rglob('*.mkv'))
all_zip = list(base.rglob('*.zip'))

print(f"\nMP4 files: {len(all_mp4)}")
print(f"AVI files: {len(all_avi)}")
print(f"MOV files: {len(all_mov)}")
print(f"MKV files: {len(all_mkv)}")
print(f"ZIP files: {len(all_zip)}")

if all_zip:
    print("\nZIP files found (may contain video files):")
    for z in all_zip[:5]:
        print(f"  {z.parent.name}/{z.name}")

# Final summary
print("\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)

extracted_videos = len(all_mp4) + len(all_avi) + len(all_mov) + len(all_mkv)
compressed_videos = len(all_zip)

if extracted_videos > 0:
    print(f"[OK] Video files are PRESENT ({extracted_videos} extracted video files)")
    print("  Status: Ready to use")
elif compressed_videos > 0:
    print(f"[ZIP] Video files are in ZIP format ({compressed_videos} ZIP files)")
    print("  Status: Need to extract ZIP files to use videos")
    print(f"  Location: Check in {base} for directories with ZIP files")
else:
    print("[MISSING] No video files found")
    print("  Status: Videos are not present in the dataset")

print("=" * 60)




