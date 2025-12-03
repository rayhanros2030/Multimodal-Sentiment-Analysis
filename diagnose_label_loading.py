"""
Diagnostic script to check why CMU-MOSI labels are not being loaded correctly.
This will help identify ID mismatches between video/audio files and labels.json.
"""

import json
from pathlib import Path
import sys

def diagnose_labels(mosi_dir: str):
    """Diagnose label loading issues"""
    
    mosi_dir = Path(mosi_dir)
    
    print("="*80)
    print("CMU-MOSI Label Loading Diagnostic")
    print("="*80)
    
    # Find labels.json
    labels_json_path = mosi_dir / 'labels.json'
    if not labels_json_path.exists():
        # Try in subdirectories
        possible_paths = [
            mosi_dir / 'labels' / 'labels.json',
            mosi_dir / 'label' / 'labels.json',
        ]
        for p in possible_paths:
            if p.exists():
                labels_json_path = p
                break
    
    if not labels_json_path.exists():
        print(f"\n❌ ERROR: labels.json not found!")
        print(f"   Searched in: {mosi_dir}")
        print(f"   Please ensure labels.json exists in the dataset root or labels/ folder.")
        return
    
    print(f"\n✓ Found labels.json at: {labels_json_path}")
    
    # Load labels.json
    try:
        with open(labels_json_path, 'r') as f:
            labels_json = json.load(f)
        print(f"✓ Loaded {len(labels_json)} entries from labels.json")
    except Exception as e:
        print(f"❌ ERROR loading labels.json: {e}")
        return
    
    # Show sample keys from labels.json
    print(f"\nSample keys from labels.json (first 10):")
    for i, key in enumerate(list(labels_json.keys())[:10]):
        print(f"  {i+1}. '{key}' -> {labels_json[key]}")
    
    # Find video files
    possible_video_dirs = [
        mosi_dir / 'MOSI-VIDS',
        mosi_dir / 'MOSI-Videos',
        mosi_dir / 'videos',
        mosi_dir / 'video',
        mosi_dir
    ]
    
    video_files = []
    video_dir = None
    for vd in possible_video_dirs:
        if vd.exists():
            mp4_files = list(vd.rglob('*.mp4'))
            avi_files = list(vd.rglob('*.avi'))
            mov_files = list(vd.rglob('*.mov'))
            mkv_files = list(vd.rglob('*.mkv'))
            all_videos = list(set(mp4_files + avi_files + mov_files + mkv_files))
            if all_videos:
                video_files = all_videos
                video_dir = vd
                break
    
    print(f"\n✓ Found {len(video_files)} video files in {video_dir}")
    
    # Find audio files
    possible_audio_dirs = [
        mosi_dir / 'MOSI-AUDIO',
        mosi_dir / 'MOSI-VIDS',
        mosi_dir / 'audios',
        mosi_dir / 'audio',
        mosi_dir / 'MOSI-Audios',
        mosi_dir
    ]
    
    audio_files = []
    audio_dir = None
    for ad in possible_audio_dirs:
        if ad.exists():
            wav_files = list(ad.rglob('*.wav'))
            if wav_files:
                audio_files = wav_files
                audio_dir = ad
                break
    
    print(f"✓ Found {len(audio_files)} audio files in {audio_dir}")
    
    # Check ID matching
    print(f"\n{'='*80}")
    print("ID Matching Analysis")
    print(f"{'='*80}")
    
    # Extract IDs from files
    video_ids = {f.stem: f for f in video_files[:20]}  # Check first 20
    audio_ids = {f.stem: f for f in audio_files[:20]}
    
    print(f"\nSample video IDs (first 10):")
    for i, vid_id in enumerate(list(video_ids.keys())[:10]):
        print(f"  {i+1}. '{vid_id}'")
    
    print(f"\nSample audio IDs (first 10):")
    for i, aud_id in enumerate(list(audio_ids.keys())[:10]):
        print(f"  {i+1}. '{aud_id}'")
    
    print(f"\nSample label.json keys (first 10):")
    label_keys = list(labels_json.keys())[:10]
    for i, key in enumerate(label_keys):
        print(f"  {i+1}. '{key}' -> {labels_json[key]}")
    
    # Try to match
    print(f"\n{'='*80}")
    print("Matching Test (first 20 video files)")
    print(f"{'='*80}")
    
    matched_count = 0
    unmatched_samples = []
    
    for vid_id, vid_file in list(video_ids.items())[:20]:
        matched = False
        
        # Direct match
        if vid_id in labels_json:
            print(f"✓ '{vid_id}' -> Direct match in labels.json (value: {labels_json[vid_id]})")
            matched_count += 1
            matched = True
        else:
            # Partial match
            for key in labels_json.keys():
                if vid_id in key or key in vid_id:
                    print(f"✓ '{vid_id}' -> Partial match with '{key}' in labels.json (value: {labels_json[key]})")
                    matched_count += 1
                    matched = True
                    break
        
        if not matched:
            unmatched_samples.append(vid_id)
            print(f"❌ '{vid_id}' -> NO MATCH in labels.json")
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"{'='*80}")
    print(f"Total labels.json entries: {len(labels_json)}")
    print(f"Matched samples (out of 20 tested): {matched_count}")
    print(f"Unmatched samples: {len(unmatched_samples)}")
    
    if unmatched_samples:
        print(f"\nUnmatched sample IDs:")
        for sid in unmatched_samples[:10]:
            print(f"  - '{sid}'")
    
    # Check label value formats
    print(f"\n{'='*80}")
    print("Label Value Analysis")
    print(f"{'='*80}")
    
    label_values = [v for v in labels_json.values()]
    non_zero_count = sum(1 for v in label_values if v != 0.0 and v != 0 and v != "0")
    zero_count = len(label_values) - non_zero_count
    
    print(f"Total label values: {len(label_values)}")
    print(f"Non-zero labels: {non_zero_count}")
    print(f"Zero labels: {zero_count}")
    
    if label_values:
        try:
            numeric_values = [float(v) for v in label_values if isinstance(v, (int, float, str)) and str(v).replace('.','').replace('-','').isdigit()]
            if numeric_values:
                print(f"Label range: [{min(numeric_values):.2f}, {max(numeric_values):.2f}]")
                print(f"Label mean: {np.mean(numeric_values):.2f}")
        except:
            pass
    
    print(f"\n{'='*80}")
    print("Recommendations:")
    print(f"{'='*80}")
    
    if matched_count == 0:
        print("1. ❌ CRITICAL: No matches found between file IDs and labels.json keys!")
        print("   - Check if labels.json uses different ID format (e.g., with/without extensions)")
        print("   - Verify that labels.json corresponds to the same dataset version")
        print("   - Consider manually checking a few IDs to understand the format")
    elif matched_count < 10:
        print(f"1. ⚠️  WARNING: Only {matched_count}/20 samples matched!")
        print("   - Most IDs are not matching - check ID format differences")
    else:
        print(f"1. ✓ {matched_count}/20 samples matched - this is good!")
    
    if zero_count == len(label_values):
        print("2. ❌ CRITICAL: All labels in labels.json are zero!")
        print("   - Check if labels.json contains actual sentiment values")
        print("   - Verify labels.json format is correct")
    elif zero_count > non_zero_count:
        print(f"2. ⚠️  WARNING: {zero_count} zero labels vs {non_zero_count} non-zero")
        print("   - Many labels are zero - this might be expected, but verify")

if __name__ == '__main__':
    import numpy as np
    
    if len(sys.argv) < 2:
        print("Usage: python diagnose_label_loading.py <CMU-MOSI_Dataset_Path>")
        print("Example: python diagnose_label_loading.py \"C:/Users/PC/Downloads/CMU-MOSI Dataset\"")
        sys.exit(1)
    
    diagnose_labels(sys.argv[1])




