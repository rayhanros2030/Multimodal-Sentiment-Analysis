#!/usr/bin/env python3
"""
Verify CMU-MOSEI Data Loading
==============================

This script checks if the CMU-MOSEI data being loaded contains real values
or placeholder/dummy values.
"""

import h5py
import numpy as np
from pathlib import Path

def verify_mosei_data(mosei_dir: str):
    """Verify CMU-MOSEI data files contain real values"""
    
    print("=" * 80)
    print("CMU-MOSEI DATA VERIFICATION")
    print("=" * 80)
    print()
    
    mosei_path = Path(mosei_dir)
    
    # File paths
    visual_path = mosei_path / 'visuals' / 'CMU_MOSEI_VisualOpenFace2.csd'
    audio_path = mosei_path / 'acoustics' / 'CMU_MOSEI_COVAREP.csd'
    text_path = mosei_path / 'languages' / 'CMU_MOSEI_TimestampedWordVectors.csd'
    labels_path = mosei_path / 'labels' / 'CMU_MOSEI_Labels.csd'
    
    files_to_check = {
        'Visual': visual_path,
        'Audio': audio_path,
        'Text': text_path,
        'Labels': labels_path
    }
    
    all_valid = True
    
    for name, file_path in files_to_check.items():
        print(f"Checking {name} file: {file_path.name}")
        print("-" * 80)
        
        if not file_path.exists():
            print(f"  ERROR: File not found!")
            all_valid = False
            continue
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Find the correct key
                keys = list(f.keys())
                print(f"  Found keys: {keys}")
                
                # Try common key names
                key_to_use = None
                for key_name in [name.lower(), name.upper(), 'OpenFace_2', 'COVAREP', 'glove_vectors', 'All Labels', 'Sentiment']:
                    if key_name in f:
                        key_to_use = key_name
                        break
                
                if not key_to_use and len(keys) > 0:
                    key_to_use = keys[0]
                
                if not key_to_use:
                    print(f"  ERROR: No valid key found in file")
                    all_valid = False
                    continue
                
                print(f"  Using key: {key_to_use}")
                feature_group = f[key_to_use]
                
                if 'data' not in feature_group:
                    print(f"  ERROR: 'data' group not found")
                    all_valid = False
                    continue
                
                data_group = feature_group['data']
                video_ids = list(data_group.keys())[:5]  # Check first 5 videos
                
                print(f"  Found {len(list(data_group.keys()))} video IDs")
                print(f"  Checking first 5 samples:")
                
                for vid_id in video_ids:
                    try:
                        video_group = data_group[vid_id]
                        
                        if 'features' in video_group:
                            features = video_group['features'][:]
                            print(f"    Video ID: {vid_id}")
                            print(f"      Shape: {features.shape}")
                            print(f"      Data type: {features.dtype}")
                            print(f"      Min value: {np.min(features):.6f}")
                            print(f"      Max value: {np.max(features):.6f}")
                            print(f"      Mean value: {np.mean(features):.6f}")
                            print(f"      Std value: {np.std(features):.6f}")
                            print(f"      Contains NaN: {np.any(np.isnan(features))}")
                            print(f"      Contains Inf: {np.any(np.isinf(features))}")
                            
                            # Check if values are all zeros or very uniform (placeholder indicators)
                            if np.all(features == 0):
                                print(f"      WARNING: All values are zero - possible placeholder!")
                                all_valid = False
                            elif np.std(features) < 1e-6:
                                print(f"      WARNING: Very low variance - possible placeholder!")
                            else:
                                print(f"      OK: Real data detected (non-zero, varied values)")
                        else:
                            print(f"    Video ID: {vid_id}: No 'features' found")
                            all_valid = False
                        
                        print()
                    except Exception as e:
                        print(f"    Error processing {vid_id}: {e}")
                        all_valid = False
                
        except Exception as e:
            print(f"  ERROR: Could not read file: {e}")
            all_valid = False
        
        print()
    
    print("=" * 80)
    if all_valid:
        print("VERIFICATION RESULT: All files appear to contain REAL data")
        print("(No placeholder or dummy values detected)")
    else:
        print("VERIFICATION RESULT: Some issues detected - check warnings above")
    print("=" * 80)
    
    return all_valid

if __name__ == "__main__":
    # Update this path to your CMU-MOSEI directory
    mosei_dir = r"C:\Users\PC\Downloads\CMU-MOSEI"
    
    print("Verifying CMU-MOSEI data files...")
    print(f"Directory: {mosei_dir}")
    print()
    
    verify_mosei_data(mosei_dir)

