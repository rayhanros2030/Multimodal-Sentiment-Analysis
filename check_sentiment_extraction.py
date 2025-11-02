#!/usr/bin/env python3
"""
Check Sentiment Extraction from CMU-MOSEI Labels
==================================================
"""

import h5py
import numpy as np
from pathlib import Path

def check_sentiment_extraction():
    """Check how sentiment is extracted from labels"""
    
    labels_path = Path(r"C:\Users\PC\Downloads\CMU-MOSEI") / 'labels' / 'CMU_MOSEI_Labels.csd'
    
    print("Checking CMU-MOSEI Labels Structure")
    print("=" * 80)
    
    with h5py.File(labels_path, 'r') as f:
        feature_group = f['All Labels']
        data_group = feature_group['data']
        
        # Check first few videos
        video_ids = list(data_group.keys())[:10]
        
        print(f"\nChecking {len(video_ids)} sample videos:\n")
        
        for vid_id in video_ids:
            video_group = data_group[vid_id]
            features = video_group['features'][:]
            
            print(f"Video ID: {vid_id}")
            print(f"  Shape: {features.shape}")
            print(f"  Raw values: {features}")
            
            # Try different extraction methods
            print("  Extraction methods:")
            
            # Method 1: features[0, 0] (current code)
            try:
                val1 = float(features[0, 0])
                print(f"    features[0, 0] = {val1:.4f}")
            except:
                print(f"    features[0, 0] = FAILED")
            
            # Method 2: Mean of first column
            try:
                val2 = float(np.mean(features[:, 0]))
                print(f"    Mean of column 0 = {val2:.4f}")
            except:
                print(f"    Mean of column 0 = FAILED")
            
            # Method 3: Mean of all
            try:
                val3 = float(np.mean(features))
                print(f"    Mean of all = {val3:.4f}")
            except:
                print(f"    Mean of all = FAILED")
            
            # Check which column might be sentiment
            print(f"  Column analysis (if shape > 1):")
            if len(features.shape) > 1 and features.shape[1] > 1:
                for col in range(features.shape[1]):
                    col_data = features[:, col]
                    print(f"    Column {col}: min={np.min(col_data):.4f}, max={np.max(col_data):.4f}, mean={np.mean(col_data):.4f}")
            
            print()

if __name__ == "__main__":
    check_sentiment_extraction()

