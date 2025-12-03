"""
Direct download and extraction of CMU-MOSI labels.
Downloads the .csd file directly and extracts labels using h5py.
"""

import urllib.request
import h5py
import numpy as np
import json
from pathlib import Path

def download_and_extract_labels(output_path="labels.json"):
    """
    Download CMU-MOSI Opinion Labels directly and extract sentiment values.
    """
    
    print("="*80)
    print("Downloading CMU-MOSI Labels")
    print("="*80)
    
    # URL for labels
    labels_url = "http://immortal.multicomp.cs.cmu.edu/CMU-MOSI/labels/CMU_MOSI_Opinion_Labels.csd"
    
    # Download directory
    download_dir = Path("C:/Users/PC/Downloads/cmumosi_labels")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    csd_file = download_dir / "CMU_MOSI_Opinion_Labels.csd"
    
    # Download file if not exists
    if not csd_file.exists():
        print(f"\nDownloading labels from CMU servers...")
        print(f"URL: {labels_url}")
        print(f"Destination: {csd_file}")
        print("This may take a few minutes...\n")
        
        try:
            urllib.request.urlretrieve(labels_url, csd_file)
            print(f"[OK] Download complete!")
        except Exception as e:
            print(f"ERROR downloading: {e}")
            return None
    else:
        print(f"\n[OK] Found existing labels file: {csd_file}")
    
    # Extract labels from .csd file
    print(f"\nExtracting labels from {csd_file}...")
    
    labels_dict = {}
    
    try:
        with h5py.File(csd_file, 'r') as f:
            print("\nExploring file structure...")
            print(f"Top-level keys: {list(f.keys())}")
            
            # Try different possible structures
            data_group = None
            
            # Structure 1: Direct 'data' key
            if 'data' in f:
                data_group = f['data']
            # Structure 2: 'Opinion Segment Labels' -> 'data'
            elif 'Opinion Segment Labels' in f:
                opinion_group = f['Opinion Segment Labels']
                if 'data' in opinion_group:
                    data_group = opinion_group['data']
                    print(f"Found structure: Opinion Segment Labels -> data")
            
            if data_group is None:
                # Try to find any group that contains data
                for key in f.keys():
                    if isinstance(f[key], h5py.Group):
                        group = f[key]
                        print(f"Checking group '{key}': {list(group.keys())}")
                        if 'data' in group:
                            data_group = group['data']
                            print(f"Found data in group: {key}")
                            break
            
            if data_group is None:
                print("ERROR: Could not find 'data' group in .csd file!")
                print(f"Available keys: {list(f.keys())}")
                # Try to explore recursively
                def explore_group(g, prefix="", depth=0):
                    if depth > 3:
                        return
                    for key in g.keys():
                        print(f"{prefix}{key}")
                        if isinstance(g[key], h5py.Group):
                            explore_group(g[key], prefix + "  ", depth+1)
                
                print("\nFull structure:")
                explore_group(f)
                return None
            
            print(f"Found 'data' group with {len(data_group.keys())} entries")
            
            for video_id in data_group.keys():
                try:
                    video_group = data_group[video_id]
                    
                    if 'features' in video_group:
                        features = np.array(video_group['features'])
                        
                        # Extract sentiment value
                        # Features shape: (n_segments, 1) or (n_segments, n_features)
                        if len(features.shape) == 2:
                            # Average across segments if multiple
                            if features.shape[0] > 0:
                                sentiment_value = float(np.mean(features[:, 0]))
                            else:
                                sentiment_value = 0.0
                        elif len(features.shape) == 1:
                            sentiment_value = float(np.mean(features)) if len(features) > 0 else 0.0
                        else:
                            sentiment_value = float(features[0]) if features.size > 0 else 0.0
                        
                        # Clean video ID - remove brackets and indices
                        clean_id = video_id.split('[')[0].strip()
                        # Extract base video ID (e.g., "video_001" from "video_001[0]")
                        if '_' in clean_id:
                            parts = clean_id.split('_')
                            if len(parts) >= 2:
                                clean_id = f"{parts[-2]}_{parts[-1]}"
                        
                        # Store label (if multiple segments, average them)
                        if clean_id in labels_dict:
                            # Average if duplicate
                            labels_dict[clean_id] = (labels_dict[clean_id] + sentiment_value) / 2
                        else:
                            labels_dict[clean_id] = sentiment_value
                except Exception as e:
                    print(f"  Warning: Could not process {video_id}: {e}")
                    continue
    
    except Exception as e:
        print(f"ERROR reading .csd file: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"\n[OK] Extracted {len(labels_dict)} unique video labels")
    
    # Show sample labels
    print("\nSample labels (first 15):")
    for i, (vid_id, label) in enumerate(list(labels_dict.items())[:15]):
        print(f"  {i+1:2d}. {vid_id:30s}: {label:7.4f}")
    
    # Save to JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(labels_dict, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"[OK] Labels saved to: {output_file}")
    print(f"{'='*80}")
    print(f"  Total labels: {len(labels_dict)}")
    print(f"  Non-zero labels: {sum(1 for v in labels_dict.values() if abs(v) > 1e-6)}")
    print(f"  Zero labels: {sum(1 for v in labels_dict.values() if abs(v) <= 1e-6)}")
    
    if labels_dict:
        values = [v for v in labels_dict.values() if abs(v) > 1e-6]
        if values:
            print(f"  Non-zero label range: [{min(values):.2f}, {max(values):.2f}]")
            print(f"  Non-zero label mean: {np.mean(values):.4f}")
    
    return labels_dict

if __name__ == '__main__':
    import sys
    
    output_path = "C:/Users/PC/Downloads/CMU-MOSI Dataset/labels.json"
    
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    
    download_and_extract_labels(output_path)

