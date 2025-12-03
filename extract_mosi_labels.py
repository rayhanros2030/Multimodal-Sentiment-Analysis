"""
Extract CMU-MOSI labels and create labels.json compatible with your script.
This script uses the CMU-MultimodalSDK to extract labels from the Opinion Segment Labels.
"""

import sys
import json
from pathlib import Path

def extract_labels_using_sdk(mosi_dir=None, output_path="labels.json"):
    """
    Extract CMU-MOSI labels using the CMU-MultimodalSDK
    
    Args:
        mosi_dir: Directory containing CMU-MOSI dataset (optional, SDK will download if needed)
        output_path: Path to save labels.json
    """
    
    try:
        # Try importing the SDK - try multiple methods
        try:
            from mmsdk import mmdatasdk
        except ImportError:
            # Try using extracted SDK
            import sys
            sdk_path = Path(__file__).parent.parent / 'CMU-MultimodalSDK-extracted' / 'CMU-MultimodalSDK-main'
            if sdk_path.exists():
                sys.path.insert(0, str(sdk_path))
                from mmsdk import mmdatasdk
                print(f"  Using SDK from extracted folder: {sdk_path}")
            else:
                raise ImportError("Cannot find mmsdk. Please install: pip install mmsdk")
        
        import numpy as np
        
        print("="*80)
        print("Extracting CMU-MOSI Labels using CMU-MultimodalSDK")
        print("="*80)
        
        # Load labels using SDK
        print("\nLoading CMU-MOSI labels...")
        
        # If mosi_dir is provided, check if labels are already downloaded
        labels_path = None
        if mosi_dir:
            labels_path = Path(mosi_dir) / 'cmumosi' / 'Opinion Segment Labels.csd'
            if not labels_path.exists():
                # Also try alternative paths
                alt_paths = [
                    Path(mosi_dir) / 'Opinion Segment Labels.csd',
                    Path(mosi_dir) / 'labels' / 'Opinion Segment Labels.csd',
                    Path(mosi_dir) / 'CMU_MOSI_Opinion_Labels.csd',
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        labels_path = alt_path
                        break
        
        if labels_path and labels_path.exists():
            print(f"  Found labels at: {labels_path}")
            labels_dataset = mmdatasdk.mmdataset(str(labels_path.parent))
        else:
            print("  Downloading labels from CMU servers...")
            print("  This may take a few minutes...")
            # Use absolute path to avoid SDK bug
            download_dir = Path.cwd() / 'cmumosi_labels'
            download_dir.mkdir(exist_ok=True)
            labels_dataset = mmdatasdk.mmdataset(
                mmdatasdk.cmu_mosi.labels,
                str(download_dir.absolute())
            )
        
        # Extract labels
        print("\nExtracting labels from dataset...")
        labels_dict = {}
        
        if "Opinion Segment Labels" in labels_dataset.computational_sequences:
            label_seq = labels_dataset["Opinion Segment Labels"]
            
            print(f"  Found {len(label_seq.data)} videos with labels")
            
            for video_id, label_data in label_seq.data.items():
                # label_data contains 'intervals' and 'features'
                # Features contain the sentiment values
                intervals = np.array(label_data["intervals"])
                features = np.array(label_data["features"])
                
                # Extract sentiment value - usually the first feature or average
                if len(features.shape) == 2:
                    # Multiple segments per video - take average or first
                    sentiment_value = float(np.mean(features[:, 0])) if features.shape[1] > 0 else 0.0
                elif len(features.shape) == 1:
                    sentiment_value = float(features[0]) if len(features) > 0 else 0.0
                else:
                    sentiment_value = float(features[0]) if features.size > 0 else 0.0
                
                # Clean video ID (remove brackets and indices)
                clean_id = video_id.split('[')[0].strip()
                
                # Store label
                if clean_id not in labels_dict:
                    labels_dict[clean_id] = sentiment_value
                else:
                    # If duplicate, average them
                    labels_dict[clean_id] = (labels_dict[clean_id] + sentiment_value) / 2
        
        else:
            print("  ERROR: 'Opinion Segment Labels' not found in dataset!")
            print(f"  Available sequences: {list(labels_dataset.computational_sequences.keys())}")
            return None
        
        print(f"\n  Extracted {len(labels_dict)} unique video labels")
        
        # Show sample labels
        print("\nSample labels (first 10):")
        for i, (vid_id, label) in enumerate(list(labels_dict.items())[:10]):
            print(f"  {i+1}. {vid_id}: {label:.4f}")
        
        # Save to JSON
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(labels_dict, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✓ Labels saved to: {output_file}")
        print(f"{'='*80}")
        print(f"  Total labels: {len(labels_dict)}")
        print(f"  Non-zero labels: {sum(1 for v in labels_dict.values() if v != 0.0)}")
        print(f"  Zero labels: {sum(1 for v in labels_dict.values() if v == 0.0)}")
        
        if labels_dict:
            values = list(labels_dict.values())
            print(f"  Label range: [{min(values):.2f}, {max(values):.2f}]")
            print(f"  Label mean: {np.mean(values):.4f}")
        
        return labels_dict
        
    except ImportError:
        print("ERROR: CMU-MultimodalSDK not installed!")
        print("\nPlease install it using:")
        print("  pip install mmsdk")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_labels_from_csd_file(csd_file_path, output_path="labels.json"):
    """
    Extract labels directly from a .csd (computational sequence data) file.
    This uses h5py to read the HDF5 format that .csd files use.
    """
    
    try:
        import h5py
        import numpy as np
        
        print("="*80)
        print("Extracting CMU-MOSI Labels from .csd file")
        print("="*80)
        
        csd_path = Path(csd_file_path)
        
        if not csd_path.exists():
            print(f"ERROR: File not found: {csd_path}")
            return None
        
        print(f"\nReading: {csd_path}")
        
        labels_dict = {}
        
        with h5py.File(csd_path, 'r') as f:
            # .csd files have a specific structure
            # Typically: data -> video_id -> features and intervals
            
            print("\nExploring file structure...")
            
            def explore_h5(obj, prefix=""):
                """Recursively explore HDF5 structure"""
                if isinstance(obj, h5py.Group):
                    for key in obj.keys():
                        print(f"{prefix}{key}/")
                        explore_h5(obj[key], prefix + "  ")
                elif isinstance(obj, h5py.Dataset):
                    print(f"{prefix}[Dataset: shape={obj.shape}, dtype={obj.dtype}]")
            
            explore_h5(f)
            
            # Try to find label data
            # Common structure: data -> video_id -> features/intervals
            if 'data' in f:
                data_group = f['data']
                print(f"\nFound 'data' group with {len(data_group.keys())} entries")
                
                for video_id in data_group.keys():
                    video_group = data_group[video_id]
                    
                    if 'features' in video_group:
                        features = video_group['features'][:]
                        intervals = video_group['intervals'][:] if 'intervals' in video_group else None
                        
                        # Extract sentiment value
                        if len(features.shape) == 2:
                            sentiment_value = float(np.mean(features[:, 0])) if features.shape[1] > 0 else 0.0
                        elif len(features.shape) == 1:
                            sentiment_value = float(features[0]) if len(features) > 0 else 0.0
                        else:
                            sentiment_value = float(features[0]) if features.size > 0 else 0.0
                        
                        # Clean video ID
                        clean_id = video_id.split('[')[0].strip()
                        
                        labels_dict[clean_id] = sentiment_value
        
        if not labels_dict:
            print("\nERROR: Could not extract labels from .csd file!")
            print("Please check the file structure manually.")
            return None
        
        print(f"\n  Extracted {len(labels_dict)} labels")
        
        # Show sample labels
        print("\nSample labels (first 10):")
        for i, (vid_id, label) in enumerate(list(labels_dict.items())[:10]):
            print(f"  {i+1}. {vid_id}: {label:.4f}")
        
        # Save to JSON
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(labels_dict, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✓ Labels saved to: {output_file}")
        print(f"{'='*80}")
        
        return labels_dict
        
    except ImportError:
        print("ERROR: h5py not installed!")
        print("\nPlease install it using:")
        print("  pip install h5py")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract CMU-MOSI labels')
    parser.add_argument('--mosi_dir', type=str, help='CMU-MOSI dataset directory')
    parser.add_argument('--csd_file', type=str, help='Path to Opinion Segment Labels.csd file')
    parser.add_argument('--output', type=str, default='labels.json', help='Output JSON file path')
    parser.add_argument('--method', type=str, choices=['sdk', 'csd'], default='sdk',
                       help='Extraction method: sdk (uses CMU-MultimodalSDK) or csd (reads .csd directly)')
    
    args = parser.parse_args()
    
    if args.method == 'sdk':
        extract_labels_using_sdk(args.mosi_dir, args.output)
    elif args.method == 'csd':
        if not args.csd_file:
            print("ERROR: --csd_file required when using --method csd")
            sys.exit(1)
        extract_labels_from_csd_file(args.csd_file, args.output)
    else:
        print(f"ERROR: Unknown method: {args.method}")
        sys.exit(1)

