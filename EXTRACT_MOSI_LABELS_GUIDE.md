# How to Extract CMU-MOSI Labels

## Problem:
Your transfer learning script found only 1/93 samples with labels because the ID matching between files and `labels.json` is failing.

## Solution:
The CMU-MultimodalSDK provides the official way to access CMU-MOSI labels. The labels are stored in "Opinion Segment Labels" which are downloaded as `.csd` files (HDF5 format).

## Method 1: Using CMU-MultimodalSDK (Recommended)

### Step 1: Install SDK
```powershell
pip install mmsdk
```

### Step 2: Run extraction script
```powershell
python "C:\Users\PC\Downloads\Github Folders 2\extract_mosi_labels.py" --method sdk --output "C:\Users\PC\Downloads\CMU-MOSI Dataset\labels.json"
```

This will:
1. Download labels from CMU servers if not already downloaded
2. Extract sentiment labels for each video
3. Save them as `labels.json` compatible with your script

## Method 2: Direct .csd file reading

If you already have the `.csd` file downloaded:

1. Find the file `CMU_MOSI_Opinion_Labels.csd` (or `Opinion Segment Labels.csd`)
2. Run:
```powershell
python "C:\Users\PC\Downloads\Github Folders 2\extract_mosi_labels.py" --method csd --csd_file "path/to/CMU_MOSI_Opinion_Labels.csd" --output "C:\Users\PC\Downloads\CMU-MOSI Dataset\labels.json"
```

## Method 3: Manual ID matching fix

If you want to fix the ID matching in your current `labels.json`:

1. Run the diagnostic script first:
```powershell
python "C:\Users\PC\Downloads\Github Folders 2\diagnose_label_loading.py" "C:\Users\PC\Downloads\CMU-MOSI Dataset"
```

2. Based on the output, create a mapping script to fix ID mismatches

## Expected Output:

After extraction, `labels.json` should contain:
- ~93-200 entries (depends on segment vs video level)
- Video IDs matching your file names
- Sentiment values in range [-3.0, 3.0]
- Most labels should be non-zero

Example `labels.json` structure:
```json
{
  "video_001": 1.5,
  "video_002": -0.8,
  "video_003": 2.1,
  ...
}
```

## Troubleshooting:

1. **SDK not installing**: Try `pip install --upgrade pip` first
2. **Download timeout**: The SDK downloads from CMU servers - ensure internet connection
3. **ID mismatch**: Run diagnostic script to see actual IDs in both files
4. **Permission errors**: Make sure you have write access to output directory




