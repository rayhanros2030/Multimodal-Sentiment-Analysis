# PowerShell script to run the fixed transfer learning script
# This version includes proper train/val/test split (no data leakage)

python "C:\Users\PC\Downloads\Github Folders 2\train_mosei_test_mosi_with_adapters.py" `
  --mosei_dir "C:/Users/PC/Downloads/CMU-MOSEI" `
  --mosi_dir "C:/Users/PC/Downloads/CMU-MOSI Dataset" `
  --mosi_samples 93 `
  --adapter_epochs 75 `
  --fine_tune_epochs 20




