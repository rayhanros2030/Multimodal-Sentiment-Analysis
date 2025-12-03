# Test Different Modality Combinations with Plots

## 🎯 **What This Script Does:**

1. **Tests different modality combinations:**
   - All (Visual + Audio + Text)
   - Audio + Text
   - Audio + Visual
   - Text + Visual
   - Audio only
   - Visual only
   - Text only

2. **Tracks full training metrics:**
   - Train Loss, Train MAE, Train Correlation
   - Val Loss, Val MAE, Val Correlation

3. **Generates training curves:**
   - Loss plots
   - MAE plots
   - Correlation plots
   - All saved as PNG files

4. **Same structure:**
   - Train on MOSEI
   - Train adapters
   - Fine-tune with full tracking
   - Test on held-out set

---

## 🚀 **How to Run:**

### **Test All Modalities:**
```powershell
python "C:\Users\PC\Downloads\Github Folders 2\train_modality_combinations_with_plots.py" --combination "all"
```

### **Test Audio + Text:**
```powershell
python "C:\Users\PC\Downloads\Github Folders 2\train_modality_combinations_with_plots.py" --combination "audio+text"
```

### **Test Audio + Visual:**
```powershell
python "C:\Users\PC\Downloads\Github Folders 2\train_modality_combinations_with_plots.py" --combination "audio+visual"
```

### **Test Text + Visual:**
```powershell
python "C:\Users\PC\Downloads\Github Folders 2\train_modality_combinations_with_plots.py" --combination "text+visual"
```

### **Test Single Modalities:**
```powershell
python "C:\Users\PC\Downloads\Github Folders 2\train_modality_combinations_with_plots.py" --combination "audio"
python "C:\Users\PC\Downloads\Github Folders 2\train_modality_combinations_with_plots.py" --combination "visual"
python "C:\Users\PC\Downloads\Github Folders 2\train_modality_combinations_with_plots.py" --combination "text"
```

---

## 📊 **Output Files:**

For each combination, you'll get:

1. **Training Curves:** `training_curves_{combination}.png`
   - 6 subplots showing Loss, MAE, Correlation for Train and Val
   
2. **Results:** `results_{combination}.json`
   - Test metrics (Correlation, MAE, MSE)
   - Full training history (all metrics per epoch)

3. **Model:** `model_{combination}.pth`
   - Trained model weights

---

## 📈 **What the Graphs Show:**

### **Top Row:**
- **Loss:** Training and validation loss over epochs
- **MAE:** Training and validation MAE over epochs
- **Correlation:** Training and validation correlation over epochs

### **Bottom Row:**
- Combined comparison plots for all metrics

---

## ⚙️ **Options:**

```powershell
--mosei_dir "path/to/MOSEI"          # MOSEI dataset path
--mosi_dir "path/to/MOSI"            # MOSI dataset path
--combination "all"                   # Modality combination
--mosi_samples 93                     # Number of MOSI samples
--adapter_epochs 75                   # Adapter training epochs
--fine_tune_epochs 20                 # Fine-tuning epochs
--skip_training                       # Skip MOSEI training (use existing)
--skip_adapters                       # Skip adapter training (use existing)
```

---

## 📋 **Example: Test All Combinations**

```powershell
# All modalities
python train_modality_combinations_with_plots.py --combination "all"

# Audio + Text
python train_modality_combinations_with_plots.py --combination "audio+text"

# Audio + Visual
python train_modality_combinations_with_plots.py --combination "audio+visual"

# Text + Visual
python train_modality_combinations_with_plots.py --combination "text+visual"
```

---

## 🎯 **Expected Output:**

For each run, you'll see:
- Training progress on MOSEI
- Adapter training progress
- Fine-tuning with metrics printed every 5 epochs
- Test results on held-out set
- Training curves saved as PNG
- Results saved as JSON

---

## 💡 **Tips:**

1. **First run:** Use `--combination "all"` to get baseline
2. **Then test:** Other combinations to compare
3. **Skip training:** Use `--skip_training` and `--skip_adapters` for faster iteration
4. **Compare results:** Look at both correlation and MAE

---

## 📊 **What to Analyze:**

1. **Correlation:** Higher is better
2. **MAE:** Lower is better
3. **Training curves:** Check for overfitting (train >> val)
4. **Convergence:** Check if metrics plateau

---

## ✅ **Ready to Run!**

The script is ready. Just run it with your desired combination!




