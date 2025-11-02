#!/usr/bin/env python3
"""
CMU-MOSEI Modality Combination Testing
======================================

Tests different modality combinations:
- Text + Visual (no Audio)
- Text + Audio (no Visual)
- Visual + Audio (no Text)
- Individual modalities
"""

import os
import sys
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
from scipy.stats import pearsonr
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import time
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== FLEXIBLE MODEL ARCHITECTURE ====================

class FlexibleMultimodalModel(nn.Module):
    """Flexible multimodal model that can handle any combination of modalities"""
    
    def __init__(self, visual_dim=713, audio_dim=74, text_dim=300, 
                 hidden_dim=224, embed_dim=112, dropout=0.65,
                 use_visual=True, use_audio=True, use_text=True):
        super().__init__()
        
        self.use_visual = use_visual
        self.use_audio = use_audio
        self.use_text = use_text
        
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        
        # Create encoders only for used modalities
        if self.use_visual:
            self.visual_encoder = self._create_encoder(visual_dim, hidden_dim, embed_dim, dropout)
        if self.use_audio:
            self.audio_encoder = self._create_encoder(audio_dim, hidden_dim, embed_dim, dropout)
        if self.use_text:
            self.text_encoder = self._create_encoder(text_dim, hidden_dim, embed_dim, dropout)
        
        # Count number of active modalities
        num_modalities = sum([self.use_visual, self.use_audio, self.use_text])
        
        if num_modalities > 1:
            # Cross-modal attention
            self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=0.5, batch_first=True)
        
        # Fusion layers (adapts to number of modalities)
        fusion_input_dim = embed_dim * num_modalities
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def _create_encoder(self, input_dim, hidden_dim, embed_dim, dropout):
        """Create encoder"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
    
    def forward(self, visual=None, audio=None, text=None):
        encodings = []
        
        if self.use_visual and visual is not None:
            v_enc = self.visual_encoder(visual)
            encodings.append(v_enc)
        
        if self.use_audio and audio is not None:
            a_enc = self.audio_encoder(audio)
            encodings.append(a_enc)
        
        if self.use_text and text is not None:
            t_enc = self.text_encoder(text)
            encodings.append(t_enc)
        
        if len(encodings) == 0:
            raise ValueError("At least one modality must be enabled")
        
        # Cross-modal attention if multiple modalities
        if len(encodings) > 1:
            features = torch.stack(encodings, dim=1)
            attended_features, _ = self.cross_attention(features, features, features)
            # Use attended features for fusion
            concat_features = torch.cat(encodings, dim=-1)
        else:
            concat_features = encodings[0]
        
        output = self.fusion_layers(concat_features)
        return output.squeeze(-1)

class ImprovedCorrelationLoss(nn.Module):
    """Improved loss function"""
    
    def __init__(self, alpha=0.3, beta=0.7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def pearson_correlation_loss(self, pred, target):
        """Stable Pearson correlation loss"""
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        
        numerator = (pred_centered * target_centered).mean()
        pred_std = torch.sqrt((pred_centered ** 2).mean() + 1e-8)
        target_std = torch.sqrt((target_centered ** 2).mean() + 1e-8)
        denominator = pred_std * target_std
        
        correlation = numerator / denominator
        return (1 - correlation) ** 2
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        corr_loss = self.pearson_correlation_loss(pred, target)
        
        total_loss = self.alpha * (mse_loss + mae_loss) / 2 + self.beta * corr_loss
        
        return total_loss, {
            'mse': mse_loss.item(),
            'mae': mae_loss.item(),
            'corr': corr_loss.item()
        }

# ==================== DATA LOADING ====================

class MOSEIDataset(Dataset):
    """CMU-MOSEI Dataset Loader"""
    
    def __init__(self, mosei_dir: str, max_samples: int = None):
        self.mosei_dir = Path(mosei_dir)
        self.max_samples = max_samples
        
        print(f"Loading CMU-MOSEI from: {self.mosei_dir}")
        self.samples = self._load_mosei_data()
        print(f"Loaded {len(self.samples)} MOSEI samples")
    
    def _load_mosei_data(self) -> List[Dict]:
        """Load MOSEI data"""
        samples = []
        
        visual_path = self.mosei_dir / 'visuals' / 'CMU_MOSEI_VisualOpenFace2.csd'
        audio_path = self.mosei_dir / 'acoustics' / 'CMU_MOSEI_COVAREP.csd'
        text_path = self.mosei_dir / 'languages' / 'CMU_MOSEI_TimestampedWordVectors.csd'
        labels_path = self.mosei_dir / 'labels' / 'CMU_MOSEI_Labels.csd'
        
        if not all([visual_path.exists(), audio_path.exists(), text_path.exists(), labels_path.exists()]):
            print(f"ERROR: MOSEI files not found!")
            return []
        
        print("  Loading visual features...")
        visual_data = self._load_csd_file(visual_path, 'OpenFace_2') or self._load_csd_file(visual_path, 'Visual')
        print("  Loading audio features...")
        audio_data = self._load_csd_file(audio_path, 'COVAREP') or self._load_csd_file(audio_path, 'Audio')
        print("  Loading text features...")
        text_data = self._load_csd_file(text_path, 'glove_vectors') or self._load_csd_file(text_path, 'Text')
        print("  Loading labels...")
        labels_data = self._load_csd_file(labels_path, 'All Labels') or self._load_csd_file(labels_path, 'Sentiment')
        
        common_ids = set(visual_data.keys()) & set(audio_data.keys()) & set(text_data.keys()) & set(labels_data.keys())
        print(f"Found {len(common_ids)} common video IDs")
        
        total_attempted = 0
        skipped = 0
        for vid_id in list(common_ids)[:self.max_samples] if self.max_samples else common_ids:
            total_attempted += 1
            try:
                visual_feat = self._extract_features(visual_data[vid_id], 713)
                audio_feat = self._extract_features(audio_data[vid_id], 74)
                text_feat = self._extract_features(text_data[vid_id], 300)
                sentiment = self._extract_sentiment(labels_data[vid_id])
                
                visual_feat = self._clean_features(visual_feat)
                audio_feat = self._clean_features(audio_feat)
                text_feat = self._clean_features(text_feat)
                sentiment = self._clean_sentiment(sentiment)
                
                if (np.all(visual_feat == 0) and np.all(audio_feat == 0) and np.all(text_feat == 0)):
                    skipped += 1
                    continue
                
                samples.append({
                    'audio': audio_feat,
                    'visual': visual_feat,
                    'text': text_feat,
                    'sentiment': sentiment
                })
            except Exception as e:
                skipped += 1
                continue
        
        print(f"Successfully created {len(samples)} valid samples out of {total_attempted} attempted ({skipped} skipped)")
        return samples
    
    def _clean_features(self, features: np.ndarray) -> np.ndarray:
        """Clean features"""
        features = np.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
        features = np.clip(features, -500, 500)
        return features
    
    def _clean_sentiment(self, sentiment: float) -> float:
        """Clean sentiment value"""
        if np.isnan(sentiment) or np.isinf(sentiment):
            return 0.0
        return float(np.clip(sentiment, -3.0, 3.0))
    
    def _load_csd_file(self, path: Path, key: str) -> Dict:
        """Load .csd file"""
        data = {}
        try:
            with h5py.File(path, 'r') as f:
                keys_to_try = [key, key.lower(), key.upper(), 'data']
                found_key = None
                
                for k in keys_to_try:
                    if k in f:
                        found_key = k
                        break
                
                if not found_key:
                    found_key = list(f.keys())[0] if len(f.keys()) > 0 else None
                
                if found_key:
                    feature_group = f[found_key]
                    if 'data' in feature_group:
                        data_group = feature_group['data']
                        for video_id in data_group.keys():
                            try:
                                video_group = data_group[video_id]
                                if 'features' in video_group:
                                    features = video_group['features'][:]
                                    data[video_id] = {'features': features}
                            except Exception:
                                continue
        except Exception as e:
            print(f"  Error loading {path.name}: {e}")
        return data
    
    def _extract_features(self, data: Dict, target_dim: int) -> np.ndarray:
        """Extract and pad features"""
        if data is None or 'features' not in data:
            return np.zeros(target_dim, dtype=np.float32)
        
        features = data['features']
        
        if len(features.shape) > 1:
            features = np.mean(features, axis=0) if features.shape[0] > 1 else features[0]
        
        features = features.flatten()
        
        if len(features) > target_dim:
            features = features[:target_dim]
        elif len(features) < target_dim:
            features = np.pad(features, (0, target_dim - len(features)), mode='constant', constant_values=0)
        
        return features.astype(np.float32)
    
    def _extract_sentiment(self, data: Dict) -> float:
        """Extract sentiment score - Uses mean of all segments"""
        if data is None or 'features' not in data:
            return 0.0
        
        features = data['features']
        
        try:
            if len(features.shape) > 1:
                sentiment = float(np.mean(features[:, 0])) if features.shape[1] > 0 else 0.0
            else:
                sentiment = float(features[0]) if len(features) > 0 else 0.0
        except:
            try:
                sentiment = float(np.mean(features))
            except:
                sentiment = 0.0
        
        return sentiment
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'audio': torch.FloatTensor(sample['audio']),
            'visual': torch.FloatTensor(sample['visual']),
            'text': torch.FloatTensor(sample['text']),
            'sentiment': torch.FloatTensor([sample['sentiment']])
        }

class TransformedSubset(torch.utils.data.Subset):
    """Wrap a Subset to apply scalers on-the-fly without leakage."""
    def __init__(self, subset, audio_scaler, visual_scaler, text_scaler, use_audio=True, use_visual=True, use_text=True):
        super().__init__(subset.dataset, subset.indices)
        self.audio_scaler = audio_scaler
        self.visual_scaler = visual_scaler
        self.text_scaler = text_scaler
        self.use_audio = use_audio
        self.use_visual = use_visual
        self.use_text = use_text
    
    def __getitem__(self, idx):
        item = self.dataset[self.indices[idx]]
        
        if self.use_audio:
            item['audio'] = torch.FloatTensor(self.audio_scaler.transform(item['audio'].numpy().reshape(1, -1))).flatten()
        if self.use_visual:
            item['visual'] = torch.FloatTensor(self.visual_scaler.transform(item['visual'].numpy().reshape(1, -1))).flatten()
        if self.use_text:
            item['text'] = torch.FloatTensor(self.text_scaler.transform(item['text'].numpy().reshape(1, -1))).flatten()
        
        return item

def calculate_metrics(predictions, targets):
    """Calculate MAE and Pearson correlation"""
    mae = torch.nn.functional.l1_loss(predictions, targets).item()
    
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    if len(pred_np) > 1:
        correlation, _ = pearsonr(pred_np.flatten(), target_np.flatten())
        correlation = float(correlation) if not np.isnan(correlation) else 0.0
    else:
        correlation = 0.0
    
    return mae, correlation

def train_modality_combination(combination_name, use_visual, use_audio, use_text):
    """Train model with specific modality combination"""
    
    print("=" * 80)
    print(f"TRAINING: {combination_name}")
    print("=" * 80)
    print(f"Modalities: Visual={use_visual}, Audio={use_audio}, Text={use_text}")
    print()
    
    # Load dataset
    mosei_dir = r"C:\Users\PC\Downloads\CMU-MOSEI"
    mosei_dataset = MOSEIDataset(mosei_dir)
    
    if len(mosei_dataset) == 0:
        print("ERROR: No samples loaded!")
        return None
    
    # Split dataset
    total_size = len(mosei_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        mosei_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    # Fit scalers on TRAIN ONLY
    print("Fitting scalers on train split...")
    train_audio = []
    train_visual = []
    train_text = []
    for idx in train_dataset.indices:
        s = mosei_dataset[idx]
        train_audio.append(s['audio'].numpy())
        train_visual.append(s['visual'].numpy())
        train_text.append(s['text'].numpy())
    
    audio_scaler = RobustScaler().fit(np.vstack(train_audio)) if use_audio else None
    visual_scaler = RobustScaler().fit(np.vstack(train_visual)) if use_visual else None
    text_scaler = RobustScaler().fit(np.vstack(train_text)) if use_text else None
    
    # Wrap subsets
    train_dataset = TransformedSubset(train_dataset, audio_scaler, visual_scaler, text_scaler, use_audio, use_visual, use_text)
    val_dataset = TransformedSubset(val_dataset, audio_scaler, visual_scaler, text_scaler, use_audio, use_visual, use_text)
    test_dataset = TransformedSubset(test_dataset, audio_scaler, visual_scaler, text_scaler, use_audio, use_visual, use_text)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    model = FlexibleMultimodalModel(
        use_visual=use_visual,
        use_audio=use_audio,
        use_text=use_text,
        hidden_dim=224,
        embed_dim=112,
        dropout=0.65
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = ImprovedCorrelationLoss(alpha=0.3, beta=0.7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.04)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=7)
    
    best_correlation = -1.0
    num_epochs = 50  # Fewer epochs for faster testing
    
    print(f"\nStarting training...")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print()
    
    train_losses = []
    train_maes = []
    train_corrs = []
    val_losses = []
    val_maes = []
    val_corrs = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_preds = []
        train_targets = []
        train_loss_epoch = 0.0
        
        for batch in train_loader:
            visual = batch['visual'].to(device) if use_visual else None
            audio = batch['audio'].to(device) if use_audio else None
            text = batch['text'].to(device) if use_text else None
            sentiment = batch['sentiment'].to(device).squeeze()
            
            optimizer.zero_grad()
            output = model(visual=visual, audio=audio, text=text)
            loss, loss_components = criterion(output, sentiment)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            train_loss_epoch += loss.item()
            train_preds.append(output.detach())
            train_targets.append(sentiment.detach())
        
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_loss = train_loss_epoch / len(train_loader)
        train_mae, train_corr = calculate_metrics(train_preds, train_targets)
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss_epoch = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                visual = batch['visual'].to(device) if use_visual else None
                audio = batch['audio'].to(device) if use_audio else None
                text = batch['text'].to(device) if use_text else None
                sentiment = batch['sentiment'].to(device).squeeze()
                
                output = model(visual=visual, audio=audio, text=text)
                loss, _ = criterion(output, sentiment)
                
                val_loss_epoch += loss.item()
                val_preds.append(output.detach())
                val_targets.append(sentiment.detach())
        
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_loss = val_loss_epoch / len(val_loader)
        val_mae, val_corr = calculate_metrics(val_preds, val_targets)
        
        # Scheduler step
        scheduler.step(val_corr)
        
        # Track best
        if val_corr > best_correlation:
            best_correlation = val_corr
        
        # Store metrics
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        train_corrs.append(train_corr)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        val_corrs.append(val_corr)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train Corr: {train_corr:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val Corr: {val_corr:.4f} | Best: {best_correlation:.4f}")
    
    print()
    print("=" * 80)
    print(f"RESULTS: {combination_name}")
    print("=" * 80)
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Train MAE:  {train_maes[-1]:.4f}")
    print(f"Final Train Corr: {train_corrs[-1]:.4f}")
    print(f"Final Val Loss:   {val_losses[-1]:.4f}")
    print(f"Final Val MAE:    {val_maes[-1]:.4f}")
    print(f"Final Val Corr:   {val_corrs[-1]:.4f}")
    print(f"Best Val Corr:    {best_correlation:.4f}")
    print("=" * 80)
    print()
    
    return {
        'combination': combination_name,
        'modalities': {'visual': use_visual, 'audio': use_audio, 'text': use_text},
        'train_loss': train_losses[-1],
        'train_mae': train_maes[-1],
        'train_corr': train_corrs[-1],
        'val_loss': val_losses[-1],
        'val_mae': val_maes[-1],
        'val_corr': val_corrs[-1],
        'best_val_corr': best_correlation
    }

def main():
    """Test different modality combinations"""
    
    combinations = [
        ("Text + Visual", False, True, True),
        ("Text + Audio", True, False, True),
        ("Visual + Audio", True, True, False),
        ("Text Only", False, False, True),
        ("Visual Only", False, True, False),
        ("Audio Only", True, False, False),
        ("All Modalities", True, True, True),
    ]
    
    results = []
    
    for name, use_audio, use_visual, use_text in combinations:
        result = train_modality_combination(name, use_visual, use_audio, use_text)
        if result:
            results.append(result)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL MODALITY COMBINATIONS")
    print("=" * 80)
    print(f"{'Combination':<20} {'Train Loss':<12} {'Train MAE':<12} {'Train Corr':<12} {'Val Loss':<12} {'Val MAE':<12} {'Val Corr':<12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['combination']:<20} {r['train_loss']:<12.4f} {r['train_mae']:<12.4f} {r['train_corr']:<12.4f} "
              f"{r['val_loss']:<12.4f} {r['val_mae']:<12.4f} {r['val_corr']:<12.4f}")
    
    print("=" * 80)
    
    # Save results
    with open('modality_combination_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to: modality_combination_results.json")

if __name__ == "__main__":
    main()

