#!/usr/bin/env python3
"""
Complete Optimized Three-Modality Training
==========================================

This is a FULL, runnable script that optimizes 3-modality fusion
to beat Text+Visual performance. Uses:
1. Hierarchical attention (pairwise then global)
2. Better audio preprocessing (quality filtering + cleaning)
3. Gated fusion mechanism
4. Deeper architecture
5. Better training strategy
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

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== OPTIMIZED THREE-MODALITY MODEL ====================

class OptimizedThreeModalityModel(nn.Module):
    """Advanced 3-modality model designed to outperform Text+Visual"""
    
    def __init__(self, visual_dim=713, audio_dim=74, text_dim=300, 
                 hidden_dim=256, embed_dim=128, dropout=0.6):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        
        # Deeper, more powerful encoders
        self.visual_encoder = self._create_deeper_encoder(visual_dim, hidden_dim, embed_dim, dropout)
        self.audio_encoder = self._create_deeper_encoder(audio_dim, hidden_dim, embed_dim, dropout)
        self.text_encoder = self._create_deeper_encoder(text_dim, hidden_dim, embed_dim, dropout)
        
        # Hierarchical attention: pairwise then global
        # Pairwise attention between each modality pair
        self.pairwise_attention = nn.ModuleDict({
            'va': nn.MultiheadAttention(embed_dim, num_heads=4, dropout=0.4, batch_first=True),
            'vt': nn.MultiheadAttention(embed_dim, num_heads=4, dropout=0.4, batch_first=True),
            'at': nn.MultiheadAttention(embed_dim, num_heads=4, dropout=0.4, batch_first=True),
        })
        
        # Self-attention refinement for each modality
        self.self_attention = nn.ModuleDict({
            'v': nn.MultiheadAttention(embed_dim, num_heads=4, dropout=0.3, batch_first=True),
            'a': nn.MultiheadAttention(embed_dim, num_heads=4, dropout=0.3, batch_first=True),
            't': nn.MultiheadAttention(embed_dim, num_heads=4, dropout=0.3, batch_first=True),
        })
        
        # Global 3-way attention
        self.global_attention = nn.MultiheadAttention(embed_dim, num_heads=8, dropout=0.5, batch_first=True)
        
        # Gated fusion mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 3),
            nn.LayerNorm(embed_dim * 3),
            nn.Sigmoid()
        )
        
        # Enhanced fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),  # Less dropout near output
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def _create_deeper_encoder(self, input_dim, hidden_dim, embed_dim, dropout):
        """Create deeper encoder"""
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
    
    def forward(self, visual, audio, text):
        # Encode each modality
        v_enc = self.visual_encoder(visual)
        a_enc = self.audio_encoder(audio)
        t_enc = self.text_encoder(text)
        
        # Pairwise cross-modal attention with residual connections
        # Visual-Audio
        va_out, _ = self.pairwise_attention['va'](
            v_enc.unsqueeze(1), a_enc.unsqueeze(1), a_enc.unsqueeze(1)
        )
        v_enhanced = v_enc + 0.5 * va_out.squeeze(1)
        
        # Visual-Text
        vt_out, _ = self.pairwise_attention['vt'](
            v_enc.unsqueeze(1), t_enc.unsqueeze(1), t_enc.unsqueeze(1)
        )
        v_enhanced = v_enhanced + 0.5 * vt_out.squeeze(1)
        
        # Audio-Text
        at_out, _ = self.pairwise_attention['at'](
            a_enc.unsqueeze(1), t_enc.unsqueeze(1), t_enc.unsqueeze(1)
        )
        a_enhanced = a_enc + at_out.squeeze(1)
        
        # Self-attention refinement
        v_refined, _ = self.self_attention['v'](v_enhanced.unsqueeze(1), v_enhanced.unsqueeze(1), v_enhanced.unsqueeze(1))
        a_refined, _ = self.self_attention['a'](a_enhanced.unsqueeze(1), a_enhanced.unsqueeze(1), a_enhanced.unsqueeze(1))
        t_refined, _ = self.self_attention['t'](t_enc.unsqueeze(1), t_enc.unsqueeze(1), t_enc.unsqueeze(1))
        
        v_final = v_enhanced + v_refined.squeeze(1)
        a_final = a_enhanced + a_refined.squeeze(1)
        t_final = t_enc + t_refined.squeeze(1)
        
        # Global 3-way attention
        features = torch.stack([v_final, a_final, t_final], dim=1)
        global_out, attention_weights = self.global_attention(features, features, features)
        
        # Concatenate for fusion
        concat_features = torch.cat([v_final, a_final, t_final], dim=-1)
        
        # Gated fusion
        gate = self.fusion_gate(concat_features)
        gated_features = gate * concat_features
        
        # Final fusion
        output = self.fusion_layers(gated_features)
        return output.squeeze(-1)

class ImprovedCorrelationLoss(nn.Module):
    """Improved loss function"""
    
    def __init__(self, alpha=0.25, beta=0.75):
        super().__init__()
        self.alpha = alpha  # Even less weight on MSE/MAE
        self.beta = beta    # Even more weight on correlation
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

# ==================== IMPROVED DATA LOADING ====================

class ImprovedMOSEIDataset(Dataset):
    """CMU-MOSEI Dataset with better audio handling"""
    
    def __init__(self, mosei_dir: str, max_samples: int = None, filter_bad_audio=True):
        self.mosei_dir = Path(mosei_dir)
        self.max_samples = max_samples
        self.filter_bad_audio = filter_bad_audio
        
        print(f"Loading CMU-MOSEI from: {self.mosei_dir}")
        self.samples = self._load_mosei_data()
        print(f"Loaded {len(self.samples)} MOSEI samples")
    
    def _load_mosei_data(self) -> List[Dict]:
        """Load MOSEI data with improved audio handling"""
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
        bad_audio_skipped = 0
        
        for vid_id in list(common_ids)[:self.max_samples] if self.max_samples else common_ids:
            total_attempted += 1
            try:
                visual_feat = self._extract_features(visual_data[vid_id], 713)
                audio_feat = self._extract_features(audio_data[vid_id], 74)
                text_feat = self._extract_features(text_data[vid_id], 300)
                sentiment = self._extract_sentiment(labels_data[vid_id])
                
                # Audio quality filtering
                if self.filter_bad_audio:
                    audio_quality = self._assess_audio_quality(audio_feat)
                    if audio_quality < 0.4:  # Filter very bad audio
                        bad_audio_skipped += 1
                        continue
                
                visual_feat = self._clean_features(visual_feat)
                audio_feat = self._clean_audio_features(audio_feat)
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
        
        print(f"Successfully created {len(samples)} valid samples")
        print(f"  - {skipped} skipped (missing features)")
        if self.filter_bad_audio:
            print(f"  - {bad_audio_skipped} skipped (poor audio quality)")
        return samples
    
    def _assess_audio_quality(self, audio_feat):
        """Assess audio feature quality"""
        inf_count = np.isinf(audio_feat).sum()
        extreme_count = (np.abs(audio_feat) > 50).sum()
        zero_count = (audio_feat == 0).sum()
        total = len(audio_feat)
        
        quality = 1.0 - (inf_count + extreme_count * 0.5) / total - 0.3 * (zero_count / total)
        return max(0.0, quality)
    
    def _clean_audio_features(self, features: np.ndarray) -> np.ndarray:
        """Special cleaning for audio features"""
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        # More conservative clipping
        features = np.clip(features, -5, 5)
        # Normalize if too spread out
        std = np.std(features)
        if std > 3:
            features = features / (std + 1e-8) * 2  # Normalize to reasonable range
        return features
    
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
    def __init__(self, subset, audio_scaler, visual_scaler, text_scaler):
        super().__init__(subset.dataset, subset.indices)
        self.audio_scaler = audio_scaler
        self.visual_scaler = visual_scaler
        self.text_scaler = text_scaler
    
    def __getitem__(self, idx):
        item = self.dataset[self.indices[idx]]
        
        item['audio'] = torch.FloatTensor(self.audio_scaler.transform(item['audio'].numpy().reshape(1, -1))).flatten()
        item['visual'] = torch.FloatTensor(self.visual_scaler.transform(item['visual'].numpy().reshape(1, -1))).flatten()
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

def train_model():
    """Train optimized three-modality model"""
    
    print("=" * 80)
    print("OPTIMIZED THREE-MODALITY TRAINING")
    print("=" * 80)
    print()
    
    # Load dataset with audio quality filtering
    mosei_dir = r"C:\Users\PC\Downloads\CMU-MOSEI"
    mosei_dataset = ImprovedMOSEIDataset(mosei_dir, filter_bad_audio=True)
    
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
    print("\nFitting scalers on train split...")
    train_audio = []
    train_visual = []
    train_text = []
    for idx in train_dataset.indices:
        s = mosei_dataset[idx]
        train_audio.append(s['audio'].numpy())
        train_visual.append(s['visual'].numpy())
        train_text.append(s['text'].numpy())
    
    audio_scaler = RobustScaler().fit(np.vstack(train_audio))
    visual_scaler = RobustScaler().fit(np.vstack(train_visual))
    text_scaler = RobustScaler().fit(np.vstack(train_text))
    
    # Wrap subsets
    train_dataset = TransformedSubset(train_dataset, audio_scaler, visual_scaler, text_scaler)
    val_dataset = TransformedSubset(val_dataset, audio_scaler, visual_scaler, text_scaler)
    test_dataset = TransformedSubset(test_dataset, audio_scaler, visual_scaler, text_scaler)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create optimized model
    model = OptimizedThreeModalityModel(
        hidden_dim=256,
        embed_dim=128,
        dropout=0.6
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup - optimized for correlation
    criterion = ImprovedCorrelationLoss(alpha=0.25, beta=0.75)  # Even more correlation focus
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=7)
    
    best_correlation = -1.0
    best_model_state = None
    num_epochs = 100
    
    print(f"\nStarting training...")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Strategy: Hierarchical attention + Audio quality filtering + Gated fusion")
    print()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_preds = []
        train_targets = []
        train_loss_epoch = 0.0
        
        for batch in train_loader:
            visual = batch['visual'].to(device)
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            sentiment = batch['sentiment'].to(device).squeeze()
            
            optimizer.zero_grad()
            output = model(visual, audio, text)
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
                visual = batch['visual'].to(device)
                audio = batch['audio'].to(device)
                text = batch['text'].to(device)
                sentiment = batch['sentiment'].to(device).squeeze()
                
                output = model(visual, audio, text)
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
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track best
        if val_corr > best_correlation:
            best_correlation = val_corr
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train Corr: {train_corr:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val Corr: {val_corr:.4f} | Best: {best_correlation:.4f} | LR: {current_lr:.6f}")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Test evaluation
    print("\nEvaluating on test set...")
    model.eval()
    test_preds = []
    test_targets = []
    test_loss_epoch = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            visual = batch['visual'].to(device)
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            sentiment = batch['sentiment'].to(device).squeeze()
            
            output = model(visual, audio, text)
            loss, _ = criterion(output, sentiment)
            
            test_loss_epoch += loss.item()
            test_preds.append(output.detach())
            test_targets.append(sentiment.detach())
    
    test_preds = torch.cat(test_preds)
    test_targets = torch.cat(test_targets)
    test_loss = test_loss_epoch / len(test_loader)
    test_mae, test_corr = calculate_metrics(test_preds, test_targets)
    
    print()
    print("=" * 80)
    print("FINAL RESULTS - OPTIMIZED THREE-MODALITY MODEL")
    print("=" * 80)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Correlation: {test_corr:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Best Validation Correlation: {best_correlation:.4f}")
    print("=" * 80)
    print()
    
    return {
        'test_loss': test_loss,
        'test_correlation': test_corr,
        'test_mae': test_mae,
        'best_val_correlation': best_correlation
    }

if __name__ == "__main__":
    results = train_model()




