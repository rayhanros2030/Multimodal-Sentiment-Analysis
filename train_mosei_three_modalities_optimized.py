#!/usr/bin/env python3
"""
Optimized Three-Modality Training for Regeneron
===============================================

This script optimizes the three-modality (Visual + Audio + Text) model
to match or beat the Text+Visual performance by:
1. Better audio feature cleaning and preprocessing
2. Improved attention mechanism for 3 modalities
3. Better fusion strategy
4. Audio quality filtering
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

# ==================== OPTIMIZED THREE-MODALITY MODEL ====================

class OptimizedThreeModalityModel(nn.Module):
    """Optimized model specifically designed for 3 modalities"""
    
    def __init__(self, visual_dim=713, audio_dim=74, text_dim=300, 
                 hidden_dim=256, embed_dim=128, dropout=0.6):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        
        # Deeper encoders for better feature extraction
        self.visual_encoder = self._create_deeper_encoder(visual_dim, hidden_dim, embed_dim, dropout)
        self.audio_encoder = self._create_deeper_encoder(audio_dim, hidden_dim, embed_dim, dropout)
        self.text_encoder = self._create_deeper_encoder(text_dim, hidden_dim, embed_dim, dropout)
        
        # Hierarchical attention: pairwise then global
        # Step 1: Pairwise attention between modalities
        self.pairwise_attention = nn.ModuleDict({
            'va': nn.MultiheadAttention(embed_dim, num_heads=4, dropout=0.4, batch_first=True),
            'vt': nn.MultiheadAttention(embed_dim, num_heads=4, dropout=0.4, batch_first=True),
            'at': nn.MultiheadAttention(embed_dim, num_heads=4, dropout=0.4, batch_first=True),
        })
        
        # Step 2: Global fusion attention
        self.global_attention = nn.MultiheadAttention(embed_dim, num_heads=8, dropout=0.5, batch_first=True)
        
        # Enhanced fusion with gating mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.Sigmoid()
        )
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def _create_deeper_encoder(self, input_dim, hidden_dim, embed_dim, dropout):
        """Create deeper encoder for better feature extraction"""
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
        
        # Pairwise cross-modal attention
        # Visual-Audio interaction
        va_out, _ = self.pairwise_attention['va'](
            v_enc.unsqueeze(1), a_enc.unsqueeze(1), a_enc.unsqueeze(1)
        )
        v_enhanced = v_enc + va_out.squeeze(1)  # Residual
        
        # Visual-Text interaction
        vt_out, _ = self.pairwise_attention['vt'](
            v_enc.unsqueeze(1), t_enc.unsqueeze(1), t_enc.unsqueeze(1)
        )
        v_enhanced = v_enhanced + vt_out.squeeze(1)  # Accumulate
        
        # Audio-Text interaction
        at_out, _ = self.pairwise_attention['at'](
            a_enc.unsqueeze(1), t_enc.unsqueeze(1), t_enc.unsqueeze(1)
        )
        a_enhanced = a_enc + at_out.squeeze(1)
        
        # Global attention across all three
        features = torch.stack([v_enhanced, a_enhanced, t_enc], dim=1)
        global_out, _ = self.global_attention(features, features, features)
        global_out = global_out.mean(dim=1)  # Aggregate
        
        # Gated fusion
        concat_features = torch.cat([v_enhanced, a_enhanced, t_enc], dim=-1)
        gate = self.fusion_gate(concat_features)
        gated_features = gate * concat_features
        
        # Final fusion
        output = self.fusion_layers(gated_features)
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

# ==================== IMPROVED DATA LOADING ====================

class ImprovedMOSEIDataset(Dataset):
    """CMU-MOSEI Dataset with better audio cleaning"""
    
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
                
                # IMPROVED: Better audio quality filtering
                if self.filter_bad_audio:
                    audio_quality = self._assess_audio_quality(audio_feat)
                    if audio_quality < 0.3:  # Filter out very bad audio
                        bad_audio_skipped += 1
                        continue
                
                visual_feat = self._clean_features(visual_feat)
                audio_feat = self._clean_audio_features(audio_feat)  # Special cleaning for audio
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
        
        print(f"Successfully created {len(samples)} valid samples out of {total_attempted} attempted")
        print(f"  - {skipped} skipped (missing/invalid features)")
        if self.filter_bad_audio:
            print(f"  - {bad_audio_skipped} skipped (poor audio quality)")
        return samples
    
    def _assess_audio_quality(self, audio_feat):
        """Assess audio feature quality"""
        # Check for too many extreme values or zeros
        inf_count = np.isinf(audio_feat).sum()
        zero_count = (audio_feat == 0).sum()
        extreme_count = (np.abs(audio_feat) > 100).sum()
        
        total = len(audio_feat)
        quality_score = 1.0 - (inf_count + extreme_count) / total - 0.5 * (zero_count / total)
        return max(0.0, quality_score)
    
    def _clean_audio_features(self, features: np.ndarray) -> np.ndarray:
        """Special cleaning for audio features"""
        # More aggressive cleaning for audio
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        # Replace extreme values more conservatively
        features = np.clip(features, -10, 10)  # Tighter clipping for audio
        # Normalize extreme outliers
        if np.std(features) > 5:
            features = features / (np.std(features) + 1e-8)
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

# Continue with TransformedSubset and training code...
# (Copy the rest from train_mosei_only.py but use OptimizedThreeModalityModel)

print("Optimized three-modality model loaded!")
print("Key improvements:")
print("  1. Hierarchical attention (pairwise + global)")
print("  2. Better audio feature cleaning and quality filtering")
print("  3. Gated fusion mechanism")
print("  4. Deeper encoders")
print("  5. Better hyperparameters")




