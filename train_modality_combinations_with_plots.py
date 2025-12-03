"""
Test different modality combinations on CMU-MOSI with feature adapters.
Includes training/validation metrics tracking and plotting.

Usage:
    python train_modality_combinations_with_plots.py --combination "audio+text"
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import h5py
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import librosa
import cv2
import mediapipe as mp
from tqdm import tqdm
import json
import sys
from scipy.stats import pearsonr
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse

# Import original model
sys.path.append(str(Path(__file__).parent))
from train_mosei_only import RegularizedMultimodalModel, ImprovedCorrelationLoss

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import shared classes
from train_mosei_test_mosi_with_adapters import (
    FeatureAdapter, MOSEIDataset, MOSIDataset,
    train_adapters, fine_tune_end_to_end
)


class ModalityCombinationModel(nn.Module):
    """Multimodal model that supports different modality combinations - matches RegularizedMultimodalModel architecture"""
    
    def __init__(self, visual_dim: int, audio_dim: int, text_dim: int,
                 hidden_dim: int = 192, embed_dim: int = 96, dropout: float = 0.7,
                 use_visual: bool = True, use_audio: bool = True, use_text: bool = True,
                 num_layers: int = 2):
        super().__init__()
        
        self.use_visual = use_visual
        self.use_audio = use_audio
        self.use_text = use_text
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        
        # Modality encoders (matching RegularizedMultimodalModel structure)
        if self.use_visual:
            self.visual_encoder = self._create_encoder(visual_dim, hidden_dim, embed_dim, num_layers, dropout)
        
        if self.use_audio:
            self.audio_encoder = self._create_encoder(audio_dim, hidden_dim, embed_dim, num_layers, dropout)
        
        if self.use_text:
            self.text_encoder = self._create_encoder(text_dim, hidden_dim, embed_dim, num_layers, dropout)
        
        # Cross-modal attention (matching RegularizedMultimodalModel: num_heads=4, dropout=min(dropout+0.1, 0.8))
        num_modalities = sum([use_visual, use_audio, use_text])
        if num_modalities > 1:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=embed_dim, 
                num_heads=4,  # Match original
                dropout=min(dropout + 0.1, 0.8),  # Match original
                batch_first=True
            )
        
        # Fusion layers (matching RegularizedMultimodalModel: BatchNorm1d, not LayerNorm)
        self.fusion_layers = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Match original (not LayerNorm)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),  # Match original (not LayerNorm)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def _create_encoder(self, input_dim, hidden_dim, embed_dim, num_layers, dropout):
        """Create encoder with batch normalization - matching RegularizedMultimodalModel"""
        layers = []
        current_dim = input_dim
        
        layers.extend([
            nn.Linear(current_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        current_dim = hidden_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        layers.append(nn.Linear(hidden_dim, embed_dim))
        layers.append(nn.BatchNorm1d(embed_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, visual=None, audio=None, text=None):
        embeddings = []
        
        if self.use_visual and visual is not None:
            v_enc = self.visual_encoder(visual)
            if v_enc.dim() == 1:
                v_enc = v_enc.unsqueeze(0)
            embeddings.append(v_enc)
        
        if self.use_audio and audio is not None:
            a_enc = self.audio_encoder(audio)
            if a_enc.dim() == 1:
                a_enc = a_enc.unsqueeze(0)
            embeddings.append(a_enc)
        
        if self.use_text and text is not None:
            t_enc = self.text_encoder(text)
            if t_enc.dim() == 1:
                t_enc = t_enc.unsqueeze(0)
            embeddings.append(t_enc)
        
        if len(embeddings) == 0:
            raise ValueError("At least one modality must be enabled")
        
        # Stack for attention (matching RegularizedMultimodalModel)
        features = torch.stack(embeddings, dim=1)  # [batch, num_modalities, embed_dim]
        
        # Cross-modal attention
        if len(embeddings) > 1:
            attended_features, _ = self.cross_attention(features, features, features)
        else:
            attended_features = features
        
        # Concatenate embeddings (matching RegularizedMultimodalModel approach)
        concat_features = torch.cat(embeddings, dim=-1)  # [batch, embed_dim * num_modalities]
        
        # Fusion (matching RegularizedMultimodalModel)
        output = self.fusion_layers(concat_features)
        
        return output.squeeze(-1)


def train_on_mosei_combination(mosei_dir: str, epochs: int = 100, batch_size: int = 32,
                               use_visual: bool = True, use_audio: bool = True, 
                               use_text: bool = True, model_path: str = None):
    """Train model on CMU-MOSEI with specified modality combination"""
    
    print("="*80)
    print(f"Training on CMU-MOSEI: Visual={use_visual}, Audio={use_audio}, Text={use_text}")
    print("="*80)
    
    # Load dataset
    dataset = MOSEIDataset(mosei_dir)
    
    if len(dataset) == 0:
        print("ERROR: No MOSEI samples loaded!")
        return None
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = ModalityCombinationModel(
        visual_dim=713 if use_visual else 0,
        audio_dim=74 if use_audio else 0,
        text_dim=300 if use_text else 0,
        hidden_dim=192, embed_dim=96, dropout=0.7,
        use_visual=use_visual, use_audio=use_audio, use_text=use_text
    ).to(device)
    
    criterion = ImprovedCorrelationLoss(alpha=0.3, beta=0.7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=0.04)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_corr = -1.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        train_preds, train_labels = [], []
        
        for batch in train_loader:
            visual = batch['visual'].to(device).float() if use_visual else None
            audio = batch['audio'].to(device).float() if use_audio else None
            text = batch['text'].to(device).float() if use_text else None
            sentiment = batch['sentiment'].to(device).float()
            
            pred = model(visual=visual, audio=audio, text=text).squeeze()
            loss, _ = criterion(pred, sentiment)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            train_preds.extend(pred.detach().cpu().numpy())
            train_labels.extend(sentiment.cpu().numpy())
        
        # Validation
        model.eval()
        val_losses = []
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                visual = batch['visual'].to(device).float() if use_visual else None
                audio = batch['audio'].to(device).float() if use_audio else None
                text = batch['text'].to(device).float() if use_text else None
                sentiment = batch['sentiment'].to(device).float()
                
                pred = model(visual=visual, audio=audio, text=text).squeeze()
                loss, _ = criterion(pred, sentiment)
                
                val_losses.append(loss.item())
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(sentiment.cpu().numpy())
        
        # Metrics
        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_mae = np.mean(np.abs(train_preds - train_labels))
        val_mae = np.mean(np.abs(val_preds - val_labels))
        
        # Check for constant predictions (collapse)
        train_pred_std = np.std(train_preds)
        val_pred_std = np.std(val_preds)
        
        # Compute correlation with safety checks
        if len(train_preds) >= 2 and train_pred_std > 1e-6:
            try:
                train_corr, _ = pearsonr(train_preds, train_labels)
                if np.isnan(train_corr):
                    train_corr = 0.0
            except:
                train_corr = 0.0
        else:
            train_corr = 0.0
        
        if len(val_preds) >= 2 and val_pred_std > 1e-6:
            try:
                val_corr, _ = pearsonr(val_preds, val_labels)
                if np.isnan(val_corr):
                    val_corr = 0.0
            except:
                val_corr = 0.0
        else:
            val_corr = 0.0
        
        # Early stopping if model collapses
        if train_pred_std < 1e-6 or val_pred_std < 1e-6:
            print(f"\n⚠️  WARNING: Model collapsed at epoch {epoch+1}! Predictions are constant.")
            print(f"   Train pred std: {train_pred_std:.6f}, Val pred std: {val_pred_std:.6f}")
            print(f"   Stopping training early. Best validation correlation: {best_val_corr:.4f}")
            break
        
        # Early stopping if correlation becomes NaN or drops significantly
        if np.isnan(val_corr) or (best_val_corr > 0.5 and val_corr < best_val_corr - 0.2):
            print(f"\n⚠️  WARNING: Validation correlation dropped significantly at epoch {epoch+1}!")
            print(f"   Current: {val_corr:.4f}, Best: {best_val_corr:.4f}")
            print(f"   Stopping training early.")
            break
        
        scheduler.step(val_corr if not np.isnan(val_corr) else 0.0)
        
        if val_corr > best_val_corr and not np.isnan(val_corr):
            best_val_corr = val_corr
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Train Corr={train_corr:.4f}, Train MAE={train_mae:.4f}")
            print(f"              Val Loss={val_loss:.4f}, Val Corr={val_corr:.4f}, Val MAE={val_mae:.4f}")
            print(f"              Train pred std: {train_pred_std:.6f}, Val pred std: {val_pred_std:.6f}")
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model with validation correlation: {best_val_corr:.4f}")
    
    if model_path:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    return model


def fine_tune_with_tracking(model, adapters, mosi_dataset, epochs: int = 20, 
                           batch_size: int = 16, use_visual: bool = True,
                           use_audio: bool = True, use_text: bool = True):
    """Fine-tune with full metric tracking"""
    
    if len(adapters) == 5:
        visual_adapter, audio_adapter, text_adapter, mosei_stats, test_dataset = adapters
        normalize_features = True
    elif len(adapters) == 4:
        visual_adapter, audio_adapter, text_adapter, mosei_stats = adapters
        normalize_features = True
        test_dataset = None
    else:
        visual_adapter, audio_adapter, text_adapter = adapters
        mosei_stats = None
        normalize_features = False
        test_dataset = None
    
    # Split MOSI
    total_size = len(mosi_dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset_split, test_dataset_split = torch.utils.data.random_split(
        mosi_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    if test_dataset is None:
        test_dataset = test_dataset_split
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset_split, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Set all to train mode
    model.train()
    if use_visual:
        visual_adapter.train()
    if use_audio:
        audio_adapter.train()
    if use_text:
        text_adapter.train()
    
    # Create optimizer
    all_params = list(model.parameters())
    if use_visual:
        all_params += list(visual_adapter.parameters())
    if use_audio:
        all_params += list(audio_adapter.parameters())
    if use_text:
        all_params += list(text_adapter.parameters())
    
    optimizer = torch.optim.Adam(all_params, lr=0.00005, weight_decay=1e-5)  # Lower LR to prevent overfitting
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = ImprovedCorrelationLoss(alpha=0.3, beta=0.7)
    
    # Track metrics
    history = {
        'train_loss': [], 'train_mae': [], 'train_corr': [],
        'val_loss': [], 'val_mae': [], 'val_corr': []
    }
    
    # Track best model
    best_val_corr = -1.0
    best_model_state = None
    best_visual_adapter_state = None
    best_audio_adapter_state = None
    best_text_adapter_state = None
    
    print(f"Fine-tuning for {epochs} epochs...")
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        if use_visual:
            visual_adapter.train()
        if use_audio:
            audio_adapter.train()
        if use_text:
            text_adapter.train()
        
        train_losses = []
        train_preds, train_labels = [], []
        
        for batch in tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch+1}/{epochs}"):
            # Adapt features
            v_adapted = None
            a_adapted = None
            t_adapted = None
            
            if use_visual:
                v_adapted = visual_adapter(batch['visual'].to(device).float())
                if normalize_features and mosei_stats is not None:
                    v_adapted = (v_adapted - mosei_stats['visual']['mean']) / mosei_stats['visual']['std']
                    v_adapted = torch.clamp(v_adapted, -10, 10)
            
            if use_audio:
                a_adapted = audio_adapter(batch['audio'].to(device).float())
                if normalize_features and mosei_stats is not None:
                    a_adapted = (a_adapted - mosei_stats['audio']['mean']) / mosei_stats['audio']['std']
                    a_adapted = torch.clamp(a_adapted, -10, 10)
            
            if use_text:
                t_adapted = text_adapter(batch['text'].to(device).float())
                if normalize_features and mosei_stats is not None:
                    t_adapted = (t_adapted - mosei_stats['text']['mean']) / mosei_stats['text']['std']
                    t_adapted = torch.clamp(t_adapted, -10, 10)
            
            # Predict
            pred = model(visual=v_adapted, audio=a_adapted, text=t_adapted).squeeze()
            sentiment = batch['sentiment'].to(device).float()
            
            # Loss
            loss, _ = criterion(pred, sentiment)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            train_preds.extend(pred.detach().cpu().numpy())
            train_labels.extend(sentiment.cpu().numpy())
        
        # Validation
        model.eval()
        if use_visual:
            visual_adapter.eval()
        if use_audio:
            audio_adapter.eval()
        if use_text:
            text_adapter.eval()
        
        val_losses = []
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                v_adapted = None
                a_adapted = None
                t_adapted = None
                
                if use_visual:
                    v_adapted = visual_adapter(batch['visual'].to(device).float())
                    if normalize_features and mosei_stats is not None:
                        v_adapted = (v_adapted - mosei_stats['visual']['mean']) / mosei_stats['visual']['std']
                        v_adapted = torch.clamp(v_adapted, -10, 10)
                
                if use_audio:
                    a_adapted = audio_adapter(batch['audio'].to(device).float())
                    if normalize_features and mosei_stats is not None:
                        a_adapted = (a_adapted - mosei_stats['audio']['mean']) / mosei_stats['audio']['std']
                        a_adapted = torch.clamp(a_adapted, -10, 10)
                
                if use_text:
                    t_adapted = text_adapter(batch['text'].to(device).float())
                    if normalize_features and mosei_stats is not None:
                        t_adapted = (t_adapted - mosei_stats['text']['mean']) / mosei_stats['text']['std']
                        t_adapted = torch.clamp(t_adapted, -10, 10)
                
                pred = model(visual=v_adapted, audio=a_adapted, text=t_adapted).squeeze()
                sentiment = batch['sentiment'].to(device).float()
                loss, _ = criterion(pred, sentiment)
                
                val_losses.append(loss.item())
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(sentiment.cpu().numpy())
        
        # Compute metrics
        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_mae = np.mean(np.abs(train_preds - train_labels))
        val_mae = np.mean(np.abs(val_preds - val_labels))
        
        if len(train_preds) >= 2:
            train_corr, _ = pearsonr(train_preds, train_labels)
        else:
            train_corr = 0.0
        
        # Check for constant predictions
        val_pred_std = np.std(val_preds)
        
        if len(val_preds) >= 2 and val_pred_std > 1e-6:
            try:
                val_corr, _ = pearsonr(val_preds, val_labels)
                if np.isnan(val_corr):
                    val_corr = 0.0
            except:
                val_corr = 0.0
        else:
            val_corr = 0.0
        
        # Learning rate scheduling
        scheduler.step(val_corr if not np.isnan(val_corr) else 0.0)
        
        # Early stopping if overfitting (validation correlation drops significantly)
        if best_val_corr > 0.3 and val_corr < best_val_corr - 0.15:
            print(f"\n⚠️  Early stopping: Validation correlation dropped from {best_val_corr:.4f} to {val_corr:.4f}")
            print(f"   Stopping fine-tuning to prevent overfitting. Using best model.")
            break
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['train_corr'].append(train_corr)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_corr'].append(val_corr)
        
        # Save best model
        if val_corr > best_val_corr and not np.isnan(val_corr):
            best_val_corr = val_corr
            best_model_state = model.state_dict().copy()
            if use_visual:
                best_visual_adapter_state = visual_adapter.state_dict().copy()
            if use_audio:
                best_audio_adapter_state = audio_adapter.state_dict().copy()
            if use_text:
                best_text_adapter_state = text_adapter.state_dict().copy()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Corr={train_corr:.4f}, Train MAE={train_mae:.4f}")
            print(f"          Val Loss={val_loss:.4f}, Val Corr={val_corr:.4f}, Val MAE={val_mae:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if use_visual and best_visual_adapter_state is not None:
            visual_adapter.load_state_dict(best_visual_adapter_state)
        if use_audio and best_audio_adapter_state is not None:
            audio_adapter.load_state_dict(best_audio_adapter_state)
        if use_text and best_text_adapter_state is not None:
            text_adapter.load_state_dict(best_text_adapter_state)
        print(f"Fine-tuning complete! Loaded best model with validation correlation: {best_val_corr:.4f}")
    else:
        print(f"Fine-tuning complete! Best validation correlation: {max(history['val_corr']):.4f}")
    
    return (visual_adapter, audio_adapter, text_adapter, mosei_stats, test_dataset), history


def plot_training_curves(history: Dict, combination: str, save_path: str):
    """Plot training and validation curves"""
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Curves: {combination.upper()}', fontsize=16, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].plot(epochs, history['train_mae'], 'b-', label='Train MAE', linewidth=2)
    axes[0, 1].plot(epochs, history['val_mae'], 'r-', label='Val MAE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('MAE', fontsize=12)
    axes[0, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Correlation
    axes[0, 2].plot(epochs, history['train_corr'], 'b-', label='Train Correlation', linewidth=2)
    axes[0, 2].plot(epochs, history['val_corr'], 'r-', label='Val Correlation', linewidth=2)
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('Pearson Correlation', fontsize=12)
    axes[0, 2].set_title('Correlation', fontsize=14, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([-0.2, 1.0])
    
    # Combined Loss plot
    axes[1, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2, alpha=0.7)
    axes[1, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('Loss Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined MAE plot
    axes[1, 1].plot(epochs, history['train_mae'], 'b-', label='Train', linewidth=2, alpha=0.7)
    axes[1, 1].plot(epochs, history['val_mae'], 'r-', label='Val', linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('MAE', fontsize=12)
    axes[1, 1].set_title('MAE Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Combined Correlation plot
    axes[1, 2].plot(epochs, history['train_corr'], 'b-', label='Train', linewidth=2, alpha=0.7)
    axes[1, 2].plot(epochs, history['val_corr'], 'r-', label='Val', linewidth=2, alpha=0.7)
    axes[1, 2].set_xlabel('Epoch', fontsize=12)
    axes[1, 2].set_ylabel('Correlation', fontsize=12)
    axes[1, 2].set_title('Correlation Comparison', fontsize=14, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim([-0.2, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def test_on_mosi_combination(model, adapters, mosi_dataset, test_dataset,
                            use_visual: bool = True, use_audio: bool = True,
                            use_text: bool = True):
    """Test on held-out test set"""
    
    if len(adapters) == 5:
        visual_adapter, audio_adapter, text_adapter, mosei_stats, _ = adapters
        normalize_features = True
    elif len(adapters) == 4:
        visual_adapter, audio_adapter, text_adapter, mosei_stats = adapters
        normalize_features = True
    else:
        visual_adapter, audio_adapter, text_adapter = adapters
        mosei_stats = None
        normalize_features = False
    
    model.eval()
    if use_visual:
        visual_adapter.eval()
    if use_audio:
        audio_adapter.eval()
    if use_text:
        text_adapter.eval()
    
    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    predictions, labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            v_adapted = None
            a_adapted = None
            t_adapted = None
            
            if use_visual:
                v_adapted = visual_adapter(batch['visual'].to(device).float())
                if normalize_features and mosei_stats is not None:
                    v_adapted = (v_adapted - mosei_stats['visual']['mean']) / mosei_stats['visual']['std']
                    v_adapted = torch.clamp(v_adapted, -10, 10)
            
            if use_audio:
                a_adapted = audio_adapter(batch['audio'].to(device).float())
                if normalize_features and mosei_stats is not None:
                    a_adapted = (a_adapted - mosei_stats['audio']['mean']) / mosei_stats['audio']['std']
                    a_adapted = torch.clamp(a_adapted, -10, 10)
            
            if use_text:
                t_adapted = text_adapter(batch['text'].to(device).float())
                if normalize_features and mosei_stats is not None:
                    t_adapted = (t_adapted - mosei_stats['text']['mean']) / mosei_stats['text']['std']
                    t_adapted = torch.clamp(t_adapted, -10, 10)
            
            pred = model(visual=v_adapted, audio=a_adapted, text=t_adapted).squeeze()
            predictions.extend(pred.cpu().numpy())
            labels.extend(batch['sentiment'].numpy())
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    
    if len(predictions) >= 2:
        correlation, _ = pearsonr(predictions, labels)
    else:
        correlation = 0.0
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'correlation': float(correlation),
        'num_samples': len(predictions)
    }


def main():
    parser = argparse.ArgumentParser(description='Test modality combinations with plots')
    parser.add_argument('--mosei_dir', type=str, default="C:/Users/PC/Downloads/CMU-MOSEI")
    parser.add_argument('--mosi_dir', type=str, default="C:/Users/PC/Downloads/CMU-MOSI Dataset")
    parser.add_argument('--combination', type=str, default='all',
                       choices=['all', 'audio+text', 'audio+visual', 'text+visual', 'audio', 'visual', 'text'],
                       help='Modality combination to test')
    parser.add_argument('--mosi_samples', type=int, default=93)
    parser.add_argument('--adapter_epochs', type=int, default=75)
    parser.add_argument('--fine_tune_epochs', type=int, default=20)
    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--skip_adapters', action='store_true')
    
    args = parser.parse_args()
    
    # Parse combination
    use_visual = args.combination in ['all', 'audio+visual', 'text+visual', 'visual']
    use_audio = args.combination in ['all', 'audio+text', 'audio+visual', 'audio']
    use_text = args.combination in ['all', 'audio+text', 'text+visual', 'text']
    
    print("="*80)
    print(f"Testing Modality Combination: {args.combination.upper()}")
    print(f"Visual={use_visual}, Audio={use_audio}, Text={use_text}")
    print("="*80)
    
    # Step 1: Train on MOSEI
    model_path = f'model_{args.combination}.pth'
    if not args.skip_training:
        model = train_on_mosei_combination(
            args.mosei_dir, epochs=100,
            use_visual=use_visual, use_audio=use_audio, use_text=use_text,
            model_path=model_path
        )
    else:
        print("Loading existing model...")
        model = ModalityCombinationModel(
            visual_dim=713 if use_visual else 0,
            audio_dim=74 if use_audio else 0,
            text_dim=300 if use_text else 0,
            use_visual=use_visual, use_audio=use_audio, use_text=use_text
        ).to(device)
        model.load_state_dict(torch.load(model_path))
    
    # Step 2: Load MOSI
    mosi_dataset = MOSIDataset(args.mosi_dir, max_samples=args.mosi_samples)
    
    # Step 3: Train adapters
    if not args.skip_adapters:
        adapters_and_stats = train_adapters(
            args.mosei_dir, mosi_dataset, epochs=args.adapter_epochs
        )
    else:
        print("Loading existing adapters...")
        visual_adapter = FeatureAdapter(65, 713).to(device)
        audio_adapter = FeatureAdapter(74, 74).to(device)
        text_adapter = FeatureAdapter(768, 300).to(device)
        visual_adapter.load_state_dict(torch.load('visual_adapter.pth'))
        audio_adapter.load_state_dict(torch.load('audio_adapter.pth'))
        text_adapter.load_state_dict(torch.load('text_adapter.pth'))
        adapters_and_stats = (visual_adapter, audio_adapter, text_adapter, None)
    
    # Step 4: Fine-tune with tracking
    adapters_and_stats, history = fine_tune_with_tracking(
        model, adapters_and_stats, mosi_dataset, epochs=args.fine_tune_epochs,
        use_visual=use_visual, use_audio=use_audio, use_text=use_text
    )
    
    # Extract test_dataset from adapters
    if len(adapters_and_stats) >= 5:
        test_dataset = adapters_and_stats[4]
    else:
        test_dataset = None
    
    # Step 5: Plot curves
    # Save to the same folder as the script
    script_dir = Path(__file__).parent
    plot_path = script_dir / f'training_curves_{args.combination}.png'
    plot_training_curves(history, args.combination, str(plot_path))
    
    # Step 6: Test
    results = test_on_mosi_combination(
        model, adapters_and_stats, mosi_dataset, test_dataset,
        use_visual=use_visual, use_audio=use_audio, use_text=use_text
    )
    
    print("\n" + "="*80)
    print(f"Test Results ({args.combination.upper()}):")
    print("="*80)
    print(f"Correlation: {results['correlation']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    print(f"MSE: {results['mse']:.4f}")
    print(f"Samples: {results['num_samples']}")
    print("="*80)
    
    # Save results (convert numpy types to native Python types for JSON)
    script_dir = Path(__file__).parent
    results['history'] = {
        'train_loss': [float(x) for x in history['train_loss']],
        'train_mae': [float(x) for x in history['train_mae']],
        'train_corr': [float(x) if not np.isnan(x) else None for x in history['train_corr']],
        'val_loss': [float(x) for x in history['val_loss']],
        'val_mae': [float(x) for x in history['val_mae']],
        'val_corr': [float(x) if not np.isnan(x) else None for x in history['val_corr']]
    }
    # Convert results to native Python types
    results['mse'] = float(results['mse'])
    results['mae'] = float(results['mae'])
    results['correlation'] = float(results['correlation']) if not np.isnan(results['correlation']) else None
    results['num_samples'] = int(results['num_samples'])
    
    results_path = script_dir / f'results_{args.combination}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    print(f"Training curves saved to {plot_path}")


if __name__ == "__main__":
    main()

