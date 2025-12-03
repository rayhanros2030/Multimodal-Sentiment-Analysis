"""
Test ALL modality combinations on CMU-MOSI with feature adapters.
Uses the same architecture as the original script that achieved 0.64 correlation.
Generates graphs and results for each combination.

Usage:
    python test_all_modality_combinations.py --mosei_dir "path/to/MOSEI" --mosi_dir "path/to/MOSI"
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import sys
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# Import shared classes
sys.path.append(str(Path(__file__).parent))
from train_mosei_only import RegularizedMultimodalModel, ImprovedCorrelationLoss
from train_mosei_test_mosi_with_adapters import (
    FeatureAdapter, MOSEIDataset, MOSIDataset,
    train_adapters, fine_tune_end_to_end
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class ModalityCombinationModel(nn.Module):
    """Multimodal model that supports different modality combinations - matches RegularizedMultimodalModel architecture exactly"""
    
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
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    # Use RegularizedMultimodalModel directly for "all" combination (exact same as 0.64 correlation)
    # Use ModalityCombinationModel for other combinations (matches architecture)
    if use_visual and use_audio and use_text:
        # EXACT same model that achieved 0.64 correlation
        model = RegularizedMultimodalModel(
            visual_dim=713, audio_dim=74, text_dim=300,
            hidden_dim=192, embed_dim=96, dropout=0.7
        ).to(device)
    else:
        # Flexible model for other combinations (same architecture)
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
    patience = 0
    max_patience = 25
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        train_preds, train_labels = [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            visual = batch['visual'].to(device).float() if use_visual else None
            audio = batch['audio'].to(device).float() if use_audio else None
            text = batch['text'].to(device).float() if use_text else None
            sentiment = batch['sentiment'].to(device).float()
            
            # Handle RegularizedMultimodalModel vs ModalityCombinationModel forward signature
            if use_visual and use_audio and use_text:
                # RegularizedMultimodalModel expects (visual, audio, text) positional args
                pred = model(visual, audio, text).squeeze()
            else:
                # ModalityCombinationModel expects keyword args with None for disabled modalities
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
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                visual = batch['visual'].to(device).float() if use_visual else None
                audio = batch['audio'].to(device).float() if use_audio else None
                text = batch['text'].to(device).float() if use_text else None
                sentiment = batch['sentiment'].to(device).float()
                
                # Handle RegularizedMultimodalModel vs ModalityCombinationModel forward signature
                if use_visual and use_audio and use_text:
                    # RegularizedMultimodalModel expects (visual, audio, text) positional args
                    pred = model(visual, audio, text).squeeze()
                else:
                    # ModalityCombinationModel expects keyword args with None for disabled modalities
                    pred = model(visual=visual, audio=audio, text=text).squeeze()
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(sentiment.cpu().numpy())
        
        # Metrics
        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean([(p - l)**2 for p, l in zip(val_preds, val_labels)])
        train_mae = np.mean(np.abs(train_preds - train_labels))
        val_mae = np.mean(np.abs(val_preds - val_labels))
        
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
        
        scheduler.step(val_corr if not np.isnan(val_corr) else 0.0)
        
        # Early stopping if model collapses
        if val_pred_std < 1e-6:
            print(f"\n⚠️  Model collapse detected (constant predictions). Stopping early.")
            break
        
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_model_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Corr={val_corr:.4f}, Val MAE={val_mae:.4f}")
        
        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation correlation: {best_val_corr:.4f}")
    
    # Save model
    if model_path:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    return model


def fine_tune_with_tracking(model, visual_adapter, audio_adapter, text_adapter,
                            train_dataset, val_dataset, test_dataset,
                            use_visual: bool, use_audio: bool, use_text: bool,
                            epochs: int = 20, mosei_stats: Dict = None):
    """Fine-tune model with adapters, tracking metrics"""
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Collect all parameters
    all_params = list(model.parameters())
    if use_visual:
        all_params += list(visual_adapter.parameters())
    if use_audio:
        all_params += list(audio_adapter.parameters())
    if use_text:
        all_params += list(text_adapter.parameters())
    
    optimizer = torch.optim.Adam(all_params, lr=0.00001, weight_decay=1e-5)  # Lower LR to prevent overfitting
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
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
    
    normalize_features = mosei_stats is not None
    
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
        
        for batch in train_loader:
            # Adapt features
            v_adapted = None
            a_adapted = None
            t_adapted = None
            
            if use_visual:
                v_adapted = visual_adapter(batch['visual'].to(device).float())
                if normalize_features:
                    v_adapted = (v_adapted - torch.tensor(mosei_stats['visual']['mean'], device=device)) / torch.tensor(mosei_stats['visual']['std'], device=device)
                    v_adapted = torch.clamp(v_adapted, -10, 10)
            
            if use_audio:
                a_adapted = audio_adapter(batch['audio'].to(device).float())
                if normalize_features:
                    a_adapted = (a_adapted - torch.tensor(mosei_stats['audio']['mean'], device=device)) / torch.tensor(mosei_stats['audio']['std'], device=device)
                    a_adapted = torch.clamp(a_adapted, -10, 10)
            
            if use_text:
                t_adapted = text_adapter(batch['text'].to(device).float())
                if normalize_features:
                    t_adapted = (t_adapted - torch.tensor(mosei_stats['text']['mean'], device=device)) / torch.tensor(mosei_stats['text']['std'], device=device)
                    t_adapted = torch.clamp(t_adapted, -10, 10)
            
            # Predict - handle RegularizedMultimodalModel vs ModalityCombinationModel
            if use_visual and use_audio and use_text:
                # RegularizedMultimodalModel expects (visual, audio, text) positional args
                pred = model(v_adapted, a_adapted, t_adapted).squeeze()
            else:
                # ModalityCombinationModel expects keyword args with None for disabled modalities
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
                # Adapt features
                v_adapted = None
                a_adapted = None
                t_adapted = None
                
                if use_visual:
                    v_adapted = visual_adapter(batch['visual'].to(device).float())
                    if normalize_features:
                        v_adapted = (v_adapted - torch.tensor(mosei_stats['visual']['mean'], device=device)) / torch.tensor(mosei_stats['visual']['std'], device=device)
                        v_adapted = torch.clamp(v_adapted, -10, 10)
                
                if use_audio:
                    a_adapted = audio_adapter(batch['audio'].to(device).float())
                    if normalize_features:
                        a_adapted = (a_adapted - torch.tensor(mosei_stats['audio']['mean'], device=device)) / torch.tensor(mosei_stats['audio']['std'], device=device)
                        a_adapted = torch.clamp(a_adapted, -10, 10)
                
                if use_text:
                    t_adapted = text_adapter(batch['text'].to(device).float())
                    if normalize_features:
                        t_adapted = (t_adapted - torch.tensor(mosei_stats['text']['mean'], device=device)) / torch.tensor(mosei_stats['text']['std'], device=device)
                        t_adapted = torch.clamp(t_adapted, -10, 10)
                
                # Predict - handle RegularizedMultimodalModel vs ModalityCombinationModel
                if use_visual and use_audio and use_text:
                    # RegularizedMultimodalModel expects (visual, audio, text) positional args
                    pred = model(v_adapted, a_adapted, t_adapted).squeeze()
                else:
                    # ModalityCombinationModel expects keyword args with None for disabled modalities
                    pred = model(visual=v_adapted, audio=a_adapted, text=t_adapted).squeeze()
                sentiment = batch['sentiment'].to(device).float()
                
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
        
        scheduler.step(val_corr if not np.isnan(val_corr) and val_corr > -1 else -1.0)
        
        # Early stopping if overfitting or consistently negative
        if best_val_corr > 0.1 and val_corr < best_val_corr - 0.15:
            print(f"\n⚠️  Early stopping: Validation correlation dropped from {best_val_corr:.4f} to {val_corr:.4f}")
            break
        
        # Also stop if validation correlation is consistently very negative
        if epoch >= 5 and val_corr < -0.2 and train_corr > 0.1:
            print(f"\n⚠️  Early stopping: Validation correlation is negative ({val_corr:.4f}) while train is positive. Possible overfitting.")
            break
        
        # Store metrics
        history['train_loss'].append(float(train_loss))
        history['train_mae'].append(float(train_mae))
        history['train_corr'].append(float(train_corr) if not np.isnan(train_corr) else None)
        history['val_loss'].append(float(val_loss))
        history['val_mae'].append(float(val_mae))
        history['val_corr'].append(float(val_corr) if not np.isnan(val_corr) else None)
        
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
        print(f"Fine-tuning complete! Best validation correlation: {max([x for x in history['val_corr'] if x is not None]):.4f}")
    
    return (visual_adapter, audio_adapter, text_adapter, mosei_stats, test_dataset), history


def test_on_mosi_combination(model, visual_adapter, audio_adapter, text_adapter,
                             test_dataset, use_visual: bool, use_audio: bool, use_text: bool,
                             mosei_stats: Dict = None):
    """Test model on CMU-MOSI with adapted features"""
    
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    model.eval()
    if use_visual:
        visual_adapter.eval()
    if use_audio:
        audio_adapter.eval()
    if use_text:
        text_adapter.eval()
    
    predictions = []
    labels = []
    
    normalize_features = mosei_stats is not None
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Adapt features
            v_adapted = None
            a_adapted = None
            t_adapted = None
            
            if use_visual:
                v_adapted = visual_adapter(batch['visual'].to(device).float())
                if normalize_features:
                    v_adapted = (v_adapted - torch.tensor(mosei_stats['visual']['mean'], device=device)) / torch.tensor(mosei_stats['visual']['std'], device=device)
                    v_adapted = torch.clamp(v_adapted, -10, 10)
            
            if use_audio:
                a_adapted = audio_adapter(batch['audio'].to(device).float())
                if normalize_features:
                    a_adapted = (a_adapted - torch.tensor(mosei_stats['audio']['mean'], device=device)) / torch.tensor(mosei_stats['audio']['std'], device=device)
                    a_adapted = torch.clamp(a_adapted, -10, 10)
            
            if use_text:
                t_adapted = text_adapter(batch['text'].to(device).float())
                if normalize_features:
                    t_adapted = (t_adapted - torch.tensor(mosei_stats['text']['mean'], device=device)) / torch.tensor(mosei_stats['text']['std'], device=device)
                    t_adapted = torch.clamp(t_adapted, -10, 10)
            
            # Predict - handle RegularizedMultimodalModel vs ModalityCombinationModel
            if use_visual and use_audio and use_text:
                # RegularizedMultimodalModel expects (visual, audio, text) positional args
                pred = model(v_adapted, a_adapted, t_adapted).squeeze()
            else:
                # ModalityCombinationModel expects keyword args with None for disabled modalities
                pred = model(visual=v_adapted, audio=a_adapted, text=t_adapted).squeeze()
            
            sentiment = batch['sentiment'].to(device).float()
            
            predictions.extend(pred.cpu().numpy())
            labels.extend(sentiment.cpu().numpy())
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Filter out zero labels
    mask = labels != 0
    if np.sum(mask) == 0:
        print("WARNING: All labels are zero. Cannot compute metrics.")
        return {'correlation': None, 'mae': None, 'mse': None, 'samples': len(labels)}
    
    predictions = predictions[mask]
    labels = labels[mask]
    
    if len(predictions) < 2:
        print("WARNING: Less than 2 valid samples. Cannot compute correlation.")
        return {'correlation': None, 'mae': None, 'mse': None, 'samples': len(labels)}
    
    # Metrics
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    
    try:
        correlation, _ = pearsonr(predictions, labels)
        if np.isnan(correlation):
            correlation = None
    except:
        correlation = None
    
    return {
        'correlation': float(correlation) if correlation is not None else None,
        'mae': float(mae),
        'mse': float(mse),
        'samples': int(np.sum(mask))
    }


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
    train_corr = [x if x is not None else 0 for x in history['train_corr']]
    val_corr = [x if x is not None else 0 for x in history['val_corr']]
    axes[0, 2].plot(epochs, train_corr, 'b-', label='Train Correlation', linewidth=2)
    axes[0, 2].plot(epochs, val_corr, 'r-', label='Val Correlation', linewidth=2)
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('Pearson Correlation', fontsize=12)
    axes[0, 2].set_title('Correlation', fontsize=14, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Loss comparison
    axes[1, 0].plot(epochs, history['train_loss'], 'b-', alpha=0.5, label='Train', linewidth=1)
    axes[1, 0].plot(epochs, history['val_loss'], 'r-', alpha=0.5, label='Val', linewidth=1)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('Loss Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # MAE comparison
    axes[1, 1].plot(epochs, history['train_mae'], 'b-', alpha=0.5, label='Train', linewidth=1)
    axes[1, 1].plot(epochs, history['val_mae'], 'r-', alpha=0.5, label='Val', linewidth=1)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('MAE', fontsize=12)
    axes[1, 1].set_title('MAE Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Correlation comparison
    axes[1, 2].plot(epochs, train_corr, 'b-', alpha=0.5, label='Train', linewidth=1)
    axes[1, 2].plot(epochs, val_corr, 'r-', alpha=0.5, label='Val', linewidth=1)
    axes[1, 2].set_xlabel('Epoch', fontsize=12)
    axes[1, 2].set_ylabel('Correlation', fontsize=12)
    axes[1, 2].set_title('Correlation Comparison', fontsize=14, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test all modality combinations')
    parser.add_argument('--mosei_dir', type=str, required=True, help='Path to CMU-MOSEI dataset')
    parser.add_argument('--mosi_dir', type=str, required=True, help='Path to CMU-MOSI dataset')
    parser.add_argument('--mosi_samples', type=int, default=93, help='Number of MOSI samples to use')
    parser.add_argument('--train_epochs', type=int, default=100, help='Epochs for MOSEI training')
    parser.add_argument('--adapter_epochs', type=int, default=75, help='Epochs for adapter training')
    parser.add_argument('--fine_tune_epochs', type=int, default=20, help='Epochs for fine-tuning')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define all modality combinations
    combinations = [
        ("visual", True, False, False),
        ("audio", False, True, False),
        ("text", False, False, True),
        ("visual+audio", True, True, False),
        ("visual+text", True, False, True),
        ("audio+text", False, True, True),
        ("all", True, True, True),
    ]
    
    all_results = {}
    
    print("\n" + "="*80)
    print("TESTING ALL MODALITY COMBINATIONS")
    print("="*80)
    print(f"Total combinations: {len(combinations)}")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")
    
    for i, (combination_name, use_visual, use_audio, use_text) in enumerate(combinations, 1):
        print(f"\n{'='*80}")
        print(f"COMBINATION {i}/{len(combinations)}: {combination_name.upper()}")
        print(f"{'='*80}\n")
        
        try:
            # Step 1: Train on MOSEI
            model_path = output_dir / f'model_{combination_name}.pth'
            model = train_on_mosei_combination(
                args.mosei_dir, epochs=args.train_epochs,
                use_visual=use_visual, use_audio=use_audio, use_text=use_text,
                model_path=str(model_path)
            )
            
            if model is None:
                print(f"❌ Failed to train model for {combination_name}")
                all_results[combination_name] = {'error': 'Model training failed'}
                continue
            
            # Step 2: Load MOSI dataset
            mosi_dataset = MOSIDataset(args.mosi_dir, max_samples=args.mosi_samples)
            
            if len(mosi_dataset) == 0:
                print(f"❌ No MOSI samples loaded for {combination_name}")
                all_results[combination_name] = {'error': 'No MOSI samples'}
                continue
            
            # Step 3: Train adapters
            print(f"\nTraining adapters for {combination_name}...")
            adapters_result = train_adapters(
                args.mosei_dir, mosi_dataset, epochs=args.adapter_epochs
            )
            
            visual_adapter, audio_adapter, text_adapter, mosei_stats = adapters_result
            
            # Step 4: Fine-tune with tracking
            print(f"\nFine-tuning for {combination_name}...")
            
            # Split MOSI for fine-tuning (60% train, 20% val, 20% test)
            train_size = int(0.6 * len(mosi_dataset))
            val_size = int(0.2 * len(mosi_dataset))
            test_size = len(mosi_dataset) - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                mosi_dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            fine_tune_result, history = fine_tune_with_tracking(
                model, visual_adapter, audio_adapter, text_adapter,
                train_dataset, val_dataset, test_dataset,
                use_visual, use_audio, use_text,
                epochs=args.fine_tune_epochs, mosei_stats=mosei_stats
            )
            
            visual_adapter, audio_adapter, text_adapter, mosei_stats, test_dataset = fine_tune_result
            
            # Step 5: Test
            print(f"\nTesting {combination_name}...")
            test_results = test_on_mosi_combination(
                model, visual_adapter, audio_adapter, text_adapter,
                test_dataset, use_visual, use_audio, use_text, mosei_stats
            )
            
            # Step 6: Plot training curves
            plot_path = output_dir / f'training_curves_{combination_name}.png'
            plot_training_curves(history, combination_name, str(plot_path))
            
            # Store results
            all_results[combination_name] = {
                'test_results': test_results,
                'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
                'final_val_corr': history['val_corr'][-1] if history['val_corr'] else None,
                'best_val_corr': max([x for x in history['val_corr'] if x is not None]) if any(x is not None for x in history['val_corr']) else None,
            }
            
            print(f"\n✅ {combination_name.upper()} Results:")
            print(f"   Test Correlation: {test_results['correlation']:.4f}" if test_results['correlation'] is not None else "   Test Correlation: N/A")
            print(f"   Test MAE: {test_results['mae']:.4f}")
            print(f"   Test MSE: {test_results['mse']:.4f}")
            print(f"   Samples: {test_results['samples']}")
            
        except Exception as e:
            print(f"❌ Error processing {combination_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[combination_name] = {'error': str(e)}
    
    # Save summary
    summary_path = output_dir / 'all_combinations_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL COMBINATIONS")
    print("="*80)
    for combination_name, results in all_results.items():
        if 'error' in results:
            print(f"{combination_name:20s}: ERROR - {results['error']}")
        else:
            test_res = results['test_results']
            corr = test_res['correlation']
            mae = test_res['mae']
            print(f"{combination_name:20s}: Correlation={corr:.4f if corr is not None else 'N/A':>10s}, MAE={mae:.4f}")
    print("="*80)
    print(f"\nFull results saved to: {summary_path}")
    print(f"All graphs saved to: {output_dir}")


if __name__ == '__main__':
    main()

