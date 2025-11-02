"""
Train FaceMesh, BERT, and Librosa to match CMU-MOSEI feature distributions,
then test on CMU-MOSI using your original architecture.

Strategy:
1. Extract features from CMU-MOSEI using OpenFace2/COVAREP/GloVe (targets)
2. Extract features from CMU-MOSI using FaceMesh/BERT/Librosa (inputs)
3. Train adapters to map FaceMesh→OpenFace2, BERT→GloVe, Librosa→COVAREP
4. Test adapted features on CMU-MOSI with your original architecture
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import h5py
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import librosa
import cv2
import mediapipe as mp
from tqdm import tqdm
import json
import sys

# Import your original model
sys.path.append(str(Path(__file__).parent))
from train_mosei_only import RegularizedMultimodalModel, ImprovedCorrelationLoss

class FeatureAdapter(nn.Module):
    """Adapts FaceMesh/BERT/Librosa features to match CMU-MOSEI feature spaces"""
    
    def __init__(self, input_dim: int, target_dim: int, hidden_dim: int = 512):
        super().__init__()
        # Use deeper architecture for large dimension gaps (e.g., 65→713)
        if target_dim / input_dim > 5:  # Large expansion
            hidden_dim = max(hidden_dim, target_dim // 2)
        
        layers = []
        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if hidden_dim > 1 else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.3)
        ])
        
        # Hidden layer
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if hidden_dim > 1 else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.3)
        ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, target_dim))
        
        self.adaptation = nn.Sequential(*layers)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.adaptation(x).squeeze()


class CMUMOSIDataset(Dataset):
    """CMU-MOSI Dataset with FaceMesh, BERT, and Librosa extraction"""
    
    def __init__(self, data_dir: str, mode: str = 'train', max_samples: int = None):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.max_samples = max_samples
        
        # Initialize extractors
        print("Loading BERT model...")
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()
        for param in self.bert_model.parameters():
            param.requires_grad = False
        
        print("Initializing FaceMesh...")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load data
        self.samples = self._load_mosi_data()
        
        print(f"Loaded {len(self.samples)} CMU-MOSI {mode} samples")
    
    def _load_mosi_data(self) -> List[Dict]:
        """Load CMU-MOSI data with video/audio/transcript paths"""
        samples = []
        
        # CMU-MOSI structure (adjust paths as needed)
        # Common structure: videos, audios, transcripts, labels folders
        video_dir = self.data_dir / 'videos'
        audio_dir = self.data_dir / 'audios'
        transcript_dir = self.data_dir / 'transcripts'
        labels_dir = self.data_dir / 'labels'
        
        # Alternative structure
        if not video_dir.exists():
            video_dir = self.data_dir / 'video'
            audio_dir = self.data_dir / 'audio'
            transcript_dir = self.data_dir / 'transcript'
            labels_dir = self.data_dir / 'label'
        
        if not video_dir.exists():
            print(f"Warning: CMU-MOSI video directory not found at {self.data_dir}")
            print("Looking for video files in root directory...")
            video_files = list(self.data_dir.glob('*.mp4')) + list(self.data_dir.glob('*.avi'))
            audio_files = list(self.data_dir.glob('*.wav')) if audio_dir.exists() else []
            transcript_files = list(self.data_dir.glob('*.txt')) if transcript_dir.exists() else []
        else:
            video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
            audio_files = list(audio_dir.glob('*.wav')) if audio_dir.exists() else []
            transcript_files = list(transcript_dir.glob('*.txt')) if transcript_dir.exists() else []
        
        # Match files by ID
        for vid_file in video_files[:self.max_samples] if self.max_samples else video_files:
            vid_id = vid_file.stem
            
            # Find corresponding files
            audio_file = next((f for f in audio_files if f.stem == vid_id), None)
            if not audio_file and audio_dir.exists():
                audio_file = audio_dir / f"{vid_id}.wav"
            
            transcript_file = next((f for f in transcript_files if vid_id in f.stem), None)
            if not transcript_file and transcript_dir.exists():
                transcript_file = transcript_dir / f"{vid_id}.txt"
            
            label_file = None
            if labels_dir.exists():
                label_file = labels_dir / f"{vid_id}.txt"
            
            if audio_file and audio_file.exists():
                samples.append({
                    'video': vid_file,
                    'audio': audio_file if audio_file.exists() else None,
                    'transcript': transcript_file if transcript_file and transcript_file.exists() else None,
                    'label': label_file if label_file and label_file.exists() else None,
                    'id': vid_id
                })
        
        return samples
    
    def extract_facemesh_features(self, video_path: Path) -> np.ndarray:
        """Extract FaceMesh features from video"""
        if not video_path.exists():
            return np.zeros(65, dtype=np.float32)
        
        cap = cv2.VideoCapture(str(video_path))
        frame_features = []
        max_frames = 100  # Limit frames for speed
        
        frame_count = 0
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] 
                    for lm in results.multi_face_landmarks[0].landmark
                ])
                features = self._extract_emotion_features(landmarks)
                frame_features.append(features)
            else:
                frame_features.append(np.zeros(65, dtype=np.float32))
            
            frame_count += 1
        
        cap.release()
        
        if len(frame_features) == 0:
            return np.zeros(65, dtype=np.float32)
        
        # Temporal averaging
        return np.mean(frame_features, axis=0).astype(np.float32)
    
    def _extract_emotion_features(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract ~65 emotion features from 468 landmarks"""
        face_width = np.linalg.norm(landmarks[0] - landmarks[16])
        if face_width < 1e-6:
            return np.zeros(65, dtype=np.float32)
        
        normalized = landmarks / face_width
        features = []
        
        # Mouth features (most important for emotion)
        mouth_width = np.linalg.norm(normalized[61] - normalized[291])
        mouth_height = np.linalg.norm(normalized[13] - normalized[14])
        left_corner_y = normalized[61, 1]
        right_corner_y = normalized[291, 1]
        mouth_center_y = (normalized[13, 1] + normalized[14, 1]) / 2
        corner_angle = np.arctan2((left_corner_y + right_corner_y) / 2 - mouth_center_y, mouth_width / 2)
        features.extend([mouth_width, mouth_height, left_corner_y, right_corner_y, corner_angle])
        
        # Eye features
        left_eye_width = np.linalg.norm(normalized[33] - normalized[133])
        right_eye_width = np.linalg.norm(normalized[362] - normalized[263])
        inter_eye = np.linalg.norm(normalized[33] - normalized[263])
        features.extend([left_eye_width, right_eye_width, inter_eye])
        
        # Eyebrow features
        left_eyebrow_height = np.mean([normalized[21, 1], normalized[55, 1], normalized[107, 1]])
        right_eyebrow_height = np.mean([normalized[251, 1], normalized[285, 1], normalized[336, 1]])
        features.extend([left_eyebrow_height, right_eyebrow_height])
        
        # Symmetry features
        eye_symmetry = abs(left_eye_width - right_eye_width) / max(left_eye_width, right_eye_width) if max(left_eye_width, right_eye_width) > 0 else 0
        mouth_symmetry = abs(left_corner_y - right_corner_y)
        features.extend([eye_symmetry, mouth_symmetry])
        
        # Fill to 65 with additional landmark positions and distances
        while len(features) < 65:
            idx = len(features)
            if idx < len(normalized):
                features.append(np.linalg.norm(normalized[idx]))
            else:
                # Use landmark pairs
                i1, i2 = (idx % 468, (idx * 7) % 468)
                features.append(np.linalg.norm(normalized[i1] - normalized[i2]))
        
        return np.array(features[:65], dtype=np.float32)
    
    def extract_librosa_features(self, audio_path: Path) -> np.ndarray:
        """Extract Librosa audio features (same as your IEMOCAP code)"""
        if not audio_path or not audio_path.exists():
            return np.zeros(74, dtype=np.float32)
        
        try:
            y, sr = librosa.load(str(audio_path), sr=22050, duration=3.0)
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
            chroma = librosa.feature.chroma(y=y, sr=sr).mean(axis=1)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            features = np.concatenate([
                mfcc,  # 13
                chroma,  # 12
                [spectral_centroid, spectral_rolloff, zero_crossing, tempo]  # 4
            ])  # Total: 29
            
            # Pad to 74 for compatibility
            if len(features) < 74:
                features = np.pad(features, (0, 74 - len(features)), mode='constant')
            else:
                features = features[:74]
            
            return features.astype(np.float32)
        except Exception as e:
            return np.zeros(74, dtype=np.float32)
    
    def extract_bert_features(self, transcript_path: Path) -> np.ndarray:
        """Extract BERT features from transcript"""
        if not transcript_path or not transcript_path.exists():
            return np.zeros(768, dtype=np.float32)
        
        try:
            with open(transcript_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            
            if not text:
                return np.zeros(768, dtype=np.float32)
            
            inputs = self.bert_tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()  # [768]
            
            return embeddings.numpy().astype(np.float32)
        except Exception as e:
            return np.zeros(768, dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract features
        visual_feat = self.extract_facemesh_features(sample['video'])
        audio_feat = self.extract_librosa_features(sample['audio'])
        text_feat = self.extract_bert_features(sample['transcript'])
        
        # Load label
        sentiment = 0.0
        if sample['label']:
            try:
                with open(sample['label'], 'r') as f:
                    sentiment = float(f.read().strip())
            except:
                pass
        
        return {
            'visual': torch.FloatTensor(visual_feat),
            'audio': torch.FloatTensor(audio_feat),
            'text': torch.FloatTensor(text_feat),
            'sentiment': sentiment,
            'id': sample['id']
        }


class FeatureDistillationTrainer:
    """Train feature adapters using CMU-MOSEI as targets"""
    
    def __init__(self, mosei_dir: str):
        self.mosei_dir = Path(mosei_dir)
        self.mosei_targets = self._load_mosei_targets()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize adapters with appropriate capacities
        # FaceMesh (65) -> OpenFace2 (713): Large gap, needs larger hidden dim
        self.visual_adapter = FeatureAdapter(65, 713, hidden_dim=512).to(device)
        # Librosa (74) -> COVAREP (74): Same dim, simpler adapter
        self.audio_adapter = FeatureAdapter(74, 74, hidden_dim=256).to(device)
        # BERT (768) -> GloVe (300): Reduction, moderate adapter
        self.text_adapter = FeatureAdapter(768, 300, hidden_dim=384).to(device)
    
    def _load_mosei_targets(self) -> Dict:
        """Load CMU-MOSEI features as target distributions"""
        targets = {'visual': [], 'audio': [], 'text': []}
        
        try:
            visual_path = self.mosei_dir / 'visuals' / 'CMU_MOSEI_VisualOpenFace2.csd'
            audio_path = self.mosei_dir / 'acoustics' / 'CMU_MOSEI_COVAREP.csd'
            text_path = self.mosei_dir / 'languages' / 'CMU_MOSEI_TimestampedWordVectors.csd'
            
            for path, key, target_size in [
                (visual_path, 'visual', 713),
                (audio_path, 'audio', 74),
                (text_path, 'text', 300)
            ]:
                if path.exists():
                    with h5py.File(path, 'r') as f:
                        data_key = None
                        for k in ['OpenFace_2', 'COVAREP', 'glove_vectors', 'data']:
                            if k in f:
                                data_key = k
                                break
                        
                        if data_key and 'data' in f[data_key]:
                            video_ids = list(f[data_key]['data'].keys())[:1000]  # Sample 1000
                            for vid_id in video_ids:
                                try:
                                    features = f[data_key]['data'][vid_id]['features'][:]
                                    if len(features.shape) > 1:
                                        features = np.mean(features, axis=0)
                                    features = features.flatten()
                                    
                                    # Pad/truncate to target size
                                    if len(features) > target_size:
                                        features = features[:target_size]
                                    elif len(features) < target_size:
                                        features = np.pad(features, (0, target_size - len(features)))
                                    
                                    targets[key].append(features.astype(np.float32))
                                except:
                                    continue
            
            print(f"MOSEI Targets: Visual={len(targets['visual'])}, "
                  f"Audio={len(targets['audio'])}, Text={len(targets['text'])}")
        except Exception as e:
            print(f"Warning loading MOSEI: {e}")
        
        return targets
    
    def train_adapters(self, mosi_dataset: CMUMOSIDataset, epochs: int = 20, batch_size: int = 16):
        """Train adapters using MOSEI targets"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        dataloader = DataLoader(mosi_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        optimizers = {
            'visual': torch.optim.Adam(self.visual_adapter.parameters(), lr=0.001),
            'audio': torch.optim.Adam(self.audio_adapter.parameters(), lr=0.001),
            'text': torch.optim.Adam(self.text_adapter.parameters(), lr=0.001)
        }
        
        criterion = nn.MSELoss()
        
        print("\nTraining feature adapters...")
        for epoch in range(epochs):
            losses = {'visual': 0, 'audio': 0, 'text': 0}
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                # Visual adapter
                if len(self.mosei_targets['visual']) > 0:
                    v_in = batch['visual'].to(device)
                    target_v = torch.FloatTensor(
                        np.random.choice(self.mosei_targets['visual'], size=v_in.shape[0])
                    ).to(device)
                    
                    v_out = self.visual_adapter(v_in)
                    v_loss = criterion(v_out, target_v)
                    
                    optimizers['visual'].zero_grad()
                    v_loss.backward()
                    optimizers['visual'].step()
                    losses['visual'] += v_loss.item()
                
                # Audio adapter (if needed)
                if len(self.mosei_targets['audio']) > 0:
                    a_in = batch['audio'].to(device)
                    target_a = torch.FloatTensor(
                        np.random.choice(self.mosei_targets['audio'], size=a_in.shape[0])
                    ).to(device)
                    
                    a_out = self.audio_adapter(a_in)
                    a_loss = criterion(a_out, target_a)
                    
                    optimizers['audio'].zero_grad()
                    a_loss.backward()
                    optimizers['audio'].step()
                    losses['audio'] += a_loss.item()
                
                # Text adapter
                if len(self.mosei_targets['text']) > 0:
                    t_in = batch['text'].to(device)
                    target_t = torch.FloatTensor(
                        np.random.choice(self.mosei_targets['text'], size=t_in.shape[0])
                    ).to(device)
                    
                    t_out = self.text_adapter(t_in)
                    t_loss = criterion(t_out, target_t)
                    
                    optimizers['text'].zero_grad()
                    t_loss.backward()
                    optimizers['text'].step()
                    losses['text'] += t_loss.item()
            
            print(f"Epoch {epoch+1}: V={losses['visual']:.4f}, A={losses['audio']:.4f}, T={losses['text']:.4f}")
        
        print("Adapters trained!")
        return self.visual_adapter, self.audio_adapter, self.text_adapter


def test_on_mosi(mosi_dataset: CMUMOSIDataset, adapters: Tuple, model_path: str = None):
    """Test adapted features on CMU-MOSI"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    visual_adapter, audio_adapter, text_adapter = adapters
    
    # Your original model
    model = RegularizedMultimodalModel(
        visual_dim=713, audio_dim=74, text_dim=300,
        hidden_dim=192, embed_dim=96, dropout=0.7
    ).to(device)
    
    if model_path and Path(model_path).exists():
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    
    model.eval()
    visual_adapter.eval()
    audio_adapter.eval()
    text_adapter.eval()
    
    dataloader = DataLoader(mosi_dataset, batch_size=16, shuffle=False)
    
    predictions, labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            v = visual_adapter(batch['visual'].to(device))
            a = audio_adapter(batch['audio'].to(device))
            t = text_adapter(batch['text'].to(device))
            
            pred = model(v, a, t)
            predictions.extend(pred.cpu().numpy().flatten())
            labels.extend(batch['sentiment'].numpy())
    
    # Metrics
    from scipy.stats import pearsonr
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    correlation, _ = pearsonr(predictions, labels)
    mae = np.mean(np.abs(predictions - labels))
    
    print(f"\n{'='*80}")
    print(f"CMU-MOSI Test Results")
    print(f"{'='*80}")
    print(f"Correlation: {correlation:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"{'='*80}")
    
    return {'correlation': float(correlation), 'mae': float(mae)}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train FaceMesh/BERT/Librosa adapters')
    parser.add_argument('--mosei_dir', type=str, 
                       default="C:/Users/PC/Downloads/CMU-MOSEI",
                       help='Path to CMU-MOSEI directory')
    parser.add_argument('--mosi_dir', type=str,
                       default="C:/Users/PC/Downloads/CMU-MOSI Dataset",
                       help='Path to CMU-MOSI directory')
    parser.add_argument('--max_samples', type=int, default=10,
                       help='Maximum samples to process (start small for testing)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs for adapters')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pretrained model checkpoint')
    
    args = parser.parse_args()
    
    print("="*80)
    print("FaceMesh + BERT + Librosa Feature Adaptation")
    print("Using CMU-MOSEI as targets, testing on CMU-MOSI")
    print("="*80)
    print(f"MOSEI Dir: {args.mosei_dir}")
    print(f"MOSI Dir: {args.mosi_dir}")
    print(f"Max Samples: {args.max_samples}")
    print("="*80)
    
    # Check paths exist
    if not Path(args.mosei_dir).exists():
        print(f"ERROR: MOSEI directory not found: {args.mosei_dir}")
        return
    
    if not Path(args.mosi_dir).exists():
        print(f"ERROR: MOSI directory not found: {args.mosi_dir}")
        return
    
    # Initialize
    print("\nInitializing trainer...")
    trainer = FeatureDistillationTrainer(args.mosei_dir)
    
    print(f"\nLoading CMU-MOSI dataset (max {args.max_samples} samples)...")
    mosi_dataset = CMUMOSIDataset(args.mosi_dir, max_samples=args.max_samples)
    
    if len(mosi_dataset) == 0:
        print("ERROR: No CMU-MOSI samples loaded. Check dataset structure and paths.")
        return
    
    # Train adapters
    print("\n" + "="*80)
    print("Training Feature Adapters")
    print("="*80)
    adapters = trainer.train_adapters(mosi_dataset, epochs=args.epochs)
    
    # Save adapters
    print("\nSaving trained adapters...")
    torch.save(trainer.visual_adapter.state_dict(), 'visual_adapter.pth')
    torch.save(trainer.audio_adapter.state_dict(), 'audio_adapter.pth')
    torch.save(trainer.text_adapter.state_dict(), 'text_adapter.pth')
    print("Adapters saved!")
    
    # Test
    print("\n" + "="*80)
    print("Testing on CMU-MOSI")
    print("="*80)
    results = test_on_mosi(mosi_dataset, adapters, args.model_path)
    
    with open('cmumosi_adapted_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to cmumosi_adapted_results.json")
    print("="*80)


if __name__ == "__main__":
    main()
