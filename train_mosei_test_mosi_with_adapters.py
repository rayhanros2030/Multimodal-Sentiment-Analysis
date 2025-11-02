"""
Train on CMU-MOSEI pre-extracted features, then test on CMU-MOSI 
using Librosa, FaceMesh, and BERT with feature adapters.

Workflow:
1. Train model on MOSEI (OpenFace2, COVAREP, GloVe)
2. Train adapters: Librosa→COVAREP, FaceMesh→OpenFace2, BERT→GloVe
3. Extract features from MOSI using Librosa/FaceMesh/BERT
4. Adapt features and test with trained model
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
from scipy.stats import pearsonr
from sklearn.preprocessing import RobustScaler

# Import your original model
sys.path.append(str(Path(__file__).parent))
from train_mosei_only import RegularizedMultimodalModel, ImprovedCorrelationLoss

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class FeatureAdapter(nn.Module):
    """Adapts Librosa/FaceMesh/BERT features to match MOSEI feature spaces"""
    
    def __init__(self, input_dim: int, target_dim: int, hidden_dim: int = 512):
        super().__init__()
        # Use deeper architecture for large dimension gaps
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


class MOSEIDataset(Dataset):
    """CMU-MOSEI Dataset with pre-extracted features"""
    
    def __init__(self, mosei_dir: str, max_samples: int = None):
        self.mosei_dir = Path(mosei_dir)
        self.max_samples = max_samples
        self.samples = self._load_mosei_data()
        
    def _load_mosei_data(self) -> List[Dict]:
        """Load CMU-MOSEI pre-extracted features"""
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
        
        for vid_id in list(common_ids)[:self.max_samples] if self.max_samples else common_ids:
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
                    continue
                
                samples.append({
                    'visual': visual_feat,
                    'audio': audio_feat,
                    'text': text_feat,
                    'sentiment': sentiment
                })
            except Exception:
                continue
        
        print(f"Loaded {len(samples)} MOSEI samples")
        return samples
    
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
        """Extract sentiment score"""
        if data is None or 'features' not in data:
            return 0.0
        
        features = data['features']
        
        try:
            if len(features.shape) > 1:
                sentiment = float(np.mean(features[:, 0])) if features.shape[1] > 0 else 0.0
            else:
                sentiment = float(features[0]) if len(features) > 0 else 0.0
        except:
            sentiment = 0.0
        
        return sentiment
    
    def _clean_features(self, features: np.ndarray) -> np.ndarray:
        """Clean features"""
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        features = np.clip(features, -1000, 1000)
        return features
    
    def _clean_sentiment(self, sentiment: float) -> float:
        """Clean sentiment value"""
        if np.isnan(sentiment) or np.isinf(sentiment):
            return 0.0
        return float(np.clip(sentiment, -3.0, 3.0))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'visual': torch.FloatTensor(sample['visual']),
            'audio': torch.FloatTensor(sample['audio']),
            'text': torch.FloatTensor(sample['text']),
            'sentiment': sample['sentiment']
        }


class MOSIDataset(Dataset):
    """CMU-MOSI Dataset with Librosa, FaceMesh, and BERT extraction"""
    
    def __init__(self, mosi_dir: str, max_samples: int = None):
        self.mosi_dir = Path(mosi_dir)
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
        print(f"Loaded {len(self.samples)} CMU-MOSI samples")
    
    def _load_mosi_data(self) -> List[Dict]:
        """Load CMU-MOSI data with video/audio/transcript paths"""
        samples = []
        
        # CMU-MOSI structure (adjust paths as needed)
        video_dir = self.mosi_dir / "MOSI-Videos"
        audio_dir = self.mosi_dir / "MOSI-Audios"
        transcript_dir = self.mosi_dir / "MOSI-Transcript"
        labels_path = self.mosi_dir / "labels.json"
        
        # Alternative structure
        if not video_dir.exists():
            video_dir = self.mosi_dir / "videos"
            audio_dir = self.mosi_dir / "audios"
            transcript_dir = self.mosi_dir / "transcripts"
            labels_path = self.mosi_dir / "labels.json"
        
        if not video_dir.exists():
            print(f"Warning: CMU-MOSI video directory not found at {self.mosi_dir}")
            print("Looking for video files in root directory...")
            video_files = list(self.mosi_dir.glob('*.mp4')) + list(self.mosi_dir.glob('*.avi'))
            audio_files = list(self.mosi_dir.glob('*.wav')) if audio_dir.exists() else []
            transcript_files = list(self.mosi_dir.glob('*.txt')) if transcript_dir.exists() else []
        else:
            video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
            audio_files = list(audio_dir.glob('*.wav')) if audio_dir.exists() else []
            transcript_files = list(transcript_dir.glob('*.txt')) if transcript_dir.exists() else []
        
        # Load labels if available
        labels_dict = {}
        if labels_path.exists():
            try:
                with open(labels_path, 'r') as f:
                    labels_dict = json.load(f)
            except:
                pass
        
        # Match files by ID
        for vid_file in video_files[:self.max_samples] if self.max_samples else video_files:
            vid_id = vid_file.stem
            
            # Find corresponding files
            audio_file = next((f for f in audio_files if vid_id in f.stem), None)
            if not audio_file and audio_dir.exists():
                audio_file = audio_dir / f"{vid_id}.wav"
            
            transcript_file = next((f for f in transcript_files if vid_id in f.stem), None)
            if not transcript_file and transcript_dir.exists():
                transcript_file = transcript_dir / f"{vid_id}.txt"
            
            sentiment = labels_dict.get(vid_id, 0.0) if labels_dict else 0.0
            
            if audio_file and audio_file.exists():
                samples.append({
                    'video': vid_file,
                    'audio': audio_file if audio_file.exists() else None,
                    'transcript': transcript_file if transcript_file and transcript_file.exists() else None,
                    'sentiment': sentiment,
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
                i1, i2 = (idx % 468, (idx * 7) % 468)
                features.append(np.linalg.norm(normalized[i1] - normalized[i2]))
        
        return np.array(features[:65], dtype=np.float32)
    
    def extract_librosa_features(self, audio_path: Path) -> np.ndarray:
        """Extract Librosa audio features"""
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
        
        return {
            'visual': torch.FloatTensor(visual_feat),
            'audio': torch.FloatTensor(audio_feat),
            'text': torch.FloatTensor(text_feat),
            'sentiment': sample['sentiment'],
            'id': sample['id']
        }


class AdapterTrainer:
    """Train adapters using MOSEI features as targets"""
    
    def __init__(self, mosei_dir: str):
        self.mosei_dir = Path(mosei_dir)
        self.mosei_targets = self._load_mosei_targets()
        
        # Initialize adapters
        self.visual_adapter = FeatureAdapter(65, 713, hidden_dim=512).to(device)
        self.audio_adapter = FeatureAdapter(74, 74, hidden_dim=256).to(device)
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
    
    def train_adapters(self, mosi_dataset: MOSIDataset, epochs: int = 20, batch_size: int = 16):
        """Train adapters using MOSEI targets"""
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
                
                # Audio adapter
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


def train_on_mosei(mosei_dir: str, epochs: int = 100, batch_size: int = 32):
    """Train model on CMU-MOSEI pre-extracted features"""
    print("="*80)
    print("Training on CMU-MOSEI Pre-extracted Features")
    print("="*80)
    
    # Load dataset
    dataset = MOSEIDataset(mosei_dir)
    
    if len(dataset) == 0:
        print("ERROR: No MOSEI samples loaded!")
        return None
    
    # Split dataset
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = RegularizedMultimodalModel(
        visual_dim=713, audio_dim=74, text_dim=300,
        hidden_dim=192, embed_dim=96, dropout=0.7
    ).to(device)
    
    criterion = ImprovedCorrelationLoss(alpha=0.3, beta=0.7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=0.04)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=7
    )
    
    # Train
    best_val_corr = -1.0
    patience_counter = 0
    max_patience = 25
    
    print("\nStarting training...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        train_preds, train_labels = [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            visual = batch['visual'].to(device)
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            labels = batch['sentiment'].to(device)
            
            optimizer.zero_grad()
            outputs = model(visual, audio, text).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            train_losses.append(loss.item())
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                visual = batch['visual'].to(device)
                audio = batch['audio'].to(device)
                text = batch['text'].to(device)
                labels = batch['sentiment'].to(device)
                
                outputs = model(visual, audio, text).squeeze()
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_corr, _ = pearsonr(train_preds, train_labels)
        train_mae = np.mean(np.abs(np.array(train_preds) - np.array(train_labels)))
        train_loss = np.mean(train_losses)
        
        val_corr, _ = pearsonr(val_preds, val_labels)
        val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_labels)))
        
        scheduler.step(val_corr)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            patience_counter = 0
            torch.save(model.state_dict(), 'best_mosei_model.pth')
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 3 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train Corr: {train_corr:.4f} | "
                  f"Val Loss: {val_mae:.4f}, Val MAE: {val_mae:.4f}, Val Corr: {val_corr:.4f} | "
                  f"Best: {best_val_corr:.4f} | LR: {current_lr:.6f} | Patience: {patience_counter}/{max_patience}")
        
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_mosei_model.pth'))
    
    # Test evaluation
    model.eval()
    test_preds, test_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            visual = batch['visual'].to(device)
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            labels = batch['sentiment'].to(device)
            
            outputs = model(visual, audio, text).squeeze()
            test_preds.extend(outputs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_corr, _ = pearsonr(test_preds, test_labels)
    test_mae = np.mean(np.abs(np.array(test_preds) - np.array(test_labels)))
    test_loss = np.mean([criterion(torch.FloatTensor([p]), torch.FloatTensor([l])) for p, l in zip(test_preds, test_labels)])
    
    print("\n" + "="*80)
    print("MOSEI Training Results")
    print("="*80)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Correlation: {test_corr:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Best Validation Correlation: {best_val_corr:.4f}")
    print("="*80)
    
    return model


def test_on_mosi_with_adapters(
    model: RegularizedMultimodalModel,
    adapters: Tuple,
    mosi_dir: str,
    max_samples: int = None
):
    """Test trained model on CMU-MOSI using adapters"""
    print("\n" + "="*80)
    print("Testing on CMU-MOSI with Librosa, FaceMesh, and BERT")
    print("="*80)
    
    visual_adapter, audio_adapter, text_adapter = adapters
    
    # Load MOSI dataset
    mosi_dataset = MOSIDataset(mosi_dir, max_samples=max_samples)
    
    if len(mosi_dataset) == 0:
        print("ERROR: No MOSI samples loaded!")
        return
    
    dataloader = DataLoader(mosi_dataset, batch_size=16, shuffle=False)
    
    model.eval()
    visual_adapter.eval()
    audio_adapter.eval()
    text_adapter.eval()
    
    predictions, labels = [], []
    
    print("\nExtracting features and testing...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing MOSI samples"):
            # Extract features with Librosa/FaceMesh/BERT
            visual_raw = batch['visual'].to(device)  # FaceMesh: [batch, 65]
            audio_raw = batch['audio'].to(device)    # Librosa: [batch, 74]
            text_raw = batch['text'].to(device)      # BERT: [batch, 768]
            
            # Adapt features to MOSEI space
            visual_adapted = visual_adapter(visual_raw)  # [batch, 713]
            audio_adapted = audio_adapter(audio_raw)     # [batch, 74]
            text_adapted = text_adapter(text_raw)        # [batch, 300]
            
            # Predict with trained model
            pred = model(visual_adapted, audio_adapted, text_adapted)
            
            predictions.extend(pred.cpu().numpy().flatten())
            labels.extend(batch['sentiment'].numpy())
    
    # Calculate metrics
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    correlation, _ = pearsonr(predictions, labels)
    mae = np.mean(np.abs(predictions - labels))
    
    print("\n" + "="*80)
    print("CMU-MOSI Test Results (with Feature Adapters)")
    print("="*80)
    print(f"Correlation: {correlation:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Number of samples: {len(predictions)}")
    print("="*80)
    
    return {
        'correlation': float(correlation),
        'mae': float(mae),
        'predictions': predictions.tolist(),
        'labels': labels.tolist()
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train on MOSEI, test on MOSI with adapters')
    parser.add_argument('--mosei_dir', type=str, 
                       default="C:/Users/PC/Downloads/CMU-MOSEI",
                       help='Path to CMU-MOSEI directory')
    parser.add_argument('--mosi_dir', type=str,
                       default="C:/Users/PC/Downloads/CMU-MOSI Dataset",
                       help='Path to CMU-MOSI directory')
    parser.add_argument('--train_epochs', type=int, default=100,
                       help='Number of training epochs on MOSEI')
    parser.add_argument('--adapter_epochs', type=int, default=20,
                       help='Number of epochs for adapter training')
    parser.add_argument('--mosi_max_samples', type=int, default=None,
                       help='Maximum MOSI samples to process (None for all)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip MOSEI training (use existing model)')
    parser.add_argument('--skip_adapters', action='store_true',
                       help='Skip adapter training (use existing adapters)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MOSEI Training → MOSI Testing with Feature Adapters")
    print("="*80)
    print(f"MOSEI Dir: {args.mosei_dir}")
    print(f"MOSI Dir: {args.mosi_dir}")
    print("="*80)
    
    # Step 1: Train model on MOSEI
    model = None
    if not args.skip_training:
        model = train_on_mosei(args.mosei_dir, epochs=args.train_epochs)
    else:
        # Load existing model
        model = RegularizedMultimodalModel(
            visual_dim=713, audio_dim=74, text_dim=300,
            hidden_dim=192, embed_dim=96, dropout=0.7
        ).to(device)
        model.load_state_dict(torch.load('best_mosei_model.pth'))
        print("Loaded existing model from best_mosei_model.pth")
    
    # Step 2: Train adapters
    adapters = None
    if not args.skip_adapters:
        adapter_trainer = AdapterTrainer(args.mosei_dir)
        mosi_dataset_for_adapter = MOSIDataset(args.mosi_dir, max_samples=50)  # Use subset for adapter training
        adapters = adapter_trainer.train_adapters(mosi_dataset_for_adapter, epochs=args.adapter_epochs)
        
        # Save adapters
        torch.save(adapter_trainer.visual_adapter.state_dict(), 'visual_adapter.pth')
        torch.save(adapter_trainer.audio_adapter.state_dict(), 'audio_adapter.pth')
        torch.save(adapter_trainer.text_adapter.state_dict(), 'text_adapter.pth')
        print("Adapters saved!")
    else:
        # Load existing adapters
        visual_adapter = FeatureAdapter(65, 713, hidden_dim=512).to(device)
        audio_adapter = FeatureAdapter(74, 74, hidden_dim=256).to(device)
        text_adapter = FeatureAdapter(768, 300, hidden_dim=384).to(device)
        
        visual_adapter.load_state_dict(torch.load('visual_adapter.pth'))
        audio_adapter.load_state_dict(torch.load('audio_adapter.pth'))
        text_adapter.load_state_dict(torch.load('text_adapter.pth'))
        
        adapters = (visual_adapter, audio_adapter, text_adapter)
        print("Loaded existing adapters")
    
    # Step 3: Test on MOSI
    results = test_on_mosi_with_adapters(
        model, adapters, args.mosi_dir, max_samples=args.mosi_max_samples
    )
    
    # Save results
    with open('mosei_train_mosi_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to mosei_train_mosi_test_results.json")
    print("="*80)


if __name__ == "__main__":
    main()

