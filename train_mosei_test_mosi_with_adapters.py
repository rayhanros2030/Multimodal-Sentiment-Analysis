"""
Train on CMU-MOSEI pre-extracted features, then test on CMU-MOSI
using FaceMesh, BERT, and Librosa with feature adapters.

Pipeline:
1. Train model on CMU-MOSEI (OpenFace2, COVAREP, GloVe)
2. Train adapters: FaceMesh→OpenFace2, BERT→GloVe, Librosa→COVAREP
3. Test on CMU-MOSI using adapted features
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

# Import original model
sys.path.append(str(Path(__file__).parent))
from train_mosei_only import RegularizedMultimodalModel, ImprovedCorrelationLoss

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class FeatureAdapter(nn.Module):
    """Adapts FaceMesh/BERT/Librosa features to match CMU-MOSEI feature spaces"""
    
    def __init__(self, input_dim: int, target_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        layers = []
        
        # For large dimension gaps (e.g., 65→713), use deeper architecture
        if target_dim / input_dim > 5:  # Large expansion like 65→713
            # Deeper network for visual adapter
            dims = [input_dim, 128, 256, 512, 1024, target_dim]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:  # No BatchNorm/activation on last layer
                    layers.append(nn.BatchNorm1d(dims[i+1]) if dims[i+1] > 1 else nn.Identity())
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.3))
        else:
            # Standard architecture for smaller gaps
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
    
    def __init__(self, mosei_dir: str, max_samples: Optional[int] = None):
        self.mosei_dir = Path(mosei_dir)
        self.max_samples = max_samples
        self.samples = self._load_mosei_data()
        
        print(f"Loaded {len(self.samples)} CMU-MOSEI samples")
    
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
        
        # Clip to [-3, 3]
        return float(np.clip(sentiment, -3.0, 3.0))
    
    def _load_mosei_data(self) -> List[Dict]:
        """Load CMU-MOSEI data"""
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
                
                # Clean features
                visual_feat = np.nan_to_num(visual_feat, nan=0.0, posinf=1.0, neginf=-1.0)
                audio_feat = np.nan_to_num(audio_feat, nan=0.0, posinf=1.0, neginf=-1.0)
                text_feat = np.nan_to_num(text_feat, nan=0.0, posinf=1.0, neginf=-1.0)
                visual_feat = np.clip(visual_feat, -1000, 1000)
                audio_feat = np.clip(audio_feat, -1000, 1000)
                text_feat = np.clip(text_feat, -1000, 1000)
                
                if (np.all(visual_feat == 0) and np.all(audio_feat == 0) and np.all(text_feat == 0)):
                    continue
                
                samples.append({
                    'audio': audio_feat,
                    'visual': visual_feat,
                    'text': text_feat,
                    'sentiment': sentiment
                })
            except Exception as e:
                continue
        
        return samples
    
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
    """CMU-MOSI Dataset with FaceMesh, BERT, and Librosa extraction"""
    
    def __init__(self, mosi_dir: str, max_samples: Optional[int] = None):
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
        
        # Try different possible folder structures
        possible_video_dirs = [
            self.mosi_dir / 'MOSI-VIDS',  # Primary location (93 MP4 files here)
            self.mosi_dir / 'MOSI-Videos',
            self.mosi_dir / 'videos',
            self.mosi_dir / 'video',
            self.mosi_dir
        ]
        possible_audio_dirs = [
            self.mosi_dir / 'MOSI-AUDIO',  # Primary location (WAV files moved here)
            self.mosi_dir / 'MOSI-VIDS',  # Fallback location
            self.mosi_dir / 'audios',
            self.mosi_dir / 'audio',
            self.mosi_dir / 'MOSI-Audios',
            self.mosi_dir
        ]
        possible_transcript_dirs = [
            self.mosi_dir / 'transcripts',
            self.mosi_dir / 'transcript',
            self.mosi_dir / 'MOSI-Transcript',
            self.mosi_dir
        ]
        possible_label_dirs = [
            self.mosi_dir / 'labels',
            self.mosi_dir / 'label',
            self.mosi_dir
        ]
        
        video_dir = None
        audio_dir = None
        transcript_dir = None
        label_dir = None
        
        for vd in possible_video_dirs:
            if vd.exists():
                print(f"  Checking {vd}...")
                mp4_files = list(vd.rglob('*.mp4'))  # Use recursive from start
                avi_files = list(vd.rglob('*.avi'))
                # Also check other video formats
                mov_files = list(vd.rglob('*.mov'))
                mkv_files = list(vd.rglob('*.mkv'))
                # Remove duplicates
                all_video_files = list(set(mp4_files + avi_files + mov_files + mkv_files))
                
                # Check for ZIP files (common in CMU-MOSI dataset)
                zip_files = list(vd.rglob('*.zip'))
                if zip_files and not all_video_files:
                    print(f"  WARNING: Found {len(zip_files)} ZIP files but no extracted video files!")
                    print(f"  The CMU-MOSI dataset appears to be compressed.")
                    print(f"  Please extract the ZIP files in {vd} to get .mp4/.avi files.")
                    print(f"  Example ZIP files found:")
                    for zf in zip_files[:3]:
                        print(f"    - {zf.name}")
                
                if all_video_files:
                    video_dir = vd
                    print(f"Found video directory: {video_dir}")
                    print(f"  - {len(mp4_files)} MP4 files")
                    print(f"  - {len(avi_files)} AVI files")
                    print(f"  - {len(mov_files)} MOV files")
                    print(f"  - {len(mkv_files)} MKV files")
                    print(f"  - Total: {len(all_video_files)} video files")
                    break
                else:
                    print(f"  No video files found in {vd}")
        
        for ad in possible_audio_dirs:
            if ad.exists():
                wav_files = list(ad.glob('*.wav')) + list(ad.rglob('*.wav'))
                if wav_files:
                    audio_dir = ad
                    print(f"Found audio directory: {audio_dir} ({len(set(wav_files))} WAV files)")
                    break
        
        for td in possible_transcript_dirs:
            if td.exists():
                txt_files = list(td.glob('*.txt')) + list(td.rglob('*.txt'))
                textonly_files = list(td.glob('*.textonly')) + list(td.rglob('*.textonly'))
                all_transcript_files = txt_files + textonly_files
                if all_transcript_files:
                    transcript_dir = td
                    print(f"Found transcript directory: {transcript_dir}")
                    print(f"  - {len(set(txt_files))} TXT files")
                    print(f"  - {len(set(textonly_files))} .textonly files")
                    print(f"  - Total: {len(set(all_transcript_files))} transcript files")
                    break
        
        for ld in possible_label_dirs:
            if ld.exists():
                label_dir = ld
                break
        
        if not video_dir:
            print(f"Warning: CMU-MOSI video directory not found. Looking in root and recursively...")
            video_files = list(self.mosi_dir.glob('*.mp4')) + list(self.mosi_dir.glob('*.avi'))
            # Also try recursive search
            if not video_files:
                video_files = list(self.mosi_dir.rglob('*.mp4')) + list(self.mosi_dir.rglob('*.avi'))
                if video_files:
                    print(f"Found {len(video_files)} video files via recursive search")
        else:
            # Use recursive search within the found directory
            video_files = (list(video_dir.rglob('*.mp4')) + 
                          list(video_dir.rglob('*.avi')) +
                          list(video_dir.rglob('*.mov')) +
                          list(video_dir.rglob('*.mkv')))
            # Remove duplicates
            video_files = list(set(video_files))
            print(f"Using {len(video_files)} video files from {video_dir}")
        
        # If no video files, try to use audio files as primary source
        if not video_files:
            print(f"\n⚠️  No video files found. Using audio files as primary identifier...")
            # Get all audio files
            if audio_dir:
                all_audio_files = list(audio_dir.rglob('*.wav'))
            else:
                all_audio_files = list(self.mosi_dir.rglob('*.wav'))
            
            if all_audio_files:
                print(f"Found {len(all_audio_files)} audio files. Using them to create samples (video will be None).")
                # Create sample IDs from audio files
                audio_file_ids = {}
                for audio_file in all_audio_files[:self.max_samples] if self.max_samples else all_audio_files:
                    # Extract ID from filename (remove extension)
                    audio_id = audio_file.stem
                    audio_file_ids[audio_id] = audio_file
                
                # Now create samples using audio files
                for audio_id, audio_file in audio_file_ids.items():
                    # Find transcript
                    transcript_file = None
                    if transcript_dir:
                        transcript_file = (next((f for f in transcript_dir.rglob(f"*{audio_id}*.txt")), None) or
                                         next((f for f in transcript_dir.rglob(f"*{audio_id}*.textonly")), None))
                    if not transcript_file:
                        transcript_file = (next((f for f in self.mosi_dir.rglob(f"*{audio_id}*.txt")), None) or
                                         next((f for f in self.mosi_dir.rglob(f"*{audio_id}*.textonly")), None))
                    
                    # Get label
                    label_value = None
                    if label_dir and (label_dir / 'labels.json').exists():
                        try:
                            with open(label_dir / 'labels.json', 'r') as f:
                                labels_json = json.load(f)
                                if audio_id in labels_json:
                                    label_value = labels_json[audio_id]
                                else:
                                    for key in labels_json.keys():
                                        if audio_id in key or key in audio_id:
                                            label_value = labels_json[key]
                                            break
                        except:
                            pass
                    
                    samples.append({
                        'video': None,  # No video available
                        'audio': audio_file,
                        'transcript': transcript_file if transcript_file and transcript_file.exists() else None,
                        'label': label_dir / 'labels.json' if label_value is not None else None,
                        'label_value': label_value,
                        'id': audio_id
                    })
                
                print(f"Created {len(samples)} samples from audio files (video=None)")
                return samples
            else:
                print(f"\nERROR: No video files AND no audio files found!")
                return []
        
        # Match files by ID
        for vid_file in video_files[:self.max_samples] if self.max_samples else video_files:
            vid_id = vid_file.stem
            
            # Find corresponding files (try recursive search in directories)
            audio_file = None
            if audio_dir:
                audio_file = next((f for f in audio_dir.rglob(f"*{vid_id}*.wav")), None)
            if not audio_file:
                audio_file = next((f for f in self.mosi_dir.rglob(f"*{vid_id}*.wav")), None)
            
            transcript_file = None
            if transcript_dir:
                # Try both .txt and .textonly extensions
                transcript_file = (next((f for f in transcript_dir.rglob(f"*{vid_id}*.txt")), None) or
                                 next((f for f in transcript_dir.rglob(f"*{vid_id}*.textonly")), None))
            if not transcript_file:
                transcript_file = (next((f for f in self.mosi_dir.rglob(f"*{vid_id}*.txt")), None) or
                                 next((f for f in self.mosi_dir.rglob(f"*{vid_id}*.textonly")), None))
            
            label_file = None
            if label_dir:
                label_file = next((f for f in label_dir.rglob(f"*{vid_id}*.txt")), None)
            
            # Try to load label from JSON if available
            label_value = None
            if label_dir and (label_dir / 'labels.json').exists():
                try:
                    with open(label_dir / 'labels.json', 'r') as f:
                        labels_json = json.load(f)
                        # Try different ID formats (with/without extension, partial matches)
                        if vid_id in labels_json:
                            label_value = labels_json[vid_id]
                        else:
                            # Try to find partial match
                            for key in labels_json.keys():
                                if vid_id in key or key in vid_id:
                                    label_value = labels_json[key]
                                    break
                except Exception as e:
                    pass
            
            # Be more lenient - allow samples with just audio + transcript (no video needed for testing)
            # Match audio files by ID if no video files found
            if not video_files:
                # Use audio files as primary identifier
                audio_file = next((f for f in self.mosi_dir.rglob(f"*{vid_id}*.wav")), None)
                if audio_file and audio_file.exists():
                    # Get transcript for this audio ID
                    transcript_file = (next((f for f in self.mosi_dir.rglob(f"*{vid_id}*.txt")), None) or
                                     next((f for f in self.mosi_dir.rglob(f"*{vid_id}*.textonly")), None))
                    
                    samples.append({
                        'video': None,  # No video file
                        'audio': audio_file,
                        'transcript': transcript_file if transcript_file and transcript_file.exists() else None,
                        'label': label_file if label_file and label_file.exists() else (label_dir / 'labels.json' if label_value is not None else None),
                        'label_value': label_value,
                        'id': vid_id
                    })
            elif vid_file and vid_file.exists():
                samples.append({
                    'video': vid_file,
                    'audio': audio_file if audio_file and audio_file.exists() else None,
                    'transcript': transcript_file if transcript_file and transcript_file.exists() else None,
                    'label': label_file if label_file and label_file.exists() else (label_dir / 'labels.json' if label_value is not None else None),
                    'label_value': label_value,
                    'id': vid_id
                })
        
        return samples
    
    def extract_facemesh_features(self, video_path: Optional[Path]) -> np.ndarray:
        """Extract FaceMesh features from video"""
        if not video_path or not video_path.exists():
            # Return zeros if no video available
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
    
    def extract_bert_features(self, transcript_path: Optional[Path]) -> np.ndarray:
        """Extract BERT features from transcript"""
        if not transcript_path or not transcript_path.exists():
            return np.zeros(768, dtype=np.float32)
        
        try:
            # Handle both .txt and .textonly files
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
        
        # Extract features (with fallbacks if files missing)
        # Visual: Only if video file exists, otherwise use zeros
        visual_feat = self.extract_facemesh_features(sample.get('video')) if sample.get('video') else np.zeros(65, dtype=np.float32)
        # Audio: Extract if file exists
        audio_feat = self.extract_librosa_features(sample['audio']) if sample['audio'] else np.zeros(74, dtype=np.float32)
        # Text: Extract if transcript exists
        text_feat = self.extract_bert_features(sample['transcript']) if sample['transcript'] else np.zeros(768, dtype=np.float32)
        
        # Load label
        sentiment = 0.0
        # First try label_value from JSON
        if 'label_value' in sample and sample['label_value'] is not None:
            try:
                sentiment = float(sample['label_value'])
            except:
                pass
        # Otherwise try label file
        elif sample['label']:
            try:
                if isinstance(sample['label'], Path) and sample['label'].name == 'labels.json':
                    # Label is the JSON file, value already in label_value
                    pass
                else:
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


def train_on_mosei(mosei_dir: str, epochs: int = 100, batch_size: int = 32, 
                   model_path: str = 'best_mosei_trained_model.pth'):
    """Train model on CMU-MOSEI pre-extracted features"""
    
    print("="*80)
    print("STEP 1: Training on CMU-MOSEI Pre-extracted Features")
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = RegularizedMultimodalModel(
        visual_dim=713, audio_dim=74, text_dim=300,
        hidden_dim=192, embed_dim=96, dropout=0.7
    ).to(device)
    
    criterion = ImprovedCorrelationLoss(alpha=0.3, beta=0.7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=0.04)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=7)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Starting training for {epochs} epochs...")
    
    best_val_corr = -1.0
    patience = 0
    max_patience = 25
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            visual = batch['visual'].to(device).float()
            audio = batch['audio'].to(device).float()
            text = batch['text'].to(device).float()
            sentiment = batch['sentiment'].to(device).float()
            
            optimizer.zero_grad()
            pred = model(visual, audio, text).squeeze()
            loss, loss_dict = criterion(pred, sentiment)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                visual = batch['visual'].to(device).float()
                audio = batch['audio'].to(device).float()
                text = batch['text'].to(device).float()
                sentiment = batch['sentiment'].to(device).float()
                
                pred = model(visual, audio, text).squeeze()
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(sentiment.cpu().numpy())
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_corr, _ = pearsonr(val_preds, val_labels)
        val_mae = np.mean(np.abs(val_preds - val_labels))
        val_loss = np.mean([(p - l)**2 for p, l in zip(val_preds, val_labels)])
        
        scheduler.step(val_corr)
        
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            torch.save(model.state_dict(), model_path)
            patience = 0
        else:
            patience += 1
        
        if epoch % 3 == 0:
            train_loss = np.mean(train_losses)
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val Corr: {val_corr:.4f} | "
                  f"Best: {best_val_corr:.4f} | Patience: {patience}/{max_patience}")
        
        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(model_path))
    
    # Test evaluation
    model.eval()
    test_preds, test_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            visual = batch['visual'].to(device).float()
            audio = batch['audio'].to(device).float()
            text = batch['text'].to(device).float()
            sentiment = batch['sentiment'].to(device).float()
            
            pred = model(visual, audio, text).squeeze()
            test_preds.extend(pred.cpu().numpy())
            test_labels.extend(sentiment.cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_corr, _ = pearsonr(test_preds, test_labels)
    test_mae = np.mean(np.abs(test_preds - test_labels))
    test_loss = np.mean([(p - l)**2 for p, l in zip(test_preds, test_labels)])
    
    print("\n" + "="*80)
    print("MOSEI Training Results:")
    print("="*80)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Correlation: {test_corr:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Best Validation Correlation: {best_val_corr:.4f}")
    print("="*80)
    
    return model


def train_adapters(mosei_dir: str, mosi_dataset: MOSIDataset, 
                   epochs: int = 50, batch_size: int = 16):
    """Train adapters to map FaceMesh/BERT/Librosa to MOSEI feature space"""
    
    print("\n" + "="*80)
    print("STEP 2: Training Feature Adapters (Improved)")
    print("="*80)
    
    # Load MOSEI targets
    mosei_dataset = MOSEIDataset(mosei_dir)
    mosei_targets = {'visual': [], 'audio': [], 'text': []}
    
    # Sample more for better clustering
    sample_size = min(2000, len(mosei_dataset.samples))
    for sample in mosei_dataset.samples[:sample_size]:
        mosei_targets['visual'].append(sample['visual'].astype(np.float32))
        mosei_targets['audio'].append(sample['audio'].astype(np.float32))
        mosei_targets['text'].append(sample['text'].astype(np.float32))
    
    # Convert to numpy arrays
    mosei_targets['visual'] = np.array(mosei_targets['visual'])
    mosei_targets['audio'] = np.array(mosei_targets['audio'])
    mosei_targets['text'] = np.array(mosei_targets['text'])
    
    print(f"MOSEI Targets: Visual={len(mosei_targets['visual'])}, "
          f"Audio={len(mosei_targets['audio'])}, Text={len(mosei_targets['text'])}")
    
    # Use K-means clustering for better target selection
    print("\nComputing K-means clusters for target selection...")
    n_clusters = min(100, len(mosei_targets['visual']) // 10)
    
    visual_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    visual_kmeans.fit(mosei_targets['visual'])
    visual_cluster_centers = torch.FloatTensor(visual_kmeans.cluster_centers_).to(device)
    
    audio_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    audio_kmeans.fit(mosei_targets['audio'])
    audio_cluster_centers = torch.FloatTensor(audio_kmeans.cluster_centers_).to(device)
    
    text_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    text_kmeans.fit(mosei_targets['text'])
    text_cluster_centers = torch.FloatTensor(text_kmeans.cluster_centers_).to(device)
    
    print(f"Created {n_clusters} clusters for each modality")
    
    # Compute MOSEI feature statistics for normalization
    print("\nComputing MOSEI feature statistics for normalization...")
    mosei_stats = {
        'visual': {
            'mean': torch.FloatTensor(mosei_targets['visual'].mean(axis=0)).to(device),
            'std': torch.FloatTensor(mosei_targets['visual'].std(axis=0) + 1e-8).to(device)
        },
        'audio': {
            'mean': torch.FloatTensor(mosei_targets['audio'].mean(axis=0)).to(device),
            'std': torch.FloatTensor(mosei_targets['audio'].std(axis=0) + 1e-8).to(device)
        },
        'text': {
            'mean': torch.FloatTensor(mosei_targets['text'].mean(axis=0)).to(device),
            'std': torch.FloatTensor(mosei_targets['text'].std(axis=0) + 1e-8).to(device)
        }
    }
    print("MOSEI statistics computed")
    
    # Initialize adapters
    visual_adapter = FeatureAdapter(65, 713, hidden_dim=512).to(device)
    audio_adapter = FeatureAdapter(74, 74, hidden_dim=256).to(device)
    text_adapter = FeatureAdapter(768, 300, hidden_dim=384).to(device)
    
    dataloader = DataLoader(mosi_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Use different learning rates: higher for visual (needs more training)
    optimizers = {
        'visual': torch.optim.Adam(visual_adapter.parameters(), lr=0.001, weight_decay=1e-5),
        'audio': torch.optim.Adam(audio_adapter.parameters(), lr=0.0005, weight_decay=1e-5),
        'text': torch.optim.Adam(text_adapter.parameters(), lr=0.0005, weight_decay=1e-5)
    }
    
    # Add learning rate schedulers
    schedulers = {
        'visual': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers['visual'], mode='min', factor=0.7, patience=5),
        'audio': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers['audio'], mode='min', factor=0.7, patience=5),
        'text': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers['text'], mode='min', factor=0.7, patience=5)
    }
    
    criterion = nn.MSELoss()
    
    print("\nTraining adapters...")
    best_losses = {'visual': float('inf'), 'audio': float('inf'), 'text': float('inf')}
    
    for epoch in range(epochs):
        losses = {'visual': 0, 'audio': 0, 'text': 0}
        batch_counts = {'visual': 0, 'audio': 0, 'text': 0}
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Visual adapter - use K-means nearest cluster
            if len(mosei_targets['visual']) > 0:
                v_in = batch['visual'].to(device).float()
                
                # For visual: cluster first 65 dims of MOSEI features to match FaceMesh input dim
                # Then use full 713-dim cluster centers as targets
                if visual_cluster_centers.shape[1] >= 65:
                    # Use first 65 dims for distance computation
                    v_in_for_dist = v_in  # [batch, 65]
                    centers_for_dist = visual_cluster_centers[:, :65]  # [n_clusters, 65]
                else:
                    v_in_for_dist = v_in[:, :visual_cluster_centers.shape[1]]
                    centers_for_dist = visual_cluster_centers
                
                distances = torch.cdist(v_in_for_dist.unsqueeze(1), centers_for_dist.unsqueeze(0)).squeeze(1)
                nearest_clusters = torch.argmin(distances, dim=1)
                target_v = visual_cluster_centers[nearest_clusters]  # [batch, 713]
                
                v_out = visual_adapter(v_in)
                v_loss = criterion(v_out, target_v)
                
                optimizers['visual'].zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(visual_adapter.parameters(), 1.0)
                optimizers['visual'].step()
                losses['visual'] += v_loss.item()
                batch_counts['visual'] += 1
            
            # Audio adapter - use K-means nearest cluster
            if len(mosei_targets['audio']) > 0:
                a_in = batch['audio'].to(device).float()
                
                # Find nearest cluster center
                distances = torch.cdist(a_in.unsqueeze(1), audio_cluster_centers.unsqueeze(0)).squeeze(1)
                nearest_clusters = torch.argmin(distances, dim=1)
                target_a = audio_cluster_centers[nearest_clusters]
                
                a_out = audio_adapter(a_in)
                a_loss = criterion(a_out, target_a)
                
                optimizers['audio'].zero_grad()
                a_loss.backward()
                torch.nn.utils.clip_grad_norm_(audio_adapter.parameters(), 1.0)
                optimizers['audio'].step()
                losses['audio'] += a_loss.item()
                batch_counts['audio'] += 1
            
            # Text adapter - use K-means nearest cluster
            if len(mosei_targets['text']) > 0:
                t_in = batch['text'].to(device).float()
                
                # Find nearest cluster center (use first 768 dims if text clusters have more)
                if text_cluster_centers.shape[1] >= 768:
                    t_in_for_dist = t_in
                    centers_for_dist = text_cluster_centers[:, :768]
                else:
                    t_in_for_dist = t_in[:, :text_cluster_centers.shape[1]]
                    centers_for_dist = text_cluster_centers
                
                distances = torch.cdist(t_in_for_dist.unsqueeze(1), centers_for_dist.unsqueeze(0)).squeeze(1)
                nearest_clusters = torch.argmin(distances, dim=1)
                target_t = text_cluster_centers[nearest_clusters]
                
                t_out = text_adapter(t_in)
                t_loss = criterion(t_out, target_t)
                
                optimizers['text'].zero_grad()
                t_loss.backward()
                torch.nn.utils.clip_grad_norm_(text_adapter.parameters(), 1.0)
                optimizers['text'].step()
                losses['text'] += t_loss.item()
                batch_counts['text'] += 1
        
        # Average losses
        if batch_counts['visual'] > 0:
            losses['visual'] /= batch_counts['visual']
        if batch_counts['audio'] > 0:
            losses['audio'] /= batch_counts['audio']
        if batch_counts['text'] > 0:
            losses['text'] /= batch_counts['text']
        
        # Update schedulers
        schedulers['visual'].step(losses['visual'])
        schedulers['audio'].step(losses['audio'])
        schedulers['text'].step(losses['text'])
        
        # Track best losses
        if losses['visual'] < best_losses['visual']:
            best_losses['visual'] = losses['visual']
        if losses['audio'] < best_losses['audio']:
            best_losses['audio'] = losses['audio']
        if losses['text'] < best_losses['text']:
            best_losses['text'] = losses['text']
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Visual: {losses['visual']:.4f} (best: {best_losses['visual']:.4f}, lr: {optimizers['visual'].param_groups[0]['lr']:.6f})")
            print(f"  Audio: {losses['audio']:.4f} (best: {best_losses['audio']:.4f}, lr: {optimizers['audio'].param_groups[0]['lr']:.6f})")
            print(f"  Text: {losses['text']:.4f} (best: {best_losses['text']:.4f}, lr: {optimizers['text'].param_groups[0]['lr']:.6f})")
    
    print("\nAdapters trained!")
    print(f"Final losses - Visual: {losses['visual']:.4f}, Audio: {losses['audio']:.4f}, Text: {losses['text']:.4f}")
    return visual_adapter, audio_adapter, text_adapter, mosei_stats


def fine_tune_end_to_end(model: RegularizedMultimodalModel, adapters: Tuple,
                        mosi_dataset: MOSIDataset, epochs: int = 20, batch_size: int = 16):
    """Fine-tune adapters + model together on MOSI with sentiment loss"""
    
    print("\n" + "="*80)
    print("STEP 3.5: End-to-End Fine-tuning on CMU-MOSI")
    print("="*80)
    
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
    
    # Set all to train mode
    model.train()
    visual_adapter.train()
    audio_adapter.train()
    text_adapter.train()
    
    # Create combined optimizer
    all_params = list(model.parameters()) + list(visual_adapter.parameters()) + \
                 list(audio_adapter.parameters()) + list(text_adapter.parameters())
    optimizer = torch.optim.Adam(all_params, lr=0.0001, weight_decay=1e-5)
    
    # Use sentiment loss
    criterion = ImprovedCorrelationLoss(alpha=0.3, beta=0.7)
    
    # Split MOSI for fine-tuning (60% train, 20% val, 20% test - held out)
    total_size = len(mosi_dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        mosi_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Store test dataset for later use
    mosi_dataset._test_dataset = test_dataset
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Fine-tuning for {epochs} epochs...")
    print(f"Train: {train_size} samples, Val: {val_size} samples, Test (held out): {test_size} samples")
    best_val_corr = -1.0
    
    for epoch in range(epochs):
        model.train()
        visual_adapter.train()
        audio_adapter.train()
        text_adapter.train()
        
        epoch_losses = []
        
        for batch in tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch+1}/{epochs}"):
            # Adapt features
            v_adapted = visual_adapter(batch['visual'].to(device).float())
            a_adapted = audio_adapter(batch['audio'].to(device).float())
            t_adapted = text_adapter(batch['text'].to(device).float())
            
            # Normalize if stats available
            if normalize_features and mosei_stats is not None:
                v_adapted = (v_adapted - mosei_stats['visual']['mean']) / mosei_stats['visual']['std']
                a_adapted = (a_adapted - mosei_stats['audio']['mean']) / mosei_stats['audio']['std']
                t_adapted = (t_adapted - mosei_stats['text']['mean']) / mosei_stats['text']['std']
                v_adapted = torch.clamp(v_adapted, -10, 10)
                a_adapted = torch.clamp(a_adapted, -10, 10)
                t_adapted = torch.clamp(t_adapted, -10, 10)
            
            # Predict
            pred = model(v_adapted, a_adapted, t_adapted).squeeze()
            sentiment = batch['sentiment'].to(device).float()
            
            # Loss
            loss, loss_dict = criterion(pred, sentiment)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Validation
        model.eval()
        visual_adapter.eval()
        audio_adapter.eval()
        text_adapter.eval()
        
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                v_adapted = visual_adapter(batch['visual'].to(device).float())
                a_adapted = audio_adapter(batch['audio'].to(device).float())
                t_adapted = text_adapter(batch['text'].to(device).float())
                
                if normalize_features and mosei_stats is not None:
                    v_adapted = (v_adapted - mosei_stats['visual']['mean']) / mosei_stats['visual']['std']
                    a_adapted = (a_adapted - mosei_stats['audio']['mean']) / mosei_stats['audio']['std']
                    t_adapted = (t_adapted - mosei_stats['text']['mean']) / mosei_stats['text']['std']
                    v_adapted = torch.clamp(v_adapted, -10, 10)
                    a_adapted = torch.clamp(a_adapted, -10, 10)
                    t_adapted = torch.clamp(t_adapted, -10, 10)
                
                pred = model(v_adapted, a_adapted, t_adapted).squeeze()
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(batch['sentiment'].numpy())
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        if len(val_preds) >= 2:
            val_corr, _ = pearsonr(val_preds, val_labels)
            val_mae = np.mean(np.abs(val_preds - val_labels))
            
            if val_corr > best_val_corr:
                best_val_corr = val_corr
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                avg_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Corr={val_corr:.4f}, Val MAE={val_mae:.4f}")
    
    print(f"\nFine-tuning complete! Best validation correlation: {best_val_corr:.4f}")
    # Return test dataset for proper evaluation
    return visual_adapter, audio_adapter, text_adapter, mosei_stats, test_dataset


def test_on_mosi(model: RegularizedMultimodalModel, adapters: Tuple, 
                 mosi_dataset: MOSIDataset, model_path: str, test_dataset=None):
    """Test adapted features on CMU-MOSI"""
    
    print("\n" + "="*80)
    print("STEP 3: Testing on CMU-MOSI with Adapted Features")
    print("="*80)
    
    if len(adapters) == 5:
        visual_adapter, audio_adapter, text_adapter, mosei_stats, _ = adapters
        normalize_features = True
        print("Using feature normalization to match MOSEI statistics")
    elif len(adapters) == 4:
        visual_adapter, audio_adapter, text_adapter, mosei_stats = adapters
        normalize_features = True
        print("Using feature normalization to match MOSEI statistics")
    else:
        visual_adapter, audio_adapter, text_adapter = adapters
        mosei_stats = None
        normalize_features = False
    
    # Load trained model
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded trained model from {model_path}")
    
    model.eval()
    visual_adapter.eval()
    audio_adapter.eval()
    text_adapter.eval()
    
    # Use test dataset if provided (held out), otherwise use full dataset
    if test_dataset is not None:
        print(f"\nUsing held-out test set: {len(test_dataset)} samples")
        dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    else:
        print(f"\nUsing full dataset: {len(mosi_dataset)} samples (WARNING: may include fine-tuning data)")
        dataloader = DataLoader(mosi_dataset, batch_size=16, shuffle=False)
    
    predictions, labels = [], []
    sample_ids = []
    
    test_size = len(test_dataset) if test_dataset is not None else len(mosi_dataset)
    print(f"\nTesting on {test_size} CMU-MOSI samples...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing on CMU-MOSI"):
            # Adapt features
            v_adapted = visual_adapter(batch['visual'].to(device).float())
            a_adapted = audio_adapter(batch['audio'].to(device).float())
            t_adapted = text_adapter(batch['text'].to(device).float())
            
            # Normalize adapted features to match MOSEI feature distribution
            if normalize_features and mosei_stats is not None:
                v_adapted = (v_adapted - mosei_stats['visual']['mean']) / mosei_stats['visual']['std']
                a_adapted = (a_adapted - mosei_stats['audio']['mean']) / mosei_stats['audio']['std']
                t_adapted = (t_adapted - mosei_stats['text']['mean']) / mosei_stats['text']['std']
                
                # Clip extreme values to prevent outliers
                v_adapted = torch.clamp(v_adapted, -10, 10)
                a_adapted = torch.clamp(a_adapted, -10, 10)
                t_adapted = torch.clamp(t_adapted, -10, 10)
            
            # Predict
            pred = model(v_adapted, a_adapted, t_adapted)
            
            # Diagnostic: Check first batch predictions
            if len(predictions) == 0:
                print(f"\nSample adapted feature ranges:")
                print(f"  Visual adapted: [{v_adapted.min():.4f}, {v_adapted.max():.4f}], mean={v_adapted.mean():.4f}")
                print(f"  Audio adapted: [{a_adapted.min():.4f}, {a_adapted.max():.4f}], mean={a_adapted.mean():.4f}")
                print(f"  Text adapted: [{t_adapted.min():.4f}, {t_adapted.max():.4f}], mean={t_adapted.mean():.4f}")
                print(f"\nFirst 5 predictions vs labels:")
                for i in range(min(5, len(pred))):
                    print(f"  Pred={pred[i].item():.4f}, Label={batch['sentiment'][i].item():.4f}")
            
            predictions.extend(pred.cpu().numpy().flatten())
            labels.extend(batch['sentiment'].numpy())
            if 'id' in batch:
                sample_ids.extend(batch['id'])
    
    print(f"\nLoaded {len(predictions)} predictions and {len(labels)} labels")
    print(f"Label statistics:")
    print(f"  Non-zero labels: {np.sum(np.array(labels) != 0.0)}")
    print(f"  Zero labels: {np.sum(np.array(labels) == 0.0)}")
    if len(labels) > 0:
        print(f"  Label range: [{np.min(labels):.2f}, {np.max(labels):.2f}]")
        print(f"  Label mean: {np.mean(labels):.2f}")
    
    # Diagnostic: Check prediction variance
    if len(predictions) > 0:
        pred_array = np.array(predictions)
        print(f"\nPrediction statistics:")
        print(f"  Prediction range: [{np.min(pred_array):.4f}, {np.max(pred_array):.4f}]")
        print(f"  Prediction mean: {np.mean(pred_array):.4f}")
        print(f"  Prediction std: {np.std(pred_array):.4f}")
        print(f"  Label std: {np.std(labels):.4f}")
    
    # Metrics
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Remove zero labels (missing labels)
    mask = labels != 0.0
    if np.sum(mask) > 0:
        predictions = predictions[mask]
        labels = labels[mask]
    else:
        print("WARNING: No valid labels found (all labels are zero)")
        print("Returning default metrics...")
        return {
            'mse': float('nan'),
            'correlation': float('nan'),
            'mae': float('nan'),
            'num_samples': 0
        }
    
    # Check if we have enough samples for correlation
    if len(predictions) < 2:
        print(f"WARNING: Only {len(predictions)} valid sample(s) - cannot compute correlation")
        print("Computing MAE and MSE only...")
        if len(predictions) == 1:
            mae = np.abs(predictions[0] - labels[0])
            mse = (predictions[0] - labels[0]) ** 2
            correlation = float('nan')
        else:
            mae = float('nan')
            mse = float('nan')
            correlation = float('nan')
    else:
        correlation, _ = pearsonr(predictions, labels)
        mae = np.mean(np.abs(predictions - labels))
        mse = np.mean((predictions - labels) ** 2)
    
    print(f"\n{'='*80}")
    print(f"CMU-MOSI Test Results (with adapted features):")
    print(f"{'='*80}")
    if not np.isnan(mse):
        print(f"Test Loss (MSE): {mse:.4f}")
    else:
        print(f"Test Loss (MSE): N/A (insufficient samples)")
    
    if not np.isnan(correlation):
        print(f"Test Correlation: {correlation:.4f}")
    else:
        print(f"Test Correlation: N/A (requires at least 2 samples)")
    
    if not np.isnan(mae):
        print(f"Test MAE: {mae:.4f}")
    else:
        print(f"Test MAE: N/A (insufficient samples)")
    
    print(f"Number of valid samples: {len(predictions)}")
    print(f"{'='*80}")
    
    return {
        'mse': float(mse),
        'correlation': float(correlation),
        'mae': float(mae),
        'num_samples': len(predictions)
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
    parser.add_argument('--mosei_samples', type=int, default=None,
                       help='Maximum MOSEI samples for training')
    parser.add_argument('--mosi_samples', type=int, default=50,
                       help='Maximum MOSI samples for testing (start small)')
    parser.add_argument('--train_epochs', type=int, default=100,
                       help='Epochs for training on MOSEI')
    parser.add_argument('--adapter_epochs', type=int, default=50,
                       help='Epochs for training adapters')
    parser.add_argument('--fine_tune_epochs', type=int, default=20,
                       help='Epochs for end-to-end fine-tuning')
    parser.add_argument('--skip_fine_tuning', action='store_true',
                       help='Skip end-to-end fine-tuning')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip MOSEI training (use existing model)')
    parser.add_argument('--model_path', type=str, default='best_mosei_trained_model.pth',
                       help='Path to save/load trained model')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Train on CMU-MOSEI, Test on CMU-MOSI with Feature Adapters")
    print("="*80)
    print(f"MOSEI Dir: {args.mosei_dir}")
    print(f"MOSI Dir: {args.mosi_dir}")
    print(f"MOSEI Samples: {args.mosei_samples or 'All'}")
    print(f"MOSI Samples: {args.mosi_samples}")
    print("="*80)
    
    # Check paths
    if not Path(args.mosei_dir).exists():
        print(f"ERROR: MOSEI directory not found: {args.mosei_dir}")
        return
    
    if not Path(args.mosi_dir).exists():
        print(f"ERROR: MOSI directory not found: {args.mosi_dir}")
        return
    
    # Step 1: Train on MOSEI
    if not args.skip_training:
        model = train_on_mosei(
            args.mosei_dir, 
            epochs=args.train_epochs,
            model_path=args.model_path
        )
    else:
        print("\nSkipping MOSEI training (using existing model)...")
        model = RegularizedMultimodalModel(
            visual_dim=713, audio_dim=74, text_dim=300,
            hidden_dim=192, embed_dim=96, dropout=0.7
        ).to(device)
    
    if model is None:
        print("ERROR: Model training failed!")
        return
    
    # Step 2: Load MOSI dataset
    print(f"\nLoading CMU-MOSI dataset (max {args.mosi_samples} samples)...")
    mosi_dataset = MOSIDataset(args.mosi_dir, max_samples=args.mosi_samples)
    
    if len(mosi_dataset) == 0:
        print("ERROR: No CMU-MOSI samples loaded. Check dataset structure and paths.")
        return
    
    # Step 3: Train adapters
    adapters_and_stats = train_adapters(
        args.mosei_dir,
        mosi_dataset,
        epochs=args.adapter_epochs
    )
    
    # Step 3.5: End-to-end fine-tuning (optional but recommended)
    if not args.skip_fine_tuning:
        print("\n" + "="*80)
        print("OPTIONAL: End-to-End Fine-tuning")
        print("="*80)
        print("This fine-tunes adapters + model together for sentiment prediction.")
        print("This should significantly improve correlation!")
        print("="*80)
        
        adapters_and_stats = fine_tune_end_to_end(
            model, adapters_and_stats, mosi_dataset,
            epochs=args.fine_tune_epochs
        )
    
    # Save adapters
    print("\nSaving trained adapters...")
    if len(adapters_and_stats) == 5:
        visual_adapter, audio_adapter, text_adapter, mosei_stats, test_dataset = adapters_and_stats
    else:
        visual_adapter, audio_adapter, text_adapter, mosei_stats = adapters_and_stats
        test_dataset = None
    torch.save(visual_adapter.state_dict(), 'visual_adapter.pth')
    torch.save(audio_adapter.state_dict(), 'audio_adapter.pth')
    torch.save(text_adapter.state_dict(), 'text_adapter.pth')
    print("Adapters saved!")
    
    # Step 4: Test on MOSI (using held-out test set if available)
    results = test_on_mosi(model, adapters_and_stats, mosi_dataset, args.model_path, test_dataset=test_dataset)
    
    # Save results
    with open('mosei_to_mosi_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to mosei_to_mosi_results.json")
    print("="*80)


if __name__ == "__main__":
    main()

