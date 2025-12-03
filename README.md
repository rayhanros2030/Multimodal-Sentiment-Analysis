# Cross-Domain Multimodal Sentiment Analysis through Feature Space Adaptation

This repository contains the implementation of a transfer learning framework for multimodal sentiment analysis that enables cross-dataset generalization through feature adaptation. The system trains on CMU-MOSEI using pre-extracted features and tests on CMU-MOSI using real-time extracted features, bridging the gap between different feature extraction paradigms through learned feature adapter networks.

## Overview

This project addresses the challenge of deploying multimodal sentiment analysis models trained on datasets with pre-extracted features (e.g., CMU-MOSEI) to real-world scenarios requiring real-time feature extraction. The framework employs feature adapter networks that learn mappings between different feature spaces, enabling a model trained on one feature extraction paradigm to work with another without retraining the entire model.

## Key Contributions

- Transfer learning framework for cross-dataset multimodal sentiment analysis
- Feature adapter networks that map between different feature extraction paradigms
- Cross-domain evaluation from CMU-MOSEI (training) to CMU-MOSI (testing)
- Real-time feature extraction using modern tools (MediaPipe FaceMesh, BERT, Librosa)

## Methodology

### Training Phase (CMU-MOSEI)
- **Visual Features**: OpenFace2 (713 dimensions)
- **Audio Features**: COVAREP (74 dimensions)
- **Text Features**: GloVe word vectors (300 dimensions)
- **Model**: Multimodal fusion model with cross-modal attention

### Testing Phase (CMU-MOSI)
- **Visual Features**: MediaPipe FaceMesh (65 dimensions) → adapted to OpenFace2 space (713 dimensions)
- **Audio Features**: Librosa (74 dimensions) → adapted to COVAREP space (74 dimensions)
- **Text Features**: BERT embeddings (768 dimensions) → adapted to GloVe space (300 dimensions)

### Feature Adaptation
The framework employs three feature adapter networks:
1. **Visual Adapter**: Maps FaceMesh features (65-dim) to OpenFace2 space (713-dim)
2. **Text Adapter**: Maps BERT embeddings (768-dim) to GloVe space (300-dim)
3. **Audio Adapter**: Maps Librosa features (74-dim) to COVAREP space (74-dim)

Each adapter is a neural network trained to minimize the mean squared error between adapted features and target MOSEI features.

### Model Architecture
- Modality-specific encoders transform features into unified 96-dimensional representations
- Cross-modal multi-head attention (4 heads) models bidirectional interactions
- Fusion layers with residual connections combine attended features
- Output: Continuous sentiment score in range [-3, +3]

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Dependencies
```bash
pip install torch torchvision torchaudio
pip install transformers
pip install librosa
pip install opencv-python
pip install mediapipe
pip install scikit-learn
pip install scipy
pip install h5py
pip install tqdm
```

## Dataset Setup

### CMU-MOSEI
Download CMU-MOSEI dataset and organize it as:
```
CMU-MOSEI/
  ├── visuals/
  │   └── CMU_MOSEI_VisualOpenFace2.csd
  ├── acoustics/
  │   └── CMU_MOSEI_COVAREP.csd
  ├── languages/
  │   └── CMU_MOSEI_TimestampedWordVectors.csd
  └── labels/
      └── CMU_MOSEI_Labels.csd
```

### CMU-MOSI
Organize CMU-MOSI dataset with:
- Video files (MP4 format)
- Audio files (WAV format)
- Transcript files
- Label files

## Usage

### Training Base Model on CMU-MOSEI
```bash
python train_mosei_only.py
```

This script:
- Trains the multimodal fusion model on CMU-MOSEI
- Uses pre-extracted features (OpenFace2, COVAREP, GloVe)
- Saves the trained model as `best_mosei_trained_model.pth`

### Transfer Learning: Train on MOSEI, Test on MOSI
```bash
python train_mosei_test_mosi_with_adapters.py
```

This script:
1. Loads the pre-trained model from CMU-MOSEI
2. Trains feature adapters to map MOSI features to MOSEI feature space
3. Tests the adapted features on CMU-MOSI dataset
4. Evaluates performance using correlation and MAE metrics

### Real-time Feature Extraction
```bash
python train_facemesh_bert_librosa_cmumosi.py
```

This script demonstrates:
- FaceMesh feature extraction from video frames
- BERT text embeddings from transcripts
- Librosa audio features from audio files
- Integration with the adaptation framework

## Project Structure

### Main Scripts
- `train_mosei_only.py`: Train base model on CMU-MOSEI
- `train_mosei_test_mosi_with_adapters.py`: Main transfer learning script
- `train_facemesh_bert_librosa_cmumosi.py`: Real-time feature extraction and training
- `FACEMESH_FEATURE_EXTRACTION.py`: MediaPipe FaceMesh implementation

### Key Components
- `FeatureAdapter`: Neural network for feature space adaptation
- `RegularizedMultimodalModel`: Base multimodal fusion model
- `ImprovedCorrelationLoss`: Loss function optimizing for correlation

## Results

The system achieves strong performance on cross-domain sentiment analysis:
- **Correlation**: 0.6360 (with all three modalities)
- Demonstrates effectiveness of combining visual, audio, and text modalities
- Validates the feature adaptation approach for cross-dataset generalization

## Model Files

The repository includes trained model checkpoints:
- `best_mosei_trained_model.pth`: Base model trained on CMU-MOSEI
- `visual_adapter.pth`: Visual feature adapter (FaceMesh → OpenFace2)
- `text_adapter.pth`: Text feature adapter (BERT → GloVe)
- `audio_adapter.pth`: Audio feature adapter (Librosa → COVAREP)

## Citation

If you use this code in your research, please cite:

```
Cross-Domain Multimodal Sentiment Analysis through Feature Space Adaptation
Rayhan Roswendi
December 2025
```

## License

This project is open source and available under the MIT License.

## Acknowledgments

- CMU-MOSEI dataset: [CMU Multimodal SDK](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/)
- CMU-MOSI dataset: [CMU-MOSI](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/)
- MediaPipe: [MediaPipe](https://mediapipe.dev/)
- BERT: [Hugging Face Transformers](https://huggingface.co/transformers/)
