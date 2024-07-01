import os
import librosa
import numpy as np
import pandas as pd
import random
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    SR = 16000  # Sample rate expected by Wav2Vec2
    ROOT_FOLDER = './'
    BATCH_SIZE = 32
    SEED = 42

CONFIG = Config()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CONFIG.SEED)

# Load data
df = pd.read_csv('./train.csv')
train, val = train_test_split(df, test_size=0.2, random_state=CONFIG.SEED)

# Initialize Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)

def extract_wav2vec_features(df):
    features = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_input, _ = librosa.load(row['path'], sr=CONFIG.SR)
        input_values = processor(audio_input, return_tensors='pt', sampling_rate=CONFIG.SR).input_values
        input_values = input_values.to(device)

        with torch.no_grad():
            outputs = model(input_values).last_hidden_state
            # Taking the mean of the hidden states
            features.append(outputs.mean(dim=1).cpu().numpy())
    return np.concatenate(features, axis=0)

# Extract features
train_features = extract_wav2vec_features(train)
val_features = extract_wav2vec_features(val)

# Add the extracted features to the dataframe
train['features'] = list(train_features)
val['features'] = list(val_features)

# Simple EDA
print("Train Features Shape:", train_features.shape)
print("Validation Features Shape:", val_features.shape)

# Example of feature vectors
print("Example of extracted feature vector for the first train sample:", train['features'].iloc[0])

# You can visualize the features, for example using t-SNE or PCA for dimensionality reduction.
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_pca(features, labels, title):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()

train_labels = [0 if label == 'fake' else 1 for label in train['label']]
val_labels = [0 if label == 'fake' else 1 for label in val['label']]

plot_pca(train_features, train_labels, 'Train Features PCA')
plot_pca(val_features, val_labels, 'Validation Features PCA')
