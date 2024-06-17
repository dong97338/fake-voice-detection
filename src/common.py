# common.py
import torch
from torch import nn
import torchaudio
import timm
import numpy as np
import os
from tqdm.auto import tqdm

# ASVSpoof Dataset class
class ASVSpoofDataset(torch.utils.data.Dataset):
    def __init__(self, audio_dir_path, num_samples, filename2label, transforms):
        super().__init__()
        self.audio_dir_path = audio_dir_path
        self.num_samples = num_samples
        self.audio_file_names = [name + '.flac' for name in filename2label.keys()]
        self.labels, self.label2id = self.encode_labels(filename2label)
        self.transforms = transforms
        
    def __getitem__(self, index):
        signal, sr = torchaudio.load(os.path.join(self.audio_dir_path, self.audio_file_names[index]))
        signal = self.preprocess(signal)
        signal = self.transforms(signal)
        return signal, self.labels[index]
    
    def __len__(self):
        return len(self.labels)
    
    def encode_labels(self, filename2label):
        labels = list(filename2label.values())
        label2id = {'spoof': 0, 'bonafide': 1}
        return [label2id[label] for label in labels], label2id
    
    def preprocess(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdims=True)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        if self.num_samples > signal.shape[1]:
            pad_last_dim = (0, self.num_samples - signal.shape[1])
            signal = torch.nn.functional.pad(signal, pad_last_dim)
        return signal

# Model class
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('resnet200d.ra2_in1k', pretrained=True, in_chans=1)
        for param in list(self.model.parameters())[:39]:
            param.requires_grad = False
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(self.model.num_features, 1), nn.Sigmoid())
        
    def forward(self, inputs):
        x = self.features(inputs)
        x = self.custom_layers(x)
        return x

# Utility function to load labels
def get_labels(path):
    with open(path, 'r') as file:
        text = file.read().splitlines()
    return {item.split(' ')[1]: item.split(' ')[-1] for item in tqdm(text)}

# EER function
def EER(labels, outputs):
    from sklearn.metrics import roc_curve
    fpr, tpr, threshold = roc_curve(labels, outputs, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer
