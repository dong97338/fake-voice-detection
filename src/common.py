import numpy as np
import torch
from torch import nn
import torchaudio
import timm
import os
from tqdm.auto import tqdm

# ASVSpoof Dataset class
class ASVSpoofDataset(torch.utils.data.Dataset):
    def __init__(self, audio_dir_path, num_samples, filename2label, transforms=None):
        super().__init__()
        self.audio_dir_path = audio_dir_path
        self.num_samples = num_samples
        self.audio_file_names = [name + '.flac' for name in filename2label.keys()]
        self.labels, self.label2id = self.encode_labels(filename2label)
        self.transforms = transforms
        
    def __getitem__(self, index):
        signal, sr = torchaudio.load(os.path.join(self.audio_dir_path, self.audio_file_names[index]))
        signal = self.preprocess(signal)
        if self.transforms:
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

# Model classes
class ResNetModel(nn.Module):
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

class ASTModel(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self.extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.pooling = nn.AdaptiveAvgPool2d((1, self.model.config.hidden_size))
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)  # Add a classification layer

    def forward(self, inputs):
        inputs = inputs.cpu()  # Move to CPU
        inputs = self.extractor(inputs.squeeze(1).numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {key: value.to(next(self.model.parameters()).device) for key, value in inputs.items()}  # Move to the same device as the model
        outputs = self.model(**inputs)
        pooled_output = self.pooling(outputs.logits.unsqueeze(1)).squeeze(1)  # Pool the output
        logits = self.classifier(pooled_output)  # Apply the classification layer
        return torch.sigmoid(logits)  # Use sigmoid activation for binary classification

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
