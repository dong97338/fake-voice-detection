import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torchaudio
import timm
from torch import nn
from tqdm.auto import tqdm
from transformers import ASTFeatureExtractor, ASTModel


@dataclass
class ASVSpoofDataset(torch.utils.data.Dataset):
    audio_dir_path: str
    num_samples: int
    filename2label: dict
    transforms: nn.Module = None
    augment: bool = False
    audio_file_names: list = field(init=False)
    labels: list = field(init=False)
    label2id: dict = field(default_factory=lambda: {'spoof': 0, 'bonafide': 1})

    def __post_init__(self):
        self.audio_file_names = [name + '.flac' for name in self.filename2label.keys()]
        self.labels = [self.label2id[label] for label in self.filename2label.values()]

    def __getitem__(self, index):
        signal, sr = torchaudio.load(os.path.join(self.audio_dir_path, self.audio_file_names[index]))
        signal = self.preprocess(signal)
        if self.transforms:
            signal = self.transforms(signal)
        if self.augment:
            signal = self.audio_augment(signal)
        return signal, self.labels[index]

    def __len__(self):
        return len(self.labels)

    def preprocess(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdims=True)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        if self.num_samples > signal.shape[1]:
            pad_last_dim = (0, self.num_samples - signal.shape[1])
            signal = torch.nn.functional.pad(signal, pad_last_dim)
        return signal

    def audio_augment(self, signal):
        signal = self.trim_audio(signal)
        signal = self.time_shift(signal)
        signal = self.add_noise(signal)
        signal = self.crop_or_pad(signal)
        return signal

    def trim_audio(self, audio, epsilon=0.15):
        # Find indices where the absolute value of the signal is greater than epsilon
        non_silent_indices = torch.where(torch.abs(audio) > epsilon)[1]
        if len(non_silent_indices) > 0:
            start_idx = non_silent_indices[0].item()
            end_idx = non_silent_indices[-1].item() + 1
            audio = audio[:, start_idx:end_idx]
        return audio

    def time_shift(self, signal, shift_max=0.2, prob=0.5):
        if random.random() < prob:
            shift = int(random.uniform(-shift_max, shift_max) * signal.shape[1])
            signal = torch.roll(signal, shift)
        return signal

    def add_noise(self, signal, noise_factor=0.005, prob=0.5):
        if random.random() < prob:
            noise = torch.randn_like(signal) * noise_factor
            signal = signal + noise
        return signal

    def crop_or_pad(self, audio, target_len=64, pad_mode='constant'):
        if target_len is None:
            target_len = self.num_samples
        audio_len = audio.shape[1]
        if audio_len < target_len:  # if audio_len is smaller than target_len then use Padding
            diff_len = (target_len - audio_len)
            pad1 = random.randint(0, diff_len)  # select random location for padding
            pad2 = diff_len - pad1
            padding = (pad1, pad2)
            audio = torch.nn.functional.pad(audio, (0, 0, padding[0], padding[1]), mode=pad_mode)  # apply padding
        elif audio_len > target_len:  # if audio_len is larger than target_len then use Cropping
            diff_len = (audio_len - target_len)
            idx = random.randint(0, diff_len)  # select random location for cropping
            audio = audio[:, :, idx: (idx + target_len)]
        return audio


    def gaussian_noise(self, audio, std_min=0.0025, std_max=0.025, prob=0.5):
        if random.random() < prob:
            std = random.uniform(std_min, std_max)
            noise = torch.randn_like(audio) * std
            audio = audio + noise
        return audio

    def normalize(self, audio):
        mean = torch.mean(audio)
        std = torch.std(audio)
        audio = (audio - mean) / (std + 1e-6)
        return audio


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


class ASTModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.pooling = nn.AdaptiveAvgPool2d((1, self.model.config.hidden_size))
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, inputs):
        inputs = inputs.cpu()
        inputs_np = inputs.squeeze(1).numpy()
        inputs = self.extractor(inputs_np, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {key: value.to(next(self.model.parameters()).device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        pooled_output = self.pooling(outputs.last_hidden_state.unsqueeze(1)).squeeze(1)
        logits = self.classifier(pooled_output)
        return torch.sigmoid(logits).squeeze(-1)


def get_labels(path):
    with open(path, 'r') as file:
        text = file.read().splitlines()
    return {item.split(' ')[1]: item.split(' ')[-1] for item in tqdm(text)}


def EER(labels, outputs):
    from sklearn.metrics import roc_curve
    fpr, tpr, threshold = roc_curve(labels, outputs, pos_label=1)
    fnr = 1 - tpr
    return fpr[np.nanargmin(np.absolute((fnr - fpr)))]
