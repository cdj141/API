# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:32:32 2024

@author: 陈东杰
"""

import matplotlib.pyplot as plt 
import soundata
import librosa
import soundfile as sf
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

# Mapping of category names to classIDs
CLASS_NAME_TO_ID = {
    "air_conditioner": 0,
    "car_horn": 1,
    "children_playing": 2,
    "dog_bark": 3,
    "drilling": 4,
    "engine_idling": 5,
    "gun_shot": 6,
    "jackhammer": 7,
    "siren": 8,
    "street_music": 9,
}

# Load UrbanSound8K dataset
def load_urbansound8k():
    dataset = soundata.initialize("urbansound8k")
    dataset.validate()  # Validate data set integrity
    print("Dataset loaded and validated successfully!")
    return dataset

# Acquire audio data, divide training and test sets according to fold
def get_data_by_fold(dataset, fold, target_class_id):
    train_clips = []
    test_clips = []

    for clip_id in dataset.clip_ids:
        clip = dataset.clip(clip_id)
        audio, sr = clip.audio
        clip_fold = clip.fold  # Get fold information
        labels = clip.tags.labels if clip.tags and clip.tags.labels else []  # Get the list of tags

        # Mapping category names to numeric IDs
        class_id = CLASS_NAME_TO_ID[labels[0]] if labels and labels[0] in CLASS_NAME_TO_ID else None

        if clip_fold == fold:
            test_clips.append((audio, sr, class_id))
        else:
            train_clips.append((audio, sr, class_id))

    # Separate screening of target and noise audio
    train_target = [(audio, sr) for audio, sr, cid in train_clips if cid == target_class_id]
    train_noise = [(audio, sr) for audio, sr, cid in train_clips if cid != target_class_id]
    test_target = [(audio, sr) for audio, sr, cid in test_clips if cid == target_class_id]
    test_noise = [(audio, sr) for audio, sr, cid in test_clips if cid != target_class_id]

    return train_target, train_noise, test_target, test_noise

def normalize_audio(audio):
    """
    Normalises the audio signal to a maximum absolute amplitude of 1.0.

    Args.
        audio (numpy.ndarray): input audio data.

    Returns.
        numpy.ndarray: the normalised audio data.
    """
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude > 0:
        audio = audio / max_amplitude
    return audio

def create_mixed_audio(
    target_clips, noise_clips, fixed_length=5, sample_rate=44100,
    num_targets=1, num_noises=5, target_weight=3.0, noise_weight=0.5, random_seed=42
):
    if random_seed is not None:
        random.seed(random_seed)  # Setting the random seed
        np.random.seed(random_seed)

    mixed_audio = []
    target_audio = []
    fixed_samples = fixed_length * sample_rate  # fixed sample size (statistics)

    for idx in range(len(target_clips)):
        # Random selection of multiple target and noise audio
        selected_targets = random.sample(target_clips, min(num_targets, len(target_clips)))
        selected_noises = random.sample(noise_clips, min(num_noises, len(noise_clips)))

        # Initialise frequency domain mixing
        mixed_fft = np.zeros(fixed_samples, dtype=complex)
        target_fft_sum = np.zeros(fixed_samples, dtype=complex)

        # Accumulate the frequency domain components of multiple target audios
        for target, sr in selected_targets:
            target = adjust_audio_length(target, fixed_samples)

            # Enhance target audio amplitude
            target = target * 1.5

            # Fourier transform to frequency domain
            target_fft = np.fft.fft(target)

            # Use bandpass filters to preserve the core frequency range of the target audio
            freqs = np.fft.fftfreq(len(target_fft), d=1/sample_rate)
            band_pass_filter = (freqs > 500) & (freqs < 5000)
            target_fft = target_fft * band_pass_filter

            # Accumulate target audio
            target_fft_sum += target_fft * target_weight
            mixed_fft += target_fft * target_weight

        # Accumulate the frequency domain components of multiple noisy audios
        for noise, sr in selected_noises:
            noise = adjust_audio_length(noise, fixed_samples)

            # Fourier transform to frequency domain
            noise_fft = np.fft.fft(noise)

            # Use a low-pass filter to attenuate the high-frequency component of the noise
            freqs = np.fft.fftfreq(len(noise_fft), d=1/sample_rate)
            low_pass_filter = freqs < 500
            noise_fft = noise_fft * low_pass_filter

            # Cumulative Noise Audio
            mixed_fft += noise_fft * noise_weight

        # Fourier inversion back to the time domain
        mixed = np.fft.ifft(mixed_fft).real
        target_sum = np.fft.ifft(target_fft_sum).real

        # normalisation
        mixed = normalize_audio(mixed)
        target_sum = normalize_audio(target_sum)

        # Add to results list
        mixed_audio.append((mixed, sample_rate))
        target_audio.append((target_sum, sample_rate))

    return mixed_audio, target_audio

def adjust_audio_length(audio, target_length):
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        pad_width = target_length - len(audio)
        return np.pad(audio, (0, pad_width), mode="constant")
    return audio

# data set class
class AudioDataset(Dataset):
    def __init__(self, mixed_audio, target_audio, fixed_length=5, sample_rate=44100):
        self.mixed_audio = mixed_audio
        self.target_audio = target_audio
        self.fixed_length = fixed_length
        self.sample_rate = sample_rate
        self.fixed_samples = int(fixed_length * sample_rate)

    def __len__(self):
        return len(self.mixed_audio)

    def __getitem__(self, idx):
        mixed, sr = self.mixed_audio[idx]
        target, _ = self.target_audio[idx]

        mixed = self._adjust_length(mixed)
        target = self._adjust_length(target)

        return (
            torch.tensor(mixed).unsqueeze(0).float(),
            torch.tensor(target).unsqueeze(0).float(),
        )

    def _adjust_length(self, audio):
        if len(audio) > self.fixed_samples:
            return audio[:self.fixed_samples]
        elif len(audio) < self.fixed_samples:
            pad_width = self.fixed_samples - len(audio)
            return np.pad(audio, (0, pad_width), mode="constant")
        return audio

# Wave-U-Net model
class AdaptiveWaveUNet(nn.Module):
    def __init__(self, base_channels=16, num_layers=5, kernel_size=15):
        super(AdaptiveWaveUNet, self).__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.dim_match_layers = nn.ModuleList()

        # Dynamically build encoders
        for i in range(num_layers):
            in_channels = 1 if i == 0 else base_channels * (2 ** (i - 1))
            out_channels = base_channels * (2 ** i)
            self.encoders.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2)
            )

        # Dynamically build decoder and channel alignment layers
        for i in range(num_layers - 1, -1, -1):
            in_channels = base_channels * (2 ** i)
            skip_channels = base_channels * (2 ** i)
            out_channels = base_channels * (2 ** (i - 1)) if i > 0 else 1
            self.decoders.append(
                nn.ConvTranspose1d(
                    in_channels + skip_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    output_padding=1,
                )
            )
            self.dim_match_layers.append(nn.Conv1d(skip_channels, in_channels, kernel_size=1))

    def forward(self, x):
        skips = []

        # Encoder section
        for encoder in self.encoders:
            x = F.relu(encoder(x))
            skips.append(x)

        # Decoder section
        for i, decoder in enumerate(self.decoders):
            if i < len(skips):
                skip = skips[-(i + 1)]

                # Time Dimension Alignment Logic
                if skip.shape[2] != x.shape[2]:
                    min_len = min(skip.shape[2], x.shape[2])
                    skip = skip[:, :, :min_len]
                    x = x[:, :, :min_len]

                skip = self.dim_match_layers[i](skip)  # channel alignment
                x = torch.cat([x, skip], dim=1)
            x = F.relu(decoder(x))

        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = load_urbansound8k()
    target_class_id = 6  # The classID of the "gun_shot" category.

    num_folds = 10
    for fold in range(1, num_folds + 1):
        print(f"Processing fold {fold}...")
        train_target, train_noise, test_target, test_noise = get_data_by_fold(dataset, fold, target_class_id)

        print(f"Train target clips: {len(train_target)}, Train noise clips: {len(train_noise)}")
        print(f"Test target clips: {len(test_target)}, Test noise clips: {len(test_noise)}")

        if len(train_target) == 0 or len(train_noise) == 0:
            print(f"Skipping fold {fold} due to insufficient training data.")
            continue

        if len(test_target) == 0 or len(test_noise) == 0:
            print(f"Skipping fold {fold} due to insufficient testing data.")
            continue

        train_mixed_audio, train_target_audio = create_mixed_audio(train_target, train_noise)
        test_mixed_audio, test_target_audio = create_mixed_audio(test_target, test_noise)

        if len(train_mixed_audio) == 0 or len(test_mixed_audio) == 0:
            print(f"Skipping fold {fold} due to insufficient mixed audio.")
            continue

        train_dataset = AudioDataset(train_mixed_audio, train_target_audio)
        test_dataset = AudioDataset(test_mixed_audio, test_target_audio)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        model = AdaptiveWaveUNet(base_channels=16, num_layers=5, kernel_size=15).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        for epoch in range(5):
            model.train()
            total_train_loss = 0
            for mixed, target in train_loader:
                mixed, target = mixed.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(mixed)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Fold {fold}, Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for mixed, target in test_loader:
                mixed, target = mixed.to(device), target.to(device)
                output = model(mixed)
                total_test_loss += loss_fn(output, target).item()
        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Fold {fold}, Test Loss: {avg_test_loss:.4f}")

        # Save the first three mixes of the test set, separated audio and enhanced audio
        for idx in range(min(5, len(test_mixed_audio))):
            mixed_audio, sr = test_mixed_audio[idx]
            sf.write(f"test_mixed_audio_fold{fold}_{idx}.wav", mixed_audio, sr)

            # Separate and enhance the current sample
            example_mixed_tensor = torch.tensor(mixed_audio).unsqueeze(0).unsqueeze(0).float().to(device)
            separated_audio = model(example_mixed_tensor).squeeze().detach().cpu().numpy()
            enhanced_audio = librosa.effects.preemphasis(separated_audio)

            # Save separated and enhanced audio
            sf.write(f"test_separated_audio_fold{fold}_{idx}.wav", separated_audio, sr)
            sf.write(f"test_enhanced_audio_fold{fold}_{idx}.wav", enhanced_audio, sr)
            print(f"Fold {fold}, Test Sample {idx}: Mixed, separated, and enhanced audio saved.")

            # Visualisation of audio waveforms
            visualize_audio_waveforms(mixed_audio, separated_audio, enhanced_audio, sr, fold, idx)


def visualize_audio_waveforms(mixed_audio, separated_audio, enhanced_audio, sr, fold, idx):
    time_mixed = np.linspace(0, len(mixed_audio) / sr, num=len(mixed_audio))
    time_separated = np.linspace(0, len(separated_audio) / sr, num=len(separated_audio))
    time_enhanced = np.linspace(0, len(enhanced_audio) / sr, num=len(enhanced_audio))

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(time_mixed, mixed_audio)
    plt.title(f"Fold {fold}, Test Sample {idx}: Mixed Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 3)
    plt.plot(time_separated, separated_audio)
    plt.title(f"Fold {fold}, Test Sample {idx}: Enhanced Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 2)
    plt.plot(time_enhanced, enhanced_audio)
    plt.title(f"Fold {fold}, Test Sample {idx}: Separated Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(f"audio_waveform_fold{fold}_sample{idx}.png")
    plt.close()
    print(f"Waveform visualization saved for fold {fold}, sample {idx}.")


if __name__ == "__main__":
    main()