"""Préparation commune de FER2013 pour les notebooks du projet."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


DOSSIER_DONNEES = Path(__file__).resolve().parents[1] / "data"


def creer_dataloaders(batch_size=64):
    """Charge FER2013 et renvoie les lots d'entraînement, de validation et de test."""
    train_dir = DOSSIER_DONNEES / "train"
    test_dir = DOSSIER_DONNEES / "test"
    if not (train_dir.is_dir() and test_dir.is_dir()):
        import kagglehub

        kagglehub.dataset_download("msambare/fer2013", output_dir=str(DOSSIER_DONNEES))

    transformation = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])
    donnees = datasets.ImageFolder(train_dir, transform=transformation)
    test = datasets.ImageFolder(test_dir, transform=transformation)

    taille_train = int(0.85 * len(donnees))
    train, validation = random_split(
        donnees,
        [taille_train, len(donnees) - taille_train],
        generator=torch.Generator().manual_seed(42),
    )
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(validation, batch_size=batch_size, shuffle=False),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )
