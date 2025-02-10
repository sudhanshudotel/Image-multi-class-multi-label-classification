import random
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision.models import inception_v3
from tqdm import tqdm
import os
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

# Configuration parameters
OR_PATH = os.getcwd()
os.chdir("..")
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
os.chdir(OR_PATH)

N_EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-4
IMAGE_SIZE = 299
CHANNELS = 3
NICKNAME = "Gloria"
THRESHOLD = 0.5
SAVE_MODEL = True

# Classes to augment
AUGMENT_CLASSES = ["class11", "class13", "class14", "class6", "class1", "class10", "class2", "class8", "class9", "class15", "class3", "class5", "class12", "class7", "class16"]
TARGET_SAMPLES = 200000
AUG_PER_IMAGE = 50

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Dynamic Threshold Optimization Function
def optimize_thresholds(y_true, y_pred_proba, class_names):
    optimal_thresholds = {}
    for i, class_name in enumerate(class_names):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_pred_proba[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_thresholds[class_name] = thresholds[optimal_idx]
        print(f"Class {class_name}: Optimal Threshold = {optimal_thresholds[class_name]:.4f}, F1-Score = {f1_scores[optimal_idx]:.4f}")
    return optimal_thresholds

# Validation function with threshold optimization
def validate_with_dynamic_thresholds(model, test_loader, criterion, class_names):
    model.eval()
    all_preds_proba, all_targets = [], []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Validation"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds_proba = torch.sigmoid(outputs).cpu().numpy()
            all_preds_proba.append(preds_proba)
            all_targets.append(targets.cpu().numpy())

    all_preds_proba = np.vstack(all_preds_proba)
    all_targets = np.vstack(all_targets)

    # Optimize thresholds
    optimal_thresholds = optimize_thresholds(all_targets, all_preds_proba, class_names)

    # Apply optimal thresholds
    pred_labels = np.zeros_like(all_preds_proba)
    for i, class_name in enumerate(class_names):
        pred_labels[:, i] = (all_preds_proba[:, i] >= optimal_thresholds[class_name]).astype(int)

    # Calculate F1-Score with optimized thresholds
    f1_macro = f1_score(all_targets, pred_labels, average='macro')
    print(f"Validation F1 Macro (Dynamic Thresholds): {f1_macro:.4f}")
    return optimal_thresholds, f1_macro

# Dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, data_dir, transforms=None):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img_path = os.path.join(self.data_dir, row['id'])

        # Ensure only one '.jpg' is appended, if necessary
        if not img_path.endswith('.jpg'):
            img_path += ".jpg"

        img = cv2.imread(img_path)

        # Ensure valid image
        if img is None:
            print(f"[WARNING] Image not found or unreadable: {img_path}")
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)  # Use blank image
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        label = torch.FloatTensor([int(e) for e in row['target_class'].split(",")])

        if self.transforms:
            img = self.transforms(img)

        return img, label


# Augmentation function
def augment_image(img, augmentations, count):
    augmented_images = []
    for _ in range(count):
        augmented_images.append(augmentations(img))
    return augmented_images

# Augmenting dataset
def augment_dataset(dataframe, data_dir, augment_classes, target_samples, aug_per_image):
    augmented_data = []
    augmentation_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    for _, row in dataframe.iterrows():
        if row['target'] in augment_classes:  # Replacing 'class' with 'target'
            current_samples = len(dataframe[dataframe['target'] == row['target']])  # Replacing 'class' with 'target'
            deficit = target_samples - current_samples
            if deficit <= 0:
                continue

            img_path = os.path.join(data_dir, row['id'])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transforms.ToTensor()(img)

            # Generate augmented images
            for aug_img in augment_image(img, augmentation_transforms, min(aug_per_image, deficit)):
                new_id = f"{row['id']}_aug_{len(augmented_data)}"
                augmented_data.append({
                    "id": new_id,
                    "target": row['target'],  # Replacing 'class' with 'target'
                    "target_class": row['target_class']
                })
                # Save augmented image
                aug_img_pil = transforms.ToPILImage()(aug_img)
                aug_img_pil.save(os.path.join(data_dir, new_id + ".jpg"))

    return pd.DataFrame(augmented_data)


# Weighted sampler
def create_weighted_sampler(dataframe):
    class_counts = dataframe['target'].value_counts()  # Replace 'class' with 'target'
    class_weights = 1.0 / class_counts
    sample_weights = dataframe['target'].map(class_weights).values  # Replace 'class' with 'target'
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


# Model definition
# def model_definition():
#     model = inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
#     model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
#     model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, OUTPUTS_a)
#
#     for param in model.parameters():
#         param.requires_grad = True
#
#     model = model.to(device)
#
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
#     criterion = nn.BCEWithLogitsLoss()
#     scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
#
#     return model, optimizer, criterion, scheduler

# Model definition
def model_definition():
    # Load InceptionV3 with pre-trained weights
    model = inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, OUTPUTS_a)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze Mixed_6x and Mixed_7x blocks
    for name, param in model.named_parameters():
        if "Mixed_6" in name or "Mixed_7" in name:  # Unfreeze Mixed_6 and Mixed_7 blocks
            param.requires_grad = True

    model = model.to(device)

    # Define optimizer, loss function, and scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    return model, optimizer, criterion, scheduler


# Training and testing function
# Training and testing function (Updated to include dynamic thresholds)
def train_and_test(train_loader, test_loader):
    model, optimizer, criterion, scheduler = model_definition()
    best_metric = 0
    optimal_thresholds = None

    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{N_EPOCHS} [Training]"):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs, aux_outputs = model(images)
            loss_main = criterion(outputs, targets)
            loss_aux = criterion(aux_outputs, targets)
            loss = loss_main + 0.4 * loss_aux

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Average Training Loss: {avg_train_loss:.4f}")

        # Validation with dynamic thresholds
        current_thresholds, f1_macro = validate_with_dynamic_thresholds(
            model, test_loader, criterion, class_names
        )

        scheduler.step(f1_macro)

        # Save the best model and thresholds
        if f1_macro > best_metric and SAVE_MODEL:
            best_metric = f1_macro
            optimal_thresholds = current_thresholds
            torch.save(model.state_dict(), f"model_{NICKNAME}.pt")
            print(f"Model saved with F1 Macro: {f1_macro:.4f}")

    return optimal_thresholds

# Main script
# Main script
if __name__ == '__main__':
    for file in os.listdir(PATH + os.path.sep + "excel"):
        if file.endswith('.xlsx'):
            FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    df = pd.read_excel(FILE_NAME)

    # Verify DataFrame columns
    required_columns = ['id', 'target', 'split', 'target_class']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns in the DataFrame: {required_columns}")

    class_names = df['target'].str.split(',').explode().unique()
    OUTPUTS_a = len(class_names)

    # Augment dataset
    augmented_df = augment_dataset(df, DATA_DIR, AUGMENT_CLASSES, TARGET_SAMPLES, AUG_PER_IMAGE)

    # Combine datasets
    full_df = pd.concat([df, augmented_df])

    # Create sampler
    sampler = create_weighted_sampler(full_df)

    # Create DataLoader for training
    train_ds = CustomDataset(full_df, DATA_DIR, transforms=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)

    # Create DataLoader for testing
    test_ds = CustomDataset(df[df['split'] == 'test'], DATA_DIR, transforms=transforms.ToTensor())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Train and test the model
    optimal_thresholds = train_and_test(train_loader, test_loader)
    print(f"Optimal Thresholds: {optimal_thresholds}")


