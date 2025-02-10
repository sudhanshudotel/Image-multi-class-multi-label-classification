import random
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision.models import efficientnet_b2
from tqdm import tqdm
import os

# Configuration parameters
OR_PATH = os.getcwd()
os.chdir("..")
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
os.chdir(OR_PATH)

N_EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-4
IMAGE_SIZE = 260
CHANNELS = 3
NICKNAME = "Gloria"
THRESHOLD = 0.5
SAVE_MODEL = True

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Dataset class
class Dataset(data.Dataset):
    def __init__(self, list_IDs, type_data, target_type):
        self.type_data = type_data
        self.list_IDs = list_IDs
        self.target_type = target_type

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        y = xdf_dset.target_class.get(ID) if self.type_data == 'train' else xdf_dset_test.target_class.get(ID)
        labels_ohe = [int(e) for e in y.split(",")] if self.target_type == 2 else np.zeros(OUTPUTS_a)
        y = torch.FloatTensor(labels_ohe)

        file = DATA_DIR + (xdf_dset.id.get(ID) if self.type_data == 'train' else xdf_dset_test.id.get(ID))
        img = cv2.imread(file)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = torch.FloatTensor(img).permute(2, 0, 1)  # Rearrange to (Channels, Height, Width)
        return img, y

# Read data function
def read_data(target_type):
    list_of_ids = list(xdf_dset.index)
    list_of_ids_test = list(xdf_dset_test.index)

    params_train = {'batch_size': BATCH_SIZE, 'shuffle': True}
    params_test = {'batch_size': BATCH_SIZE, 'shuffle': False}

    train_set = Dataset(list_of_ids, 'train', target_type)
    train_loader = data.DataLoader(train_set, **params_train)

    test_set = Dataset(list_of_ids_test, 'test', target_type)
    test_loader = data.DataLoader(test_set, **params_test)

    return train_loader, test_loader

# Save model summary
def save_model_summary(model):
    with open(f'summary_{NICKNAME}.txt', "w") as f:
        print(model, file=f)

# Model definition
def model_definition():
    model = efficientnet_b2(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, OUTPUTS_a)  # Modify final layer
    # Ensure the classifier layer is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
    # model.summary()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    save_model_summary(model)
    return model, optimizer, criterion, scheduler

# for name, parma in model.parameters():
#     print(f"Name: {name}, Param: {parma.requires_grad}")
# Training and testing function
def train_and_test(train_loader, test_loader, save_on):
    model, optimizer, criterion, scheduler = model_definition()

    best_metric = 0
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{N_EPOCHS} [Training]"):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Average Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        test_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{N_EPOCHS} [Validation]"):
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                preds = (torch.sigmoid(outputs) > THRESHOLD).float()

                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        f1_macro = f1_score(all_targets, all_preds, average='macro')
        print(f"Epoch {epoch + 1}: Validation Loss: {avg_test_loss:.4f}, F1 Macro: {f1_macro:.4f}")

        scheduler.step(f1_macro)

        # Save model if metric improves
        if f1_macro > best_metric and SAVE_MODEL:
            best_metric = f1_macro
            torch.save(model.state_dict(), f"model_{NICKNAME}.pt")
            print(f"Model saved with F1 Macro: {f1_macro:.4f}")

if __name__ == '__main__':
    for file in os.listdir(PATH + os.path.sep + "excel"):
        if file.endswith('.xlsx'):
            FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    xdf_data = pd.read_excel(FILE_NAME)
    class_names = xdf_data['target'].str.split(',').explode().unique()
    OUTPUTS_a = len(class_names)

    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()
    xdf_dset_test = xdf_data[xdf_data["split"] == 'test'].copy()

    train_loader, test_loader = read_data(target_type=2)
    train_and_test(train_loader, test_loader, save_on='f1_macro')
