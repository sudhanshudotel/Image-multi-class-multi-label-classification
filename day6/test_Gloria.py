# import random
# from sklearn.preprocessing import MultiLabelBinarizer
# import cv2
# import pandas as pd
# from sklearn.metrics import f1_score
# import torch
# import torch.nn as nn
# import numpy as np
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from torchvision.models import inception_v3
# from tqdm import tqdm
# import os
# import argparse
#
# # Configuration parameters
# BATCH_SIZE = 32
# IMAGE_SIZE = 299
# NICKNAME = "Gloria"
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
# # Predefined optimal thresholds for each class
# OPTIMAL_THRESHOLDS = {
#     "class4": 0.5918,
#     "class11": 0.7074,
#     "class17": 0.8678,
#     "class10": 0.3023,
#     "class14": 0.9927,
#     "class13": 0.2647,
#     "class6": 0.9808,
#     "class9": 0.6657,
#     "class8": 0.8448,
#     "class2": 0.8137,
#     "class1": 0.4411,
#     "class15": 0.9831,
#     "class12": 0.4765,
#     "class3": 0.4295,
#     "class5": 0.9706,
#     "class7": 0.9986,
#     "class16": 0.3383,
# }
#
# # Dataset class
# class CustomDataset(Dataset):
#     def __init__(self, dataframe, data_dir, transforms=None):
#         self.dataframe = dataframe
#         self.data_dir = data_dir
#         self.transforms = transforms
#
#     def __len__(self):
#         return len(self.dataframe)
#
#     def __getitem__(self, index):
#         row = self.dataframe.iloc[index]
#         img_path = os.path.join(self.data_dir, row['id'])
#
#         if not img_path.endswith('.jpg'):
#             img_path += ".jpg"
#
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"[WARNING] Image not found or unreadable: {img_path}")
#             img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
#         else:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
#
#         label = torch.FloatTensor([int(e) for e in row['target_class'].split(",")])
#
#         if self.transforms:
#             img = self.transforms(img)
#
#         return img, label
#
# # Read data function
# def read_data(dataframe, data_dir):
#     transforms_ = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.ToTensor(),
#     ])
#     test_ds = CustomDataset(dataframe, data_dir, transforms=transforms_)
#     test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
#     return test_loader
#
# # Model definition
# def model_definition():
#     model = inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
#     model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
#     model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, OUTPUTS_a)
#     model.load_state_dict(torch.load(f'model_{NICKNAME}.pt', map_location=device))
#     model = model.to(device)
#
#     # Save model summary
#     with open(f'summary_{NICKNAME}.txt', 'w') as summary_file:
#         print(model, file=summary_file)
#
#     return model
#
# # Testing function with dynamic thresholds
# def test_model(test_loader):
#     model = model_definition()
#     model.eval()
#
#     pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
#     total_loss = 0
#     criterion = nn.BCEWithLogitsLoss()
#
#     with torch.no_grad():
#         for images, labels in tqdm(test_loader, desc="Testing"):
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#
#             preds_proba = torch.sigmoid(outputs).cpu().numpy()
#             pred_logits = np.vstack((pred_logits, preds_proba))
#             real_labels = np.vstack((real_labels, labels.cpu().numpy()))
#
#     avg_loss = total_loss / len(test_loader)
#     pred_logits = pred_logits[1:]
#     real_labels = real_labels[1:]
#
#     # Apply dynamic thresholds
#     pred_labels = np.zeros_like(pred_logits)
#     for i, class_name in enumerate(class_names):
#         threshold = OPTIMAL_THRESHOLDS.get(class_name, 0.5)  # Default to 0.5 if no optimal threshold
#         pred_labels[:, i] = (pred_logits[:, i] >= threshold).astype(int)
#
#     f1_macro = f1_score(real_labels, pred_labels, average='macro')
#     print(f"Binary Cross-Entropy Loss: {avg_loss:.4f}")
#     print(f"F1 Macro: {f1_macro:.4f}")
#
#     return pred_labels
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Test a trained model for multi-label classification.")
#     parser.add_argument("--path", type=str, required=True, help="Path to the data directory (e.g., /home/ubuntu/Exam2-v6)")
#     parser.add_argument("--split", type=str, required=True, help="Dataset split to test (e.g., 'test')")
#     args = parser.parse_args()
#
#     PATH = args.path
#     DATA_DIR = os.path.join(PATH, 'Data')
#     SPLIT = args.split
#
#     for file in os.listdir(os.path.join(PATH, "excel")):
#         if file.endswith('.xlsx'):
#             FILE_NAME = os.path.join(PATH, "excel", file)
#
#     df = pd.read_excel(FILE_NAME)
#
#     # Verify DataFrame columns
#     required_columns = ['id', 'target', 'split', 'target_class']
#     if not all(col in df.columns for col in required_columns):
#         raise ValueError(f"Missing required columns in the DataFrame: {required_columns}")
#
#     class_names = df['target'].str.split(',').explode().unique()
#     OUTPUTS_a = len(class_names)
#
#     test_df = df[df['split'] == SPLIT]
#     test_loader = read_data(test_df, DATA_DIR)
#     predictions = test_model(test_loader)
#
#     test_df['predictions'] = [" ".join(map(str, row)) for row in predictions]
#     test_df.to_excel(f'results_{NICKNAME}.xlsx', index=False)


# import random
# from sklearn.preprocessing import MultiLabelBinarizer
# import cv2
# import pandas as pd
# from sklearn.metrics import f1_score
# import torch
# import torch.nn as nn
# import numpy as np
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from torchvision.models import inception_v3
# from tqdm import tqdm
# import os
# import argparse
#
# # Configuration parameters
# BATCH_SIZE = 32
# IMAGE_SIZE = 299
# NICKNAME = "Gloria"
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
# # Predefined optimal thresholds for each class
# OPTIMAL_THRESHOLDS = {
#     "class4": 0.5918,
#     "class11": 0.7074,
#     "class17": 0.8678,
#     "class10": 0.3023,
#     "class14": 0.9927,
#     "class13": 0.2647,
#     "class6": 0.9808,
#     "class9": 0.6657,
#     "class8": 0.8448,
#     "class2": 0.8137,
#     "class1": 0.4411,
#     "class15": 0.9831,
#     "class12": 0.4765,
#     "class3": 0.4295,
#     "class5": 0.9706,
#     "class7": 0.9986,
#     "class16": 0.3383,
# }
#
# # Dataset class
# class CustomDataset(Dataset):
#     def __init__(self, dataframe, data_dir, transforms=None):
#         self.dataframe = dataframe
#         self.data_dir = data_dir
#         self.transforms = transforms
#
#     def __len__(self):
#         return len(self.dataframe)
#
#     def __getitem__(self, index):
#         row = self.dataframe.iloc[index]
#         img_path = os.path.join(self.data_dir, row['id'])
#
#         if not img_path.endswith('.jpg'):
#             img_path += ".jpg"
#
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"[WARNING] Image not found or unreadable: {img_path}")
#             img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
#         else:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
#
#         label = torch.FloatTensor([int(e) for e in row['target_class'].split(",")])
#
#         if self.transforms:
#             img = self.transforms(img)
#
#         return img, label
#
# # Read data function
# def read_data(dataframe, data_dir):
#     transforms_ = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.ToTensor(),
#     ])
#     test_ds = CustomDataset(dataframe, data_dir, transforms=transforms_)
#     test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
#     return test_loader
#
# # Model definition
# def model_definition():
#     model = inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
#     model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
#     model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, OUTPUTS_a)
#     model.load_state_dict(torch.load(f'model_{NICKNAME}.pt', map_location=device))
#     model = model.to(device)
#
#     # Save model summary
#     with open(f'summary_{NICKNAME}.txt', 'w') as summary_file:
#         print(model, file=summary_file)
#
#     return model
#
# # Testing function with dynamic thresholds
# def test_model(test_loader):
#     model = model_definition()
#     model.eval()
#
#     pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
#     total_loss = 0
#     criterion = nn.BCEWithLogitsLoss()
#
#     with torch.no_grad():
#         for images, labels in tqdm(test_loader, desc="Testing"):
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#
#             preds_proba = torch.sigmoid(outputs).cpu().numpy()
#             pred_logits = np.vstack((pred_logits, preds_proba))
#             real_labels = np.vstack((real_labels, labels.cpu().numpy()))
#
#     avg_loss = total_loss / len(test_loader)
#     pred_logits = pred_logits[1:]
#     real_labels = real_labels[1:]
#
#     # Apply dynamic thresholds
#     pred_labels = np.zeros_like(pred_logits)
#     for i, class_name in enumerate(class_names):
#         threshold = OPTIMAL_THRESHOLDS.get(class_name, 0.5)  # Default to 0.5 if no optimal threshold
#         pred_labels[:, i] = (pred_logits[:, i] >= threshold).astype(int)
#
#     f1_macro = f1_score(real_labels, pred_labels, average='macro')
#     print(f"Binary Cross-Entropy Loss: {avg_loss:.4f}")
#     print(f"F1 Macro: {f1_macro:.4f}")
#
#     return pred_labels
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Test a trained model for multi-label classification.")
#     parser.add_argument("--path", type=str, required=True, help="Path to the data directory (e.g., /home/ubuntu/Exam2-v6)")
#     parser.add_argument("--split", type=str, required=True, help="Dataset split to test (e.g., 'test')")
#     args = parser.parse_args()
#
#     PATH = args.path
#     DATA_DIR = os.path.join(PATH, 'Data')
#     SPLIT = args.split
#
#     for file in os.listdir(os.path.join(PATH, "excel")):
#         if file.endswith('.xlsx'):
#             FILE_NAME = os.path.join(PATH, "excel", file)
#
#     df = pd.read_excel(FILE_NAME)
#
#     # Verify DataFrame columns
#     required_columns = ['id', 'target', 'split', 'target_class']
#     if not all(col in df.columns for col in required_columns):
#         raise ValueError(f"Missing required columns in the DataFrame: {required_columns}")
#
#     class_names = df['target'].str.split(',').explode().unique()
#     OUTPUTS_a = len(class_names)
#
#     # Prepare the test DataFrame and DataLoader
#     test_df = df[df['split'] == SPLIT]
#     test_loader = read_data(test_df, DATA_DIR)
#
#     # Get predictions from the model
#     predictions = test_model(test_loader)
#
#     # Convert predictions to a string format
#     xfinal_pred_labels = [",".join(map(str, row)) for row in predictions]
#     test_df['results'] = xfinal_pred_labels
#
#     # Print statement for verification
#     print("Sample of the prediction results:")
#     print(test_df[['id', 'results']].head())
#
#     # Save the results to an Excel file
#     test_df.to_excel(f'results_{NICKNAME}.xlsx', index=False)

import random
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3
from tqdm import tqdm
import os
import argparse

# Configuration parameters
BATCH_SIZE = 32
IMAGE_SIZE = 299
NICKNAME = "Gloria"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Predefined optimal thresholds for each class
# OPTIMAL_THRESHOLDS = {
#     "class4": 0.5918,
#     "class11": 0.7074,
#     "class17": 0.8678,
#     "class10": 0.3023,
#     "class14": 0.9927,
#     "class13": 0.2647,
#     "class6": 0.9808,
#     "class9": 0.6657,
#     "class8": 0.8448,
#     "class2": 0.8137,
#     "class1": 0.4411,
#     "class15": 0.9831,
#     "class12": 0.4765,
#     "class3": 0.4295,
#     "class5": 0.9706,
#     "class7": 0.9986,
#     "class16": 0.3383,
# }

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

        if not img_path.endswith('.jpg'):
            img_path += ".jpg"

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Image not found or unreadable: {img_path}")
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        label = torch.FloatTensor([int(e) for e in row['target_class'].split(",")])

        if self.transforms:
            img = self.transforms(img)

        return img, label

# Read data function
def read_data(dataframe, data_dir):
    transforms_ = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    test_ds = CustomDataset(dataframe, data_dir, transforms=transforms_)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader

# Model definition
def model_definition(OUTPUTS_a):
    model = inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, OUTPUTS_a)
    model.load_state_dict(torch.load(f'model_{NICKNAME}.pt', map_location=device))
    model = model.to(device)

    # Save model summary
    with open(f'summary_{NICKNAME}.txt', 'w') as summary_file:
        print(model, file=summary_file)

    return model

# Testing function with dynamic thresholds
# def test_model(test_loader, class_names, OUTPUTS_a):
#     model = model_definition(OUTPUTS_a)
#     model.eval()
#
#     pred_logits = np.empty((0, OUTPUTS_a))
#     real_labels = np.empty((0, OUTPUTS_a))
#     total_loss = 0
#     criterion = nn.BCEWithLogitsLoss()
#
#     with torch.no_grad():
#         for images, labels in tqdm(test_loader, desc="Testing"):
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#
#             preds_proba = torch.sigmoid(outputs).cpu().numpy()
#             pred_logits = np.vstack((pred_logits, preds_proba))
#             real_labels = np.vstack((real_labels, labels.cpu().numpy()))
#
#     avg_loss = total_loss / len(test_loader)
#
#     # Apply dynamic thresholds
#     pred_labels = np.zeros_like(pred_logits)
#     for i, class_name in enumerate(class_names):
#         threshold = OPTIMAL_THRESHOLDS.get(class_name, 0.5)  # Default to 0.5 if no optimal threshold
#         pred_labels[:, i] = (pred_logits[:, i] >= threshold).astype(int)
#
#     f1_macro = f1_score(real_labels, pred_labels, average='macro')
#     print(f"Binary Cross-Entropy Loss: {avg_loss:.4f}")
#     print(f"F1 Macro: {f1_macro:.4f}")
#
#     return pred_labels

# Testing function with a single threshold
def test_model(test_loader, class_names, OUTPUTS_a):
    model = model_definition(OUTPUTS_a)
    model.eval()

    pred_logits = np.empty((0, OUTPUTS_a))
    real_labels = np.empty((0, OUTPUTS_a))
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds_proba = torch.sigmoid(outputs).cpu().numpy()
            pred_logits = np.vstack((pred_logits, preds_proba))
            real_labels = np.vstack((real_labels, labels.cpu().numpy()))

    avg_loss = total_loss / len(test_loader)

    # Apply a single threshold of 0.5 across all classes
    pred_labels = (pred_logits >= 0.5).astype(int)

    f1_macro = f1_score(real_labels, pred_labels, average='macro')
    print(f"Binary Cross-Entropy Loss: {avg_loss:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")

    return pred_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a trained model for multi-label classification.")
    parser.add_argument("--path", type=str, required=True, help="Path to the data directory (e.g., /home/ubuntu/Exam2-v6)")
    parser.add_argument("--split", type=str, required=True, help="Dataset split to test (e.g., 'test')")
    args = parser.parse_args()

    PATH = args.path
    DATA_DIR = os.path.join(PATH, 'Data')
    SPLIT = args.split

    # Find the Excel file in the specified path
    excel_path = os.path.join(PATH, "excel")
    excel_files = [f for f in os.listdir(excel_path) if f.endswith('.xlsx')]
    if not excel_files:
        raise FileNotFoundError("No Excel file found in the specified directory.")
    FILE_NAME = os.path.join(excel_path, excel_files[0])

    df = pd.read_excel(FILE_NAME)

    # Verify DataFrame columns
    required_columns = ['id', 'target', 'split', 'target_class']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns in the DataFrame: {required_columns}")

    # Extract unique class names
    class_names = df['target'].str.split(',').explode().unique()
    class_names = sorted(class_names)  # Optional: sort class names to maintain consistent order
    OUTPUTS_a = len(class_names)

    # Prepare the test DataFrame and DataLoader
    test_df = df[df['split'] == SPLIT].reset_index(drop=True)
    test_loader = read_data(test_df, DATA_DIR)

    # Get predictions from the model
    predictions = test_model(test_loader, class_names, OUTPUTS_a)

    # Convert predictions to a string format
    xfinal_pred_labels = [",".join(map(str, row.astype(int))) for row in predictions]
    test_df['results'] = xfinal_pred_labels

    # Print statement for verification
    print("Sample of the prediction results:")
    print(test_df[['id', 'results']].head())

    # Save the results to an Excel file
    test_df.to_excel(f'results_{NICKNAME}.xlsx', index=False)
