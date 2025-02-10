# import random
# from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
# import torch
# import torch.nn as nn
# import numpy as np
# from torch.utils import data
# from torchvision.models import efficientnet_b2
# from tqdm import tqdm
# import os
# import pandas as pd
# from sklearn.preprocessing import MultiLabelBinarizer
# import cv2
# import argparse
#
# # Argument parsing
# parser = argparse.ArgumentParser(description="Test a trained model for multi-label classification.")
# parser.add_argument("--path", type=str, required=True, help="Path to the data directory (e.g., /home/ubuntu/Exam2-v6)")
# parser.add_argument("--split", type=str, required=True, help="Dataset split to test (e.g., 'test', 'validate')")
# args = parser.parse_args()
#
# # Dynamic paths based on input arguments
# PATH = args.path
# DATA_DIR = os.path.join(PATH, "Data")
# FILE_NAME = os.path.join(PATH, "excel", "train_test.xlsx")
# SPLIT = args.split
#
# # Configuration
# THRESHOLD = 0.5
# BATCH_SIZE = 16
# IMAGE_SIZE = 260
# NICKNAME = "Gloria"
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
# # Dataset class
# class Dataset(data.Dataset):
#     def __init__(self, list_IDs, type_data, target_type):
#         self.type_data = type_data
#         self.list_IDs = list_IDs
#         self.target_type = target_type
#
#     def __len__(self):
#         return len(self.list_IDs)
#
#     def __getitem__(self, index):
#         ID = self.list_IDs[index]
#         y = xdf_dset_test.target_class.get(ID)
#         labels_ohe = [int(e) for e in y.split(",")]
#         y = torch.FloatTensor(labels_ohe)
#
#         file = os.path.join(DATA_DIR, xdf_dset_test.id.get(ID))
#         img = cv2.imread(file)
#         img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
#         img = torch.FloatTensor(img).permute(2, 0, 1)
#         return img, y
#
# # Read data
# def read_data():
#     list_of_ids_test = list(xdf_dset_test.index)
#     params = {'batch_size': BATCH_SIZE, 'shuffle': False}
#     test_set = Dataset(list_of_ids_test, 'test', target_type=2)
#     test_loader = data.DataLoader(test_set, **params)
#     return test_loader
#
# # Model Definition
# def model_definition():
#     model = efficientnet_b2(weights="EfficientNet_B2_Weights.IMAGENET1K_V1")
#     model.classifier[1] = nn.Linear(model.classifier[1].in_features, OUTPUTS_a)
#     model.load_state_dict(torch.load(f'model_{NICKNAME}.pt', map_location=device))
#     model = model.to(device)
#     return model
#
# # Testing function
# def test_model(test_loader):
#     model = model_definition()
#     model.eval()
#
#     pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
#     total_loss = 0
#     criterion = nn.BCEWithLogitsLoss()
#
#     with torch.no_grad():
#         for xdata, xtarget in tqdm(test_loader, desc="Testing"):
#             xdata, xtarget = xdata.to(device), xtarget.to(device)
#             output = model(xdata)
#             loss = criterion(output, xtarget)
#             total_loss += loss.item()
#
#             pred = torch.sigmoid(output).cpu().numpy()
#             pred_logits = np.vstack((pred_logits, pred))
#             real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))
#
#     avg_loss = total_loss / len(test_loader)
#     pred_logits = pred_logits[1:]
#     real_labels = real_labels[1:]
#     pred_labels = (pred_logits >= THRESHOLD).astype(int)
#
#     f1_macro = f1_score(real_labels, pred_labels, average='macro')
#     print(f"Binary Cross-Entropy Loss: {avg_loss:.4f}")
#     print(f"F1 Macro: {f1_macro:.4f}")
#
#     # Save predictions
#     xfinal_pred_labels = [",".join(map(str, row)) for row in pred_labels]
#     xdf_dset_test['results'] = xfinal_pred_labels
#     xdf_dset_test.to_excel(f'results_{NICKNAME}.xlsx', index=False)
#
# # if __name__ == '__main__':
# #     # File paths
# #     PATH = "/home/ubuntu/Exam2-v6"
# #     DATA_DIR = os.path.join(PATH, "Data")
# #     FILE_NAME = os.path.join(PATH, "excel", "train_test.xlsx")
# #
# #     xdf_data = pd.read_excel(FILE_NAME)
# #     class_names = xdf_data['target'].str.split(',').explode().unique()
# #     OUTPUTS_a = len(class_names)
# #
# #     xdf_dset_test = xdf_data[xdf_data["split"] == 'test'].copy()
# #
# #     test_loader = read_data()
# #     test_model(test_loader)
# if __name__ == '__main__':
#     # Parse arguments and set paths
#     parser = argparse.ArgumentParser(description="Test a trained model for multi-label classification.")
#     parser.add_argument("--path", type=str, required=True, help="Path to the data directory (e.g., /home/ubuntu/Exam2-v6)")
#     parser.add_argument("--split", type=str, required=True, help="Dataset split to test (e.g., 'test', 'validate')")
#     args = parser.parse_args()
#
#     PATH = args.path
#     DATA_DIR = os.path.join(PATH, "Data")
#     FILE_NAME = os.path.join(PATH, "excel", "train_test.xlsx")
#     SPLIT = args.split
#
#     xdf_data = pd.read_excel(FILE_NAME)
#     class_names = xdf_data['target'].str.split(',').explode().unique()
#     OUTPUTS_a = len(class_names)
#
#     xdf_dset_test = xdf_data[xdf_data["split"] == SPLIT].copy()
#
#     test_loader = read_data()
#     test_model(test_loader)

import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torchvision.models import efficientnet_b2
from tqdm import tqdm
import os
import argparse

'''
LAST UPDATED 11/10/2021, lsdr
'''

# Argument parsing
parser = argparse.ArgumentParser(description="Test a trained model for multi-label classification.")
parser.add_argument("--path", type=str, required=True, help="Path to the data directory (e.g., /home/ubuntu/Exam2-v6)")
parser.add_argument("--split", type=str, required=True, help="Dataset split to test (e.g., 'test', 'validate')")
args = parser.parse_args()

# Paths and configurations
PATH = args.path
DATA_DIR = os.path.join(PATH, 'Data')
FILE_NAME = os.path.join(PATH, 'excel', 'train_test.xlsx')
SPLIT = args.split

BATCH_SIZE = 16
IMAGE_SIZE = 260
NICKNAME = "Gloria"
THRESHOLD = 0.5
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Dataset class
class Dataset(data.Dataset):
    def __init__(self, list_IDs, type_data, target_type):
        self.list_IDs = list_IDs
        self.type_data = type_data
        self.target_type = target_type

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        y = xdf_dset_test.target_class.get(ID)
        labels_ohe = [int(e) for e in y.split(",")]
        y = torch.FloatTensor(labels_ohe)

        file = os.path.join(DATA_DIR, xdf_dset_test.id.get(ID))
        img = cv2.imread(file)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = torch.FloatTensor(img).permute(2, 0, 1)
        return img, y

# Read data
def read_data():
    list_of_ids_test = list(xdf_dset_test.index)
    params = {'batch_size': BATCH_SIZE, 'shuffle': False}
    test_set = Dataset(list_of_ids_test, 'test', target_type=2)
    test_loader = data.DataLoader(test_set, **params)
    return test_loader

# Model definition
def model_definition():
    model = efficientnet_b2(weights="EfficientNet_B2_Weights.IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, OUTPUTS_a)
    model.load_state_dict(torch.load(f'model_{NICKNAME}.pt', map_location=device))
    model = model.to(device)

    # Save model summary
    with open(f'summary_{NICKNAME}.txt', 'w') as summary_file:
        print(model, file=summary_file)

    return model

# Testing function
def test_model(test_loader):
    model = model_definition()
    model.eval()

    pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for xdata, xtarget in tqdm(test_loader, desc="Testing"):
            xdata, xtarget = xdata.to(device), xtarget.to(device)
            output = model(xdata)
            loss = criterion(output, xtarget)
            total_loss += loss.item()

            pred = torch.sigmoid(output).cpu().numpy()
            pred_logits = np.vstack((pred_logits, pred))
            real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

    avg_loss = total_loss / len(test_loader)
    pred_logits = pred_logits[1:]
    real_labels = real_labels[1:]
    pred_labels = (pred_logits >= THRESHOLD).astype(int)

    f1_macro = f1_score(real_labels, pred_labels, average='macro')
    print(f"Binary Cross-Entropy Loss: {avg_loss:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")

    # Save predictions
    xfinal_pred_labels = [",".join(map(str, row)) for row in pred_labels]
    xdf_dset_test['results'] = xfinal_pred_labels
    xdf_dset_test.to_excel(f'results_{NICKNAME}.xlsx', index=False)

if __name__ == '__main__':
    # Load dataset
    xdf_data = pd.read_excel(FILE_NAME)
    class_names = xdf_data['target'].str.split(',').explode().unique()
    OUTPUTS_a = len(class_names)

    xdf_dset_test = xdf_data[xdf_data["split"] == SPLIT].copy()

    # Create data loader and test the model
    test_loader = read_data()
    test_model(test_loader)
