import random
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torchvision.models import inception_v3
from tqdm import tqdm
import os
import argparse

# Paths and configurations
BATCH_SIZE = 32
IMAGE_SIZE = 299
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
        y = xdf_dset_test.loc[ID, 'target_class']
        labels_ohe = [int(e) for e in y.split(",")]
        y = torch.FloatTensor(labels_ohe)

        file_path = os.path.join(DATA_DIR, xdf_dset_test.loc[ID, 'id'])
        img = cv2.imread(file_path)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = torch.FloatTensor(img).permute(2, 0, 1)
        return img, y

# Read data function
def read_data():
    list_of_ids_test = list(xdf_dset_test.index)
    params = {'batch_size': BATCH_SIZE, 'shuffle': False}
    test_set = Dataset(list_of_ids_test, 'test', target_type=2)
    test_loader = data.DataLoader(test_set, **params)
    return test_loader

# Model definition
def model_definition():
    # Load InceptionV3 with pre-trained weights
    model = inception_v3(weights="IMAGENET1K_V1", aux_logits=True)

    # Adjust the number of output classes for both the main classifier and auxiliary classifier
    model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)  # Main classifier
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, OUTPUTS_a)  # Auxiliary classifier

    # Load the trained weights
    model.load_state_dict(torch.load(f'model_{NICKNAME}.pt', map_location=device))
    model = model.to(device)

    # Save model summary
    with open(f'summary_{NICKNAME}.txt', 'w') as summary_file:
        print(model, file=summary_file)

    return model


# Testing function
def test_model(test_loader, list_of_metrics, list_of_agg):
    model = model_definition()
    model.eval()

    pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for xdata, xtarget in tqdm(test_loader, desc="Testing"):
            xdata, xtarget = xdata.to(device), xtarget.to(device)

            # During evaluation, the model only returns the main logits
            outputs = model(xdata)
            loss = criterion(outputs, xtarget)
            total_loss += loss.item()

            pred = torch.sigmoid(outputs).cpu().numpy()
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
    import argparse

    parser = argparse.ArgumentParser(description="Test a trained model for multi-label classification.")
    parser.add_argument("--path", type=str, required=True,
                        help="Path to the data directory (e.g., /home/ubuntu/Exam2-v6)")
    parser.add_argument("--split", type=str, required=True, help="Dataset split to test (e.g., 'test', 'validate')")
    args = parser.parse_args()

    # Set the data directory
    PATH = args.path
    DATA_DIR = os.path.join(PATH, 'Data') + os.path.sep
    SPLIT = args.split

    # Load the Excel file
    for file in os.listdir(os.path.join(PATH, "excel")):
        if file.endswith('.xlsx'):
            FILE_NAME = os.path.join(PATH, "excel", file)

    # Read and process the dataset
    xdf_data = pd.read_excel(FILE_NAME)
    class_names = xdf_data['target'].str.split(',').explode().unique()
    OUTPUTS_a = len(class_names)

    # Filter the dataset for the specified split
    xdf_dset_test = xdf_data[xdf_data["split"] == SPLIT].copy()

    # Create the data loader
    test_ds = read_data()

    # Define metrics
    list_of_metrics = ['f1_macro']
    list_of_agg = ['avg']

    # Run the testing function
    test_model(test_ds, list_of_metrics, list_of_agg)
