# This is a sample Python script.
# This is a sample Python script.
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
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
import os
import argparse

'''
LAST UPDATED 11/10/2021, lsdr
'''

## Process images in parallel

## folder "Data" images
## folder "excel" excel file , whatever is there is the file
## get the classes from the excel file
## folder "Documents" readme file


parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
parser.add_argument("--split", default=False, type=str, required=True)  # validate, test, train

args = parser.parse_args()

PATH = args.path
DATA_DIR = args.path + os.path.sep + 'Data' + os.path.sep
SPLIT = args.split


BATCH_SIZE = 30
LR = 0.001

## Image processing
CHANNELS = 3
IMAGE_SIZE = 100

NICKNAME = "Gloria"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5


#---- Define the model ---- #
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#
#
#     def forward(self, x):
#         return #res model
#---- Define the model ---- #
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pad1 = nn.ZeroPad2d(2)

        self.conv2 = nn.Conv2d(16, 128, (3, 3))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(128, OUTPUTS_a)  # OUTPUTS_a must match the number of output classes
        self.act = torch.relu

    def forward(self, x):
        x = self.pad1(self.convnorm1(self.act(self.conv1(x))))
        x = self.act(self.conv2(self.act(x)))
        return self.linear(self.global_avg_pool(x).view(-1, 128))



## ------------------ Data Loadaer definition

# class Dataset(data.Dataset):
#     '''
#     From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
#     '''
#     def __init__(self, list_IDs, type_data, target_type):
#         #Initialization'
#
#     def __len__(self):
#         #Denotes the total number of samples'
#         return len(self.list_IDs)
#
#     def __getitem__(self, index):
#         #Generates one sample of data'
#         # Select sample
#
#         return X, y
# class Dataset(data.Dataset):
#     '''
#     Dataset class for loading image data and targets.
#     '''
#     def __init__(self, list_IDs, type_data):
#         # Initialization
#         self.list_IDs = list_IDs  # IDs correspond to indices of the Excel file
#         self.type_data = type_data
#
#     def __len__(self):
#         # Returns the total number of samples
#         return len(self.list_IDs)
#
#     def __getitem__(self, index):
#         # Generates one sample of data
#         ID = self.list_IDs[index]
#
#         # Load labels
#         if self.type_data == 'test':
#             y = xdf_dset_test.loc[ID, 'target_class']
#         else:
#             y = xdf_dset.loc[ID, 'target_class']
#
#         # Convert labels to tensor
#         labels_ohe = [int(e) for e in y.split(",")]
#         y = torch.FloatTensor(labels_ohe)
#
#         # Load image
#         if self.type_data == 'test':
#             file_path = os.path.join(DATA_DIR, xdf_dset_test.loc[ID, 'id'])
#         else:
#             file_path = os.path.join(DATA_DIR, xdf_dset.loc[ID, 'id'])
#
#         img = cv2.imread(file_path)  # Load image
#         img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))  # Resize to IMAGE_SIZE
#         img = torch.FloatTensor(img)  # Convert to tensor
#         img = img.permute(2, 0, 1)  # Rearrange to (Channels, Height, Width)
#
#         return img, y
class Dataset(data.Dataset):
    '''
    Dataset class for loading image data and targets.
    '''
    def __init__(self, list_IDs, type_data, target_type=None):  # Added `target_type` with a default value
        # Initialization
        self.list_IDs = list_IDs
        self.type_data = type_data
        self.target_type = target_type  # Store target_type if needed

    def __len__(self):
        # Returns the total number of samples
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Generates one sample of data
        ID = self.list_IDs[index]

        # Load labels
        if self.type_data == 'test':
            y = xdf_dset_test.loc[ID, 'target_class']
        else:
            y = xdf_dset.loc[ID, 'target_class']

        # Convert labels to tensor
        labels_ohe = [int(e) for e in y.split(",")]
        y = torch.FloatTensor(labels_ohe)

        # Load image
        if self.type_data == 'test':
            file_path = os.path.join(DATA_DIR, xdf_dset_test.loc[ID, 'id'])
        else:
            file_path = os.path.join(DATA_DIR, xdf_dset.loc[ID, 'id'])

        img = cv2.imread(file_path)  # Load image
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))  # Resize to IMAGE_SIZE
        img = torch.FloatTensor(img)  # Convert to tensor
        img = img.permute(2, 0, 1)  # Rearrange to (Channels, Height, Width)

        return img, y



def read_data(target_type):
    ## Only the training set
    ## xdf_dset ( data set )
    ## read the data data from the file

    # ---------------------- Parameters for the data loader --------------------------------

    list_of_ids_test = list(xdf_dset_test.index)


    # Datasets
    partition = {
        'test' : list_of_ids_test
    }

    # Data Loader

    params = {'batch_size': BATCH_SIZE,
              'shuffle': False}

    test_set = Dataset(partition['test'], 'test', target_type)
    test_generator = data.DataLoader(test_set, **params)

    ## Make the channel as a list to make it variable

    return test_generator


# def model_definition(pretrained=False):
#     # Define a Keras sequential model
#     # Compile the model
#
#     if pretrained == True:
#         ## pretrained model
#     else:
#         model = CNN()
#
#     model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))
#     model = model.to(device)
#
#     print(model, file=open('summary_{}.txt'.format(NICKNAME), "w"))
#
#     criterion = nn.BCEWithLogitsLoss()
#
#     return model, criterion

def model_definition(pretrained=False):
    # Define a model
    if pretrained == True:
        # Load a pre-trained model
        model = models.resnet18(pretrained=True)  # Example: ResNet18
        model.fc = nn.Linear(model.fc.in_features, OUTPUTS_a)  # Adjust the final layer
    else:
        # Use custom CNN model
        model = CNN()

    # Load the trained model weights
    model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))
    model = model.to(device)

    # Print model summary to a file
    print(model, file=open('summary_{}.txt'.format(NICKNAME), "w"))

    # Define loss function
    criterion = nn.BCEWithLogitsLoss()

    return model, criterion


# def test_model(test_ds, list_of_metrics, list_of_agg , pretrained = False):
#     # Create the test instructions to
#     # Load the model
#     # Create the loop to validate the data
#     # You can use a dataloader
#
#     model, criterion  = model_definition(pretrained)
#     model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))
#
#     pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
#
#     model.eval()
#
#
#     #  Create the evalution
#     #  Run the statistics
#     #  Save the results in the Excel file
#     # Remember to wirte a string con the result (NO A COLUMN FOR each )
#
#
#     xdf_dset_test['results'] = xfinal_pred_labels
#     xdf_dset_test.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)

# def test_model(test_ds, list_of_metrics, list_of_agg, pretrained=False):
#     # Create the test instructions to
#     # Load the model
#     model, criterion = model_definition(pretrained)
#     model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))
#
#     pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
#
#     model.eval()
#
#     # To store final predictions
#     xfinal_pred_labels = []
#
#     # Create the evaluation loop
#     with torch.no_grad():
#         for xdata, _ in tqdm(test_ds, desc="Testing"):
#             xdata = xdata.to(device)
#
#             # Get model predictions
#             output = model(xdata)
#
#             # Convert logits to probabilities
#             pred = torch.sigmoid(output).cpu().numpy()
#
#             # Apply threshold to generate binary predictions (e.g., 0 or 1)
#             pred_labels = (pred >= THRESHOLD).astype(int)
#
#             # Convert binary predictions to comma-separated strings
#             for row in pred_labels:
#                 xfinal_pred_labels.append(",".join(map(str, row)))
#
#     # Save predictions to the DataFrame
#     xdf_dset_test['results'] = xfinal_pred_labels
#
#     # Save the DataFrame to an Excel file
#     xdf_dset_test.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)
#     print(f"Results saved to results_{NICKNAME}.xlsx")

def save_model_summary(model):
    """
    Save the model architecture to a summary file.
    """
    with open('summary_{}.txt'.format(NICKNAME), 'w') as summary_file:
        print(model, file=summary_file)


# def test_model(test_ds, list_of_metrics, list_of_agg, pretrained=False):
#     # Load the model and criterion
#     model, criterion = model_definition(pretrained)
#     model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))
#
#     # Initialize variables for metrics
#     pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
#     total_loss = 0
#     steps = 0
#
#     model.eval()
#
#     # To store final predictions
#     xfinal_pred_labels = []
#
#     # Evaluation loop
#     with torch.no_grad():
#         for xdata, xtarget in tqdm(test_ds, desc="Testing"):
#             xdata, xtarget = xdata.to(device), xtarget.to(device)
#
#             # Get model predictions
#             output = model(xdata)
#
#             # Calculate BCE loss
#             loss = criterion(output, xtarget)
#             total_loss += loss.item()
#             steps += 1
#
#             # Convert logits to probabilities
#             pred = torch.sigmoid(output).cpu().numpy()
#
#             # Apply threshold to generate binary predictions (e.g., 0 or 1)
#             pred_labels = (pred >= THRESHOLD).astype(int)
#
#             # Store predictions
#             pred_logits = np.vstack((pred_logits, pred))
#             real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))
#
#             # Convert binary predictions to comma-separated strings
#             for row in pred_labels:
#                 xfinal_pred_labels.append(",".join(map(str, row)))
#
#     # Compute average loss
#     avg_loss = total_loss / steps
#
#     # Save predictions to the DataFrame
#     xdf_dset_test['results'] = xfinal_pred_labels
#     xdf_dset_test.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)
#
#     # Evaluate metrics
#     pred_logits = pred_logits[1:]  # Remove the initial zero row
#     real_labels = real_labels[1:]
#     pred_labels = (pred_logits >= THRESHOLD).astype(int)
#
#     metrics_results = metrics_func(list_of_metrics, list_of_agg, real_labels, pred_labels)
#
#     # Print metrics
#     print(f"Binary Cross-Entropy Loss (Average): {avg_loss:.4f}")
#     for metric, value in metrics_results.items():
#         print(f"{metric}: {value:.4f}")
#
#     print(f"Results saved to results_{NICKNAME}.xlsx")

def test_model(test_ds, list_of_metrics, list_of_agg, pretrained=False):
    # Load the model and criterion
    model, criterion = model_definition(pretrained)
    model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))

    # Save model summary
    save_model_summary(model)

    # Initialize variables for metrics
    pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))
    total_loss = 0
    steps = 0

    model.eval()

    # To store final predictions
    xfinal_pred_labels = []

    # Evaluation loop
    with torch.no_grad():
        for xdata, xtarget in tqdm(test_ds, desc="Testing"):
            xdata, xtarget = xdata.to(device), xtarget.to(device)

            # Get model predictions
            output = model(xdata)

            # Calculate BCE loss
            loss = criterion(output, xtarget)
            total_loss += loss.item()
            steps += 1

            # Convert logits to probabilities
            pred = torch.sigmoid(output).cpu().numpy()

            # Apply threshold to generate binary predictions (e.g., 0 or 1)
            pred_labels = (pred >= THRESHOLD).astype(int)

            # Store predictions
            pred_logits = np.vstack((pred_logits, pred))
            real_labels = np.vstack((real_labels, xtarget.cpu().numpy()))

            # Convert binary predictions to comma-separated strings
            for row in pred_labels:
                xfinal_pred_labels.append(",".join(map(str, row)))

    # Compute average loss
    avg_loss = total_loss / steps

    # Save predictions to the DataFrame
    xdf_dset_test['results'] = xfinal_pred_labels
    xdf_dset_test.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)

    # Evaluate metrics
    pred_logits = pred_logits[1:]  # Remove the initial zero row
    real_labels = real_labels[1:]
    pred_labels = (pred_logits >= THRESHOLD).astype(int)

    metrics_results = metrics_func(list_of_metrics, list_of_agg, real_labels, pred_labels)

    # Print metrics
    print(f"Binary Cross-Entropy Loss (Average): {avg_loss:.4f}")
    for metric, value in metrics_results.items():
        print(f"{metric}: {value:.4f}")

    print(f"Results saved to results_{NICKNAME}.xlsx")
    print(f"Summary saved to summary_{NICKNAME}.txt")



def metrics_func(metrics, aggregates, y_true, y_pred):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        return res

    def hamming_metric(y_true, y_pred):
        res = hamming_loss(y_true, y_pred)
        return res

    xcont = 1
    xsum = 0
    res_dict = {}
    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
             # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet =accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet =matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet =hamming_metric(y_true, y_pred)
        else:
            xmet = 0

        res_dict[xm] = xmet

        xsum = xsum + xmet
        xcont = xcont +1

    if 'sum' in aggregates:
        res_dict['sum'] = xsum
    if 'avg' in aggregates and xcont > 0:
        res_dict['avg'] = xsum/xcont
    # Ask for arguments for each metric

    return res_dict

def process_target(target_type):
    '''
        1- Binary   target = (1,0)
        2- Multiclass  target = (1...n, text1...textn)
        3- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
    :return:
    '''

    dict_target = {}
    xerror = 0

    if target_type == 2:
        target = np.array(xdf_data['target'].apply( lambda x : x.split(",")))
        final_target = mlb.fit_transform(target)
        xfinal = []
        if len(final_target) ==0:
            xerror = 'Could not process Multilabel'
        else:
            class_names = mlb.classes_
            for i in range(len(final_target)):
                joined_string = ",".join( str(e) for e in final_target[i])
                xfinal.append(joined_string)
            xdf_data['target_class'] = xfinal

    if target_type == 1:
        xtarget = list(np.array(xdf_data['target'].unique()))
        le = LabelEncoder()
        le.fit(xtarget)
        final_target = le.transform(np.array(xdf_data['target']))
        class_names=(xtarget)
        xdf_data['target_class'] = final_target

    ## We add the column to the main dataset

    return class_names


# if __name__ == '__main__':
#
#     for file in os.listdir(PATH+os.path.sep + "excel"):
#         if file[-5:] == '.xlsx':
#             FILE_NAME = PATH+ os.path.sep+ "excel" + os.path.sep + file
#
#     # Reading and filtering Excel file
#     xdf_data = pd.read_excel(FILE_NAME)
#
#     ## Processing Train dataset
#     ## Target_type = 1  Multiclass   Target_type = 2 MultiLabel
#     class_names = process_target(target_type = 2)
#
#     ## Balancing classes , all groups have the same number of observations
#     xdf_dset_test= xdf_data[xdf_data["split"] == SPLIT].copy()
#
#     ## read_data creates the dataloaders, take target_type = 2
#
#     test_ds = read_data(target_type = 2)
#
#     OUTPUTS_a = len(class_names)
#
#     list_of_metrics = ['f1_macro']
#     list_of_agg = ['avg']
#
#     test_model(test_ds, list_of_metrics, list_of_agg, pretrained=False)

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
    class_names = process_target(target_type=2)

    # Filter the dataset for the specified split
    xdf_dset_test = xdf_data[xdf_data["split"] == SPLIT].copy()

    # Create the data loader
    test_ds = read_data(target_type=2)

    # Set the number of output classes
    OUTPUTS_a = len(class_names)

    # Define metrics
    list_of_metrics = ['f1_macro']
    list_of_agg = ['avg']

    # Run the testing function
    test_model(test_ds, list_of_metrics, list_of_agg, pretrained=False)
