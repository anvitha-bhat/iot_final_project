import torch
from torch import Tensor
from torch.nn import Linear, MSELoss, functional as F
from torch.optim import SGD
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import classification_report
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# This file runs a pre-trained audio classifier on the saved feature file of ASV spoof 2017 dataset and report results


# load data
data_path = ".\data"

# each row is a sample
# label 1 for genuine data, 0 for recorded data
print("loading test data")
with open(os.path.join(data_path,'combined_void_phoneme_training_features.npy'), 'rb') as f:
    features = np.load(f)
with open(os.path.join(data_path,'training_labels.npy'), 'rb') as f:
    labels = np.load(f)
    
# standardization
scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=19)

# convert into tensor
Xtest=torch.from_numpy(X_test).float().to(device)
ytest=torch.from_numpy(y_test).to(device)



# define neural networks

# This is a basic 2-layer fully connected network
# It should be broad to push it into overparameterization regime
# best parameters so far: 70000 hidden neurons, 5e-3 learning rate, 99.5% test accuracy

# N is number of samples; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        # define layers and activation function
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H).to(device)
        self.linear2 = torch.nn.Linear(H, D_out).to(device)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # define forward pass
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        y_pred = self.linear2(x)
        return y_pred
    
    
class TenLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TenLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H).to(device)
        self.linear2 = torch.nn.Linear(H, H).to(device)
        self.linear3 = torch.nn.Linear(H, H).to(device)
        self.linear4 = torch.nn.Linear(H, H).to(device)
        self.linear5 = torch.nn.Linear(H, H).to(device)
        self.linear6 = torch.nn.Linear(H, H).to(device)
        self.linear7 = torch.nn.Linear(H, H).to(device)
        self.linear8 = torch.nn.Linear(H, H).to(device)
        self.linear9 = torch.nn.Linear(H, H).to(device)
        self.linear10 = torch.nn.Linear(H, D_out).to(device)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.relu(self.linear3(x))
        x = self.dropout(x)
        x = self.relu(self.linear4(x))
        x = self.dropout(x)
        x = self.relu(self.linear5(x))
        x = self.dropout(x)
        x = self.relu(self.linear6(x))
        x = self.dropout(x)
        x = self.relu(self.linear7(x))
        x = self.dropout(x)
        x = self.relu(self.linear8(x))
        x = self.dropout(x)
        x = self.relu(self.linear9(x))
        x = self.dropout(x)
        y_pred = self.linear10(x)
        return y_pred



# load pre-trained model from disk
f = open("results.pkl", "rb")
print("loading pre-trained model from disk")
results = pickle.load(f)
f.close()


acc_training, acc_test, model = results[("TwoLayerNet", 1e-2)]

# run the pre-trained model to classify test set
print("running the pre-trained model to classify test set")
y_pred = model(Xtest)
y_pred_tag = torch.round(torch.sigmoid(y_pred))

# produce detailed report
# False Positive Rate  = 1 - True negative rate
# False negative rate = 1 - True positive rate
target_names = ['genuine', 'recorded']
print("test results:")
print(classification_report(y_test, y_pred_tag.cpu().detach().numpy(), target_names=target_names, digits=4))