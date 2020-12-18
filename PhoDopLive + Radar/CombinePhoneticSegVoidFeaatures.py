import os
import numpy as np

# This file reads void and phoneme features and combine them together to create input data for NNs


# load data
data_path = ".\data"
training_set_description_path = ".\data\protocol_V2\ASVspoof2017_V2_train.trn.txt"

# load data
# each row is a sample
# label 1 for genuine data, 0 for recorded data
with open(os.path.join(data_path,'training_features_for_all_samples_PCA.npy'), 'rb') as f:
    training_featuures = np.load(f)
with open(os.path.join(data_path,'training_labels.npy'), 'rb') as f:
    training_labels = np.load(f)

f = open(training_set_description_path, "r")
training_set_description = f.read().split('\n')
f.close()

# void data column 0:97 is void feature 
# void data column 97 is label
with open(os.path.join(data_path,'void_feature_label_train.npy'), 'rb') as f:
    void_data = np.load(f)

# check labels. They should be the same
# print(np.sum(void_data[:,97].reshape(-1)- training_labels.reshape(-1)))

# print(training_featuures.shape)
# print(void_data.shape)

combined_training_features = np.concatenate((void_data[:,0:97], training_featuures), axis=1)

# save data
with open(os.path.join(data_path,'combined_void_phoneme_training_features.npy'), 'wb') as f:
    np.save(f, combined_training_features)

print("The phoneme-based features and void features are combined and saved in combined_void_phoneme_training_features.npy. The resulting data have dimension: " + str(combined_training_features.shape))