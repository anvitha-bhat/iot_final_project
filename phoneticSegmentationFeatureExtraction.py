import pocketsphinx
from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *
from pocketsphinx import Pocketsphinx, get_model_path, get_data_path
import os
import numpy as np
from scipy.io import wavfile
from sklearn import preprocessing
from python_speech_features import mfcc, logfbank
import matplotlib.pyplot as plt

# phonetic segmentation using pocketsphiinx
# feature extraction: MFCC + filterbank energy + Void

model_path = get_model_path()
training_data_path = ".\data\ASVspoof2017_V2_train"
dev_data_path = ".\data\ASVspoof2017_V2_dev"
data_path = ".\data"
dictionary = "ASVspoof2017_1.dict"
training_set_description_path = ".\data\protocol_V2\ASVspoof2017_V2_train.trn.txt"
dev_set_description_path = ".\data\protocol_V2\ASVspoof2017_V2_dev.trl.txt"
fileName = "T_1000003.wav"
phonemes_path = ".\data\PasswordPhonemes.txt"

f = open(phonemes_path, "r")
all_possible_phonemes =  list(set(f.read().split()))
f.close()
all_possible_phonemes.sort()
print("all possible phonemes: ", all_possible_phonemes)


# generate phoneme to index mapping for creating features
# group similar phonemes together. For example, 'AA' and 'AH'.
# some elements are set to zero because they are extremely unfrequent
phoneme_to_index = {'AA':0, 'AE':0, 'AH':0, 'AO':0, 'AW':0, 'AY':0, 'B':1, 'CH':2, 'D':3, 'DH':3, 'EH':4, 'ER':4, 
                    'EY':4, 'F':-1, 'G':-1, 'HH':-1, 'IH':8, 'IY':8, 'JH':-1, 'K':10, 'L':11, 'M':12, 'N':12, 'NG':12, 
                    'OW':14, 'OY':14, 'P':13, 'R':9, 'S':7, 'SH':7, 'T':6, 'TH':16, 'UW':-1, 'V':15, 'W':5, 
                    'Y':17, 'Z':7, 'ZH':7}

# read data description file
f = open(training_set_description_path, "r")
training_set_description = f.read().split('\n')
f.close()

f = open(dev_set_description_path, "r")
dev_set_description = f.read().split('\n')
f.close()


# # count occurence of each phoneme
# counts = {}
# for p in all_possible_phonemes:
#     counts[p] = 0
# # add occurence count
# for record in training_set_description:
#     if len(record)>40:
#         dictionary = "ASVspoof2017_"+ str(int(record.split(" ")[3][1:])) + ".dict"
#         f = open(os.path.join(data_path, dictionary), "r")
#         dict =  f.read()
#         f.close()
#         for line in dict.split("\n"):
#             for w in line.split()[1:]:
#                 counts[w] += 1
# print(counts)



# feature extraction through mfcc and log filter bank energy
def extract_mfcc_filterbank_features(file_path, show=False):

    sampling_frequency, signal = wavfile.read(file_path)


    mfcc_features = mfcc(signal, sampling_frequency)
    mfcc_features = preprocessing.scale(mfcc_features)
    filterbank_features = logfbank(signal, sampling_frequency, nfilt=5)
    filterbank_features = preprocessing.scale(filterbank_features)

    if (show):
        print('Sampling rate: ', sampling_frequency)
        print('Number of samples: ', signal.shape[0])
        print('Number of windows: ', features_mfcc.shape[0])
#         print('Length of each feature: ', features_mfcc.shape[1])

        plt.matshow(mfcc_features.T)
        plt.title('MFCC')

        plt.matshow(filterbank_features.T)
        plt.title('Filterbank')
    
    return mfcc_features, filterbank_features

def word_seg(file_path, verbose=True):
    config = {
        'hmm': os.path.join(model_path, 'en-us'),
        'lm': os.path.join(model_path, 'en-us.lm.bin'),
        'dict': os.path.join(data_path, dictionary)
    }
    ps = Pocketsphinx(**config)

    ps.decode(
        audio_file=file_path,
        buffer_size=2048,
        no_search=False,
        full_utt=False
    )

    segs_and_phonemes = [(seg.word, ps.lookup_word(seg.word), seg.start_frame,  seg.end_frame)  for seg in ps.seg()]
    
    if (verbose):
        print("hypothesis: " + ps.hypothesis()) 

        # print("word, prob, start_frame, end_frame")
        print('Detailed segments:', *ps.segments(detailed=True), sep='\n')
        
        print('Segments Phonemes:', segs_and_phonemes)
        
    return segs_and_phonemes

def phoneme_seg(file_path, verbose=True):
    # Create a decoder with a certain model
    config = pocketsphinx.Decoder.default_config()
    config.set_string('-hmm', os.path.join(model_path, 'en-us'))
    config.set_string('-allphone', os.path.join(model_path, "en-us-phone.lm.bin"))
    config.set_string('-dict', os.path.join(data_path, dictionary))
    config.set_float('-lw', 2.0)
    config.set_float('-beam', 1e-20)
    config.set_float('-pbeam', 1e-20)
    decoder = Decoder(config)

    # Decode streaming data
    buffer = bytearray(2048)
    f = open(file_path, 'rb')
    decoder.start_utt()
    while f.readinto(buffer):
        decoder.process_raw(buffer, False, False)
    decoder.end_utt()
    f.close()
    
    segs = [(seg.word, seg.start_frame, seg.end_frame) for seg in decoder.seg()]
    
    if (verbose):
        print('Best phonetic segments:', segs)
        
    return segs

def extract_feature(mfcc_features, filterbank_features,  word_segments, phoneme_segments):
    # feature array
    # first (mfcc_features.shape[1]+filterbank_features.shape[1])) elements are for the first phoneme
    feature_length_for_each_phoneme = (mfcc_features.shape[1]+filterbank_features.shape[1])
    result = np.zeros((1, (max(phoneme_to_index.values())+1)*feature_length_for_each_phoneme), dtype=np.float64)

    # check if a phoneme is inside a word
    def inside(p, w):
        p_start = p[1]
        p_end = p [2]
        w_start = w[2]
        w_end = w[3]
        if p_start >= w_start-2 and p_end <= w_end +2:
            return True
        return False

    # find corresponding phonemes in each word
    m_features = {}
    f_features = {}
    phoneme_counts = {}
    for w in word_segments[1:-1]:
        phonemes = []
        for p in phoneme_segments:
            if inside(p, w):
                phonemes.append(p)
        # if phonemes are matched
        if len(w[1].split(" ")) == len(phonemes):
            # add extracted features to feature dictionary
            index = 0
            for correct_phonemes in w[1].split(" "):
                if correct_phonemes in phoneme_to_index and phoneme_to_index[correct_phonemes] != -1:
                    if not correct_phonemes in phoneme_counts:
                        phoneme_counts[correct_phonemes] = phonemes[index][2] - phonemes[index][1] + 1
                        m_features[correct_phonemes] = np.sum(mfcc_features[phonemes[index][1]:phonemes[index][2]+1], axis=0)
                        f_features[correct_phonemes] = np.sum(filterbank_features[phonemes[index][1]:phonemes[index][2]+1], axis=0)
                    else:
                        phoneme_counts[correct_phonemes] += phonemes[index][2] - phonemes[index][1] + 1
                        m_features[correct_phonemes] += np.sum(mfcc_features[phonemes[index][1]:phonemes[index][2]+1], axis=0)
                        f_features[correct_phonemes] += np.sum(filterbank_features[phonemes[index][1]:phonemes[index][2]+1], axis=0)
                index += 1

    # average all phoneme features for each phoneme and add it to feature array
    for p in phoneme_counts.keys():
        if p in phoneme_to_index and phoneme_to_index[p] != -1:
            m_features[p] = m_features[p] / phoneme_counts[p]
            f_features[p] = f_features[p] / phoneme_counts[p]
            result[0,phoneme_to_index[p]*feature_length_for_each_phoneme:(phoneme_to_index[p]+1)*feature_length_for_each_phoneme] = np.concatenate((m_features[p], f_features[p]), axis=0).reshape(1,-1)
        
    return result





features_for_all_samples = None
labels = None
sample_count = 0

# generate Phoneme segmentation and features for each training file
for record in training_set_description:
    # if this line is not empty
    if len(record)>10:
        fileName = record.split(" ")[0]
        label = 1 if record.split(" ")[1] == 'genuine' else 0  # label is 1 for genuine data
        dictionary = "ASVspoof2017_"+ str(int(record.split(" ")[3][1:])) + ".dict"
        
       
        # extract features of entire audio
        mfcc_features, filterbank_features = extract_mfcc_filterbank_features(os.path.join(training_data_path, fileName), show=False)

        # phoneme segmentation
        # word segmentation is much more accurate, so we perform word segmentation first and use it to guide phoneme extraction
        word_segments = word_seg(os.path.join(training_data_path, fileName), verbose=False)
        phoneme_segments = phoneme_seg(os.path.join(training_data_path, fileName), verbose=False)

        # extract features for each phoneme
        result = extract_feature(mfcc_features, filterbank_features,  word_segments, phoneme_segments)
        
        # add feature to features_for_all_samples
        if sample_count == 0:
            feature_length_for_each_phoneme = (mfcc_features.shape[1]+filterbank_features.shape[1])
            features_for_all_samples = np.zeros((len(training_set_description), (max(phoneme_to_index.values())+1)*feature_length_for_each_phoneme), dtype=np.float64)
            labels = np.zeros((len(training_set_description), 1), dtype=np.int32)
            
        features_for_all_samples[sample_count,:] = result
        labels[sample_count] = label
        sample_count += 1
        
        if sample_count%100 == 0:
            print("processing sample " + str(sample_count))




# change 0 into mean values of non-zero elements for each column
column_mean = np.true_divide(features_for_all_samples.sum(0),(features_for_all_samples!=0).sum(0))
inds = np.where(features_for_all_samples == 0)
# replace the index
features_for_all_samples[inds] = np.take(column_mean, inds[1])



# save data
with open(os.path.join(data_path,'training_features_for_all_samples.npy'), 'wb') as f:
    np.save(f, features_for_all_samples)
# 1 for genuine data, 0 for recorded data
with open(os.path.join(data_path,'training_labels.npy'), 'wb') as f:
    np.save(f, labels)



# load data
with open(os.path.join(data_path,'training_features_for_all_samples.npy'), 'rb') as f:
    temp = np.load(f)
print(temp)



# dimensional reduction to reduce size and speed up training/testing
# This aligns with the goal of void

# using vanilla PCA
all_data = temp
# all_data = np.concatenate((temp, temp2), axis=0)

covariance = np.cov(all_data.T)
eigen_values, eigen_vectors = np.linalg.eig(covariance)
eigen_values = eigen_values.real
# find out how many components are redundent
cumulative_variance = []
sum_eigen_values = sum(eigen_values)
for i in eigen_values:
     cumulative_variance.append((i/sum_eigen_values))
cumulative_variance = np.cumsum(cumulative_variance)
print(cumulative_variance)



# use 165 components for 95% variance
projection = (eigen_vectors.T[:][:165]).T
projected_features_training = temp.dot(projection)
print(projected_features_training)
# projected_features_dev = temp2.dot(projection)
# print(projected_features_dev)

# save data
with open(os.path.join(data_path,'training_features_for_all_samples_PCA.npy'), 'wb') as f:
    np.save(f, projected_features_training)
# with open(os.path.join(data_path,'dev_features_for_all_samples_PCA.npy'), 'wb') as f:
#     np.save(f, projected_features_dev)
with open(os.path.join(data_path,'projection_PCA.npy'), 'wb') as f:
    np.save(f, projection)