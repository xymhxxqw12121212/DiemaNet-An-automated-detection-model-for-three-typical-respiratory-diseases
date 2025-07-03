# DiemaNet:An automated detection model for three typical respiratory diseases
############Dataset
The dataset folder contains the original dataset. (See the instructions below for how to obtain the data.)

############Audio Segmentation
Run splitSegment.py to segment the cough recordings in the dataset folder into shorter cough clips based on a silence threshold. The segmented data will be saved in the datasetSegment folder.

############Visualization of Segmentation
The statistics.py script generates waveform comparison plots before and after segmentation of cough signals.

############Data Augmentation
Copy the segmented data from the datasetSegment folder into the datasetSegAug folder. Then, run augaudio.py to perform data augmentation on the COPD, COVID, and asthma samples in datasetSegment. The augmented data will be saved in the datasetSegAug folder.

############Train/Test Split
Run split_train_test.py to split the dataset into training and testing sets with a 4:1 ratio. The resulting data will be saved in the train_test_data folder.

############Acoustic Feature Extraction
Run make_npz.py to extract acoustic features including MFCC, GFCC, OpenL3, and Wav2Vec2.0.

############For MFCC extraction, the window_size parameter is set to 6 ms if using the DiffRes module, and 10 ms otherwise.

############For OpenL3 extraction, the hop_size parameter is set to 0.06 if using DiffRes, and 0.1 otherwise.
(The relevant code sections are labeled with comments.)
Features extracted without using the DiffRes module are saved in the feature folder, while those with DiffRes are saved in the feature1s_6ms folder.

############Model Training and Testing without DiffRes
There are four scripts for training and testing:

4s_train_convnet.py and 4s_test_convnet.py are used to train and test models without the DiffRes module, including:

Single-branch model: GoogleNet

Dual-branch models: TCCNN and TCCNN-E (as described in the paper)

To run 4s_train_convnet.py, uncomment the feature loading section relevant to your experiment, and select the desired model by commenting/uncommenting accordingly. The trained model will be saved in the modelsave_balanced_dataset folder as best_network.pth.

4s_test_convnet.py follows the same logic. It evaluates the trained model, generating a results.csv file containing the precision, recall, F1-score, and accuracy for each class, along with a confusion matrix plot.

############Model Training and Testing with DiffRes

train_diemanet.py and test_diemanet.py are used to train and test models that include the DiffRes module, mainly TCCNN-D and DiemaNet.

These scripts follow the same usage pattern as above. After training with train_diemanet.py, the model will be saved as best_network.pth. Running test_diemanet.py will generate results.csv and the confusion matrix image confusion_matrix2.png.

############Additional Notes:
The openl3-main folder contains the OpenL3 dependency. Install it by running setup.py inside that directory.
The pydiffres folder contains the DiffRes dependency.
The wav2vec_model folder contains the pre-downloaded Wav2Vec2.0 pretrained model.

The model folder includes all model architecture definitions. The key dual-branch models used in the paper—TCCNN, TCCNN-D, TCCNN-E, and DiemaNet—are defined in the CH2_model.py file.

The utils.py file contains utility functions used across various scripts.

The dataset, extracted features, confusion matrix images, and trained models (including best-performing checkpoints) can be accessed through the provided [cloud storage link].
