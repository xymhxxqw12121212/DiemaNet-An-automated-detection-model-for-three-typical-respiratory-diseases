# DiemaNet:An automated detection model for three typical respiratory diseases
1. Dataset: The dataset folder contains the original dataset. (See other instructions for how to get the data.)

2. Audio segmentation: Run splitSegment.py to segment the cough recordings in the dataset folder into shorter cough segments based on the silence threshold. The segmented data will be saved in the datasetSegment folder.

3. Segmentation visualization: The statistics.py script generates a waveform comparison chart of the cough signal before and after segmentation.

4. Data enhancement: Copy the segmented data from the datasetSegment folder to the datasetSegAug folder. Then, run augaudio.py to perform data enhancement on the COPD, COVID, and asthma samples in datasetSegment. The enhanced data will be saved in the datasetSegAug folder.

5. Training/testing split: Run split_train_test.py to split the dataset into training and test sets in a ratio of 4:1. The generated data will be saved in the train_test_data folder.

6. Acoustic feature extraction: Run make_npz.py to extract acoustic features, including MFCC, GFCC, OpenL3, and Wav2Vec2.0. For MFCC extraction, set the window_size parameter to 6 milliseconds if the DiffRes module is used, otherwise to 10 milliseconds.
For OpenL3 extraction, set the hop_size parameter to 0.06 if the DiffRes module is used, otherwise to 0.1. (The relevant code snippets have been commented.) Features extracted without the DiffRes module are saved in the feature folder, while features extracted with the DiffRes module are saved in the feature1s_6ms folder.

7. Model training and testing without DiffRes module There are four scripts for training and testing:
4s_train_convnet.py and 4s_test_convnet.py are used to train and test models without DiffRes module, including: single-branch model: GoogleNet, dual-branch model: TCCNN and TCCNN-E (as described in the paper).
To run 4s_train_convnet.py, uncomment the feature loading section related to your experiment and select the desired model by commenting/uncommenting accordingly. The trained model will be saved as best_network.pth in the modelsave_balanced_dataset folder.
4s_test_convnet.py follows the same logic. It evaluates the trained model and generates a results.csv file containing the precision, recall, F1 score and accuracy for each category, as well as a confusion matrix plot.
Model training and testing with DiffRes:
train_diemanet.py and test_diemanet.py are used to train and test models containing DiffRes modules, mainly TCCNN-D and DiemaNet. The usage pattern of these scripts is the same as above. After training with train_diemanet.py, the model will be saved as the best_network.pth file. Running test_diemanet.py will generate the results.csv file and the confusion matrix image confusion_matrix2.png.

Additional notes: 1. The pydiffres folder contains the DiffRes dependencies. The model folder contains all model architecture definitions. The key two-branch models used in the paper - TCCNN, TCCNN-D, TCCNN-E, and DiemaNet - are defined in the CH2_model.py file.
2. The utils.py file is a related dependency, defining some functions required by other run files
3. You can access the dataset, extracted features, confusion matrix images, and the best trained model (including the best performance checkpoint) through the provided [cloud storage link], as well as information about AST, openl3-main dependencies, Wav2Vec2.0 pre-trained models, and OpenL3 dependencies. (Please run setup.py in this directory to install the openl3-main dependencies. The wav2vec_model folder contains the pre-downloaded Wav2Vec2.0 pre-trained model)
