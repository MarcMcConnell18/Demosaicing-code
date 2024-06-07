# Demosaicing-code
I only put the code here. There are 2 different methods, DnCNN and denoiser prior.
In DnCNN file, data.py is for generating .npy file for training data, data1.py is for generating .npy files for validation data. main1.py is the main code for training and test after every epoch. models.py is the DnCNN model. dncnn.h5 is the best model that I trained before, you can use test_model.py to test images by using dncnn.h5 model.
In Denoiser Prior file, the demo_no_notebook.py is the main code to apply the algorithm.
