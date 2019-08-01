# Interpreting machine learning models in neuroimaging
This repository includes Matlab and Python codes and sample fMRI data used in **Interpreting machine learning models in neuroimaging: Towards a unified framework**.  
## Dependencies
To run the **Matlab scripts** `protocol_scripts_allsteps.m`: 

+ [CanlabCore tools](https://github.com/canlab/CanlabCore) - For full functionality, make sure to install:
	1. Matlab Statistics and Machine Learning toolbox
	2. Matlab Signal Processing toolbox
	3. [Statistical Parametric Mapping (SPM) toolbox](https://www.fil.ion.ucl.ac.uk/spm/) 

The Matlab script `protocol_scripts_allsteps.m` was tested on macOS High Sierra with Matlab 9.5 R2018b.

To run the codes related to the **deep learning analyses** `cnn_lrp.ipynb`:  

+ [Python](https://www.python.org/downloads/) 3.6.5 or higher with a module Numpy
+ [Keras 2.2.4](https://keras.io) (Python Deep Learning library) 
+ [TensorFlow 1.8.0](https://www.tensorflow.org/install/) 
+ [Scikit-learn 0.20.3](https://scikit-learn.org/stable/install.html) 
+ [Pandas 0.25.0](https://pandas.pydata.org/pandas-docs/stable/install.html) 
+ [Scipy 1.3.0](https://www.tensorflow.org/install/) 
+ [Nilearn 0.5.2](http://nilearn.github.io/introduction.html#installation) 
+ [Nipype 1.2.0](https://nipype.readthedocs.io/en/latest/users/install.html) 
+ [iNNvestigate](https://github.com/albermax/innvestigate) (Keras explanation toolbox)
+ **Other dependencies:** CUDA, CuDNN (Nvidia GPU libraries)


## Installation
After downloading and installing all dependencies, download this repository using the following command line:  
```
$ git clone https://github.com/cocoanlab/interpret_ml_neuroimaging
```  

Make sure to set your Matlab path to include all required toolboxes.  

## How to run the codes
### Linear models 
The Matlab script `protocol_scripts_allsteps.m` contains all analyses performed on linear models.  
A user is only required to set the path in the following code line:
```
basedir = 'path/to/directory/interpret_ml_neuroimaging';
```  

The rest of the code can be run automatically section by section without an input from the user. The code uses a sample fMRI dataset from [Woo et al., 2014](https://www.nature.com/articles/ncomms6380), and the path to the data directory is set in the script.  

The script includes **7 steps** of model development and interpretation. 

In **Step 1**, a Support Vector Machine (SVM) Model is built. The output is an `fmri_data` object with a `.weight_obj` field which contains the model weights in `.dat` field. 

**Step 2** returns the cross-validated performance using either 8-fold cross-validation or leave-one-subject-out cross-validation. 

In **Step 3**, the previously built model is tested for potential confounds. The output can be found in `stats_nuistest.pred_outcome_r`. 

**Step 4** includes four options: option **A** - Bootstrap tests, option **B** - Recursive Feature Elimination, and option **C** - "Virtual lesion" analysis. Options **A** and **B** return significant voxels in the SVM model, whereas option **C** tests significance of large-scale resting state networks in predictions. Sample results of the "virtual lesion" analysis are provided in the script. 

In **Step 5**, an example of generalizability testing is provided. Here we test the performance of the Neurologic Pain Signature (NPS), [Wager et al., 2013](https://www.nejm.org/doi/full/10.1056/NEJMoa1204471) and Stimulus Intensity Independent Pain Signature-1 (SIIPS1), [Woo et al., 2017](https://www.nature.com/articles/ncomms14211) on the sample dataset. 

**Step 6** returns the posterior probability of observing overlaps of the boostrap thresholded model with large-scale functional networks. Finally, 

**Step 7** illustrates a representational analysis performed with NPS and SIIPS1 and the sample dataset.  

The typical **run time** of the whole script with the sample data on a standard desktop computer is about **12 to 15 hours**.

### Nonlinear models  
To run the nonlinear example, run `cnn_lrp.ipynb`

**Part0** is Description about how anaysis will be run.

**Part1** Initializing Data Matrix

**Part2** Loading Data Function

**Part3** CNN Model

**Part4** Model Training example

**Part5** Leave-One_Subject-Out Cross-validation

**Part6** Total Accuracy

**Part7** Layer-wise Relevance Propagation

**Part8** Visualization

## Instructions for use
We encourage researchers to use the provided codes and apply them to their own data. To do so, set the Matlab path to the directory of the dataset to be analysed. The format of input data should be either NIfTI format (.nii) or Analyze format (.img and .hdr).
