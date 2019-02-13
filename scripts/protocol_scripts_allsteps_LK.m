%% Step 1: Model building (Timing 20 mins – a few hours)

%% read image files or load data
basedir = '/Users/clinpsywoo/Nas/xain/publish_scripts';

% read images
gray_matter_mask = which('gray_matter_mask.img');
heat_imgs = filenames(fullfile(basedir, 'data', 'images', 'sub*', 'heat_*.nii'));
rejection_imgs = filenames(fullfile(basedir, 'data', 'images', 'sub*', 'rejection_*.nii'));
data = fmri_data([heat_imgs; rejection_imgs], gray_matter_mask);

% load data already stored in fmri_data object
% data_file = fullfile(basedir, 'data', 'dat_obj', 'dpsp_heat_rejection.mat'); %heat vs. rejection instead + change in the methods
% load(data_file);

%% OPTIONAL: You can apply a mask

mask = fullfile(basedir, 'masks', 'neurosynth_mask_Woo2014.nii');
data = apply_mask(data, mask);

%% Outcome

data.Y = [ones(numel(heat_imgs),1); -ones(numel(rejection_imgs),1)]; % heat: 1, rejection: -1

%% Training

[~, stats] = predict(data, 'algorithm_name', 'cv_svm', 'error_type', 'mcr');


%% Step 2: Cross-validated performance assessment (Timing 5 mins-20 mins)

%% Training SVM with leave-one-subject-out (LOSO) cross-validation

n_folds = [repmat(1:59, 8,1) repmat(1:59, 8,1)];
n_folds = n_folds(:); 

[~, stats_loso] = predict(data, 'algorithm_name', 'cv_svm', 'nfolds', n_folds, 'error_type', 'mcr');

%% Training SVM with 8-fold cross-validation

[~, stats_8fold] = predict(data, 'algorithm_name', 'cv_svm', 'nfolds', 8, 'error_type', 'mcr');

% CRITICAL: stratified sampling

%% ROC plot

% LOSO cross-validation
ROC_loso = roc_plot(stats_loso.dist_from_hyperplane_xval, data.Y == 1, 'threshold', 0);

% 8-fold cross-validation
ROC_8fold = roc_plot(stats_8fold.dist_from_hyperplane_xval, data.Y == 1, 'threshold', 0);

%% Step 3: Analysis of confounds (Timing 5 mins-20 mins)

% Load nuisance data that are previously saved
nuisance_file = fullfile(basedir, 'data', 'nuisance.mat'); 
                    % Example nuisance data: Mean framewise displacement (z-scored; roll, pitch, yaw, x, y, z; 6 columns)
load(nuisance_file);

% prepare fmri_data object variable 
dat_nuistest = fmri_data;
dat_nuistest.dat = R';
dat_nuistest.Y = stats_loso.dist_from_hyperplane_xval;

% train a regression model
[~, stats_nuistest] = predict(dat_nuistest, 'algorithm_name', 'cv_multregress', 'nfolds', 1, 'error_type', 'mse');

% output: stats_nuistest.pred_outcome_r = 0.3273

%% Step 4: 

