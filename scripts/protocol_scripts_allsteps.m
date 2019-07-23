%% Step 1: Model building (Timing 20 mins – a few hours)

%% read image files or load dat
basedir = 'path/to/directory/interpret_ml_neuroimaging';
gray_matter_mask = which('gray_matter_mask.img');

%% read images
heat_imgs = filenames(fullfile(basedir, 'data', 'derivatives', 'trial_images', 'sub*', 'heat_*.nii'));
rejection_imgs = filenames(fullfile(basedir, 'data', 'derivatives', 'trial_images', 'sub*', 'rejection_*.nii'));
data = fmri_data([heat_imgs; rejection_imgs], gray_matter_mask);

% % load data already stored in fmri_data object
% % // couldn't save the heat_rejection_datobj.mat file in the github due
% % // to the file size limit.
% data_file = fullfile(basedir, 'data', 'dat_obj', 'heat_rejection_datobj.mat'); %heat vs. rejection instead + change in the methods
% load(data_file);

%% OPTIONAL: You can apply a mask

mask = fullfile(basedir, 'masks', 'neurosynth_mask_Woo2014.nii');
data = apply_mask(data, mask);

%% Outcome

data.Y = [ones(numel(heat_imgs),1); -ones(numel(rejection_imgs),1)]; % heat: 1, rejection: -1

%% Training

[~, stats] = predict(data, 'algorithm_name', 'cv_svm', 'nfolds', 1, 'error_type', 'mcr');

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
nuisance_file = fullfile(basedir, 'data', 'derivatives', 'nuisance.mat'); 
                    % Example nuisance data: Mean framewise displacement (z-scored; roll, pitch, yaw, x, y, z; 6 columns)
load(nuisance_file);

% prepare fmri_data object variable 
dat_nuistest = fmri_data;
dat_nuistest.dat = R';
dat_nuistest.Y = stats_loso.dist_from_hyperplane_xval;

% train a regression model
[~, stats_nuistest] = predict(dat_nuistest, 'algorithm_name', 'cv_multregress', 'nfolds', 1, 'error_type', 'mse');

% output: stats_nuistest.pred_outcome_r = 0.3273

%% Step 4: Identify important features

%% 4A: Bootstrap tests

[~, stats_boot] = predict(data, 'algorithm_name', 'cv_svm', 'nfolds', 1, 'error_type', 'mcr', 'bootweights', 'bootsamples', 10000);

data_threshold = threshold(stats_boot.weight_obj, .05, 'fdr'); 

%% 4B: Recursive Feature Elimination (RFE)

out = svm_rfe(data, 'n_removal', 5000, 'n_finalfeat', 30000, 'algorithm_name', 'cv_svm', 'nfolds', n_folds, 'error_type', 'mcr'); 

%% Visualization of bootstrap test and RFE results

% visualizing bootstrap test results
orthviews(data_threshold);
% writing thresholded bootstrap test result image as a Nifti file
mkdir(basedir, 'results')
data_threshold.fullpath = fullfile(basedir, 'results', 'svm_bootstrap_results_fdr05.nii'); 
write(data_threshold, 'thresh');

% visualizing RFE results
orthviews(out.smallestnfeat_stats.weight_obj);
% writing RFE analysis results as a Nifti file
out.smallestnfeat_stats.weight_obj.fullpath = fullfile(basedir, 'results', 'svm_RFE_results_nfeat_30000.nii'); 
write(out.smallestnfeat_stats.weight_obj);

%% 4C: Virtual lesion analysis

%% Load mask - BucknerLab_wholebrain
img = fullfile(basedir, 'masks', 'Bucknerlab_7clusters_all_combined.nii');
mask = fmri_data(img, gray_matter_mask);
network_names = {'Visual', 'Somatomotor', 'dAttention', 'vAttention', 'Limbic', 'Frontoparietal', 'Default'};

%% One network removed for prediction

for network_i = 1:7
    
    for subj_i = 1:59
        
        masked_weights = stats_loso.other_output_cv{subj_i,1} .* double(mask.dat ~= network_i);
        dat_subj = data.dat(:, n_folds==subj_i);
        
        pexp(:,subj_i) = masked_weights' * dat_subj;
        
    end
    
    pexp_sorted = [reshape(pexp(1:8,:), numel(heat_imgs), 1);reshape(pexp(9:16,:), numel(rejection_imgs), 1)];
    roc = roc_plot(pexp_sorted, data.Y==1);
    
    out.num_vox(network_i,1) = sum(mask.dat ~= network_i);
    out.acc(network_i,1) = roc.accuracy;
    out.se(network_i,1) = roc.accuracy_se;
    out.p(network_i,1) = roc.accuracy_p;
end

table(network_names', out.num_vox, out.acc, out.se, out.p, 'VariableNames', {'Networks', 'NumVox', 'Accuracy', 'SE', 'P'})

%         Networks          NumVox      Accuracy       SE        P
%     ________________    __________    ________    _________    _
% 
%     'Visual'            1.8506e+05    0.83581      0.012057    0
%     'Somatomotor'        1.843e+05    0.92691     0.0084717    0
%     'dAttention'        1.9061e+05    0.92585      0.008528    0
%     'vAttention'        1.8861e+05    0.92903     0.0083576    0
%     'Limbic'            1.9184e+05    0.92691     0.0084717    0
%     'Frontoparietal'     1.809e+05    0.91843     0.0089083    0
%     'Default'           1.6997e+05    0.91949     0.0088554    0

%% Only one network used for prediction

for network_i = 1:7
    
    for subj_i = 1:59
        
        masked_weights = stats_loso.other_output_cv{subj_i,1} .* double(mask.dat == network_i);
        dat_subj = data.dat(:, n_folds==subj_i);
        
        pexp(:,subj_i) = masked_weights' * dat_subj;
        
    end
    
    pexp_sorted = [reshape(pexp(1:8,:), numel(heat_imgs), 1);reshape(pexp(9:16,:), numel(rejection_imgs), 1)];
    roc = roc_plot(pexp_sorted, data.Y==1);
    
    out.num_vox(network_i,1) = sum(mask.dat == network_i);
    out.acc(network_i,1) = roc.accuracy;
    out.se(network_i,1) = roc.accuracy_se;
    out.p(network_i,1) = roc.accuracy_p;
end

table(network_names', out.num_vox, out.acc, out.se, out.p, 'VariableNames', {'Networks', 'NumVox', 'Accuracy', 'SE', 'P'})

%         Networks        NumVox    Accuracy       SE           P     
%     ________________    ______    ________    ________    __________
% 
%     'Visual'            21740     0.82309      0.01242             0
%     'Somatomotor'       22503     0.68114     0.015168             0
%     'dAttention'        16192     0.65148     0.015509             0
%     'vAttention'        18194     0.75212     0.014053             0
%     'Limbic'            14959     0.55508     0.016175    0.00079248
%     'Frontoparietal'    25902     0.77225      0.01365             0
%     'Default'           36828     0.70127     0.014897             0


%% Step 5: Test generalizability

% a priori models: NPS and SIIPS1
nps = which('weights_NSF_grouppred_cvpcr.img');
siips = which('nonnoc_v11_4_137subjmap_weighted_mean.nii');

% load contrast image data
cont_imgs{1} = filenames(fullfile(basedir, 'data', 'derivatives', 'contrast_images', 'heat*nii'), 'char');
cont_imgs{2} = filenames(fullfile(basedir, 'data', 'derivatives', 'contrast_images', 'warmth*nii'), 'char');
cont_imgs{3} = filenames(fullfile(basedir, 'data', 'derivatives', 'contrast_images', 'rejection*nii'), 'char');
cont_imgs{4} = filenames(fullfile(basedir, 'data', 'derivatives', 'contrast_images', 'friend*nii'), 'char');

data_test = fmri_data(cont_imgs, gray_matter_mask);

% calculate pattern expression values
pexp_nps = apply_mask(data_test, nps, 'pattern_expression', 'ignore_missing');
pexp_siips = apply_mask(data_test, siips, 'pattern_expression', 'ignore_missing');

% reshape pexp values to have differnt conditions in different columns
pexp_nps = reshape(pexp_nps, 59, 4);
pexp_siips = reshape(pexp_siips, 59, 4);

% NPS for pain vs. warmth
roc_nps_pain_warmth = roc_plot([pexp_nps(:,1);pexp_nps(:,2)], [true(59,1);false(59,1)], 'twochoice');
% NPS for rejection vs. friend
roc_nps_rejection_friend = roc_plot([pexp_nps(:,3);pexp_nps(:,4)], [true(59,1);false(59,1)], 'twochoice');

% SIIPS1 for pain vs. warmth
roc_siips_pain_warmth = roc_plot([pexp_siips(:,1);pexp_siips(:,2)], [true(59,1);false(59,1)], 'twochoice');
% SIIPS1 for rejection vs. friend
roc_siips_rejection_friend = roc_plot([pexp_siips(:,3);pexp_siips(:,4)], [true(59,1);false(59,1)], 'twochoice');

%% Step 6: Biology-level assessment

%% 6A: Relationship with large-scale functional networks (bootstrap results)

% load Bucknerlab network mask and prepare a voxels x network matrix 
img = fullfile(basedir, 'masks', 'Bucknerlab_7clusters_all_combined.nii');
mask = fmri_data(img, gray_matter_mask);
network_names = {'Visual', 'Somatomotor', 'dAttention', 'vAttention', 'Limbic', 'Frontoparietal', 'Default'};

dat = [mask.dat==1 mask.dat==2 mask.dat==3 mask.dat==4 mask.dat==5 mask.dat==6 mask.dat==7];

% prepare the thresholded pattern map based on bootstrap tests
pattern_thresh = stats_boot.weight_obj.dat .* double(stats_boot.weight_obj.sig);

% calculate posterior probability of observing thresholded regions given each network
overlap_pos = canlab_pattern_similarity(dat, pattern_thresh>0, 'posterior_overlap', 'ignore_missing');
overlap_neg = canlab_pattern_similarity(dat, pattern_thresh<0, 'posterior_overlap', 'ignore_missing');

% % plotting (polar plot)
% colors = {[1 0 0], [0 0 1]};
% tor_polar_plot({[overlap_pos overlap_neg].*100}, colors, {network_names}, 'fixedrange', [0 24]);

%% Step 7: Representational analysis

nps_acc = zeros(4,4); 
siips_acc = zeros(4,4); 

for i = 1:4
    for j = 1:4
        if i < j
            roc_nps = roc_plot([pexp_nps(:,i);pexp_nps(:,j)], [true(59,1);false(59,1)], 'twochoice', 'noplot'); 
            nps_acc(i,j) = roc_nps.accuracy;
            nps_acc(j,i) = roc_nps.accuracy; % make it symmetric
            
            roc_siips = roc_plot([pexp_siips(:,i);pexp_siips(:,j)], [true(59,1);false(59,1)], 'twochoice', 'noplot'); 
            siips_acc(i,j) = roc_siips.accuracy;
            siips_acc(j,i) = roc_siips.accuracy; % make it symmetric
        end
    end
end

%% correlation between two confusion matrices
r = corr(nps_acc(tril(true(4,4),-1)), siips_acc(tril(true(4,4),-1)));
% 0.7100

%% Network plots

% vectorize the accuracy matrices
nps_acc_vec = nps_acc(tril(true(4,4),-1));
siips_acc_vec = siips_acc(tril(true(4,4),-1));

% make the accuracy values into weights
w_nps = (1-nps_acc_vec)./sum(1-nps_acc_vec);
w_siips = (1-siips_acc_vec)./sum(1-siips_acc_vec);

% draw network plots
[i, j] = find(tril(true(4,4),-1));
subplot(1,2,1);
G_nps = graph(i,j,w_nps);
plot(G_nps,'Layout','force', 'WeightEffect', 'inverse', 'LineWidth',w_nps*10);

subplot(1,2,2);
G_siips = graph(i,j,w_siips);
plot(G_siips,'Layout','force', 'WeightEffect', 'inverse', 'LineWidth',w_siips*10);


%% hierarchical clustering
subplot(1,2,1);
Z_nps = linkage(nps_acc(tril(true(4,4),-1))');
h_nps = dendrogram(Z_nps);

subplot(1,2,2);
Z_siips = linkage(siips_acc(tril(true(4,4),-1))');
h_siips = dendrogram(Z_siips);