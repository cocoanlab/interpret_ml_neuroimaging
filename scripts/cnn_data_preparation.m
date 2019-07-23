%% Set directories

basedir = '/path/to/directory/interpret_ml_neuroimaging/data';
savedir = '/path/to/directory/interpret_ml_neuroimaging/data';

gray_matter_mask = which('gray_matter_mask.img');


%% Load data

heat_imgs = filenames(fullfile(basedir, 'data', 'derivatives', 'trial_images', 'sub*', 'heat_*.nii'));
rejection_imgs = filenames(fullfile(basedir, 'data', 'derivatives', 'trial_images', 'sub*', 'rejection_*.nii'));

data_heat = fmri_data(heat_imgs, gray_matter_mask);
data_rej = fmri_data(rejection_imgs, gray_matter_mask);

data_heat = remove_empty(data_heat);
data_rej = remove_empty(data_rej);

%% Reconstruct data into 4-D and save
recon_dat_heat = reconstruct_image(data_heat);

savename = fullfile(savedir, 'dpsp_hot_masked.mat');
save(savename, 'recon_dat_heat', '-v7.3');


recon_dat_rej = reconstruct_image(data_rej);

savename = fullfile(savedir, 'dpsp_rej_masked.mat');
save(savename, 'recon_dat_rej', '-v7.3');





