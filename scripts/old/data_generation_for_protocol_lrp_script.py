
import time
startTime = time.time()
# coding: utf-8

"""
Load the data from .mat
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

social_data_path = 'data/'
hot = 'dpsp_hot_masked.mat'
social = 'dpsp_rej_masked.mat'


f1 = h5py.File(social_data_path+hot,'r')
hot_data = np.array(f1["recon_dat"]) 

f2 = h5py.File(social_data_path+social,'r')
social_data = np.array(f2["recon_dat"]) 


"""
check the data shape and image
"""
print (hot_data.shape)
print (social_data.shape)



"""
matching 0, 1 to hot and social data
"""
hot_y = np.zeros(hot_data.shape[0]).tolist()
social_y = np.ones(social_data.shape[0]).tolist()


# train data
train_X = hot_data.reshape([len(hot_data),68,95,79])
train_X = np.vstack((train_X,social_data.reshape([len(social_data),68,95,79])))
train_y = hot_y + social_y




# coding: utf-8
"""
Generating Cross validation data 
"""

X = train_X
y = np.asarray(train_y)
print('X shape',X.shape)
print('y shape',y.shape)
    

"""
make sets : set_ 1 = (hot 8 + social 8) total 59 sets
"""
total_sets = {}
for i in range(1,60): #60
    j = i + 59
    name_X = 'set_X_'+str(i)
    name_y = 'set_y_'+str(i)
    total_sets[name_X] = X[(i-1)*8:i*8]
    total_sets[name_X] = np.vstack((total_sets[name_X],X[(j-1)*8:j*8]))
    total_sets[name_y] = y[(i-1)*8:i*8]
    total_sets[name_y] = np.vstack((total_sets[name_y],y[(j-1)*8:j*8])).reshape((total_sets[name_X].shape[0]))

print('total_sets[set_X_1].shape',total_sets['set_X_1'].shape)
print('total_sets[set_y_1].shape',total_sets['set_y_1'].shape) 
#print('total_sets.shape',total_sets.shape)     
    
"""
make cross set
"""

import h5py
save_file_name = '190220_social_physical_masked_cross.hdf5'
hdf5_file = h5py.File(save_file_name, 'w')

total_cross = {}
for i in range(1,60): #60
    name_X = 'set_X_'+str(i)
    name_y = 'set_y_'+str(i)
    name = 'cross_'+str(i)
    
    total_cross[name] = {}
    total_cross[name]['X_test'] = total_sets[name_X]
    total_cross[name]['y_test'] = total_sets[name_y]
    
    start = True
    for j in range(1,60): #60
        if i == j :
            continue
        else:
            name_X = 'set_X_'+str(j)
            name_y = 'set_y_'+str(j)
            if start:
                total_cross[name]['X_train'] = total_sets[name_X]
                total_cross[name]['y_train'] = total_sets[name_y]
                start = False
            else:
                total_cross[name]['X_train'] = np.vstack((total_cross[name]['X_train'],total_sets[name_X]))
                total_cross[name]['y_train'] = np.vstack((total_cross[name]['y_train'],total_sets[name_y]))
                #.reshape((total_cross[name]['X_train'].shape[0]))
   
    crossX = name + '_X_train'
    crossy = name + '_y_train'
    crossXte = name + '_X_test'
    crossyte = name + '_y_test'
    
    total_cross[name]['y_train']=total_cross[name]['y_train'].reshape((total_cross[name]['X_train'].shape[0],1))
    total_cross[name]['y_test']=total_cross[name]['y_test'].reshape((total_cross[name]['X_test'].shape[0],1))
    

    
    hdf5_file.create_dataset(crossX, total_cross[name]['X_train'].shape, np.float32)
    hdf5_file[crossX][...] = total_cross[name]['X_train']
    hdf5_file.create_dataset(crossy, total_cross[name]['y_train'].shape, np.float32)
    hdf5_file[crossy][...] = total_cross[name]['y_train'].reshape((total_cross[name]['X_train'].shape[0],1)) 

    hdf5_file.create_dataset(crossXte, total_cross[name]['X_test'].shape, np.float32)
    hdf5_file[crossXte][...] = total_cross[name]['X_test']
    hdf5_file.create_dataset(crossyte, total_cross[name]['y_test'].shape, np.float32)
    hdf5_file[crossyte][...] = total_cross[name]['y_test'].reshape((total_cross[name]['X_test'].shape[0],1)) 

endTime = time.time() - startTime
print(endTime) 
np.save('190220_social_physical_data_generation_final_time.npy',endTime)
        
