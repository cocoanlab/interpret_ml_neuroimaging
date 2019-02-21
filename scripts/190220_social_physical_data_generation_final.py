
import time
startTime = time.time()
# coding: utf-8
#

"""
Download the mat data from the url
"""
import urllib.request

print('Beginning file download with urllib2...')

url_hot = 'http://115.145.189.31:5000/fsdownload/danc6RnED/rejection_sharedata/dpsp_hot_masked.mat' 
urllib.request.urlretrieve(url_hot, 'dpsp_hot_masked.mat') 

url_rej = 'http://115.145.189.31:5000/fsdownload/danc6RnED/rejection_sharedata/dpsp_rej_masked.mat' 
urllib.request.urlretrieve(url_rej, 'dpsp_rej_masked.mat') 



"""
Load the data from .mat
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# social_data_path = '190220_social_physical_data'
social_data_path = ''
hot = 'dpsp_hot_masked.mat'
social = 'dpsp_rej_masked.mat'



#f1.close()
# hot_load = scipy.io.loadmat(hot)["recon_dat"]
# hot_data = np.array(hot_load) 

# social_load = scipy.io.loadmat(social)["recon_dat"]
# social_data = np.array(social_load) 


f1 = h5py.File(hot,'r')
hot_data = np.array(f1["recon_dat"]) 

f2 = h5py.File(social,'r')
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


test_data = {}
train_data = {}
test_all_data = {}

# train data
train_X = hot_data.reshape([len(hot_data),68,95,79])
train_X = np.vstack((train_X,social_data.reshape([len(social_data),68,95,79])))
train_y = hot_y + social_y


train_data['X'] = train_X
train_data['y'] = np.asarray(train_y)

print('train_data[x].shape : ',train_data['X'].shape)
print('train_data[y].shape : ',train_data['y'].shape)

# test_all
social_all_X = np.average(social_data,0).reshape([1,68,95,79])
hot_all_X = np.average(hot_data,0).reshape([1,68,95,79])

social_all_y = np.ones([1]).tolist()
hot_all_y = np.zeros([1]).tolist()


print('hot_all_X.shape : ',hot_all_X.shape)
print('social_all_X.shape : ',social_all_X.shape)

test_all_data['X'] = np.vstack((hot_all_X,social_all_X))
test_all_data['y'] = np.asarray(hot_all_y + social_all_y)
print('----test_all_y: ',test_all_data['y'])

# test_each data - hot

j=0
X = hot_data[j:j+8]
print('each X shape',X.shape)
avg_X = np.average(X,0).reshape([1,68,95,79])
print('each avg_X shape',avg_X.shape)
hot_X = avg_X
j = j+8

while j<(472-7):
    X = hot_data[j:j+8]
    avg_X = np.average(X,0).reshape([1,68,95,79])
    hot_X = np.vstack((hot_X,avg_X))
    j = j+8
print('each hot_X shape',hot_X.shape)    
hot_y = np.zeros(hot_X.shape[0]).tolist()

# test data - social

j=0
X = social_data[j:j+8]
avg_X = np.average(X,0).reshape([1,68,95,79])
social_X = avg_X
j = j+8

while j<(472-7):
    X = social_data[j:j+8]
    avg_X = np.average(X,0).reshape([1,68,95,79])
    social_X = np.vstack((social_X,avg_X))
    j = j+8
print('each social_X shape',social_X.shape)      
social_y = np.ones(social_X.shape[0]).tolist()

# test data - stack

test_data['X'] = np.vstack((hot_X.reshape([len(hot_X),68,95,79]),social_X.reshape([len(hot_X),68,95,79])))
test_data['y'] = np.asarray(hot_y+social_y)

print('test_data[x].shape : ',test_data['X'].shape)
print('test_data[y].shape : ',test_data['y'].shape)



hdf5_file = h5py.File('190220_Social_Physical_masked_data_train.h5', 'w')
hdf5_file.create_dataset("X", train_data['X'].shape, np.float32)
hdf5_file["X"][...] = train_data['X']

hdf5_file.create_dataset("y", train_data['y'].shape, np.float32)
hdf5_file["y"][...] = train_data['y']


np.save('190220_Social_Physical_masked_data_test_each.npy',test_data)
np.save('190220_Social_Physical_masked_data_test_all.npy',test_all_data)




# coding: utf-8
"""
Generating Cross validation data 
"""


import h5py
import numpy as np
import matplotlib.pyplot as plt


"""
read sub_X_dic as a file 
"""

# Load
f1 = h5py.File('190220_Social_Physical_masked_data_train.h5','r')
X = np.array(f1["X"])
y = np.array(f1['y'])
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
        
"""
### when making whole datasets, not cross data
"""


#sets = sub_dic['sub1_X']
# yy = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
# sets_y = yy

# for i in range(2,60):
#     name = 'sub'+str(i)+'_X'
#     sets = np.vstack((sets,sub_dic[name]))
#     sets_y = np.vstack((sets_y,yy))


# import h5py
# save_file_name = '180319_social_physical_all.hdf5'
# hdf5_file = h5py.File(save_file_name, 'w')

# print('test file shape+++++++++++++++++++++++ :',str(sets.shape),str(sets_y.shape))
# hdf5_file.create_dataset(X, sets.shape, np.float32)
# hdf5_file[X][...] = sets
# hdf5_file.create_dataset(y, (len(sets),1), np.float32)
# hdf5_file[y][...] = sets_y.reshape(len(train_X),1)



# Save
#np.save('180108_social_physical_cross_X_dic20.npy', cross_X_dic)











