# run train.py before visualise the results

import numpy as np
import os
import matplotlib.pyplot as plt

path_to_data = './data/datasets-promise12' 
path_to_save = './result' 

# specify these to plot the results w.r.t. the images
step = 2000
idx_case = 27
idx_slice = 6

image = np.load(os.path.join(path_to_data, "image_test%02d.npy" % idx_case))[::2, ::2, ::2]
label = np.load(os.path.join(path_to_save, "label_test%02d_step%06d.npy" % (idx_case, step)))[..., 0]
print(label.shape)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(image[idx_slice,:,:], cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(label[idx_slice,:,:], cmap='gray')
plt.show()
