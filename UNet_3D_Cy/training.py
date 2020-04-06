import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.optimizers import Adam, SGD
from tqdm import tqdm
import random
from metrics import dice_coef, dice_coef_loss
import yaml

# Hyperparameters
config = yaml.safe_load(open("hyper_parameters.yaml"))
path_to_data = config['path_to_data']
path_to_save = config['path_to_save']

N = config['N']
N_test = config['N_test']

img_width = config['img_width']
img_height = config['img_height']
img_thickness = config['img_thickness']
img_channels = config['img_channels']

learning_rate = float(config['learning_rate'])
epochs = config['epochs']
val_size = config['val_size']
dropout = config['dropout']
batch_size = config['batch_size']
patience = config['patience']

f = config['f']

# Load Training Data
X_train = np.zeros((N, img_thickness, img_height, img_width, img_channels), dtype=np.float32)
Y_train = np.zeros((N, img_thickness, img_height, img_width, img_channels), dtype=np.float32)

print('','','')
print('','','')
print('Loading Training Data')

for n in tqdm(range(N)):
    image = np.load(os.path.join(path_to_data, "image_train%02d.npy" % n))
    label = np.load(os.path.join(path_to_data, "label_train%02d.npy" % n))
    X_train[n] = image[:,:,:,np.newaxis]
    Y_train[n] = label[:,:,:,np.newaxis]

print('Loaded Training Data')

print(X_train.shape)

# Load Testing Data
X_test = np.zeros((N_test, img_thickness, img_height, img_width, img_channels), dtype=np.float32)

print('','','')
print('','','')
print('Loading Testing Data')

for n in tqdm(range(N_test)):
    image = np.load(os.path.join(path_to_data, "image_test%02d.npy" % n))
    X_test[n] = image[:,:,:,np.newaxis]

print('Loaded Testing Data')
print('','','')
print('','','')

# Randomly Show an Image

# image_x = random.randint(0, N-1)
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# x_index = X_train[image_x,idx_slice,:,:]
# tf.print(x_index.shape)
# plt.imshow(np.squeeze(x_index), cmap='gray')
# ax2 = fig.add_subplot(122)
# ax1.title.set_text('Clinical Image')
# y_index = Y_train[image_x,idx_slice,:,:]
# plt.imshow(np.squeeze(y_index), cmap='gray')
# ax2.title.set_text('Real Mask')
# plt.show()

# UNet Model
inputs = tf.keras.layers.Input((img_thickness, img_width, img_height, img_channels))

# Convert integers in image matrix to floating point 
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

# Encoding
c1 = tf.keras.layers.Conv3D(f[1], (3, 3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(dropout)(c1)
c2 = tf.keras.layers.Conv3D(f[1], (3, 3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling3D((2,2,2))(c1)


c2 = tf.keras.layers.Conv3D(f[2], (3, 3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(dropout)(c2)
c2 = tf.keras.layers.Conv3D(f[2], (3, 3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling3D((2,2,2))(c2)


c3 = tf.keras.layers.Conv3D(f[3], (3, 3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(dropout)(c3)
c3 = tf.keras.layers.Conv3D(f[3], (3, 3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling3D((2,2,2))(c3)


c4 = tf.keras.layers.Conv3D(f[4], (3, 3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(dropout)(c4)
c4 = tf.keras.layers.Conv3D(f[4], (3, 3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling3D((2,2,2))(c4)


c5 = tf.keras.layers.Conv3D(f[5], (3, 3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(dropout)(c5)
c5 = tf.keras.layers.Conv3D(f[5], (3, 3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(c5)
p5 = tf.keras.layers.MaxPooling3D((2,2,2))(c5)

# Decoding Layers
u6 = tf.keras.layers.Conv3DTranspose(f[6], (2, 2, 2), strides=(2, 2, 2), padding='same',)(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv3D(f[6], (3, 3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(dropout)(c6)
c6 = tf.keras.layers.Conv3D(f[6], (3, 3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(c6)


u7 = tf.keras.layers.Conv3DTranspose(f[7], (2, 2, 2), strides=(2, 2, 2), padding='same',)(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv3D(f[7], (3, 3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(dropout)(c7)
c7 = tf.keras.layers.Conv3D(f[7], (3, 3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(c7)


u8 = tf.keras.layers.Conv3DTranspose(f[8], (2, 2, 2), strides=(2, 2, 2), padding='same',)(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv3D(f[8], (3, 3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(dropout)(c8)
c8 = tf.keras.layers.Conv3D(f[8], (3, 3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(c8)


u9 = tf.keras.layers.Conv3DTranspose(f[9], (2, 2, 2), strides=(2, 2, 2), padding='same',)(c8)
u9 = tf.keras.layers.concatenate([u9, c1])
c9 = tf.keras.layers.Conv3D(f[9], (3, 3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(dropout)(c9)
c9 = tf.keras.layers.Conv3D(f[9], (3, 3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(c9)


outputs = tf.keras.layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer=tf.optimizers.Adam(learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
model.summary()

# Checkpoints and Callbacks
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_pros_segmentation.h5',
                                                  verbose=1, save_best_only=True)
callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=patience, monitor='loss'),
            tf.keras.callbacks.TensorBoard(log_dir='logs')]
results = model.fit(X_train, Y_train, validation_split=val_size, batch_size=batch_size,
                    epochs=epochs, callbacks=callbacks) 

# Save the output masks

idx = random.randint(0, N)

preds_train = model.predict(X_train[:int(X_train.shape[0]*(1-val_size))], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*(1-val_size)):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

print('','','')
print('','','')
print('Saving 3D Segmentation Training Masks')

for ix in tqdm(range(len(preds_train))):
    for iy in range(img_thickness):
        fig = plt.figure()
        fig.suptitle(f'3D Segmentation Training Masks (ix={ix+1}, slice={iy+1})', fontsize=12)
        ax1 = fig.add_subplot(131)
        plt.imshow(np.squeeze(X_train[ix,iy,:,:]))
        ax2 = fig.add_subplot(132)
        plt.imshow(np.squeeze(Y_train[ix,iy,:,:]))
        ax3 = fig.add_subplot(133)
        plt.imshow(np.squeeze(preds_train_t[ix,iy,:,:]))
        ax1.title.set_text('Clinical Image')
        ax2.title.set_text('Real Mask')
        ax3.title.set_text('Predicted Mask')
        plt.savefig(f'plots_training/Training_Masks_ix_{ix+1}_slice_{iy+1}.png')
        plt.close()

print('Finished Saving')
print('','','')
print('','','')
print('Saving 2D Segmentation Training Mask Overlays')

for ix in tqdm(range(len(preds_train))):
    for iy in range(img_thickness):
        fig = plt.figure()
        fig.suptitle(f'2D Segmentation Training Mask Overlay (ix={ix+1}, slice={iy+1})', fontsize=12)
        ax1 = fig.add_subplot()
        plt.imshow(np.squeeze(X_train[ix]))
        plt.contour(np.squeeze(Y_train[ix]),1,colors='yellow',linewidths=0.5)
        plt.contour(np.squeeze(preds_train_t[ix]),1,colors='red',linewidths=0.5)
        plt.savefig(f'plots_training_overlay/Training_Overlay_ix_{ix+1}_slice_{iy+1}.png')
        plt.close()

print('Finished Saving')
print('','','')
print('','','')
print('Saving 3D Segmentation Testing Masks')

for ix in tqdm(range(len(preds_test))):
    for iy in range(img_thickness):
        fig = plt.figure()
        fig.suptitle(f'3D Segmentation Testing Masks (ix={ix+1}, slice={iy+1})', fontsize=12)
        ax1 = fig.add_subplot(121)
        plt.imshow(np.squeeze(X_test[ix,iy,:,:]))
        ax3 = fig.add_subplot(122)
        plt.imshow(np.squeeze(preds_test_t[ix,iy,:,:]))
        ax1.title.set_text('Clinical Image')
        ax2.title.set_text('Real Mask')
        ax3.title.set_text('Predicted Mask')
        plt.savefig(f'plots_testing/Testing_Masks_ix_{ix+1}_slice_{iy+1}.png')
        plt.close()

print('Finished Saving')
print('','','')
print('','','')

print('Training Script has sucessfully completed')
