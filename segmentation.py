import tensorflow as tf
import os
import numpy as np
from braintools import ReadImage
import braintools
import random
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import jaccard_score
import keras.backend as K

seed =50
np.random.seed = seed

# FUNCION DE PRE-PROCESAMIENTO
def extract_brain_region(image, brain_mask, background=0):
	''' find the boundary of the brain region, return the resized brain image and the index of the boundaries'''    
	# Tomo la imagen original, la mascara del cerebro y encuentro lo que es cerebro en la imagen tomando todo lo que no sea fondo
	# encuentro bordes con los indices maximos y minimos en cada eje para el área donde está el cerebro y saco un nuevo corte con slice solo del cerebro
	# para cada eje 
	brain = np.where(brain_mask != background)
	#print brain
	min_z = int(np.min(brain[0]))
	max_z = int(np.max(brain[0]))+1
	min_y = int(np.min(brain[1]))
	max_y = int(np.max(brain[1]))+1
	min_x = int(np.min(brain[2]))
	max_x = int(np.max(brain[2]))+1
	# resize image
	resizer = (slice(min_z, max_z), slice(min_y, max_y), slice(min_x, max_x))
	return image[resizer], [[min_z, max_z], [min_y, max_y], [min_x, max_x]]


# CARGA DE DATOS

# Defino carpeta de entrenamiento
TRAIN_PATH = 'data/train/train'
TEST_PATH = 'data/validation'

# Defino dimensiones de las imagenes
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANELS = 240

# Obtengo los nombres de cada una de las subcarpetas o pacientes
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# Obtengo los nombres de cada modalidad

# Creo un arreglo que contenga todos los datos
# Datos de entrenamiento
x_train = np.zeros((len(train_ids)*IMG_CHANELS*1, IMG_WIDTH, IMG_HEIGHT, 1), dtype=np.float32)

plt.show()
# Anotaciones de entrenamiento
y_train = np.zeros((len(train_ids)*IMG_CHANELS*1, IMG_WIDTH, IMG_HEIGHT, 1), dtype=np.bool)

print('Ajustando tamano de imagenes y las mascaras de entrenamiento')
modalidades = ['_t1.nii.gz']
#modalidades = ['_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz', '_flair.nii.gz']

# Voy a recorrer cada id y genero una barra de progreso
cont_img = 0
for n, id_ in tqdm(enumerate(train_ids), total = len(train_ids)):

    # La anotación se mantiene para las 4 modalidades
    mask = ReadImage(TRAIN_PATH + '/' + id_ + '/' + id_ + '_seg.nii.gz')
    # Primero solo quiero hallar el tumor completo, no sus regiones
    mask = (mask>0).astype(np.uint8)

    for modalidad in modalidades:
        
        # Guardo cada corte de cada modalidad en x_train con su respectiva anotacion en y_train
        img = ReadImage(TRAIN_PATH + '/' + id_ + '/' + id_ + modalidad) 
        
        # Encuentro solo la región del cerebro
        brain_mask = (img != img[0, 0, 0])
        mask, bbox_seg1 = extract_brain_region(mask, brain_mask, 0)
        mask = resize(mask, (IMG_WIDTH, IMG_HEIGHT, IMG_CHANELS), mode='constant', preserve_range=True)
        img, bbox = extract_brain_region(img, brain_mask, 0)
        
        # Hago un resize de 128x128 (Luego se puede explorar tomar parches de 128 con la escala original)
        img = resize(img, (IMG_WIDTH, IMG_HEIGHT, IMG_CHANELS), mode='constant', preserve_range=True)
        
        # creo un for que recorra hasta 240 y cree así imagenes de 128x128x1 con su anotacion
        for i in range(0, IMG_CHANELS):
            x_train[cont_img,:,:,0] = img[:,:,i]
            y_train[cont_img,:,:,0] = mask[:,:,i]
            cont_img +=1

    
#plt.imshow(x_train[100,:,:,0])
#plt.show()

# PARA TEST

x_test = np.zeros((len(test_ids)*IMG_CHANELS*1, IMG_WIDTH, IMG_HEIGHT, 1), dtype=np.float32)
y_test= np.zeros((len(test_ids)*IMG_CHANELS*1, IMG_WIDTH, IMG_HEIGHT, 1), dtype=np.bool)
print('Ajustando tamano de imagenes de test y anotaciones')

cont_img = 0
for n, id_ in tqdm(enumerate(test_ids), total = len(test_ids)):
    
    # La anotación se mantiene para las 4 modalidades
    mask = ReadImage(TEST_PATH + '/' + id_ + '/' + id_ + '_seg.nii.gz')
    # Primero solo quiero hallar el tumor completo, no sus regiones
    mask = (mask>0).astype(np.uint8)

    for modalidad in modalidades:
        
        # Guardo cada corte de cada modalidad en x_train con su respectiva anotacion en y_train
        img = ReadImage(TEST_PATH + '/' + id_ + '/' + id_ + modalidad) 
        
        # Encuentro solo la región del cerebro
        brain_mask = (img != img[0, 0, 0])
        mask, bbox_seg1 = extract_brain_region(mask, brain_mask, 0)
        mask = resize(mask, (IMG_WIDTH, IMG_HEIGHT, IMG_CHANELS), mode='constant', preserve_range=True)
        img, bbox = extract_brain_region(img, brain_mask, 0)
        
        # Hago un resize de 128x128 (Luego se puede explorar tomar parches de 128 con la escala original)
        img = resize(img, (IMG_WIDTH, IMG_HEIGHT, IMG_CHANELS), mode='constant', preserve_range=True)
        
        # creo un for que recorra hasta 240 y cree así imagenes de 128x128x1 con su anotacion
        for i in range(0, IMG_CHANELS):
            x_test[cont_img,:,:,0] = img[:,:,i]
            y_test[cont_img,:,:,0] = mask[:,:,i]
            cont_img +=1

print('Done!')
# -----------------------------------------------------------
# MODELO
# CONSTRUCCIÓN DEL MODELO
# probar concatenando 3 modalidades
inputs = tf.keras.layers.Input((IMG_WIDTH,IMG_HEIGHT,1))

# Convierto valores enteros en punto flotantes al dividir cada pixel en 255
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

# Contraction path
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides = (2,2), padding= 'same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides = (2,2), padding= 'same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides = (2,2), padding= 'same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides = (2,2), padding= 'same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis = 3)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1,1), activation= 'sigmoid')(c9)

model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Para hallar las predicciones
batch_s = 3
max_epoch = 4

# 1) Guardo el mejor modelo
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_brain_segmentation.h5', verbose =1, save_best_only =True)
# 2) Guardo los logs en una carpeta 'logs' y pido monitorear el valor de loss
callbacks = [tf.keras.callbacks.EarlyStopping(patience =2, monitor='val_loss'), tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(x_train,y_train, validation_split = 0.1, batch_size = batch_s, epochs = max_epoch, callbacks = callbacks)    

#---------------------------------------------------------------------------------
# EVALUACIÓN DEL MODELO

preds_test = model.predict(x_test, verbose=1)
preds_test_t = (preds_test>0.39).astype(np.uint8)

imshow(x_test[800,:,:,0], cmap='gray')
plt.show()
imshow((y_test[800,:,:,0]))
plt.show()
imshow(preds_test_t[800,:,:,0], cmap='gray')
plt.show()

#----------------------------------------------------------------------------------------s
# METRICA
dice = np.sum(preds_test_t[y_test==1])*2.0 / (np.sum(preds_test_t) + np.sum(y_test))
print ('Dice similarity score is: ' + format(dice))
# 22.7 con 0.4
# 22.8 con 0.39
# ------------------------------------------------------------------------------------
# CAMBIOS PARA AGREGAR:

# Cambiar learning rate
# Cambiar a pytorch
# Como extraer el accuracy del modelo despues de realizar las predicciones en test
# Guardar modelo
# Intentar manejarlo como una imagen RGB para este caso, donde cada canal sea un slide de una modalidad diferente (se ignoraria una modalidad)

################################


preds_train = model.predict(x_train, verbose=1)
preds_train_t = (preds_train>0.1).astype(np.uint8)

imshow(x_train[800,:,:,0], cmap='gray')
plt.show()
imshow((y_train[800,:,:,0]))
plt.show()
imshow(preds_train_t[800,:,:,0], cmap='gray')
plt.show()

#----------------------------------------------------------------------------------------s
# METRICA
dice = np.sum(preds_train_t[y_train==1])*2.0 / (np.sum(preds_train_t) + np.sum(y_train))
print ('Dice similarity score is: ' + format(dice))

# 41.42 con 0.39