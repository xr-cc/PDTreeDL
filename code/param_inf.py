import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

import tensorflow.keras as keras
from tensorflow.keras import layers

from tensorflow.keras.layers import Lambda, Input, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy, mape
import tensorflow.keras.backend as K
from sklearn.metrics import mean_squared_error, r2_score


LB = 'ParamInf'

FILE_PATH = "data/BD_50/"
INPUT_FORM = "flat"  
PARAMS_TO_INFER = ['R_nought','transmission_rate','infectious_time']

N_TAXA = 98  
N_TREE_PER_FILE = 1000
NUM_FILES = 100

# path to save result files
SAVE_PATH = os.getcwd()+"/output_{}_{}_{}_nf{}/".format(LB,INPUT_FORM,"-".join(PARAMS_TO_INFER),NUM_FILES)

LEARNING_RATE = 0.001
N_EPOCHS = 10
BATCH_SIZE = 32

DATASET_SIZE = N_TREE_PER_FILE*NUM_FILES

MAT_SIZE = 100 #400 #
if INPUT_FORM=="flat":
  INPUT_SHAPE = (MAT_SIZE**2//2,2)
else:
  INPUT_SHAPE = (MAT_SIZE, MAT_SIZE, 1)

# print(INPUT_SHAPE)

LABEL_SHAPE = (1,)

TRI_IDX = np.triu_indices(N_TAXA, k = 0)
TRI_IDX = [[TRI_IDX[0][i],TRI_IDX[1][i]] for i in range(len(TRI_IDX[0]))]

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


################################### Functions ######################################

# helper functions for performance measure 
# mean relative error
def MRE(y_true,y_pred):
  return np.mean(np.divide(np.abs(y_pred-y_true), y_true, out=np.ones_like(y_true), where=y_true!=0))
# mean relative bias
def MRB(y_true,y_pred):
  return np.mean(np.divide(y_pred-y_true, y_true, out=np.ones_like(y_true), where=y_true!=0))


def loadMatData(file_idx, pad=0, expand_dims=0, mat_type="F", normalize=True, lower=True):
  dataset = tf.data.TextLineDataset(FILE_PATH+"FWvec/{}mat_{}.txt".format(mat_type,file_idx))
  dataset = dataset.map(lambda string: tf.strings.to_number(tf.compat.v1.string_split([string]).values, tf.float32))
  dataset = dataset.map(lambda line: tf.tensor_scatter_nd_update(tf.zeros([N_TAXA,N_TAXA], dtype=tf.float32), TRI_IDX, line))
  if lower:
    dataset = dataset.map(lambda mat: tf.transpose(mat, perm=[1,0]))
  if normalize:
    dataset = dataset.map(lambda line: tf.math.divide(line,N_TAXA//2))
  if pad:
    dataset = dataset.map(lambda mat: tf.pad(mat, tf.constant([[0, pad], [0, pad]]), "CONSTANT"))
  if expand_dims:
    dataset = dataset.map(lambda mat: tf.expand_dims(mat, axis=-1))  
  return dataset


def loadFlatData(file_idx, pad=0, mat_type="F", normalize=True, lower=True):
  dataset = tf.data.TextLineDataset(FILE_PATH+"FWvec/{}mat_{}.txt".format(mat_type,file_idx))
  dataset = dataset.map(lambda string: tf.strings.to_number(tf.compat.v1.string_split([string]).values, tf.float32))
  
  if normalize:
    dataset = dataset.map(lambda line: tf.math.divide(line,N_TAXA//2))
  # padding
  if pad:
    dataset = dataset.map(lambda line: tf.pad(line, tf.constant([[0, pad]]), "CONSTANT"))
  return dataset


def loadParam(file_idx,param_lb):
    # Read file
    params_table = pd.read_csv(FILE_PATH+"info/info_{}.txt".format(file_idx),delimiter='\t')
    lb_temp = tf.compat.as_str_any(param_lb)
    y_lb = np.array(params_table[lb_temp].values)

    for i in range(len(params_table)):
        y = [y_lb[i]]
        yield y


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# plot training history loss
def plotLoss(history,loss="loss"):
  fig = plt.figure()
  plt.plot(history.history[loss])
  plt.title(loss)
  plt.ylabel('loss')
  plt.xlabel('epoch')
  return fig

def plotReg(y_test,y_pred,param='R_nought'):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(y_test, y_pred, alpha=0.5)
  lb = min(np.min(y_test), np.min(y_pred))
  ub = max(np.max(y_test), np.max(y_pred))
  ax.plot([lb,ub],[lb,ub], c='k')
  ax.axis('equal')
  plt.title(param)
  plt.xlabel('ground truth')
  plt.ylabel('prediction')
  return fig, ax

def plotReco(X_test, X_reco, num_test = 10):
  fig, axes = plt.subplots(2, num_test, figsize=(5*num_test/2,5))
  for i in range(num_test):
      input_i = X_test[i]
      r_i = X_reco[i]
      axes[0,i].imshow(np.squeeze(input_i, axis=2), cmap='gray_r')
      axes[1,i].imshow(np.squeeze(np.clip(r_i,0,1), axis=2), cmap='gray_r')

  return fig

print("-------START-------")

########################################################################################

print("--------------Creating Data Pipeline and Network-----")

############ Load Data using data API

for file_idx in range(1,NUM_FILES+1):

  if INPUT_FORM=="flat":
    ## load two flattened vector and concatenate
    dataset_W = loadFlatData(file_idx, pad=INPUT_SHAPE[0]-N_TAXA*(N_TAXA+1)//2, mat_type="W", normalize=True, lower=False)
    dataset_F = loadFlatData(file_idx, pad=INPUT_SHAPE[0]-N_TAXA*(N_TAXA+1)//2, mat_type="F", normalize=True, lower=True)
    # combine F and W
    dataset = tf.data.Dataset.zip((dataset_W, dataset_F))
    dataset = dataset.map(lambda l1,l2: tf.transpose(tf.stack([l1,l2], 0),perm=[1,0]))
  elif INPUT_FORM=="trigF":
    ## load a triangular matrix only
    # load matrix data
    dataset = loadMatData(file_idx, pad=MAT_SIZE-N_TAXA, expand_dims=-1, mat_type="F",normalize=True, lower=True)
  elif INPUT_FORM=="trigW":
    dataset = loadMatData(file_idx, pad=MAT_SIZE-N_TAXA, expand_dims=-1, mat_type="W",normalize=True, lower=True)
  else:
    ## load two triangular matrices and concatenate to a full one
    dataset_W = loadMatData(file_idx, pad=MAT_SIZE-N_TAXA, expand_dims=-1,mat_type="W", normalize=True, lower=False)
    dataset_F = loadMatData(file_idx, pad=MAT_SIZE-N_TAXA, expand_dims=-1,mat_type="F", normalize=True, lower=True)
    # combine F and W
    dataset = tf.data.Dataset.zip((dataset_W, dataset_F))
    dataset = dataset.map(lambda mat1,mat2: tf.math.add(mat1,mat2)) 


  ### load parameters 
  ds_list = []
  for param in PARAMS_TO_INFER:
    ds_lb = tf.data.Dataset.from_generator(loadParam, args=(file_idx,param), output_types=(tf.float32), output_shapes=((1,)))
    ds_list.append(ds_lb)
  
  if file_idx==1:
    # combine
    ds = tf.data.Dataset.zip((dataset,)+tuple(ds_list))
    # ds.element_spec
    ds = ds.map(lambda mat, *params: {**{"tree_encoding": mat}, **{"{}".format(PARAMS_TO_INFER[i]):l for i,l in enumerate(params)}})
  else:
    # append
    ds2 = tf.data.Dataset.zip((dataset,)+tuple(ds_list))
    ds2 = ds2.map(lambda mat, *params: {**{"tree_encoding": mat}, **{"{}".format(PARAMS_TO_INFER[i]):l for i,l in enumerate(params)}})
    ds = ds.concatenate(ds2)  

print("Dataset:")
print(ds.element_spec)
print("-------------------------------")

############ Process Data


train_val_test_ratio = [0.8,0.1,0.1]

train_size = int(train_val_test_ratio[0] * DATASET_SIZE)
val_size = int(train_val_test_ratio[1] * DATASET_SIZE)
test_size = int(train_val_test_ratio[2] * DATASET_SIZE)

ds = ds.shuffle(buffer_size=DATASET_SIZE//2)
train_data = ds.take(train_size)
test_data= ds.skip(train_size)
val_data = test_data.skip(val_size)
test_data = test_data.take(test_size)


############ Build Model

latent_dim = 2
intermediate_dim = 32
print_model=True

param_inputs = []
for param in PARAMS_TO_INFER:
  param_inputs.append(keras.Input(shape=LABEL_SHAPE, name=param))

encoder_inputs = keras.Input(shape=INPUT_SHAPE, name='tree_encoding')


fil1, fil2, fily = 32, 32, 16 

if INPUT_FORM=="flat":
  
  ### CONV 1D Model

  x = layers.Conv1D(filters=fil1, kernel_size=3, strides=2, activation='relu', padding="same", input_shape=INPUT_SHAPE)(encoder_inputs)
  x = layers.Conv1D(filters=fil2, kernel_size=10, strides=2, activation='relu', padding="same", input_shape=INPUT_SHAPE)(x)
  intermediate_shape = K.int_shape(x)[1:]
  x = layers.Flatten()(x)
  x = layers.Dense(intermediate_dim, activation="relu")(x)

  # q(z|x)
  z_mean = layers.Dense(latent_dim, name='z_mean')(x)
  z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
  z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

  # parameter inference
  y_predictions = []
  for param in PARAMS_TO_INFER:
    pre_pred = layers.Dense(fily, activation="relu")(z_mean)
    y_predictions.append(layers.Dense(1, name='{}_pred'.format(param))(pre_pred))

  encoder = keras.Model([encoder_inputs]+param_inputs, [z_mean, z_log_var, z]+y_predictions, name='encoder')

  latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
  x = layers.Dense(np.prod(intermediate_shape), activation="relu")(latent_inputs)
  x = layers.Reshape(intermediate_shape)(x)
  x = layers.Conv1DTranspose(fil2, 10, activation="relu", strides=1, padding="same")(x)
  x = layers.UpSampling1D(size=2)(x)
  x = layers.Conv1DTranspose(fil1, 3, activation="relu", strides=1, padding="same")(x)
  x = layers.UpSampling1D(size=2)(x)
  decoder_outputs = layers.Conv1DTranspose(2, 3, activation="sigmoid", padding="same")(x)
  decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

else:

  ## CONV 2D Model

  x = layers.Conv2D(fil1, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
  x = layers.Conv2D(fil2, 3, activation="relu", strides=1, padding="same")(x)
  intermediate_shape = K.int_shape(x)[1:]
  x = layers.Flatten()(x)
  x = layers.Dense(intermediate_dim, activation="relu")(x)

  # q(z|x)
  z_mean = layers.Dense(latent_dim, name='z_mean')(x)
  z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
  z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

  # parameter inference
  y_predictions = []
  for param in PARAMS_TO_INFER:
    pre_pred = layers.Dense(fily, activation="relu")(z_mean)
    y_predictions.append(layers.Dense(1, name='{}_pred'.format(param))(pre_pred))

  # instantiate encoder model
  encoder = keras.Model([encoder_inputs]+param_inputs, [z_mean, z_log_var, z]+y_predictions, name='encoder')

  latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
  x = layers.Dense(np.prod(intermediate_shape), activation="relu")(latent_inputs)
  x = layers.Reshape(intermediate_shape)(x)
  x = layers.Conv2DTranspose(fil2, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Conv2DTranspose(fil1, 3, activation="relu", strides=1, padding="same")(x)
  decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
  decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")


if print_model:
  print("############## ENCODER ##############")
  print(encoder.summary())
  print("#####################################")
  print("############## DECODER ##############")
  print(decoder.summary())
  print("#####################################")
# save model plot
keras.utils.plot_model(encoder, SAVE_PATH+"encoder_{}.png".format(INPUT_FORM), show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=48)
keras.utils.plot_model(decoder, SAVE_PATH+"decoder_{}.png".format(INPUT_FORM), show_shapes=True, show_dtype=True, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=48)

# instantiate VAE model
outputs = decoder(encoder([encoder_inputs]+param_inputs)[2])
vae = keras.Model([encoder_inputs]+param_inputs, outputs, name='vae_model')

# customize loss of VAE model
recstr_loss = binary_crossentropy(K.flatten(encoder_inputs),K.flatten(outputs))
kl_loss = -0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
label_losses = []
all_label_loss = 0
for i_param, param in enumerate(PARAMS_TO_INFER): 
  l = 0.01*mape(param_inputs[i_param],y_predictions[i_param])
  all_label_loss += l
  label_losses.append(l) 

# vae_loss = K.mean(reconstruction_loss+kl_loss+label_loss)
vae_loss = K.mean(recstr_loss+kl_loss+all_label_loss)
vae.add_loss(vae_loss)

# add metrics
vae.add_metric(recstr_loss, name='recstr_loss', aggregation='mean')
vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')
for i_param, param in enumerate(PARAMS_TO_INFER): 
  vae.add_metric(label_losses[i_param], name='{}_loss'.format(param), aggregation='mean')

### END OF MODEL
print("############## FULL MODEL ##############")
print(vae.summary())
print("########################################")


############ Train Model

opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
vae.compile(optimizer=opt)
history = vae.fit(train_data.batch(BATCH_SIZE), validation_data=val_data.batch(BATCH_SIZE), epochs=N_EPOCHS)#, steps_per_epoch=50)

# Plot Loss
for loss_type in ["loss","recstr_loss","kl_loss"]+["{}_loss".format(param) for param in PARAMS_TO_INFER]:
  fig = plotLoss(history,loss=loss_type)
  fig.savefig(SAVE_PATH+"{}_{}_{}.png".format(loss_type,INPUT_FORM,LB))


# Predict and Evaluate
print("--------------Evaluating Performances-----")

# get test data
all_test = list(test_data.as_numpy_iterator())
X_test = np.array([i['tree_encoding'] for i in all_test])
print("testing:", X_test.shape)
y_tests = []
for param in PARAMS_TO_INFER:
  y_tests.append(np.array([i[param] for i in all_test]))
  
[z_mean, z_log_var, z, *y_pred] = encoder.predict({**{"tree_encoding": X_test}, **{"{}".format(PARAMS_TO_INFER[i]):np.zeros_like(l) for i,l in enumerate(y_tests)}})

# evaluate parameter inference
fig_regr_list = []
for i_yp, y_p in enumerate(y_pred):
  print(PARAMS_TO_INFER[i_yp])
  y1 = y_tests[i_yp]
  y2 = y_pred[i_yp]
  print("mse:{}, r2:{}".format(mean_squared_error(y1,y2),r2_score(y1,y2)))
  print("MRE:{}, MRB: {}".format(MRE(y1,y2),MRB(y1,y2)))
  fig_regr, ax_regr = plotReg(y1,y2,param=PARAMS_TO_INFER[i_yp])
  textstr = '\n'.join((
    "MRE:{:.4f}".format(MRE(y1,y2)),
    "MRB:{:.4f}".format(MRB(y1,y2)),
    "MSE:{:.4f}".format(mean_squared_error(y1,y2)),
    "R2: {:.4f}".format(r2_score(y1,y2)))
  )
  ax_regr.text(0.05, 0.95, textstr, transform=ax_regr.transAxes, fontsize=10,verticalalignment='top')
  fig_regr_list.append(fig_regr)
  fig_regr.savefig(SAVE_PATH+"regr_{}_{}_{}.png".format(PARAMS_TO_INFER[i_yp],INPUT_FORM,LB))

  if INPUT_FORM=="flat":
    y_arr = np.hstack((y1,y2))
    np.savetxt(SAVE_PATH+'y_{}.out'.format(PARAMS_TO_INFER[i_yp]), y_arr, delimiter=',')

# evaluate reconstruction 
code = z_mean
X_reco = decoder.predict(code) 

if INPUT_FORM=="flat":
  print("reconstruction:", code.shape, X_reco.shape)
  err_reco = mse(X_test,X_reco).numpy()
  print("MSE:",err_reco.sum(axis=1).mean())
  np.savetxt(SAVE_PATH+'latent_space.out', code, delimiter=',')
else:
  fig_recstr = plotReco(X_test, X_reco, num_test = 10)
  fig_recstr.savefig(SAVE_PATH+'recstr_({})_{}_{}.png'.format("-".join(PARAMS_TO_INFER),INPUT_FORM,LB))



