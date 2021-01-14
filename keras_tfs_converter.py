#!/usr/bin/env python
import tensorflow as tf
import os
import numpy as np
from keras import backend as K
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def dice_coef_loss(mask, mask_pred, smooth=1, weight=1):
    """
    加权后的dice coefficient;
    mask.shape=(h,w)
    """
    #正样本中正确预测的数量
    intersection = (mask * mask_pred).sum()
    #正样本数量，预测为正样本的数量
    union=mask.sum()+weight*mask_pred.sum()
    #dice_coef
    dice_coef=(2. * intersection + smooth) / (union + smooth)
    #dice_coef_loss
    dice_coef_loss=1-dice_coef
    return dice_coef_loss
def export_model(model, export_model_dir, model_version ):
  """
  :param export_model_dir: type string, save dir for exported model
  :param model_version: type int best
  :return:no return
  """
  with tf.get_default_graph().as_default():
    # prediction_signature
    last_conv_layer = model.get_layer('mixed10')
    pool = model.get_layer('global_average_pooling2d_1')
    #iterate = K.function([input_layer],[pool_grads, last_conv_layer.output[0], model.output])

    tensor_info_input = tf.saved_model.utils.build_tensor_info(model.input)
    tensor_info_output = tf.saved_model.utils.build_tensor_info(model.output)
    tensor_pool_grads = tf.saved_model.utils.build_tensor_info(pool.output)
    tensor_last_conv_layer = tf.saved_model.utils.build_tensor_info(last_conv_layer.output)

    #conv_values = iterate([model.input])
    #def_output = tf.saved_model.utils.build_tensor_info(tf.concat(0,[tensor_info_output, tensor_pool_grads, tensor_last_conv_layer]))
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
          inputs={'se': tensor_info_input},
          # Tensorflow.TensorInfo
          #outputs={'result': tensor_info_output,'pooled':tensor_pool_grads,'conv':tensor_last_conv_layer},
          #outputs={'result': [tensor_info_output, tensor_pool_grads, tensor_last_conv_layer]},
          outputs={'pooled': tensor_pool_grads,'result':tensor_info_output,'conv': tensor_last_conv_layer},
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
    )
    print('step1 => prediction_signature created successfully')
    # set-up a builder
    export_path_base = export_model_dir
    export_path = os.path.join( tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(model_version))
    )
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
      # tags:SERVING,TRAINING,EVAL,GPU,TPU
      sess=K.get_session(),
      tags=[tf.saved_model.tag_constants.SERVING],
      signature_def_map={'prediction_signature': prediction_signature,},
    )
    print('step2 => Export path(%s) ready to export trained model' % export_path,
        '\n starting to export model...')
    builder.save(as_text=True)
    print('Done exporting!')

if __name__ == '__main__':
  #model = keras_model()
  #model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.1), metrics=['accuracy'])
  #model.load_weights('./model_data/weights.hdf5')
  #model.summary()
 # model = load_model('/projects/myopia-lzh/data/tanh-regression2.6-best_weights-mixed.h5')
  #model = load_model('/home/ljm/EXCW_segment_model1-0-weights.best.hdf5')
  model = load_model('./fundus_softmax_model2-1.hdf5')
  print('-'*50)
  export_model(model, './export_model/fundus', 1)
