import os
import cv2
import grpc
import numpy as np
from PIL import Image
from tensorflow.contrib.util import make_tensor_proto
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from medfirstviewsvr.utils.report_img_process import img_adjust, auto_canny

#TFS_URL = '192.168.1.143'
TFS_URL = '192.168.0.202'
CHANNEL = grpc.insecure_channel('{host}:{port}'.format(host=TFS_URL, port=8500))
STUB = prediction_service_pb2_grpc.PredictionServiceStub(CHANNEL)

def get_fundus_img(img):
  """ 获取img，并返回299*299格式

  :param img: cv2格式的img
  :return: resize过后的img
  """
  scale = float(img.shape[1]) / 800
  display_img = img
  img = cv2.resize(img, (800, int(800 / img.shape[1] * img.shape[0])))
  res, bbox, status = img_adjust(img)
  img = cv2.resize(res, (299, 299))
  if not status:
    return img, display_img
  display_img = display_img[int(scale * bbox[0]):int(scale * bbox[1]), int(scale * bbox[2]):int(scale * bbox[3])]
  predict_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return predict_img, display_img

def tfs_request(tfs_model, version, request_data):
  request = predict_pb2.PredictRequest()
  request.model_spec.name = tfs_model
  request.model_spec.signature_name = 'prediction_signature'
  request.model_spec.version.value = version
  request.inputs['se'].CopyFrom(make_tensor_proto(request_data))
  try:
      lr_result = STUB.Predict(request, 20.0)
  except:
      import traceback
      traceback.print_exc()
      return 'error', 0
  return lr_result, 1

def lr_predict(data_array):
    LR_CLASS_NAMES = ['左眼', '右眼']
    lr_result, lr_status = tfs_request('lr_eyes', 1, data_array.reshape((1, 299, 299, 3)))
    rep_lr_result = np.array(lr_result.outputs['result'].float_val).reshape((2,))  # 左右眼预测结果
    lr_result = LR_CLASS_NAMES[int(np.argmax(rep_lr_result, axis=0))]
    return lr_result

def quality(img):
    upload_img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (299, 299))
    upload_img = upload_img.astype('float32')
    predict_data = upload_img / 255.
    predict_data -= [0.485, 0.456, 0.406]
    predict_data /= [0.229, 0.224, 0.225]
    quality_result, quality_status = tfs_request('fundus_quality', 1, predict_data[np.newaxis, ...])
    rep_quality = np.array(quality_result.outputs['result'].float_val).reshape((3,))
    fundus_quality = np.argmax(rep_quality, axis=0)
    print(rep_quality, fundus_quality)
    return fundus_quality

if __name__ == '__main__':
    datadir = 'quality'
    result_lr = []
    for name in os.listdir(datadir):
        imgpath = os.path.join(datadir, name)
        try:
            img = cv2.imread(imgpath)
            #predict_img, display_img = get_fundus_img(img)
            #result = lr_predict(predict_img.astype('float32'))
            result = quality(img)
            result_lr.append(name + ', ' + str(result))
        except Exception as e:
            print(imgpath, e)
            continue
    with open('quality.txt', 'w') as fw:
        fw.write('\n'.join(result_lr))
    '''
    imgpath = 'os.jpeg'
    img = cv2.imread(imgpath)
    predict_img, display_img = get_fundus_img(img)
    result = lr_predict(predict_img.astype('float32'))
    print(result)
    '''
