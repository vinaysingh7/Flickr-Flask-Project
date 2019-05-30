import os
from pathlib import Path
import glob
import numpy as np
import heapq
import multiprocessing as mp
import sys
from pymongo import MongoClient


import tensorflow as tf

from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import pprint
import pandas as pd
import pickle
import time
import flask
from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
import urllib.request
from flask_cors import CORS


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
CORS(app)
api = Api(app)

image_request = api.model('Results', 
                            {'imageURL': fields.String(required=True,
                                                    decription='Image Url',
                                                    help='Sepal Length cannot be blank'),
                            'imagesCount': fields.String(required=True,
                                                    decription='Image Count',
                                                    help='Sepal Length cannot be blank')})
                                            

name_space = api.namespace('/', description='Prediction APIs')

model = None
model_extractfeatures = None
graph = None
fin_centroids = None
collection = None
result_list = []
path = os.getcwd()
images_path = Path(path).parent
@name_space.route('/results')
class Results(Resource):

  def read_centroids(path):
    f = open(path, "rb")
    centroids = pickle.load(f)
    f.close()
    return centroids

  def to_float_arr(arr):
    return np.array(list(map(float, arr)))

  def centroid_to_dict(centroids):
    d = {}
    for c in centroids:
      if c[0] not in d.keys():
        inner_dict = {}
        inner_dict[c[1]] = (Results.to_float_arr(np.asarray(c[2][1:-2].split(', '))))
        d[c[0]] = inner_dict
      else:
        d[c[0]][c[1]] = (Results.to_float_arr(np.asarray(c[2][1:-2].split(', '))))
    return d

  def calculate_dist(a, b):
    return np.linalg.norm(a-b)

  def calculate_closest(fin_centroids, pt, split):
    minimum = ""
    dist = 10000000000
    distances = {}
    for key, value in fin_centroids[split].items():
      curr_dist = Results.calculate_dist(value, pt)
      distances[key] = curr_dist
      if curr_dist < dist:
        dist = curr_dist
        minimum = key
    return minimum, distances

  def get_features(img_path):

    img = image.load_img(img_path, target_size=(224, 224), interpolation='bicubic', color_mode='rgb')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='caffe')
    fc2_features = model_extractfeatures.predict(x)
    fc2_features = np.squeeze(fc2_features)
    return fc2_features
  
  def find_min(k, matrix, database):

    def calculate_dist_to_entry(matrix, entry):
      dist = 0
      for split, centroid in zip(range(len(entry)), entry):
        dist = dist + matrix[str(split)][centroid]
      return dist 

    res = []
    db = pd.read_pickle(database)
    i =0
    for x in range(len(db)):
        dist =  calculate_dist_to_entry(matrix, db[1].iloc[x])
        if len(res) < k:
          heapq.heappush(res,(-dist, db[0].iloc[x])) 
        else:
          if -dist > res[0][0]:
            heapq.heappop(res)
            heapq.heappush(res,(-dist, db[0].iloc[x]))
        i = i+1
    return res

  def log_result(result):
      # This is called whenever foo_pool(i) returns a result.
      # result_list is modified only by the main process, not the pool workers.
      global result_list
      result_list.append(result)

  def apply_async_with_callback(matrix, k):
      global result_list
      result_list = []
      files = glob.glob("data/database-*")
      pool = mp.Pool(8)
      for i, f in enumerate(files):
          pool.apply_async(Results.find_min, args = (k, matrix[1], f ), callback = Results.log_result)

      pool.close()
      pool.join()
      return result_list

  def find_min_local(K, data):
    res = []
    for sub_data in data:
      for k in sub_data:
        res.append(k)
    return sorted(res, reverse=True)[0:K]

  def init():
      global model
      global model_extractfeatures
      global graph
      global collection
      model = vgg16.VGG16(weights='imagenet', include_top=True)
      model_extractfeatures = Model(model.input, outputs=model.get_layer('fc2').output)
      graph = tf.get_default_graph()
      global fin_centroids
      fin_centroids = Results.centroid_to_dict(Results.read_centroids("data/centroids.pkl"))
      client = MongoClient()
      db = client.flickrurls
      collection = db["final_data"]

  def options(self):
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

  @api.expect(image_request)
  def post(self):
    try:
      result = request.json
      print(result)
      url = result['imageURL']
      images_count = int(result['imagesCount'])
      image_name = 'query_image.jpg'
      
      
      images_path = Path.cwd().parent / 'reactUI/flickr-react/src/images'

      address = str(images_path) + '/' + image_name
      urllib.request.urlretrieve(url,address)
      
      img = image.load_img(address, target_size=(224, 224), interpolation='bicubic', color_mode='rgb')
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x, mode='caffe')
      
      with graph.as_default():
        fc2_features = model_extractfeatures.predict(x)
      fc2_features = np.squeeze(fc2_features)        
      query = np.array_split(fc2_features,16)
      comp = []
      dist = {}
      for i, sub in zip(range(16),query):
        calc = Results.calculate_closest(fin_centroids, sub, str(i))
        dist[str(i)] = calc[1]
        comp.append(calc[0])
      print(time.ctime())
      md5s = Results.apply_async_with_callback((np.asarray(comp), dist), images_count)
      print(time.ctime())

      yolo = Results.find_min_local(images_count, md5s)

      finals =[]
      print(yolo)
      print(time.ctime())
      
      for md5 in yolo:
        finals.append(collection.find_one({ "md5": str(md5[1]) })['url'])
      example=finals

      print(example)
      return {
        'searchImage': address,
        'similarImagesList': example
      }, 200
    except Exception as error:
      return {"status", "Could not find similar images"}, 500

if __name__ == "__main__":
  Results.init()
  app.run(threaded=False)
