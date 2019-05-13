import os
import glob
import numpy as np
import heapq
import multiprocessing as mp
import sys


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
import subprocess

# pp = pprint.PrettyPrinter(indent=2)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = vgg16.VGG16(weights='imagenet', include_top=True)
model_extractfeatures = Model(model.input, outputs=model.get_layer('fc2').output)


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
      inner_dict[c[1]] = (to_float_arr(np.asarray(c[2][1:-2].split(', '))))
      d[c[0]] = inner_dict
    else:
      d[c[0]][c[1]] = (to_float_arr(np.asarray(c[2][1:-2].split(', '))))
  return d

def calculate_dist(a, b):
  return np.linalg.norm(a-b)


def calculate_closest(fin_centroids, pt, split):
  minimum = ""
  dist = 10000000000
  distances = {}
  for key, value in fin_centroids[split].items():
    curr_dist = calculate_dist(value, pt)
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

def quantize(img_path, fin_centroids):
  feat = get_features(img_path)
  query = np.array_split(feat,16)
  comp = []
  dist = {}
  for i, sub in zip(range(16),query):
    calc = calculate_closest(fin_centroids, sub, str(i))
    dist[str(i)] = calc[1]
    comp.append(calc[0])
  return np.asarray(comp), dist

  
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
      if len(res) <= k:
        heapq.heappush(res,(-dist, db[0].iloc[x])) 
      else:
        if -dist > res[0][0]:
          heapq.heappop(res)
          heapq.heappush(res,(-dist, db[0].iloc[x]))
      i = i+1
  return res


def get_MD5(k,img_path="plane.png", db_path="data/points.csv", centroids_path="data/centroids.pkl"):
  fin_centroids = centroid_to_dict(read_centroids(centroids_path))
  matrix = quantize(img_path, fin_centroids)

  return apply_async_with_callback(matrix)

result_list = []
def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)

def apply_async_with_callback(matrix):
    files = glob.glob("data/database-*")
    pool = mp.Pool(8)
    for i, f in enumerate(files):
        pool.apply_async(find_min, args = (5, matrix[1], f ), callback = log_result)

    pool.close()
    pool.join()
    return result_list

try:
    md5s = get_MD5(sys.argv[1], sys.argv[2])
except IndexError:
    # print(sys.argv[1])
    print("Provide K and Image url as input args")

# print("End:", time.ctime())



def find_min_local(K, data):
  res = []
  for sub_data in data:
    for k in sub_data:
      if len(res) <= K:
        heapq.heappush(res,(k[0], k[1])) 
      else:
        if k[0] > res[0][0]:
          heapq.heappop(res)
          heapq.heappush(res,(k[0], k[1]))
  return res


# yolo = find_min_local(5, md5s)




finals =[]

for md5 in yolo:
    cmd = "LC_ALL=C fgrep -m 1 '"+ str(md5[1]) +"' /media/vinay/My\ Passport/yfcc100m/data | cut -f17"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    temp = process.communicate()[0]
    finals.append(str(temp.decode("utf-8")))
print(finals)
# print(time.ctime())
