import os
import urllib.request
import PIL.Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, flash, request, redirect, url_for, render_template
app = Flask(__name__)
ALLOWED_EXTENSIONS = list(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = '/static'
def allowed_file(filename):
	return filename[-3:] in ALLOWED_EXTENSIONS or filename[-4:] in ALLOWED_EXTENSIONS
@app.route('/')
def upload_form():
	return render_template('upload.html')

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

@app.route('/', methods=['POST'])
def upload_image():
	file = request.files['file']
	file1 = request.files['file1']
	flag1='.jpg'
	flag='.jpg'
	if '.png' in file.filename:
		flag='.png'
	if '.png' in file1.filename:
		flag1='.png'
	if file and  allowed_file(file.filename):
		res='static/content'+flag
		file.save(res)
	else:
		return "Photo's only allowed,go back to continue."
	if file1 and  allowed_file(file1.filename):
		res1='static/style'+flag1
		file1.save(res1)
	else:
		return "Photo's only allowed,go back to continue."
	content=load_img(res)
	style=load_img(res1)
	hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
	stylized_image = hub_model(tf.constant(content), tf.constant(style))[0]
	res2=tensor_to_image(stylized_image)
	temp='static/res.jpg'
	res2.save(temp)
	return render_template('upload.html',filename2=temp,filename1=res1,filename=res)

     
if __name__ == "__main__":
    app.run()