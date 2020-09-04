#!flask/bin/python
################################################################################################################################
#------------------------------------------------------------------------------------------------------------------------------
# This file implements the REST layer. It uses flask micro framework for server implementation. Calls from front end reaches
# here as json and being branched out to each projects. Basic level of validation is also being done in this file. #
#-------------------------------------------------------------------------------------------------------------------------------
################################################################################################################################
from flask import Flask, jsonify, abort, request, make_response, url_for,redirect, render_template, json
from flask_httpauth import HTTPBasicAuth
from werkzeug.utils import secure_filename
import os
import sys
import random
from tensorflow.python.platform import gfile
from six import iteritems
sys.path.append('..')
import numpy as np
from lib.src import retrieve
from lib.src.align import detect_face
import tensorflow as tf
import pickle
from tensorflow.python.platform import gfile
from base64 import b64decode
from flask_cors import CORS

app = Flask(__name__, static_url_path = "")
CORS(app)
auth = HTTPBasicAuth()

#==============================================================================================================================
#
#    Loading the stored face embedding vectors for image retrieval
#
#
#==============================================================================================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(ROOT_DIR + '/extracted_dict.pickle','rb') as f:
	feature_array = pickle.load(f)

model_exp = ROOT_DIR + '/models'
graph_fr = tf.Graph()
sess_fr = tf.Session(graph=graph_fr)
with graph_fr.as_default():
	saverf = tf.train.import_meta_graph(os.path.join(model_exp, 'model-20180402-114759.meta'))
	saverf.restore(sess_fr, os.path.join(model_exp, 'model-20180402-114759.ckpt-275'))
	pnet, rnet, onet = detect_face.create_mtcnn(sess_fr, None)
#==============================================================================================================================
#
#  This function is used to do the face recognition from video camera
#
#
#==============================================================================================================================
@app.route('/facerecognitionLive', methods=['GET', 'POST'])
def face_det():

    retrieve.recognize_face(sess_fr,pnet, rnet, onet,feature_array, False)

@app.route('/facerecognition', methods=['GET', 'POST'])
def face():
	print("Hiitttttttttttttttttt")
	header, encoded = request.form["file"].split(",", 1)
	data = b64decode(encoded)
	with open("webcameimg.png", "wb") as f:
		f.write(data)
	name = retrieve.recognize_face(sess_fr,pnet, rnet, onet,feature_array, True)
	print("Name===============", name)
	result = { 'name': name }
	response = app.response_class(
		response=json.dumps(result),
		status=200,
		mimetype='application/json'
	)

	return response;

#==============================================================================================================================
#
#                                           Main function                                                        	            #
#
#==============================================================================================================================
@app.route("/")
def main():

    return render_template("main.html")
if __name__ == '__main__':
    app.run(debug = True, host= '0.0.0.0')
