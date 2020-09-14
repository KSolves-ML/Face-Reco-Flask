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
import ssl
import json, re
from flask_request_params import bind_request_params
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain('/home/ubuntu/Face-Reco-Latest/Face-Reco-Flask/server/ssl/wild.crt', '/home/ubuntu/Face-Reco-Latest/Face-Reco-Flask/server/ssl/wild.key')


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
        header, encoded = request.form["file"].split(",", 1)
        data = b64decode(encoded)

        with open("webcameimg.png", "wb") as f:
                f.write(data)

        name = retrieve.recognize_face(sess_fr,pnet, rnet, onet,feature_array, True)
        result = { 'name': name }
        response = app.response_class(
                response=json.dumps(result),
                status=200,
                mimetype='application/json'
        )

        return response;



@app.route('/align_images', methods=['GET', 'POST'])
def alignImages():
	file1 = request.form["file1"]
	file2 = request.form["file2"]
	file3 = request.form["file3"]
	file4 = request.form["file4"]
	userName = request.form["userName"]

	path = os.path.join(ROOT_DIR+"/lib/data/images/train_raw", userName)
	print("Path====================", str(path))
	if os.path.isdir(path) == False:
		os.mkdir(path)

	for i in range(1, 5):
		if i == 1:
			header, encoded = file1.split(",", 1)
		elif i == 2:
			header, encoded = file2.split(",", 1)
		elif i == 3:
			header, encoded = file3.split(",", 1)
		elif i == 4:
			header, encoded = file4.split(",", 1)

		data = b64decode(encoded)

		with open(ROOT_DIR+"/lib/data/images/train_raw/"+userName+"/"+userName+""+str(i)+".png", "wb") as f:
			f.write(data)
	subprocess.call(['sh', ROOT_DIR + '/align_images.sh'])
	with open(ROOT_DIR + '/extracted_dict.pickle','rb') as f:
		global feature_array
		feature_array = pickle.load(f)
	name = retrieve.recognize_face(sess_fr,pnet, rnet, onet,feature_array, True)
	print("Name===============", name)
	result = { 'name': name }
	response = app.response_class(
		response=json.dumps(result),
		status=200,
		mimetype='application/json'
	)

	return response;

@app.route('/create_embeddings', methods=['GET', 'POST'])
def createEmbeddings():
	subprocess.call(['sh', ROOT_DIR+'/demo.sh'])

	result = { 'message': "Embeddings Created Successfully" }
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
    app.run(debug = True, host= '0.0.0.0', ssl_context = context)
