#!/bin/bash

# Set python path variable to the facenet src directory
# export PYTHONPATH=~/Projects/Face_Recognition/lib/src/
# # remove existing training alignment
# rm -rf ~/Projects/Face_Recognition/Face-Reco-Flask/lib/data/images/train_aligned
#
export PYTHONPATH=~/Face-Reco-Latest/Face-Reco-Flask/

python ~/Face-Reco-Latest/Face-Reco-Flask/lib/src/align/align_dataset_mtcnn.py ~/Face-Reco-Latest/Face-Reco-Flask/lib/data/images/train_raw ~/Projects/Face-Reco-Flask/lib/data/images/train_aligned --image_size 160

export PYTHONPATH=~/Face-Reco-Latest/Face-Reco-Flask
rm -rf ~/Face-Reco-Latest/Face-Reco-Flask/extracted_dict.pickle

python lib/src/create_face_embeddings.py

# export PYTHONPATH=~/Projects/Face_Recognition/Face-Reco-Flask
# rm -rf ~/home/kushal/Projects/Face_Recognition/extracted_dict.pickle
#
# python lib/src/create_face_embeddings.py
# #
# python server/rest-server.py
# # remove existing testing alignment
# rm -rf ~/Projects/facenet/data/images/test_aligned
#
# # align testing images
# python ~/Projects/facenet/src/align/align_dataset_mtcnn.py \
# ~/Projects/facenet/data/images/test_raw \
# ~/Projects/facenet/data/images/test_aligned \
# --image_size 160
#
# # Train new classifier based on training images (generates pickle)
# python ~/Projects/facenet/src/classifier.py TRAIN \
# ~/Projects/facenet/data/images/train_aligned/ \
# ~/Projects/facenet/models/20180402-114759.pb \
# ~/Projects/facenet/models/my_classifier.pkl
#
# # Test face matches using pickle file to classify matching images
# python ~/Projects/facenet/src/classifier.py CLASSIFY \
# ~/Projects/facenet/data/images/test_aligned/ \
# ~/Projects/facenet/models/20180402-114759.pb \
# ~/Projects/facenet/models/my_classifier.pkl
