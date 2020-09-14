#!/bin/bash

# Set python path variable to the facenet src directory
# export PYTHONPATH=~/Projects/Face_Recognition/lib/src/
# # remove existing training alignment
# rm -rf ~/Projects/FACE_RECOGNITION/data/images/train_aligned
# #
# python ~/Projects/Face_Recognition/lib/src/align/align_dataset_mtcnn.py ~/Projects/Face_Recognition/data/train_raw ~/Projects/Face_Recognition/data/images/train_aligned --image_size 160
export PYTHONPATH=~/Projects/Face-Reco-Flask/lib/src

python ~/Projects/Face-Reco-Flask/lib/src/align/align_dataset_mtcnn.py ~/Projects/Face-Reco-Flask/lib/data/images/train_raw ~/Projects/Face-Reco-Flask/lib/data/images/train_aligned --image_size 160
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
