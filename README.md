#Kaggle datasciencebowl
Code for <a href="http://www.kaggle.com/c/datasciencebowl">National Data Science Bowl</a> on Kaggle.

The model is built using pylearn2, along with a few utilities from sklearn. Training takes around 11 hours total, using ~2GB of GPU ram and ~20GB of system ram (mostly during predictions) with an NVIDIA GTX 980.

Data augmentation is handled during training by randomly rotating, zooming, and translating each image after each epoch. At test time, 20 random augmentations will be predicted by default.

This will achieve ~0.707/0.712 public/private scores on the leaderboard.

#Requirements
pylearn2, scikit-learn, scikit-image

#Doitnow

    python train.py
