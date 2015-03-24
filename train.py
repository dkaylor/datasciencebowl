import os
import numpy as np
from data import load_images
from realtime_augment import RealtimeAugment

from pylearn2.datasets import preprocessing
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.train import Train
from pylearn2.train_extensions import best_params
from pylearn2.termination_criteria import MonitorBased
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.models.maxout import MaxoutConvC01B, Maxout
from pylearn2.models import mlp
from pylearn2.space import Conv2DSpace
from pylearn2.utils import serial
from theano import tensor as T
from theano import function
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedKFold

#optionally set to False to make predictions on a saved model
retrain = True

#nn params
img_dim = 82
central_window_shape = img_dim
max_epochs = 250
learn_rate = .025
batch_size = 128

momentum_start = .5
momentum_end = .9
momentum_saturate = max_epochs
decay_factor = .025 * learn_rate
decay_saturate = max_epochs

view_converter_dim = 1
axes = ['b',0,1,'c']
view_converter = DefaultViewConverter(shape=[img_dim, img_dim, view_converter_dim], axes=axes)

#image augment params
scale_diff = .2
translation = 9.
center_shape = (img_dim-2, img_dim-2)
preprocess = [preprocessing.GlobalContrastNormalization(sqrt_bias=10.,use_std=True),
                    preprocessing.LeCunLCN([img_dim, img_dim], batch_size=5000)]

as_grey = True

#number of random test augmentations to predict
test_examples = 20

#convolutional layers
l1= MaxoutConvC01B(layer_name='l1', 
                tied_b=1,
                num_channels=32, num_pieces=2, pad=0, 
                kernel_shape=[4,4], pool_shape=[2,2], pool_stride=[2,2],
                max_kernel_norm= 1.9365, irange=.025)
l2= MaxoutConvC01B(layer_name='l2',
                tied_b=1,
                num_channels=64, num_pieces=2, pad=3, 
                kernel_shape=[4,4], pool_shape=[2,2], pool_stride=[2,2],
                max_kernel_norm= 1.9365, irange=.025)
l3 = MaxoutConvC01B(layer_name='l3',
                tied_b=1,
                num_channels=128, num_pieces=2, pad=3,
                kernel_shape=[3,3], pool_shape=[2,2], pool_stride=[2,2],
                max_kernel_norm= 1.9365, irange=.025)
l4 = MaxoutConvC01B(layer_name='l4',
                tied_b=1,
                num_channels=128, num_pieces=2, pad=3,
                kernel_shape=[3,3], pool_shape=[2,2], pool_stride=[2,2],
                max_kernel_norm= 1.9365, irange=.025)
l5 = MaxoutConvC01B(layer_name='l5',
                tied_b=1,
                num_channels=256, num_pieces=2, pad=2,
                kernel_shape=[3,3], pool_shape=[2,2], pool_stride=[2,2],
                max_kernel_norm= 1.9365, irange=.025)
                
l6 = MaxoutConvC01B(layer_name='l6',
                tied_b=1,
                num_channels=256, num_pieces=2, pad=2,
                kernel_shape=[3,3], pool_shape=[2,2], pool_stride=[2,2],
                max_kernel_norm= 1.9365, irange=.025)

#dense layers                                       
l7 = Maxout(layer_name='l7', num_units=1024, num_pieces=2, irange=.025)
l8 = Maxout(layer_name='l8', num_units=2048, num_pieces=2, irange=.025)
output_layer = mlp.Softmax(layer_name='y', n_classes=121, irange=.01)

layers = [l1,l2,l3,l4,l5, l6,l7, l8, output_layer]

images = []
y = []
file_names = []
dimensions = []
    
train_labels = [x for x in os.listdir("train") if os.path.isdir("{0}{1}{2}".format("train", os.sep, x))]
train_directories = ["{0}{1}{2}".format("train", os.sep, x) for x in train_labels]
train_labels, train_directories = zip(*sorted(zip(train_labels, train_directories), key=lambda x: x[0]))

for idx, folder in enumerate(train_directories):
    
    for f_name_dir in os.walk(folder):
          dir_images, fnames, dims = load_images(f_name_dir, img_dim=img_dim, as_grey=as_grey)
          images = images + dir_images
          y = y + [idx for x in dir_images]
          dimensions = dimensions + dims
          file_names = file_names + fnames

def to_one_hot(l):
    out = np.zeros((len(l), len(set(l))))
    for idx, label in enumerate(l):
        out[idx, label] = 1
    return out

y = to_one_hot(y)

def predict(model, X_test):
    model.set_batch_size(batch_size)
            
    m = X_test.X.shape[0]
    extra = batch_size - m % batch_size
    if extra > 0:
        X_test.X = np.concatenate([X_test.X, np.zeros((extra, X_test.X.shape[1]), dtype=X_test.X.dtype)], axis=0)
        
    X_m = model.get_input_space().make_theano_batch()
    Y = model.fprop(X_m)
    f = function([X_m], Y, allow_input_downcast=True)
    p = []

    for i in xrange(X_test.X.shape[0] / batch_size):
        if i % 100 == 0:
            print "predicting batch {0} of {1}".format(i, X_test.X.shape[0] / batch_size)
        x_arg = X_test.X[i*batch_size:(i+1)*batch_size,:]
        x_arg = X_test.get_topological_view(x_arg)
        p.append(f(x_arg.astype(X_m.dtype)))
        
    p = np.concatenate(p)
    p = p[:m]
    
    return p

images, y, file_names, dimensions = shuffle(images, y, file_names, dimensions, random_state=7)
  
folds = 10
fold = 0

kfold = StratifiedKFold([np.argmax(y[i]) for i in range(y.shape[0])], n_folds=folds)

for train_index, test_index in kfold:
    save_path = 'valid_best_fold%d.pkl' % fold
    print save_path
    
    images_train = images[train_index]
    y_train = y[train_index]
    images_train, y_train = shuffle(images_train, y_train, random_state=7)
    X_train = DenseDesignMatrix(X=images_train, y=y_train,view_converter=view_converter)
    
    images_test = images[test_index]
    y_test = y[test_index]
    X_test = DenseDesignMatrix(X=images_test, y=y_test,view_converter=view_converter)
            
    if retrain:
        print "training on", X_train.X.shape, 'testing on', X_test.X.shape
        trainer = sgd.SGD(learning_rate=learn_rate, batch_size=batch_size,
                          learning_rule=learning_rule.Momentum(momentum_start),
                          cost=Dropout(
                                       input_include_probs={'l1':1., 'l2':1., 'l3':1., 'l4':1., 'l5':1., 'l6':1.},
                                       input_scales={'l1':1., 'l2':1., 'l3':1., 'l4':1., 'l5':1., 'l6':1.}
                                       ),
                          termination_criterion=EpochCounter(max_epochs=max_epochs),
                          monitoring_dataset={'train':X_train, 'valid':X_test},
                          )
        
        
        input_space = Conv2DSpace(shape=(central_window_shape, central_window_shape),
                    axes = axes,
                    num_channels = 1)
                    
        ann = mlp.MLP(layers, input_space=input_space)

        velocity = learning_rule.MomentumAdjustor(final_momentum=momentum_end,
                                          start=1,
                                          saturate=momentum_saturate)

        watcher = best_params.MonitorBasedSaveBest(channel_name='valid_y_nll',
                                                   save_path=save_path)

        decay = sgd.LinearDecayOverEpoch(start=1, saturate=decay_saturate, decay_factor=decay_factor)

        ra = RealtimeAugment(window_shape=[img_dim, img_dim], randomize=[X_train, X_test], 
                scale_diff=scale_diff, translation=translation, center_shape=center_shape, center=[X_train, X_test],
                preprocess=preprocess)
                
        train = Train(dataset=X_train, model=ann, algorithm=trainer,
                      extensions=[watcher, velocity, decay, ra])
                  
        train.main_loop()

    print "using model", save_path
    model = serial.load(save_path)
    
    print "loading test set"
    for f_name_dir in os.walk("test"):
        images_test, fnames, dims_test = load_images(f_name_dir, img_dim=img_dim, as_grey=as_grey)

    X_test = None
    p_test = np.zeros((len(images_test),121), dtype=np.float32)
    
    for example in xrange(test_examples):
        print "creating test augmentation %d" % example
        X_train = DenseDesignMatrix(X=images_train, y=y_train,view_converter=view_converter)
        X_test_ = DenseDesignMatrix(X=np.array(images_test), y=np.array((len(images_test),)), view_converter=view_converter)

        ra = RealtimeAugment(window_shape=[img_dim, img_dim], randomize=[X_train, X_test_], 
            scale_diff=scale_diff, translation=translation, center_shape=center_shape, center=[X_train, X_test_],
            preprocess=preprocess)
        ra.setup(None,None,None)

        preds = predict(model, X_test_)
        p_test += preds
        
    p_test /= test_examples

    print "writing sub to file"
    with open('sub.csv', 'w') as sub:
        sub.write("image," + ",".join(train_labels) + "\n")
        for idx, fname in enumerate(fnames):
            p_row = p_test[idx]
            sub.write("{0},{1}\n".format(fname, ",".join([str(x) for x in p_row])))
    quit()
