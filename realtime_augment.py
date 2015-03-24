from pylearn2.train_extensions import TrainExtension
from pylearn2.datasets.preprocessing import CentralWindow
from pylearn2.utils.rng import make_np_rng
from skimage.transform import AffineTransform, warp, resize
import skimage
import numpy as np
from pylearn2.datasets import preprocessing
import random
import math

class RealtimeAugment(TrainExtension):
    
    def __init__(self, window_shape, center_shape=None, central_window_shape=None, randomize=None, randomize_once=None, center=None, rotate=True,
                    scale_diff=0.0, rng=(2013, 02, 20), shear=0.0, translation=0.0, preprocess=None):
        self._window_shape = window_shape
        self._center_shape = center_shape
        self._central_window_shape = central_window_shape
        self._randomize = randomize if randomize else []
        self._randomize_once = randomize_once if randomize_once else []
        self._center = center if center else []
        self._rotate = rotate
        self._scale_diff = scale_diff
        self._shear = shear
        self._translation = translation
        self._preprocess = preprocess
        self._rng = make_np_rng(rng, which_method="random_integers")
        
    def setup(self, model, dataset, algorithm):
        
        if self._center_shape is not None:
            preprocessor = CentralWindow(self._center_shape)
            for data in self._center:
                preprocessor.apply(data)
            
        randomize_now = self._randomize + self._randomize_once
        self._original = dict((data,
            data.get_topological_view()) for data in randomize_now)

        self.randomize_datasets(randomize_now)
    
    def randomize_datasets(self, datasets):
        center_shift = np.array(self._window_shape) / 2. -0.5
        tform_center = skimage.transform.SimilarityTransform(translation=-center_shift)
        tform_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)
        if self._preprocess is not None:
            pipeline = preprocessing.Pipeline()
            #window the rotations to get rid of the uniform background
            if self._central_window_shape is not None:
                print 'adding window'
                pipeline.items.append(CentralWindow(self._central_window_shape))

            for item in self._preprocess:
                pipeline.items.append(item)
                
        im_shape = (self._window_shape[0], self._window_shape[1], 1)
        
        for d_idx, dataset in enumerate(datasets):
            
            data = self._original[dataset]
            #randomly window data
            print data.shape
            arr = np.empty((data.shape[0], self._window_shape[0], self._window_shape[1], data.shape[3]), dtype=np.float32)
            for idx, example in enumerate(data):
                scale_x = np.random.uniform(1 - self._scale_diff, 1 + self._scale_diff)
                scale_y = np.random.uniform(1 - self._scale_diff, 1 + self._scale_diff)
                translation_x = np.random.uniform(1 - self._translation, 1 + self._translation)
                translation_y = np.random.uniform(1 - self._translation, 1 + self._translation)
                shear = np.random.uniform(0. - self._shear, 0. + self._shear)
                rotation = np.random.uniform(0, 360)
                tform = AffineTransform(scale=(scale_x, scale_y), rotation=np.deg2rad(rotation), 
                    translation=(translation_x, translation_y), shear=shear)
                tform = tform_center + tform + tform_uncenter
                img = warp(example, tform, output_shape=self._window_shape)
                arr[idx] = img
            
            dataset.set_topological_view(arr, axes=dataset.view_converter.axes)
            #assumes self._randomize in in order of [train, valid/test]
            if self._preprocess is not None:
                can_fit = True
                if d_idx == 1:
                    can_fit = False

                dataset.apply_preprocessor(preprocessor=pipeline, can_fit=can_fit)

    def on_monitor(self, model, dataset, algorithm):
        model = None
        dataset = None
        algorithm = None

        self.randomize_datasets(self._randomize)
