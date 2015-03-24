from skimage.io import imread
from skimage.transform import resize
import os
import numpy as np

def load_images(f_name_dir, img_dim=74, as_grey=True, limit=None):
    images = []
    fnames = []
    dims = []
    
    for idx, fname in enumerate(f_name_dir[2]):
        if fname[-4:] != ".jpg":
            continue
        
        img_name = "{0}{1}{2}".format(f_name_dir[0], os.sep, fname)         
        image = imread(img_name, as_grey=as_grey)
        image = resize(image, (img_dim, img_dim))

        length = np.prod(image.shape)
        img_reshape = np.reshape(image, length)
        
        images.append(img_reshape)
        fnames.append(fname)
        dims.append(image.shape)
    return images, fnames, dims
