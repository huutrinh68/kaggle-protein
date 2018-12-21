import cv2
import numpy as np

from fastai.vision.image import *

def open_4_chanels(fname):
    fname = str(fname)
    # strip extention before adding color
    if fname.endswith('.png'):
        fname = fname[:-4]
    colors =  ['red','green','blue','yellow']
    img = [cv2.imread(fname+'_'+color+'.png', flags).astype(np.float32)/255
           for color in colors]
    
    x = np.stack(img, axis=-1)
    return Image(pil2tensor(x, np.float32).float())
