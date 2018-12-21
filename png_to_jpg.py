#!/usr/bin/env python
# coding: utf-8

# In[2]:


from PIL import Image, ImageOps
from PIL import ImageFile
import glob
from shutil import copyfile
from multiprocessing import Process
from multiprocessing import Pool

ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[11]:


def transfer(image_path):
    output_path = '/media/trinhnh1/3A08638408633DCF/kaggle/human-protein/input/train_jpg/'
    if str(image_path.split('/')[-1][:-4] + '.jpg') in glob.glob(output_path + '.jpg'):
        return
    im = Image.open(image_path)
    im.save(output_path + image_path.split('/')[-1][:-4] + '.jpg')
    
def png_to_jpg():
    raw_input_path = '/media/trinhnh1/3A08638408633DCF/kaggle/human-protein/input/train/'
    with Pool(40) as p:
        p.map(transfer, glob.glob(raw_input_path + '/*.png'))


# In[12]:


if __name__ == '__main__':
    png_to_jpg()


# In[ ]:




