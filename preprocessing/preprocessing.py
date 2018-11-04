from skimage.transform import resize

class ImagePreprocessor:
    def __init__(self, config):
        self.config = config
        self.basepath = config.data.root_path
        self.scaled_row_dim = int(config.data.image_rows/config.data.row_scale_factor)
        self.scaled_col_dim = int(config.data.image_cols/config.data.col_scale_factor)
        self.n_channels = config.data.n_channels
    
    
    def preprocess(self, image):
        image = self.resize(image)
        image = self.reshape(image)
        image = self.normalize(image)
        return image
    
    def resize(self, image):
        image = resize(image, (self.scaled_row_dim, self.scaled_col_dim))
        return image
    
    def reshape(self, image):
        image = np.reshape(image, (image.shape[0], image.shape[1], self.n_channels))
        return image
    
    def normalize(self, image):
        image /= 255.0 
        return image
    
    def load_image(self, image_id):
        image_rows = self.config.data.image_rows
        image_cols = self.config.data.image_cols
        image_channels = 4
        image = np.zeros(shape=(image_rows, image_cols, image_channels))
        image[:,:,0] = imread(self.basepath + image_id + "_green" + ".png")
        image[:,:,1] = imread(self.basepath + image_id + "_blue" + ".png")
        image[:,:,2] = imread(self.basepath + image_id + "_red" + ".png")
        image[:,:,3] = imread(self.basepath + image_id + "_yellow" + ".png")
        return image[:,:,0:self.config.data.n_channels]