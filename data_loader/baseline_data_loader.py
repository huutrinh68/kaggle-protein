from base.base_data_loader import BaseDataLoader
from utils.common import *

class DataGenerator(BaseDataLoader):
    def __init__(self, config):
        super(DataGenerator, self).__init__(config)

    def create_train(self, dataset_info, batch_size, shape, augment=True):
        assert shape[2] == 3
        while True:
            dataset_info = shuffle(dataset_info)
            for start in range(0, len(dataset_info), batch_size):
                end = min(start + batch_size, len(dataset_info))
                X_train_batch = []
                tmp = dataset_info[start:end]
                y_train_batch = np.zeros((len(tmp), 28))
                for i in range(len(tmp)):
                    image = self.load_image(tmp[i]["path"], shape)
                    if augment == True:
                        image = self.augment(image)
                    X_train_batch.append(image/255.)
                    y_train_batch[i][tmp[i]["labels"]] = 1
                yield np.array(X_train_batch), y_train_batch

    def load_image(self, path, shape):
        image_red_ch = Image.open(path+"_red.png")
        image_yellow_ch = Image.open(path+"_yellow.png")
        image_green_ch = Image.open(path+"_green.png")
        image_blue_ch = Image.open(path+"_blue.png")
        # image = np.stack((np.array(image_red_ch), np.array(image_yellow_ch), np.array(image_green_ch), np.array(image_blue_ch)), -1)
        image = np.stack((np.array(image_red_ch), np.array(image_yellow_ch), np.array(image_green_ch)), -1)
        image = cv2.resize(image, (shape[0], shape[1]))
        return image
    
    def augment(self, image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5)
            ])], random_order=True)
        image_aug = augment_img.augment_image(img)
        return image_aug



        
