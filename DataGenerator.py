
from tensorflow.keras.utils import Sequence
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from skimage.transform import rotate, AffineTransform, warp, resize
from sklearn.utils.class_weight import compute_sample_weight




class DataGeneratorKeras(Sequence):
    # Constructor
    def __init__(self,train = True, test=False,augmentation=False, preprocessing_fn = None, batch_size = 16,height,width,weighted_loss_function=False):
        self.train = train
        self.test = test
        self.batch_size = batch_size
        self.directory=path + 'fundus_photos/'
        self.H = height
        self.W = width

        self.augmentation = augmentation
        self.preprocessing_fn = preprocessing_fn

        if self.train:
            self.all_files = train_df
        elif self.test:
            self.all_files = test_df
        else:
            self.all_files= val_df

    # Get the length
    def __len__(self):
        return self.all_files.shape[0]//self.batch_size

    def on_epoch_end(self):
        self.all_files = self.all_files.sample(frac=1).reset_index(drop=True)

    # Getter
    def __getitem__(self, idx):
        images = np.array([],dtype=np.float32).reshape((0,self.H,self.W,3))
        labels = np.array([],dtype=np.float32).reshape((0,NUM_CLASSES))
        for i in range(self.batch_size):
            #print(self.all_files['image_id'][idx*self.batch_size+i])
            image = img_to_array(load_img(self.directory+self.all_files['image_id'][idx*self.batch_size+i] +'.jpg',target_size = (self.H,self.W)))
            image = resize(image, (self.H,self.W))

            y = self.all_files.iloc[idx*self.batch_size+i][class_names].values.astype(np.float32)

            # If there is any transform method, apply it onto the image
            if self.augmentation:
                image = rotate(image,np.random.uniform(-30,30),preserve_range=True)
                scale = np.random.uniform(1.0,1.25)
                tx = np.random.uniform(0,20)
                ty = np.random.uniform(0,20)
                image = warp(image,
                             AffineTransform(matrix=np.array([[scale, 0, tx],
                                                              [0,scale,  ty],
                                                              [0,   0,   1]])).inverse,
                             preserve_range=True)

            #RANDOM HORIZONTAL FLIPPING
            if np.random.choice([True,False]):
                image = np.flip(image,axis= 1)

            images = np.append(images,np.expand_dims(image,axis=0),axis=0)
            labels = np.append(labels,y.reshape(1,NUM_CLASSES),axis=0)

        if self.preprocessing_fn:
            images = self.preprocessing_fn(images)

        if weighted_loss_function == True:
            sample_weights = compute_sample_weight('balanced', labels)
            return images, labels, sample_weights
        else:
            return images, labels
