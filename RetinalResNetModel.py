from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Input,Dropout
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
import pickle




class RetinalResNetModel():

    def __init__(self,NUM_CLASSES=13,learning_rate=0.00001,height=1000,width=1000,l2_penalty=0.01):
        self.model = Sequential()
        self.num_classes = NUM_CLASSES
        self.learning_rate = learning_rate
        self.H = height
        self.W = width
        self.l2_penalty = l2_penalty

    def get_model(self):
        self.model.add(ResNet50(include_top=False,pooling='avg',weights='imagenet',input_shape=(self.H,self.W,3)))

        if CONFIG['L2_penalty'] != 0:
            self.model.add(Dense(self.num_classes,activation = 'sigmoid',kernel_regularizer=self.l2_penalty))
        else:
            #self.model.add(Dense(256,activation='relu',kernel_regularizer=l2(0.01)))
            #self.model.add(Dropout(0.5))
            self.model.add(Dense(self.num_classes,activation='sigmoid'))

        # Fine tuning
        self.model.layers[0].trainable = True

        self.model.compile(optimizer = tf.keras.optimizers.Adam(self.learning_rate),
                      loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics = [tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC()])

        return self.model


    def load_model(self,model):
        self.model = model

    def model_fitting(self,training_data,validation_data,num_epochs,call_backs,history_file_path):

        history = self.model.fit(training_data, epochs = num_epochs, verbose = 2, validation_data = validation_data,
                            shuffle = True, callbacks=call_backs)
        with open(history_file_path, 'wb') as file:
            pickle.dump(history.history, file)

        model.save('my_model.keras')
