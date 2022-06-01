import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout, \
                              GlobalAveragePooling2D, Input, Concatenate, Layer
from tensorflow.keras.optimizers import SGD


## 참조 : https://github.com/hskang9/Googlenet/blob/master/keras/googlenet.py -- LRN 코드
## 참조 : https://sike6054.github.io/blog/paper/second-post/
class LRN2D(Layer):
    """
    This code is adapted from pylearn2.
    License at: https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt
    """

    def __init__(self, alpha=0.0001,k=1,beta=0.75,n=3, name=None):
        if n % 2 == 0:
            raise NotImplementedError("LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__()
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        self.test = name

    def get_output(self, train):
        X = self.get_input(train)
        return tf.nn.lrn(X)

    def get_config(self):
        return {"test": self.__class__.__name__,
                "alpha": self.alpha,
                "k": self.k,
                "beta": self.beta,
                "n": self.n}


def inception(input_tensor, filter_channels):
    filter_1x1, filter_3x3_Reduce, filter_5x5_Reduce, filter_5x5, pool_proj = filter_channels
    
    branch_1 = Conv2D(filter_1x1, kernel_size=(1, 1), padding='same', strides=(1, 1), activation='relu', kernel_initializer='he_normal')(input_tensor)
    branch_1 = BatchNormalization()(branch_1)
    
    branch_2 = Conv2D(filter_3x3_Reduce, kernel_size=(1, 1), padding='same', strides=(1, 1), activation='relu', kernel_initializer='he_normal')(input_tensor)
    branch_2 = BatchNormalization()(branch_2)

    branch_3 = Conv2D(filter_5x5_Reduce, kernel_size=(1, 1),  padding='same', strides=(1, 1), activation='relu', kernel_initializer='he_normal')(input_tensor)
    branch_3 = BatchNormalization()(branch_3)
    branch_3 = Conv2D(filter_5x5, kernel_size=(5, 5), padding='same', strides=(1, 1), activation='relu', kernel_initializer='he_normal')(branch_3)
    branch_3 = BatchNormalization()(branch_3)
    
    branch_4 = MaxPooling2D(pool_size=(3, 3), padding='same', strides=1)(input_tensor)
    branch_4 = Conv2D(pool_proj, kernel_size=(1, 1), padding='same', strides=(1, 1), activation='relu', kernel_initializer='he_normal')(branch_4)
    branch_4 = BatchNormalization()(branch_4)
    
    DepthConcat = Concatenate()([branch_1, branch_2, branch_3, branch_4])
    
    return DepthConcat

#논문에 맞게 CIPIIPIIPIIPF 구조로 GoogLeNet 변형
#Inception layer는 3x3 feature maps 기반, 7개의 inception 사용
#n값의 초기값은 32이고, 이후 inception layer마다 32씩 증가
#n값에 따라 1x1, 3x3 reduce, 5x5 reduce, 5x5, pool projection은 3/4n, 1/2n, 1/8n, 1/4n, 1/4n에 대응
def GoogLeNet(model_input, classes=7):
    conv_1 = Conv2D(192, kernel_size=(3, 3), padding='same', activation='relu')(model_input)
    conv_1_normalize = BatchNormalization()(conv_1)
    poo11_norm1 = LRN2D(name='pool1/norm1')(conv_1_normalize)

    #n=32, 3/4n, 1/2n, 1/8n, 1/4n, 1/4n에 따라 1x1, 3x3 reduce, 5x5 reduce, 5x5, pool projection 24, 16, 4, 8, 8 대응            
    inception_1a = inception(poo11_norm1, [24, 16, 4, 8, 8]) 
    pool_1 = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(inception_1a) 
    
    #n=64, 3/4n, 1/2n, 1/8n, 1/4n, 1/4n에 따라 1x1, 3x3 reduce, 5x5 reduce, 5x5, pool projection 48, 32, 8, 16, 16 대응 
    inception_2a = inception(pool_1, [48, 32, 8, 16, 16])

    #n=96, 3/4n, 1/2n, 1/8n, 1/4n, 1/4n에 따라 1x1, 3x3 reduce, 5x5 reduce, 5x5, pool projection 72, 48, 12, 24, 24 대응  
    inception_2b = inception(inception_2a, [72, 48, 12, 24, 24]) 
    pool_2 = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(inception_2b) 
    
    #n=128, 3/4n, 1/2n, 1/8n, 1/4n, 1/4n에 따라 1x1, 3x3 reduce, 5x5 reduce, 5x5, pool projection 96, 64, 16, 32, 32 대응
    inception_3a = inception(pool_2, [96, 64, 16, 32, 32]) 

    #n=160, 3/4n, 1/2n, 1/8n, 1/4n, 1/4n에 따라 1x1, 3x3 reduce, 5x5 reduce, 5x5, pool projection 120, 80, 20, 40, 40 대응
    inception_3b = inception(inception_3a, [120, 80, 20, 40, 40]) 
    pool_3 = MaxPooling2D(pool_size=(3, 3), padding='same', strides=(2, 2))(inception_3b) 

    #n=192, 3/4n, 1/2n, 1/8n, 1/4n, 1/4n에 따라 1x1, 3x3 reduce, 5x5 reduce, 5x5, pool projection 144, 96, 24, 48, 48 대응
    inception_4a = inception(pool_3, [144, 96, 24, 48, 48]) 
    
    #n=224, 3/4n, 1/2n, 1/8n, 1/4n, 1/4n에 따라 1x1, 3x3 reduce, 5x5 reduce, 5x5, pool projection 168, 112, 28, 56, 56 대응
    inception_4b = inception(inception_4a, [168, 112, 28, 56, 56]) 
    
    
    avg_pool = GlobalAveragePooling2D()(inception_4b)
    linear = Dense(1000, activation='relu')(avg_pool)
    dropout = Dropout(0.4)(linear)
    model_output = Dense(classes, activation='softmax', name='main_classifier')(dropout) 
    
    model = Model(model_input, model_output)
    
    return model

def build_googlenet():
    input_shape = (40, 40, 1)
    model_input = Input(shape=input_shape)

    model = GoogLeNet(model_input, 7)
    opt = SGD(learning_rate=0.1, momentum=0.9, nesterov=True, decay=0.0001)

    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.load_weights('models/weights/googlenet_1.h5')

    return model
