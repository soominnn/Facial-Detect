from tensorflow import keras
from tensorflow.keras.optimizers import SGD


#참고 논문 : Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
#코드 : 핸즈온 머신러닝 2판(책) 14장 5절
#Resnet34는 34개 층으로 이루어져있고, 64개의 특성 맵을 출력하는 3개의 Residual Unit, 128개의 특성 맵을 출력하는 4개의 Residual Unit, 512개의 특성 맵을 출력하는 3개의 Residual Unit을 포함한다
#먼저 Residual Unit층을 구현한다.
class Residual_Unit(keras.layers.Layer):
    def __init__(self, filters, strides = 1, activation = 'relu',**kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters,3,strides = strides,padding = "same",use_bias = False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters,3,strides = 1,padding = "same",use_bias = False),
            keras.layers.BatchNormalization()
        ]
        #main_layers는 convolution과 batch normalization을 사용하는 기본적인 구조다.
        self.skip_layers = []
        if strides > 1 :
            self.skip_layers = [
                keras.layers.Conv2D(filters,1,strides = strides,padding = "same",use_bias = False),
                keras.layers.BatchNormalization()
            ]
        #skip_layers는 convolution과 batch normalization을 stride가 1보다 큰 경우에만 적용한다. 즉, 입력과 출력의 크기가 다른 경우를 의미한다. 
        #입력과 출력의 크기가 다르면 입력이 Residual Unit의 출력에 바로 더해질 수 없다.

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'activation' : self.activation,
            'main_layers' : self.main_layers,
            'skip_layers' : self.skip_layers,
        })
        return config
            
    def call(self,inputs):
       x = inputs
       for layer in self.main_layers :
            x = layer(x)
            skip = inputs
       for layer in self.skip_layers:
            skip = layer(skip)
       return self.activation(x+skip)
    #call()은 input을 main layer와 skip layer에 통과시키고 두 출력을 더하여 activation function에 통과시킨다.

def build_resnet():
    model = keras.models.Sequential()
    #Residual Unit을 준비해두었기 때문에 Residual Unit을 하나의 층처럼 취급할 수 있다. 그러므로 Sequential class를 이용해 구현한다.
    input_shape = (40, 40, 1)
    model.add(keras.layers.Input(shape=input_shape))

    prev_filters = 32
    for filters in [32]*3 + [64]*4 + [128]*6 + [256]*3 : 
        strides = 1 if filters == prev_filters else 2
        model.add(Residual_Unit(filters,strides = strides))
        prev_filters = filters
    #64개의 특성 맵을 출력하는 3개의 Residual Unit, 128개의 특성 맵을 출력하는 4개의 Residual Unit, 512개의 특성 맵을 출력하는 3개의 Residual Unit을 for문을 이용해 구현해주었다.
    #Filter 개수가 이전과 같으면 stride를 1, 아니면 2로 설정하고, filter개수를 계속 update해주었다.
    model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(7,activation = "softmax"))
    opt = SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.0001)

    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.load_weights('models/weights/resnet_1.h5')
    return model
