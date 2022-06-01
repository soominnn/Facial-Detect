from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import CosineDecay


def build_vggnet():
	#참조 : https://bskyvision.com/504
	#VGGNET을 논문에 맞게 BatchNormalization추가 및 변형 
	#VGGNET은 ReLU 함수를 이용하고 네트워크의 깊이를 깊게 하고 parameter의 수를 적게하기 위해 모든 convolution layer에서의 kernel size는 3x3, stride는 1로 통일합니다. 
	model = Sequential()
	model.add(Conv2D(input_shape=(40, 40, 1), filters=64, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(BatchNormalization())
	model.add(Conv2D(64, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu")) #40x40x64의 feature map 생성
	model.add(BatchNormalization())
	model.add(MaxPool2D(pool_size=(2,2), padding="valid")) #20x20x64로 feature map size 감소 

	model.add(Conv2D(128, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu")) #20x20x128의 feature map 생성
	model.add(BatchNormalization())
	model.add(Conv2D(128, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu"))
	model.add(BatchNormalization())
	model.add(MaxPool2D(pool_size=(2,2), padding="valid")) #10x10x128로 feature map size 감소

	model.add(Conv2D(256, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu")) #10x10x256의 feature map 생성
	model.add(BatchNormalization())
	model.add(Conv2D(256, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu"))
	model.add(BatchNormalization())
	model.add(MaxPool2D(pool_size=(2,2), padding="valid"))  #5x5x256로 feature map size 감소 

	model.add(Conv2D(512, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu")) #5x5x512의 feature map 생성 
	model.add(BatchNormalization())
	model.add(Conv2D(512, kernel_size=(3,3), padding="same", strides=(1,1), activation="relu"))
	model.add(BatchNormalization())
	model.add(MaxPool2D(pool_size=(2,2), padding="valid")) #2x2x512로 feature map size 감소

	model.add(Flatten())  #feature map을 1차원으로 flatten -> (0,2048)
	model.add(Dense(units=4096, activation="relu"))  #2048과 fully connected 되며 4096의 output
	model.add(Dropout(0.2))
	model.add(Dense(units=4096, activation="relu"))
	model.add(Dropout(0.2))
	model.add(Dense(7, activation="softmax")) #7개의 class를 분류하기 때문에 7개의 뉴런으로 구성.

	decay_steps = 225 * 200
	lr = CosineDecay(initial_learning_rate=0.01, decay_steps=decay_steps)
	opt = SGD(learning_rate=lr, momentum=0.9, nesterov=True, decay=0.0001)

	model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
	model.load_weights('models/weights/vgg_1.h5')

	return model
