from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

dataset = loadtxt('"C:\Users\Darshan\Documents\Diabatics Predictions using NN\pima-indians-diabetes.csv"v',delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]

model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x,y,epochs=150,batch_size=10)

_,accuracy = model.evaluate(x,y)
print('Accuracy : %.2f' % (accuracy*100))

model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")


Code for loading/training/output if you face 'Sequential' has no object like 'predict_classes' :-

from numpy import loadtxt
from keras.models import model_from_json
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")

predictions = (model.predict(x) > 0.5).astype("int32")
for i in range(5,15):
	print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))
