from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Dense
import keras
from sklearn.model_selection import train_test_split

model=Sequential()
model.add(Dense(units=4,activation="relu",input_dim=4))
model.add(Dense(units=3,activation="softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer="SGD",
              metrics=['accuracy'])

X,labels=load_iris(True)

y = keras.utils.to_categorical(labels, num_classes=3)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=50)

model.fit(X_train,y_train,epochs=500,shuffle=True)

print model.evaluate(X_train, y_train)
print model.evaluate(X_test, y_test)