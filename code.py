import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn
from sklearn.preprocessing import Imputer
from keras.models import Sequential
from keras.layers import Dense
import numpy
from keras.models import Sequential
from keras.layers import Dense

#loading the model
df=pd.read_csv('maternal_mortality.csv')


df.fillna(df.mean(), inplace=True)

print(df.head)

df.to_csv('new.csv')

#print shape of the data
#print(df.shape)

numpy.random.seed(5)


X = df[:,0:7]
Y = df[:,7]

#model layers
model = Sequential()
model.add(Dense(12, input_dim=7, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='softmax'))

#compiling the model

model.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

#fitting the model
model.fit(X, Y, epochs=1000, batch_size=10)

#evaluate the model

scores = model.evaluate(X,Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

