import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt
df = pd.read_csv('bolt.csv')
X = df.iloc[:, [0, 2, 3, 4]].values 
y = df.iloc[:, -1].values 
m = X.shape[1]
n = X.shape[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_X.fit(X_train)
scaler_y = StandardScaler()
scaler_y.fit(y_train.reshape(-1, 1))
X_train_norm = scaler_X.transform(X_train)
X_test_norm = scaler_X.transform(X_test)
y_train_norm = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
y_test_norm = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
model = keras.Sequential()
# For example, referring to the line below, change the number of parameters in this line.
model.add(keras.layers.Dense(8, activation='relu', input_shape=(m,)))
model.add(keras.layers.Dense(1, activation='linear'))
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=300)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999),loss="mean_squared_error")
model.fit(X_train_norm, y_train_norm ,validation_split=0.2,epochs=50000,batch_size=64,callbacks=[callback])
model.evaluate(X_test_norm, y_test_norm)
model.summary()
y_train_pred_norm = model.predict(X_train_norm)
y_test_pred_norm = model.predict(X_test_norm)
y_train_pred = scaler_y.inverse_transform(y_train_pred_norm).flatten()
y_test_pred = scaler_y.inverse_transform(y_test_pred_norm).flatten()
plt.scatter(y_train, y_train_pred, color='blue', label='Train') 
plt.scatter(y_test, y_test_pred, color='red', label='Test') 
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black', linestyle='--') 
plt.xlabel('Target Pu(kN)') 
plt.ylabel('Predicted Pu(kN)') 
plt.legend() 
plt.show() 