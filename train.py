from preprocess import preprocess

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt

import configparser

conf = configparser.ConfigParser()
conf.read("./conf.ini")

BATCH_SIZE = 2
LEARNING_RATE = 0.003#0.005
EPOCHS = 2
WEIGHT_DECAY = 0.002#0.002
ACTIVATION = "tanh"

# The arbitrary null value for missing data.
null = int(conf['data']['null'])

# How many timesteps in the future we're predicting
n_future = int(conf['data']['n_future'])
# The size of each input series (in timesteps). 
n_past = int(conf['data']['n_past'])
# The column index containing the output, AFTER removing the date column.
label_column = int(conf['data']['label_column'])
# Include the S&P percentage change (by about a month)
include_month_change = conf['data'].getboolean('include_month_change')
# Remove the label from the features
remove_label_from_input = conf['data'].getboolean('remove_label_from_input')

# Load the data
df, dataX, dataY = preprocess(n_past, n_future, label_column, null, df_dates=False, include_month_change=include_month_change, remove_label_from_input=remove_label_from_input)

print(df.head())
print("Shape:", dataX.shape)

mean = df.mean()
std = df.std()

# Train test split
trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.05, shuffle=False)

# Shuffle only the training data
trainRandomIndices = np.random.permutation(len(trainX))
trainX = trainX[trainRandomIndices]
trainY = trainY[trainRandomIndices]

# The input shape
SHAPE = (dataX.shape[1],dataX.shape[2])

# Define the model
inputs = keras.Input(shape=SHAPE, dtype="float32", name="inputs")

x = layers.LSTM(25, return_sequences=True, activation=ACTIVATION, kernel_regularizer=regularizers.L1(0.001))(inputs)
# x = layers.LSTM(40, return_sequences=True, activation=ACTIVATION)(x)
x = layers.LSTM(25, return_sequences=False, activation=ACTIVATION)(x)

outputs = layers.Dense(1, name="outputs")(x) # activation="tanh",

model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(loss="mse", run_eagerly=False, optimizer=(tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)), metrics=[keras.metrics.RootMeanSquaredError(), keras.losses.MeanAbsoluteError()])

# Train the model
history = model.fit(trainX, trainY, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)

# Evaluate the model on the test data
train_loss = model.evaluate(trainX, trainY, verbose=0)
test_loss = model.evaluate(testX, testY, verbose=0)

# print("Train loss:", train_loss)
# print("Test loss:", test_loss)

bestlossr = open('bestloss.txt', 'r')
bestloss = bestlossr.read()

# Save the model if this is a new best loss ([0] = MSE)
if (not bestloss.replace(".", "").isnumeric()) or test_loss[0] < float(bestloss):
	print("New best loss! Saving model...")
	model.save("./best-model.keras")

	bestlossw = open('bestloss.txt', 'w')
	bestlossw.write(str(test_loss[0]))
	bestlossw.close()

	meanstd = open('meanstd.txt', 'w')
	meanstd.write(str(mean.to_list()) + "\n" + str(std.to_list()))
	meanstd.close()

else: print("Model not saved, best lost still", bestloss)

bestlossr.close()

## VIZ

# Denormalize our results.
testYReg = testY * std[df.columns[label_column]] + mean[df.columns[label_column]]
testYHat = model.predict(testX) * std[df.columns[label_column]] + mean[df.columns[label_column]]

print(testYReg.shape)
print(testYHat.shape)

for i in range(0, len(testYHat)):
	print("Predicted: ", testYHat[i], "Actual: ", testYReg[i])

print("Train loss:", train_loss)
print("Test loss:", test_loss)

plt.plot(testYReg, c="blue", label="Actual")
plt.plot(testYHat, c="red", label="Predicted")

plt.ylabel("S&P Index Level")
plt.legend()
plt.show()