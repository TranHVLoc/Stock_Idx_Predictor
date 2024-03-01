from preprocess import preprocess
from tensorflow import keras
import ast
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import configparser

conf = configparser.ConfigParser()
conf.read("./conf.ini")

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

model = keras.models.load_model("./best-model.keras")
print(model.summary())

meanstdr = open('meanstd.txt', 'r')
meanstd = meanstdr.read().split('\n')

mean = ast.literal_eval(meanstd[0])
std = ast.literal_eval(meanstd[1])

print(mean[0])
print(std[0])

df, dataX, dataY = preprocess(n_past, n_future, label_column, null, include_month_change=include_month_change, remove_label_from_input=remove_label_from_input)

trainX, dataX, trainY, dataY = train_test_split(dataX, dataY, test_size=0.05, shuffle=False)

# The latest series, actually predicts the future
latest = ((df.iloc[:,1:] - mean) / std)[-n_past:].to_numpy().reshape(1, n_past, -1)
latestY = model.predict(latest) * std[label_column] + mean[label_column]

# Denormalize our results.
dataYReg = (dataY * std[label_column] + mean[label_column]).reshape(-1)
dataYHat = (model.predict(dataX) * std[label_column] + mean[label_column]).reshape(-1)

loss = model.evaluate(dataX, dataY, verbose=0)

print("Loss:", loss)

print("Latest Prediction:", dataYHat[-1], "->", latestY[0][0])

percentChange = round((latestY[0][0] - dataYHat[-1]) / dataYHat[-1] * 100, 4)
percentChange = ("+" + str(percentChange) if percentChange > 0 else str(percentChange)) + "%)"

print("\t(Change:", latestY[0][0] - dataYHat[-1], " or ", percentChange)

dataYHat = np.append(dataYHat, latestY[0][0])

# Set X axis

plt.plot(dataYReg, label="Actual", c="blue")
# plt.plot(df.iloc[n_past:, 0].to_numpy(), dataY, label="Actual", c="blue")
plt.plot(dataYHat, label="Predicted", c="red")

plt.legend()
plt.show()
