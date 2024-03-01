import pandas as pd
import numpy as np

# Returns a tuple of (df, dataX, dataY)
# Where df is the original dataframe, dataX is the formatted normalized input data, 
# and dataY is the formatted normalized output data. 
# The dataframe will not include the date column if df_dates is False.
def preprocess(n_past, n_future, label_column, null, df_dates=True, include_month_change=False, remove_label_from_input=False):
    # Load the data.
    # We'll forward fill any missing values, so long as there are no more than 5 of them consecutively,
    # then fill any remaining missing values with our null constant.
    df = pd.read_csv("./agg.csv").ffill(limit=5).fillna(null)

    if (not include_month_change and 'S&P MONTH CHANGE' in df.columns):
        df = df.drop('S&P MONTH CHANGE', axis=1)

    # Remove date column temporarily; not useful for normalizing
    ddf = df.iloc[:, 1:]

    # Mean normalize (standardization)
    mean = ddf.mean()
    std = ddf.std()
    ddf = (ddf - mean) / std

    # Convert to numpy
    data = ddf.to_numpy()

    # Format the data to a 3D shape so the LSTM can use it.
    # The resulting shapes will be as follows:
    #    dataX: (number_of_input_series, number_of_timesteps, number_of_features)
    #    dataY: (number_of_input_series, 1)
    dataX = []
    dataY = []

    for i in range(n_past, len(data) - n_future + 1):
        dataX.append(data[i - n_past:i, 0:data.shape[1]])
        dataY.append(data[i + n_future - 1:i + n_future, label_column])

    # Remove date column permanently if requested
    if not df_dates:
        df = df.iloc[:, 1:]

    # Delete S&P change, price open and price close columns from input data.
    if remove_label_from_input:	
        dataX = np.delete(dataX, label_column, axis=2)

    # Return a tuple containing (df, dataX, dataY)
    return (df,np.array(dataX),np.array(dataY))