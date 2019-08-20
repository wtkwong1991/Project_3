import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import timedelta
import numpy as np
from sklearn.ensemble import RandomForestRegressor as RF
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pandas import concat
from numpy import concatenate
from pandas import DataFrame


# convert series to supervised learning
    # Below block of code used from 'machinelearningmastery.com' , article title: 'Multivariate Time Series Forecasting with LSTMs in Keras'
    # Author: Jason Brownlee
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def trainer():
    for metal in ["Gold","Silver","Platinum","Palladium"]:
        RandFor(metal)
        # DepLern(metal)

def RandFor(metal):
    # Reads in and cleans data
    df = pd.read_csv(f"data/csv/{metal}_Data.csv",index_col="Business_Week")
    df = df.drop(df.columns[0],axis=1)
    df = df.drop(df.columns[9],axis=1)
    df[f"{metal}_Settle"] = df[f"{metal}_Settle"].fillna(method='bfill')
    df = df.dropna(axis=1)
    # Makes graphs easier to interpret and match up.
    # Ensures every date is over the proper time period for uniformity
    df["Date"] = pd.to_datetime(df["Date"],infer_datetime_format=True)
    df = df.reset_index(drop=True)
    for i in range(len(df.index)):
        if df["Date"][i].weekday() != 1:
            if((df["Date"][i]+timedelta(days=1)).weekday() == 1):
                df.at[i,"Date"] = df["Date"][i]+timedelta(days=1)
            else:
                df.at[i,"Date"] = df["Date"][i]-timedelta(days=1)
    df.set_index("Date",inplace=True)
    df = df.asfreq("W-TUE")
    df = df.dropna()

    # Scales Data
    # Scales the settle prices seperately, so that the predictions can be unscaled later.
    scaler_body = MinMaxScaler()
    scaler_gold = MinMaxScaler()
    scaler_body.fit(df.drop(f"{metal}_Settle",axis=1))
    scaler_gold.fit(pd.DataFrame(df[f"{metal}_Settle"]))
    normalized_df = [list(x) for x in scaler_body.transform(df.drop(f"{metal}_Settle",axis=1))]
    normalized_gold = [list(x) for x in scaler_gold.transform(pd.DataFrame(df[f"{metal}_Settle"]))]
    for i in range(len(normalized_gold)):
        normalized_gold[i].extend(normalized_df[i])
    normalized_df = pd.DataFrame(normalized_gold,index=df.index,columns=df.columns)

    # Creates the X and y training variables.
    X = normalized_df.drop(f"{metal}_Settle", axis=1)
    y = normalized_df[f"{metal}_Settle"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Trains and fits the RF model
    RF_Model = RF(n_estimators = 250, random_state = 0)
    RF_Model.fit(X_train.values, y_train.values.ravel())

    # Gets Predictions over entire range and current date
    # Used in buy sell long short decision and graph of EC graphs
    predictions_all = pd.DataFrame(RF_Model.predict(X),index=X.index)
    predictions_all = pd.DataFrame(scaler_gold.inverse_transform(predictions_all),index=predictions_all.index)
    predictions_all.columns = ["Relative Value"]
    predictions_all = predictions_all.reset_index()
    predictions_all["Date"] = predictions_all["Date"].apply(lambda x: x + timedelta(days=3))

    # Creates daily comparison
    daily_transform_url = f"data/csv/{metal}Prices.csv"
    daily = pd.read_csv(daily_transform_url)[["Date","Settle"]]
    daily["Date"] = pd.to_datetime(daily["Date"],infer_datetime_format=True)
    daily = daily.merge(predictions_all,on="Date",how="left")
    daily = daily.fillna(method="bfill")
    daily = daily.dropna()
    daily.to_csv(f"data/csv/{metal}_RF_Daily.csv",index=False)

    # Creates Weekly comparison with profit or loss signals
    weekly = daily.loc[[True if day.weekday() == 4 else False for day in daily["Date"]]]

    # Creates EC figure and adds signals and saves as csv
    weekly.loc[:,'Settle Delta'] = weekly['Settle'].diff(+1)
    weekly.loc[:,'Settle Delta'] = weekly['Settle Delta'].apply(lambda x:x*-1)
    weekly.loc[:,'Rel Val Delta'] = weekly['Relative Value'].diff(-1)

    Signal = np.sign(weekly['Rel Val Delta'])
    weekly2 = pd.concat([weekly, pd.DataFrame(Signal)], axis=1)
    weekly2.columns = ['Date', 'Settle', 'Relative Value', 'Settle Delta', 'Rel Val Delta', 'Signal']

    weekly2['Profit'] = weekly2['Settle Delta']*weekly2['Signal']
    weekly2['EC'] = weekly2.Profit[::-1].cumsum()
    fig = plt.figure()
    ax = weekly2['EC'].plot(title=f'{metal} RF Model Equity Curve')
    ax.invert_xaxis()
    fig.savefig(f'graphs/{metal}_RF.png')
    weekly2.to_csv(f"data/csv/{metal}_FinalTable_RF.csv",index=False)

    # Returns 1 to signal successful completion
    return 1


def DepLern(metal):
    file = f"data/csv/{metal}_Data.csv"
    df = pd.read_csv(file)
    postfitdf = df

    df1 = df.filter(['Date',f'{metal}_Settle', 'Open Interest', 'Money Manager Shorts', 'Money Manager Longs', 'Producer/Merchant/Processor/User Longs', 'Producer/Merchant/Processor/User Shorts'], axis=1)
    df1['Date'] = pd.to_datetime(df1['Date'])
    df1.set_index('Date', inplace = True)

    # load dataset
    #dataset = read_csv('pollution.csv', header=0, index_col=0)
    values = df1.values
    # integer encode direction
    #encoder = LabelEncoder()
    #values[:,4] = encoder.fit_transform(values[:,4])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[7,8,9,10,11]], axis=1, inplace=True)

    # split into train and test sets
    values = reframed.values
    n_train_days = int(len(df1)*.75)
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    # fit network
    history = model.fit(train_X, train_y, epochs=125, batch_size=48, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    # The path to our CSV file
    file2 = f"data/csv/{metal}Prices.csv"
    df3 = pd.read_csv(file2)
    df4 = df3.filter(['Date','Settle'])
    df4['Date'] = pd.to_datetime(df4['Date'])
    RelVal = inv_yhat.tolist()
    RelVal.reverse()
    WklyDate = postfitdf['Date'].iloc[::-1].tolist()
    z = {'Date': WklyDate, 'Relative Value': RelVal}
    df5 = pd.concat([pd.Series(v, name=k) for k, v in z.items()], axis=1)
    df5['Date'] = pd.to_datetime(df5['Date'])

    PredictDF = pd.merge(df4, df5, on='Date', how='outer')
    FinalPredict_DF = PredictDF.head(len(test_y))
    FinalPredict_DF2 = FinalPredict_DF.fillna(method='bfill')

    for i in range(len(FinalPredict_DF2.index)-3):
        FinalPredict_DF2.at[i,"Relative Value"] = FinalPredict_DF2["Relative Value"][i+3]
    FinalPredict_DF2 = FinalPredict_DF2.iloc[:len(FinalPredict_DF2.index)-3]

    weekly = FinalPredict_DF2.loc[[True if day.weekday() == 4 else False for day in FinalPredict_DF2["Date"]]]

    weekly.loc[:,'Settle Delta'] = weekly['Settle'].diff(+1)
    weekly.loc[:,'Settle Delta'] = weekly['Settle Delta'].apply(lambda x:x*-1)
    weekly.loc[:,'Rel Val Delta'] = weekly['Relative Value'].diff(-1)

    Signal = np.sign(weekly['Rel Val Delta'])
    weekly2 = pd.concat([weekly, pd.DataFrame(Signal)], axis=1)
    weekly2.columns = ['Date', 'Settle', 'Relative Value', 'Settle Delta', 'Rel Val Delta', 'Signal']

    weekly2['Profit'] = weekly2['Settle Delta']*weekly2['Signal']
    weekly2['EC'] = weekly2.Profit[::-1].cumsum()
    fig = plt.figure()
    ax = weekly2['EC'].plot(title=f'{metal} RNN Model Equity Curve')
    ax.invert_xaxis()
    fig.savefig(f'graphs/{metal}_DL_EC.png')

    weekly2.to_csv(f"data/csv/{metal}_DL_FinalTable.csv")

    # Returns 1 to signal successful completion
    return 1