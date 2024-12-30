def lstm_prediction(stock_symbol):
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM
    import matplotlib.pyplot as plt  # Import matplotlib for plotting

    # Function to fetch stock data using yfinance
    def fetch_stock_data(stock_symbol):
        stock_data = yf.download(stock_symbol, period="1y")
        return stock_data

    # Fetch the stock data
    og_df = fetch_stock_data(stock_symbol)
    
    # Reset index to make the Date column accessible
    todataframe = og_df.reset_index(inplace=False)

    # Print columns to verify structure
    print("\nColumns after resetting index:", todataframe.columns)
    print("\nFirst few rows of the data:", todataframe.head())

    # Flatten multi-level columns (if any)
    todataframe.columns = [col[1] if isinstance(col, tuple) else col for col in todataframe.columns]

    # Print the columns again after flattening
    print("\nColumns after flattening:", todataframe.columns)

    # Ensure 'Date' and 'Close' columns exist
    if 'Date' not in todataframe.columns or 'Close' not in todataframe.columns:
        print("ERROR: 'Date' or 'Close' column is missing in the dataset.")
        return

    # Sort the data by date and create a new DataFrame for 'Date' and 'Close'
    seriesdata = todataframe.sort_index(ascending=True, axis=0)
    new_seriesdata = pd.DataFrame(index=range(0, len(todataframe)), columns=['Date', 'Close'])

    # Fill the new DataFrame with Date and Close values
    new_seriesdata['Date'] = seriesdata['Date'].values
    new_seriesdata['Close'] = seriesdata['Close'].values

    # Set 'Date' as the index and drop it from the columns
    new_seriesdata.index = new_seriesdata['Date']
    new_seriesdata.drop('Date', axis=1, inplace=True)

    # Print the new DataFrame to verify data
    print("New DataFrame after filling Date and Close columns:")
    print(new_seriesdata.head())

    # Create train and test sets
    myseriesdataset = new_seriesdata.values
    totrain = myseriesdataset

    # Converting dataset into x_train and y_train using MinMaxScaler
    scalerdata = MinMaxScaler(feature_range=(0, 1))
    scale_data = scalerdata.fit_transform(myseriesdataset)
    x_totrain, y_totrain = [], []
    length_of_totrain = len(totrain)
    
    for i in range(60, length_of_totrain):
        x_totrain.append(scale_data[i-60:i, 0])
        y_totrain.append(scale_data[i, 0])
    
    x_totrain, y_totrain = np.array(x_totrain), np.array(y_totrain)
    x_totrain = np.reshape(x_totrain, (x_totrain.shape[0], x_totrain.shape[1], 1))
    
    # Build and compile the LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_totrain.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adadelta')

    # Train the model
    lstm_model.fit(x_totrain, y_totrain, epochs=3, batch_size=1, verbose=2)

    # Prepare input for prediction
    myinputs = new_seriesdata[len(new_seriesdata) - 60:].values
    myinputs = myinputs.reshape(-1, 1)
    myinputs = scalerdata.transform(myinputs)

    tostore_test_result = []
    for i in range(60, myinputs.shape[0]):
        tostore_test_result.append(myinputs[i-60:i, 0])
    tostore_test_result = np.array(tostore_test_result)
    tostore_test_result = np.reshape(tostore_test_result, (tostore_test_result.shape[0], tostore_test_result.shape[1], 1))

    # Predict stock price
    myclosing_priceresult = lstm_model.predict(tostore_test_result)
    myclosing_priceresult = scalerdata.inverse_transform(myclosing_priceresult)

    # Create a DataFrame for the predicted stock prices
    datelist = pd.date_range(pd.to_datetime('today').date(), periods=101)[1:]
    predicted_df = pd.DataFrame(myclosing_priceresult, columns=['Close'], index=datelist)
    
    # Combine the original and predicted data
    result_df = pd.concat([og_df, predicted_df])[['Close']]
    result_df = result_df.reset_index(inplace=False)
    result_df.columns = ['Date', 'Close']

    # Print the info of the final result dataset
    print("\n<----------------------Info of the RESULT dataset---------------------->")
    print(result_df.info())
    print("<------------------------------------------------------------------------>\n")

    # Plot the graph of original and predicted stock prices
    plt.figure(figsize=(14, 6))
    plt.plot(og_df['Date'], og_df['Close'], color='blue', label='Original Stock Price')
    plt.plot(predicted_df.index, predicted_df['Close'], color='red', label='Predicted Stock Price')
    plt.title(f"Stock Price Prediction for {stock_symbol}")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()  # This will display the graph

    # Function to convert DataFrame to JSON
    def get_json(df):
        import json
        import datetime
        def convert_timestamp(item_date_object):
            if isinstance(item_date_object, (datetime.date, datetime.datetime)):
                return item_date_object.strftime("%Y-%m-%d")
        
        dict_ = df.to_dict(orient='records')
        return json.dumps(dict_, default=convert_timestamp)

    return get_json(result_df)
