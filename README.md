# Forecasting Stock Values Using an LSTM Model
Long Short-Term Memory (LSTM) is an artificial Recurrent Neural Network (RNN) architecture. The same can be used for interpreting deep sequence trends and graphing of the same. This allows for the model to be used for classifying, processing and making predictions off of time-series data. Stock values are an example of such time-series data, which the LSTM can be used to predict. 
## Data Used
The data used in this project belongs to the [NSEPY](https://nsepy.xyz/) library, which allows for the pulling of real-time NSE (National Stock Exchange of India) data. More specifically, the specific dataset pulled from the library is that of the Bharti Airtel's stock value between 2004 and 2021. A copy of the same is available in the 'data.csv' file. 
## Working of the Code
The code, as per main.py, allows for the dataset pulled to be preprocessed and scaled before its train-test split. Before the data preprocessing, the number of time steps (initialised at 60), number of features (1) and train split percentage (0.8) are defined. After the LSTM model is built and compiled, it is trained and tested using the data split. The saved model can be found in the 'saved_model.h5' file. The final data, along with predictions and real-values, can be found in 'final_data.csv'. 
## Evaluation 
As expected the model does a sufficient job of predicting the next day of stock value given the previous 60 time-steps: 

It is, however, important to note that this model may not be viable in the real world as knowing the closing value of the next day based on the previous 60 days is not as relevant as it may seem. Moreover, models such as Facebook's [prophet](https://facebook.github.io/prophet/) are able to easily beat this simple LSTM. One manner to improve the model may be to predict the next 60-90 timesteps based on the previous 240-360 timesteps or so. This would allow the user to predict highs and lows, and buy and sell stocks accordingly. This may further be analysed in another project. 
