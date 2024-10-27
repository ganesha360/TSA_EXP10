# EXP NO: 10 - IMPLEMENTATION OF SARIMA MODEL FOR YAHOO STOCK PREDICTION

### Name: GANESH R
### Register No: 212222240029
### Date: 

## AIM:
To implement SARIMA model using python for Amazon stock prediction.

## ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
   
## PROGRAM:

### Importing the Packages:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
```

### Load the data:
```py
data = pd.read_csv('/content/yahoo_stock.csv', parse_dates=['Date'], index_col='Date')

plt.figure(figsize=(10, 5))
plt.plot(data)
plt.title('Time Series Data')
plt.show()
```

### Transformation and autocorrelation plotting:
```py
def check_stationarity(ts):
    result = adfuller(ts)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    if result[1] <= 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is non-stationary.")

check_stationarity(data['Volume'])

plot_acf(data['Volume'])
plot_pacf(data['Volume'])
plt.show()
```

### Model creation and prediction:
```py
p = 1  
d = 1  
q = 1  
P = 1  
D = 1  
Q = 1  
s = 12

model = SARIMAX(data['Volume'], order=(p, d, q), seasonal_order=(P, D, Q, s))
results = model.fit()

predictions = results.get_forecast(steps=12)  
predicted_mean = predictions.predicted_mean
conf_int = predictions.conf_int()

plt.figure(figsize=(10, 5))
plt.plot(data['Volume'], label='Observed')
plt.plot(predicted_mean, label='Predicted', color='red')
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink')
plt.title('SARIMA Predictions')
plt.legend()
plt.show()

rmse = np.sqrt(((predicted_mean - data['Volume'].iloc[-12:]) ** 2).mean())
print('Root Mean Squared Error:', rmse)
```

## OUTPUT:

### Original Data:
![image](https://github.com/user-attachments/assets/d580d4a6-5f4d-4d70-8920-788bbeb18f03)


### ACF and PACF Representation:
![image](https://github.com/user-attachments/assets/479f858f-f9ed-4fb7-a6d6-71224673f186)
![image](https://github.com/user-attachments/assets/70ca7f88-654c-4384-a2a7-2ef96dbc8508)


### SARIMA Prediction Representation:
![image](https://github.com/user-attachments/assets/f3b95a62-1c4c-4700-b70b-fffff008d389)


## RESULT:
Thus the program run successfully based on the SARIMA model for Amazon stock prediction.
