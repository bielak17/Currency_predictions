import itertools
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.")
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found. Using zeros as starting parameters.")
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", category=FutureWarning)

def get_week_of_month(date):
  first_day = date.replace(day=1)
  dom = date.day
  adjusted_dom = dom + first_day.weekday()
  return (adjusted_dom - 1)//7 + 1


class Models:
  def __init__(self, db, currency, train_months):
    self.train = None
    self.test = None
    self.target = ['Price']
    self.features = None
    self.db=db
    self.currency=currency
    self.train_months = train_months
    self.scaler = MinMaxScaler()

  def prepare_data(self):
    self.db = self.db[['data',self.currency]]

  def prepare_data_test(self):
    self.db = self.db.rename(columns={'Closing Price (USD)':'Price'})
    self.db = self.db.drop(columns=['24h Open (USD)','24h High (USD)','24h Low (USD)'])
    self.db['Date']=pd.to_datetime(self.db['Date'],format='%Y-%m-%d')
    self.db.set_index('Date',inplace=True)
    self.db=self.db.asfreq('D')
    self.db['DayOfWeek'] = self.db.index.dayofweek + 1
    self.db['DayOfMonth'] = self.db.index.day
    self.db['WeekOfMonth'] = self.db.index.map(get_week_of_month)
    self.train = self.db.iloc[-31-30*self.train_months:-31]
    self.test = self.db.iloc[-31:]
    self.features = [i for i in self.train.columns if i not in self.target]

  def best_params_ARIMA(self,p,d,q,progress_bar):
    pdq = list(itertools.product(p, d, q))
    best_mse = float('inf')
    best_params = []
    tscv = TimeSeriesSplit(n_splits=self.train_months)
    for param in pdq:
      try:
        errors = []
        for train_index, test_index in tscv.split(self.train):
          cv_train, cv_test = self.train.iloc[train_index], self.train.iloc[test_index]
          model = ARIMA(cv_train[self.target], order=param)
          predictions = model.fit().predict(start=len(cv_train), end=len(cv_train) + len(cv_test) - 1, dynamic=False)
          err = mean_squared_error(cv_test[self.target], predictions)
          errors.append(err)
        mse = np.mean(errors)
        if mse < best_mse:
          best_mse = mse
          best_params = param
        #print(f"ARIMA{param} with errors: {errors}\tmse:{mse}")
      except Exception as e:
        print(f"Error with params {param}: {e}")
      progress = progress_bar.value()
      progress += 1
      progress_bar.setValue(progress)
      QApplication.processEvents()
    if not best_params:
      print("No best params found")
      return [0,0,0]
    else:
      print(f"Best params found are: {best_params} with mse: {best_mse}")
      return best_params

  def train_ARIMA(self,params):
    prediction = []
    data = self.train['Price']
    for i in range(len(self.test)):
      model = ARIMA(data,order=params)
      model_fit = model.fit()
      pred = model_fit.predict(start=len(self.train)+i, end=len(self.train)+i, dynamic=False)
      prediction.append(pred[0])
      data = np.append(data,pred[0])
    mse = mean_squared_error(self.test['Price'], prediction)
    mape = mean_absolute_percentage_error(self.test['Price'], prediction)
    print(f"ARIMA:\nMSE: {mse}\nMAPE: {mape}")
    return prediction, mse, mape

  def best_params_RFR(self,n,d,progress_bar):
    params=list(itertools.product(n,d))
    best_mse = float('inf')
    best_params = []
    tscv = TimeSeriesSplit(n_splits=self.train_months)
    for param in params:
      model = RandomForestRegressor(n_estimators=param[0], max_depth=param[1])
      cv_score = cross_val_score(model,self.train[self.features],self.train['Price'],cv=tscv,scoring='neg_mean_squared_error')
      mse= -cv_score.mean()
      #print(f"RFR(n_est={param[0]}, max_depth={param[1]}) CV score: {cv_score}\t MSE: {mse}")
      if mse < best_mse:
          best_mse = mse
          best_params = param
      progress = progress_bar.value()
      progress += 1
      progress_bar.setValue(progress)
      QApplication.processEvents()
    print(f"\n\nBest params found are: {best_params} with mse: {best_mse}")
    return best_params

  def train_RFR(self, params):
    norm = self.scaler.fit(self.train[self.features])
    scaled_features_final = norm.transform(self.train[self.features])
    scaled_features_test = norm.transform(self.test[self.features])
    model = RandomForestRegressor(n_estimators=params[0], max_depth=params[1])
    model.fit(scaled_features_final, self.train['Price'])
    predicted_prices = model.predict(scaled_features_test)
    mse = mean_squared_error(self.test['Price'], predicted_prices)
    mape = mean_absolute_percentage_error(self.test['Price'], predicted_prices)
    print(f"RFR:\nMSE: {mse}\nMAPE: {mape}")
    return predicted_prices,mse,mape

  def best_params_SVR(self,c,epsilon,q,progress_bar):
    best_mse_linear = float('inf')
    best_mse_poly = float('inf')
    best_mse_rbf = float('inf')
    best_params_linear = []
    best_params_poly = []
    best_params_rbf = []
    params = list(itertools.product(c, epsilon))
    tscv = TimeSeriesSplit(n_splits=self.train_months)
    norm = self.scaler.fit(self.train[self.features])
    scaled_features_train = norm.transform(self.train[self.features])
    for param in params:
      model_linear = SVR(kernel='linear', C=param[0], epsilon=param[1])
      model_rbf = SVR(kernel='rbf', C=param[0], epsilon=param[1])
      cv_score_linear = cross_val_score(model_linear,scaled_features_train,self.train['Price'],cv=tscv,scoring='neg_mean_squared_error')
      mse_linear = -cv_score_linear.mean()
      cv_score_rbf = cross_val_score(model_rbf,scaled_features_train,self.train['Price'],cv=tscv,scoring='neg_mean_squared_error')
      mse_rbf = -cv_score_rbf.mean()
      #print(f"SVR (kernel=linear, C={param[0]}, epsilon={param[1]}) with CV_score: {cv_score_linear}\tmse: {mse_linear}")
      #print(f"SVR (kernel=rbf, C={param[0]}, epsilon={param[1]}) with CV_score: {cv_score_rbf}\tmse: {mse_rbf}")
      if mse_rbf < best_mse_rbf:
        best_mse_rbf = mse_rbf
        best_params_rbf = param
      if mse_linear < best_mse_linear:
        best_mse_linear = mse_linear
        best_params_linear = param
      for param2 in q:
        model_poly = SVR(kernel='poly', C=param[0], epsilon=param[1], degree=param2)
        cv_score_poly = cross_val_score(model_poly,scaled_features_train,self.train['Price'],cv=tscv,scoring='neg_mean_squared_error')
        mse_poly = -cv_score_poly.mean()
        #print(f"SVR (kernel=poly, C={param[0]}, epsilon={param[1]}, degree={param2}) with CV_score: {cv_score_poly}\tmse: {mse_poly}")
        if mse_poly < best_mse_poly:
          best_mse_poly = mse_poly
          best_params_poly = [param[0],param[1],param2]
      progress = progress_bar.value()
      progress += 1
      progress_bar.setValue(progress)
      QApplication.processEvents()
    print(f"Best SVR (kernel=linear, C={best_params_linear[0]}, epsilon={best_params_linear[1]}) with mse: {best_mse_linear}")
    print(f"Best SVR (kernel=poly, C={best_params_poly[0]}, epsilon={best_params_poly[1]}, q={best_params_poly[2]}) with mse: {best_mse_poly}")
    print(f"Best SVR (kernel=rbf, C={best_params_rbf[0]}, epsilon={best_params_rbf[1]}) with mse: {best_mse_rbf}")
    return [best_params_linear, best_params_poly, best_params_rbf]

  def train_SVR(self, params):
    mse = []
    mape = []
    kernels = ['linear','poly','rbf']
    norm = self.scaler.fit(self.train[self.features])
    scaled_features_train = norm.transform(self.train[self.features])
    scaled_features_test = norm.transform(self.test[self.features])
    model_linear = SVR(kernel='linear', C=params[0][0], epsilon=params[0][1])
    model_linear.fit(scaled_features_train,self.train['Price'])
    model_poly = SVR(kernel='poly', C=params[1][0], epsilon=params[1][1], degree=params[1][2])
    model_poly.fit(scaled_features_train,self.train['Price'])
    model_rbf = SVR(kernel='rbf', C=params[2][0], epsilon=params[2][1])
    model_rbf.fit(scaled_features_train,self.train['Price'])
    predicted_prices_linear = model_linear.predict(scaled_features_test)
    predicted_prices_poly = model_poly.predict(scaled_features_test)
    predicted_prices_rbf = model_rbf.predict(scaled_features_test)
    prediction = [predicted_prices_linear, predicted_prices_poly, predicted_prices_rbf]
    mse.append(mean_squared_error(self.test['Price'], predicted_prices_linear))
    mse.append(mean_squared_error(self.test['Price'], predicted_prices_poly))
    mse.append(mean_squared_error(self.test['Price'], predicted_prices_rbf))
    mape.append(mean_absolute_percentage_error(self.test['Price'], predicted_prices_linear))
    mape.append(mean_absolute_percentage_error(self.test['Price'], predicted_prices_poly))
    mape.append(mean_absolute_percentage_error(self.test['Price'], predicted_prices_rbf))
    for info in zip(mse,mape,kernels):
      print(f"SVR with {info[2]} kernel:\nMSE: {info[0]}\nMAPE: {info[1]}")
    return prediction, mse, mape
