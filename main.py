import json
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout
from PyQt5 import uic
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from models_class import Models
from best_params_class import Best_params_handler

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_type = None
        self.train_months = 6
        self.train = None
        self.test = None
        self.target = ['Ex_Rate']
        self.params_handler = Best_params_handler("best_params.json")
        self.arima_pred = None
        self.rfr_pred = None
        self.svrlin_pred = None
        self.svrpoly_pred = None
        self.svrrbf_pred = None
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.init_ui()

    def init_ui(self):                              #Loading UI and connecting buttons
        uic.loadUi("UI.ui",self)
        if self.graph.layout() is None:
            layout = QVBoxLayout(self.graph)
            self.graph.setLayout(layout)
        else:
            layout = self.graph.layout()
        layout.addWidget(self.canvas)
        self.Predict_button.clicked.connect(self.predict)
        self.Find_best_params_button.clicked.connect(self.find_best_params)
        self.result_table.hide()
        self.graph.hide()
        self.best_params_progress.hide()
        for row in range(self.result_table.rowCount()):                         #Changing font and size in table and adding button to save the graph
            button = QPushButton("Save graph to file")
            button.setFont(QFont("Arial", 16))
            button.clicked.connect(lambda _, r=row: self.save_graph_to_file(r))
            self.result_table.setCellWidget(row,3,button)

    def save_graph_to_file(self, row):                                      #Saving graph into the file
        combined_data = pd.concat([self.train, self.test])
        path=str(self.data_type)
        plt.figure(figsize=(10,8))
        plt.plot(combined_data.index, combined_data[self.target], label=self.data_type + " exchange rate", color='black')
        plt.plot(self.test.index, self.test[self.target], label='Real values', color='blue')
        plt.xlabel('Date')
        plt.ylabel('Exchange rate')             #Plotting the graph
        match row:                              #Adding prediction and changing name based on model used
            case 0:
                path += "_prediction_using_ARIMA_with_"
                path += (str(self.train_months) + "_months.png")
                plt.plot(self.test.index, self.arima_pred, label='Predicted values', color='red')
                plt.title(self.data_type + "_prediction_using_ARIMA_with_" + str(self.train_months) +"_months")
            case 1:
                path += "_prediction_using_RFR_with_"
                path += (str(self.train_months) + "_months.png")
                plt.plot(self.test.index, self.rfr_pred, label = 'Predicted values', color = 'red')
                plt.title(self.data_type + "_prediction_using_RFR_with_" + str(self.train_months) +"_months")
            case 2:
                path += "_prediction_using_SVR_linear_with_"
                path += (str(self.train_months) + "_months.png")
                plt.plot(self.test.index, self.svrlin_pred, label='Predicted values', color='red')
                plt.title(self.data_type + "_prediction_using_Linear_SVR_with_" + str(self.train_months) +"_months")
            case 3:
                path += "_prediction_using_SVR_polynomial_with_"
                path += (str(self.train_months) + "_months.png")
                plt.plot(self.test.index, self.svrpoly_pred, label='Predicted values', color='red')
                plt.title(self.data_type + "_prediction_using_Polynomial_SVR_with_" + str(self.train_months) +"_months")
            case 4:
                path += "_prediction_using_SVR_RBF_with_"
                path += (str(self.train_months) + "_months.png")
                plt.plot(self.test.index, self.svrrbf_pred, label='Predicted values', color='red')
                plt.title(self.data_type + "_prediction_using_RBF_SVR_with_" + str(self.train_months) +"_months")
        plt.legend()
        plt.savefig(path, dpi=200)
        print(f"Successfully saved as {path}")

    def predict(self):                                      #Prediction function
        self.result_table.show()
        self.graph.show()
        self.figure.clear()
        self.train_months = (self.Month_choose.currentIndex() + 1)*2        #Getting train_months from ComboBox
        match(self.Data_choose.currentIndex()):                             #Getting currency from ComboBox
            case 0:
                self.data_type = "1EUR"
            case 1:
                self.data_type = "1USD"
            case 2:
                self.data_type = "1CAD"
            case 3:
                self.data_type = "100KRW"
            case 4:
                self.data_type = "100CLP"
        if self.Params_choose.currentIndex() == 0:                          #Getting params from ComboBox
            best_params = self.params_handler.get_params(self.train_months,self.data_type)
            if best_params is None:
                print("\033[91mNO BEST PARAMS SAVED FOR THIS CONFIGURATION OF CURRENCY AND NUMBER OF MONTHS. CHOOSING DEFAULT PARAMETERS INSTEAD!!!\033[0m")
                params_rfr = None
                params_arima = None
                params_svr = None
            else:
                params_rfr = best_params['rfr']
                params_arima = best_params['arima']
                params_svr = best_params['svr']
        else:
            params_rfr = None
            params_arima = None
            params_svr = None
        if not params_rfr:                          #Default params from sklearn and statsmodels site
            params_rfr = [100, None]
        if not params_arima:
            params_arima = [0, 0, 0]
        if not params_svr:
            params_svr = [[1, 0.1], [1, 0.1, 3], [1, 0.1]]
        df = pd.read_csv('currency_data.csv', encoding='latin-1', sep=';')          #Loading data
        models = Models(df,self.data_type,self.train_months)
        models.prepare_data()
        self.train = models.train
        self.test = models.test
        self.target = models.target
        RFR_prediction, RFR_mse, RFR_mape = models.train_RFR(params_rfr)                    #Training models and predicting
        ARIMA_prediction, ARIMA_mse, ARIMA_mape = models.train_ARIMA(params_arima)
        SVR_prediction, SVR_mse, SVR_mape = models.train_SVR(params_svr)
        self.rfr_pred = RFR_prediction                                                      #Rounding MSE and MAPE
        RFR_mse = round(RFR_mse,4)
        RFR_mape = round(RFR_mape*100,2)
        self.arima_pred = ARIMA_prediction
        ARIMA_mse = round(ARIMA_mse,4)
        ARIMA_mape = round(ARIMA_mape*100,2)
        self.svrlin_pred = SVR_prediction[0]
        self.svrpoly_pred = SVR_prediction[1]
        self.svrrbf_pred = SVR_prediction[2]
        for i in range(3):
            SVR_mse[i] = round(SVR_mse[i],4)
            SVR_mape[i] = round(SVR_mape[i]*100,2)
        last_train_data = models.train.tail(30)                                             #Data to visualization
        zoomed_data = pd.concat([last_train_data,models.test])
        ax = self.figure.add_subplot(111)                                                   #Plotting results
        ax.plot(zoomed_data.index, zoomed_data[models.target], label = self.data_type, color = 'black')
        ax.plot(models.test.index, models.test[models.target], label = 'Real values', color = 'blue')
        ax.plot(models.test.index, RFR_prediction, label = 'Predicted values RFR', color = '#29f20a')
        ax.plot(models.test.index, ARIMA_prediction, label = 'Predicted values ARIMA', color = '#f2a90a')
        ax.plot(models.test.index, SVR_prediction[0], label='Predicted values Linear SVR', color='#f740f1')
        ax.plot(models.test.index, SVR_prediction[1], label='Predicted values Polynomial SVR', color='#4a073a')
        ax.plot(models.test.index, SVR_prediction[2], label='Predicted values RBF SVR', color='#f71616')
        ax.set_xlabel('Date')
        ax.set_ylabel('Exchange rate')
        ax.set_title(f'{self.data_type} exchange rate predictions')
        ax.legend()
        self.canvas.draw()
        self.result_table.item(0, 0).setText(str(params_arima))                     #Adding MSE, MAPE and params to table
        self.result_table.item(0, 1).setText(str(ARIMA_mse))
        self.result_table.item(0, 2).setText(str(ARIMA_mape) + "%")
        self.result_table.item(1, 0).setText(str(params_rfr))
        self.result_table.item(1, 1).setText(str(RFR_mse))
        self.result_table.item(1, 2).setText(str(RFR_mape)+"%")
        self.result_table.item(2, 0).setText(str(params_svr[0]))
        self.result_table.item(2, 1).setText(str(SVR_mse[0]))
        self.result_table.item(2, 2).setText(str(SVR_mape[0]) + "%")
        self.result_table.item(3, 0).setText(str(params_svr[1]))
        self.result_table.item(3, 1).setText(str(SVR_mse[1]))
        self.result_table.item(3, 2).setText(str(SVR_mape[1]) + "%")
        self.result_table.item(4, 0).setText(str(params_svr[2]))
        self.result_table.item(4, 1).setText(str(SVR_mse[2]))
        self.result_table.item(4, 2).setText(str(SVR_mape[2]) + "%")

    def find_best_params(self):                                           #Finding best params works similar to predict but without plotting and table
        df = pd.read_csv('currency_data.csv', encoding='latin-1', sep=';')
        self.train_months = (self.Month_choose.currentIndex() + 1)*2
        match (self.Data_choose.currentIndex()):
            case 0:
                self.data_type = "1EUR"
            case 1:
                self.data_type = "1USD"
            case 2:
                self.data_type = "1CAD"
            case 3:
                self.data_type = "100KRW"
            case 4:
                self.data_type = "100CLP"
        models = Models(df,self.data_type,self.train_months)
        models.prepare_data()
        n = [10,25,50,100,200,300,500]                                  #Params from which the best one are chosen
        depth = [2,5,15,25,50,100,None]
        p = [0,1,2,3,4]
        d = [0,1,2]
        q = [0,1,2,3,4]
        c = [0.01,0.1,1,10,100,1000]
        epsilon = [0.01,0.1,1,10,100]
        degree = [2,3,4]
        self.best_params_progress.show()                                #Progress bar logic - 1 step = 1 for loop = 1 set of params checked
        total_steps = len(n)*len(depth) + len(p)*len(d)*len(q) + len(c)*len(epsilon)
        self.best_params_progress.setMaximum(total_steps)
        self.best_params_progress.setValue(0)
        params_rfr = models.best_params_RFR(n,depth,self.best_params_progress)
        params_arima = models.best_params_ARIMA(p,d,q,self.best_params_progress)
        params_svr = models.best_params_SVR(c,epsilon,degree,self.best_params_progress)
        params = {                                                            #Saving those params into file with help of best_params_handler class
            'rfr':params_rfr,
            'arima':params_arima,
            'svr':params_svr
        }
        self.params_handler.set_params(self.train_months,self.data_type,params)
        self.best_params_progress.hide()

if __name__ == '__main__':                                      #Main - showing the application and running it until close is pressed
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
