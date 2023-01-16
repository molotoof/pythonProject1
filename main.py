import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sktime
import datetime
import pmdarima as pm
import seaborn
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.compose import AutoEnsembleForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.trend import STLForecaster
from sktime.forecasting.tbats import TBATS
from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error as smape_loss
from sktime.performance_metrics.forecasting import mean_absolute_error as mae
from sktime.performance_metrics.forecasting import median_absolute_error as mdae
from itertools import combinations

def get_data(root_path):
    dataset = pd.read_csv(root_path, delimiter=';')
    return dataset

# dataset - временной ряд
# percent - доля (вещественное значение), указывающая часть датасета, которую надо выделить для тестовой части
def splitting_dataset(dataset, percent):
    # test_size - размер тестовой части выборки
    y_train, y_test = temporal_train_test_split(dataset, test_size=percent)
    return (y_train, y_test)

def ForecastTBATS(y_train, y_test, forecast_horizon):
    forecaster = TBATS(use_box_cox=True,
        use_trend=True,
        use_damped_trend=False,
        sp=12,
        use_arma_errors=False,
        n_jobs=1)
    forecaster.fit(X=y_train["Date"], y=y_train["Data"])
    y_pred_tbats = forecaster.predict(fh=[*range(1, forecast_horizon + 1)])
    grade_tbats = [smape_loss(y_test["Data"], y_pred_tbats), mae(y_test["Data"], y_pred_tbats),
                       mdae(y_test["Data"], y_pred_tbats)]
    return (y_pred_tbats, grade_tbats)

def ForecastPolynomialTrendForecaster(y_train, y_test, forecast_horizon):
    forecaster = PolynomialTrendForecaster(degree=5)
    forecaster.fit(X=y_train["Date"], y=y_train["Data"])
    y_pred_polinom = forecaster.predict(fh=[*range(1, forecast_horizon + 1)])
    grade_polinom = [smape_loss(y_test["Data"], y_pred_polinom), mae(y_test["Data"], y_pred_polinom),
                       mdae(y_test["Data"], y_pred_polinom)]
    return (y_pred_polinom, grade_polinom)

def ForecastSTLForecaster(y_train, y_test, forecast_horizon):
    forecaster = STLForecaster(sp=12)
    forecaster.fit(X=y_train["Date"], y=y_train["Data"])
    y_pred_stlforecaster = forecaster.predict(fh=[*range(1, forecast_horizon + 1)])
    grade_stlforecaster = [smape_loss(y_test["Data"], y_pred_stlforecaster), mae(y_test["Data"], y_pred_stlforecaster),
                       mdae(y_test["Data"], y_pred_stlforecaster)]
    return (y_pred_stlforecaster, grade_stlforecaster)

def FoercastExponentialSmoothing(y_train, y_test, forecast_horizon):
    # Экспоненциальное сглаживание
    # Используемые оценки: SMAPE, MAE, MdAE
    forecaster = ExponentialSmoothing(trend='mul', seasonal='mul', sp=12)
    forecaster.fit(X=y_train["Date"], y=y_train["Data"])
    y_pred_expSmooth = forecaster.predict(fh=[*range(1, forecast_horizon+1)])
    #plot_series(y_train["Data"], y_test["Data"], y_pred_expSmooth, labels=["y", "y_test", "y_pred"])
    grade_expSmooth = [smape_loss(y_test["Data"], y_pred_expSmooth), mae(y_test["Data"], y_pred_expSmooth), mdae(y_test["Data"], y_pred_expSmooth)]
    return (y_pred_expSmooth, grade_expSmooth)

def FoercastAutoETS(y_train, y_test, forecast_horizon):
    # В основе моделей ETS лежит экспоненциальное сглаживание – метод прогнозирования, при котором значения переменной
    # за все предыдущие периоды входят в прогноз, экспоненциально теряя свой вес со временем.
    # Это позволяет модели с достаточной степенью гибко реагировать на новейшие изменения в данных, сохраняя при этом
    # информацию об историческом поведении временного ряда.
    # Модели ETS могут быть использованы для прогнозирования рядов с выраженным трендом и сезонностью.
    # В таком случае экспоненциальное сглаживание также используется для оценки вклада последних двух факторов.
    # Каждый из трех компонентов модели – ошибки (Errors), тренд (Trend), сезонный фактор (Seasonality) – может быть
    # специфицирован отдельно, откуда класс моделей и несет свое название.
    # Используемые оценки: SMAPE, MAE, MdAE
    forecaster = AutoETS(auto=True, n_jobs=-1, sp=12)
    forecaster.fit(X=y_train["Date"], y=y_train["Data"])
    y_pred_autoETS = forecaster.predict(fh=[*range(1, forecast_horizon+1)])
    plot_series(y_train["Data"], y_test["Data"], y_pred_autoETS, labels=["y", "y_test", "y_pred"])
    grade_autoETS = [smape_loss(y_test["Data"], y_pred_autoETS), mae(y_pred_autoETS, y_test["Data"]), mdae(y_pred_autoETS, y_test["Data"])]
    return (y_pred_autoETS, grade_autoETS)

def FoercastTheta(y_train, y_test, forecast_horizon):
    # Тета-метод прогнозирования.
    # Используемые оценки: SMAPE, MAE, MdAE
    forecaster = ThetaForecaster(sp=12)
    forecaster.fit(X=y_train["Date"], y=y_train["Data"])
    y_pred_theta = forecaster.predict(fh=[*range(1, forecast_horizon+1)])
    plot_series(y_train["Data"], y_test["Data"], y_pred_theta, labels=["y", "y_test", "y_pred"])
    grade_Theta = [smape_loss(y_test["Data"], y_pred_theta), mae(y_pred_theta, y_test["Data"]), mdae(y_pred_theta, y_test["Data"])]
    return (y_pred_theta, grade_Theta)

def FoercastAutoARIMA(y_train, y_test, forecast_horizon):
    # AutoARIMA.
    # Используемые оценки: SMAPE, MAE, MdAE
    forecaster = AutoARIMA(sp=12, suppress_warnings=True, error_action="ignore")
    forecaster.fit(y=y_train["Data"])
    y_pred_autoARIMA = forecaster.predict(fh=[*range(1, forecast_horizon+1)])
    plot_series(y_train["Data"], y_test["Data"], y_pred_autoARIMA, labels=["y", "y_test", "y_pred"])
    grade_autoARIMA = [smape_loss(y_test["Data"], y_pred_autoARIMA), mae(y_pred_autoARIMA, y_test["Data"]), mdae(y_pred_autoARIMA, y_test["Data"])]
    return (y_pred_autoARIMA, grade_autoARIMA)

def FoercastAutoEnsemble(y_train, y_test, forecast_horizon, forecasters):
    forecaster = AutoEnsembleForecaster(forecasters=forecasters)
    forecaster.fit(y=y_train["Data"])
    y_pred = forecaster.predict(fh=[*range(1, forecast_horizon+1)])
    plot_series(y_train["Data"], y_test["Data"], y_pred, labels=["y", "y_test", "y_pred"])
    grade_autoEnsemble = [smape_loss(y_test["Data"], y_pred), mae(y_test["Data"], y_pred), mdae(y_test["Data"], y_pred)]
    return (y_pred, grade_autoEnsemble)

def ForecastSimpleEnsemble(forecasters_pred, y_test, coef_models, forecast_horizon):
    ensemble_forecaster = []
    elem_rez = 0
    for i in range(forecast_horizon):
        for j in range(len(coef_models)):
            elem_rez = elem_rez + (coef_models[j] * forecasters_pred[j][i])
        ensemble_forecaster.append(elem_rez / sum(coef_models))
        elem_rez = 0
    ensembleForecaster = pd.Series(ensemble_forecaster, index=y_test.index)
    #ensembleForecaster = np.array(ensemble_forecaster, dtype=float)
    #plot_series(y_train["Data"], y_test["Data"], ensembleForecaster, labels=["y", "y_test", "y_pred"])
    grade_simpleEnsemble = [smape_loss(y_test["Data"], ensembleForecaster), mae(ensembleForecaster, y_test["Data"]), mdae(ensembleForecaster, y_test["Data"])]
    return (ensembleForecaster, grade_simpleEnsemble)

def ForecastGrangerEnsemble(list_forecast, y_train, y_test, forecast_horizon):
    # buff / coef_model
    try:
        coef_model = np.dot((np.dot(np.linalg.inv(np.dot(list_forecast.transpose(),list_forecast)), list_forecast.transpose())).transpose(), y_test["Data"])
        # coef_model = []
        # for i in range(buff.shape[1]):
        #     coef_model.append(0)
        #     for j in range(buff.shape[0]):
        #         coef_model[i] = coef_model[i] + buff[j][i] * y_test["Data"][y_test.index[j]]
        print(coef_model)
        # ones = np.ones(list_forecast.shape[0])
        #
        # lambdaB = (np.dot(ones.transpose(), coef_model) - 1)/(np.dot(np.dot(ones.transpose(), np.linalg.inv(np.dot(list_forecast,list_forecast.transpose()))), ones))
        #
        # coef_model_2 = coef_model - np.dot(lambdaB, np.linalg.inv(np.dot(list_forecast,list_forecast.transpose())))
        # print(coef_model_2)
        (ensembleForecaster, grade_GrangerEnsemble) = ForecastSimpleEnsemble(list_forecast, y_test, coef_model, forecast_horizon)
        ensembleForecasterGranger = pd.Series(ensembleForecaster, index=y_test.index)
        plot_series(y_train["Data"], y_test["Data"], ensembleForecasterGranger, labels=["y", "y_test", "y_pred"])
        return (ensembleForecasterGranger, grade_GrangerEnsemble)
    except Exception as inst:
        print(inst)
    return([], [0])



# выбираем горизонт прогноза и долю тестовой части из временного ряда
forecast_horizon = 21 #12, 18, 21 ,24
test_part = 0.175
# выбираем датасет для прогноза
path_dataset = "data/average_monthly_salary_tum.csv"
dataset = get_data(path_dataset)
dataset.columns = ["Date", "Data"]
dataset["Date"] = pd.to_datetime(dataset["Date"]).dt.to_period("m")
(y_train, y_test) = splitting_dataset(dataset, test_part)

# Прогнозы
# (y_pred_expSmooth, grade_expSmooth) = FoercastExponentialSmoothing(y_train, y_test, forecast_horizon)
# (y_pred_autoETS, grade_autoETS) = FoercastAutoETS(y_train, y_test, forecast_horizon)
# (y_pred_theta, grade_Theta) = FoercastTheta(y_train, y_test, forecast_horizon)
# (y_pred_autoARIMA, grade_autoARIMA) = FoercastAutoARIMA(y_train, y_test, forecast_horizon)
# (y_pred_polinom, grade_polinom) = ForecastPolynomialTrendForecaster(y_train, y_test, forecast_horizon)
# (y_pred_stlforecaster, grade_stlforecaster) = ForecastSTLForecaster(y_train, y_test, forecast_horizon)
# (y_pred_tbats, grade_tbats) = ForecastTBATS(y_train, y_test, forecast_horizon)
#(y_pred_autoEnsemble, grade_autoEnsemble) = FoercastAutoEnsemble(y_train, y_test, forecast_horizon)
#(y_pred_simpleEnsemble, grade_simpleEnsemble) = ForecastSimpleEnsemble(y_pred_expSmooth, y_pred_autoETS, y_pred_theta, y_pred_autoARIMA, y_train, y_test)
# в list_forecast формируем список моделей для метода Грейнджера
#list_forecast = np.array([y_pred_expSmooth, y_pred_autoETS, y_pred_autoARIMA])
#(y_pred_GrangerEnsemble, grade_GrangerEnsemble) = ForecastGrangerEnsemble(list_forecast, y_train, y_test)

# функция для формирования списка комбинаций для такого количества моделей
list_models = [("expSmooth", ExponentialSmoothing(trend='mul', seasonal='add', sp=12)),
               ("autoETS", AutoETS(auto=True, n_jobs=-1, sp=12)),
               ("theta", ThetaForecaster(sp=12)),
               ("autoARIMA", AutoARIMA(sp=12, suppress_warnings=True)),
               ("polinom", PolynomialTrendForecaster(degree=2)),
               ("stlforecaster", STLForecaster(sp=12)),
               ("tbats", TBATS(use_box_cox=True, use_trend=True, use_damped_trend=False, sp=12, use_arma_errors=True, n_jobs=1))]
for count_models in range(3, 4):
    temp = combinations(list_models, count_models)
    for i in list(temp):
        forecasts = list(i)
        # Метод AutoEnsemble из библиотеки sktime
        (y_pred_autoEnsemble, grade_autoEnsemble) = FoercastAutoEnsemble(y_train, y_test, forecast_horizon, forecasts)
        if (grade_autoEnsemble[0]<0.05):
            print(forecasts, grade_autoEnsemble)
# list_pred = [y_pred_expSmooth, y_pred_autoETS, y_pred_theta, y_pred_autoARIMA, y_pred_polinom, y_pred_stlforecaster, y_pred_tbats]
# list_pred_name = ["y_pred_expSmooth", "y_pred_autoETS", "y_pred_theta", "y_pred_autoARIMA", "y_pred_polinom", "y_pred_stlforecaster", "y_pred_tbats"]
# for count_models in range(5, 6):
#     temp = combinations(list_pred, count_models)
#     temp_name = list(combinations(list_pred_name, count_models))
#     j = 0
#     for i in list(temp):
#         # Метод Grandger
#         forecasts_pred = np.array(list(i))
#         (y_pred_grandger, grade_grandger) = ForecastGrangerEnsemble(forecasts_pred, y_train, y_test, forecast_horizon)
#         if (grade_grandger[0] < 0.035):
#             print(temp_name[j], grade_grandger)
#         j = j + 1

pd.options.display.max_rows = 100

# print("grade_expSmooth", grade_expSmooth)
# print("grade_autoETS", grade_autoETS)
# print("grade_Theta", grade_Theta)
# print("grade_autoARIMA", grade_autoARIMA)
# print("grade_polinom", grade_polinom)
# print("grade_stlforecaster", grade_stlforecaster)
# print("grade_tbats", grade_tbats)
# print("grade_autoEnsemble", grade_autoEnsemble)
# print("grade_simpleEnsemble", grade_simpleEnsemble)
# print("grade_GrangerEnsemble", grade_GrangerEnsemble)