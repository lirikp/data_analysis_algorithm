import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
sns.set_context("paper", font_scale=2)


plt.style.use('seaborn-ticks')
plt.rcParams.update({'font.size': 14})


X = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 2, 1, 3, 0, 5, 10, 1, 2],  # стаж
              [500, 700, 750, 600, 1450,        # средняя стоимость занятия
               800, 1500, 2000, 450, 1000],
              [1, 1, 2, 1, 2, 1, 3, 3, 1, 2]], dtype = np.float64) # квалификация репетитора

y = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1]) # подходит или нет репетитор

def calc_std_feat(x):
    res = (x - x.mean()) / x.std()
    return res

def calc_logloss(y, y_pred):
    err = np.mean(- y * np.log(y_pred) - (1.0 - y) * np.log(1.0 - y_pred))
    return err

def sigmoid(z):
    res = 1 / (1 + np.exp(-z))
    return res

def calc_mse(y, y_pred):
    err = np.mean((y - y_pred) ** 2)
    return err

z = np.linspace(-10, 10, 101)
#1 *Измените функцию calc_logloss так, чтобы нули по возможности не попадали в np.log (как вариант - np.clip).
#2 Подберите аргументы функции eval_LR_model для логистической регрессии таким образом, чтобы log loss был минимальным.

def eval_LR_model(X, y, iterations, alpha=1e-4):
    np.random.seed(42)
    w = np.random.randn(X.shape[0])
    n = X.shape[1]
    for i in range(1, iterations + 1):

        z = np.dot(w, X)
        y_pred = sigmoid(z)
        logloss_err = calc_logloss(y, y_pred)

        y_pred = np.dot(w, X)
        mse_err = calc_mse(y, y_pred)

        w -= alpha * (1 / n * np.dot((y_pred - y), X.T))
        if i % (iterations / 10) == 0:
            print(i, w, logloss_err, mse_err)
    return w

eval_LR_model(X, y, 800)
print(1)
#3 Создайте функцию calc_pred_proba, возвращающую предсказанную вероятность класса 1 (на вход подаются веса, которые уже посчитаны функцией eval_LR_model и X, на выходе - массив y_pred_proba).
#4 Создайте функцию calc_pred, возвращающую предсказанный класс (на вход подаются веса, которые уже посчитаны функцией eval_LR_model и X, на выходе - массив y_pred).
#5 Посчитайте accuracy, матрицу ошибок, precision и recall, а также F1-score.
#6 Могла ли модель переобучиться? Почему?
#7 *Создайте функции eval_LR_model_l1 и eval_LR_model_l2 с применением L1 и L2 регуляризации соответственно.
