import numpy as np


def calc_mse(y, y_pred):
    err = np.mean((y - y_pred) ** 2)
    return err


X = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # для умножения на intercept
              [1, 1, 2, 1, 3, 0, 5, 10, 1, 2]])  # стаж
# средний балл ЕГЭ (целевая переменная)
y = [45, 55, 50, 59, 65, 35, 75, 80, 50, 60, ]
n = X.shape[1]


alpha = 0.001
delta_alpha = 1e-4
attempt = 1000
delta_attempt = 100

mean_last_diff = np.inf


# Функция которая измеряет быстроту сползания значения к своему минимуму когда разница между предидущим значением
# и новым не больше level_change = 1
def tester(arr, *args, **kwargs):
    level_change = 1
    if change_alpha: #Альфа растёт вверх
        if (kwargs['mean_last_diff'] - arr[-1]) > level_change:
            return (True, False, arr[-1])
        else:
            return (False, True, arr[-1])

    elif change_attempt: #Попытки идут вниз
        if (arr[-1] - kwargs['mean_last_diff']) < 10:
            return (False, True, kwargs['mean_last_diff'])
        else:
            return (False, False, kwargs['mean_last_diff'])


def gradient_descent(*args, **kwargs):
    for_tester = []
    for i in range(kwargs["attempt"]):
        y_pred = np.dot(kwargs["w"], kwargs["X"])

        for j in range(kwargs["w"].shape[0]):
            kwargs["w"][j] -= kwargs["alpha"] * (1 / n * 2 * np.sum(kwargs["X"][j] * (y_pred - kwargs["y"])))
        if i % 100 == 0:
            e = calc_mse(kwargs["y"], y_pred)
            for_tester.append(e)
            print(i, w, e)

    return for_tester


change_attempt = False  # Разрешение на изменение количество попыток
change_alpha = True  # Разрешение на изменение альфы
arr_mean_last_diff = []
while change_attempt or change_alpha:  # Запуск цикла пока оба или одно разрешение активно
    w = np.array([1, 0.5])
    for_tester = gradient_descent(attempt=attempt, w=w, X=X, alpha=alpha, y=y)  # Расчёт массива MSE

    # Тестируем на предмет добавления альфы и уменьшения шагов
    (change_alpha, change_attempt, mean_last_diff) = tester(for_tester, alpha=alpha, attempt=attempt,
                                                            change_attempt=change_attempt,
                                                            change_alpha=change_alpha, mean_last_diff=mean_last_diff)

    arr_mean_last_diff.append(mean_last_diff)  # Складируем Последний шаг MSE из массива проверок.

    if change_attempt: #Попытки идут вниз
        attempt -= delta_attempt
        if attempt < 0:
            break;

    if change_alpha: #Альфа растёт вверх
        alpha += delta_alpha
        if alpha < 0:
            break;

print(f'Итого:\talpha={alpha}; достаточно попыток={attempt};')
#Итого:	alpha=0.002899999999999999; достаточно попыток=800;
