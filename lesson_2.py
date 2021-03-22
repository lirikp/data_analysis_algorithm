# 1. Постройте график зависимости весов всех признаков от lambda в L2-регуляризации (на данных из урока).
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 2, 1, 3, 0, 5, 10, 1, 2],  # стаж
              [500, 700, 750, 600, 1450,  # средняя стоимость занятия
               800, 1500, 2000, 450, 1000],
              [1, 1, 2, 1, 2, 1, 3, 3, 1, 2]])  # квалификация репетитора

y = [45, 55, 50, 59, 65, 35, 75, 80, 50, 60]

w = np.dot(
    np.dot(
        np.linalg.inv(
            np.dot(X, X.T)
        ),
        X),
    y)
X_norm = X.copy()
X_norm = X_norm.astype(np.float64)
X_norm[1] = (X[1] - X[1].min()) / (X[1].max() - X[1].min())
X_norm[2] = (X[2] - X[2].min()) / (X[2].max() - X[2].min())

X_st = X.copy().astype(np.float64)
X_st[1] = (X[1] - X[1].mean()) / X[1].std()
X_st[2] = (X[2] - X[2].mean()) / X[2].std()


def calc_mse(y, y_pred):
    err = np.mean((y - y_pred) ** 2)
    return err


def eval_model_reg2(X, y, iterations, alpha=1e-4, lambda_=1e-8):
    np.random.seed(42)
    W = np.random.randn(X.shape[0])
    n = X.shape[1]
    print(f'lambda={lambda_}')
    for i in range(1, iterations + 1):
        y_pred = np.dot(W, X)
        err = calc_mse(y, y_pred)
        W -= alpha * (1 / n * 2 * np.dot((y_pred - y), X.T) + 2 * lambda_ * W)
        if i % (iterations / 10) == 0:
            print(i, W, err)
    return W


level = 0.1
delta = 0.0001
lambda_ = 0.001

fig, axes = plt.subplots(1, 1)
fig.set_figwidth(12)
fig.set_figheight(12)

first = False

while lambda_ < level:
    W = eval_model_reg2(X_st, y, iterations=800, alpha=0.003, lambda_=lambda_)
    if not first:
        data = np.array([[lambda_, W[0], W[1], W[2], W[3]], ])
        first = True
    else:
        data = np.append(data, [[lambda_, W[0], W[1], W[2], W[3]]], axis=0)

    lambda_ += delta

_data = data[:, 1:5]

color = np.sort(np.random.random(len(data)))
axes.title.set_text('Зависимость f(lambda_)', )

axes.set_xlabel('lambda_')

axes.set_xlim((0.0001 - 0.01, level + 0.01))

# Vertical Lines
axes.vlines(x=data[:, 0].min(), ymin=-10, ymax=40, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
axes.vlines(x=data[:, 0].max(), ymin=-10, ymax=40, color='black', alpha=0.7, linewidth=1, linestyles='dotted')

axes.text(data[:, 0].min(), 35, f'lambda={round(data[:, 0].min(), 5)}', horizontalalignment='right',
          verticalalignment='center', fontdict={'size': 18, 'weight': 700})
axes.text(data[:, 0].max(), 35, f'lambda={round(data[:, 0].max(), 5)}', horizontalalignment='left',
          verticalalignment='center', fontdict={'size': 18, 'weight': 700})

axes.scatter(data[:, 0], _data[:, 0], c='r')
axes.text(data[:, 0].min(), _data[0:1, 0], f'NAN={round(float(_data[-1, 0]), 4)}',
          horizontalalignment='right', verticalalignment='center', fontdict={'size': 14})
axes.text(data[:, 0].max(), _data[-1, 0], f'NAN={round(float(_data[0:1, 0]), 4)}',
          horizontalalignment='left', verticalalignment='center', fontdict={'size': 14})

axes.scatter(data[:, 0], _data[:, 1], c='b')
axes.text(data[:, 0].min(), _data[0:1, 1], f'Стаж={round(float(_data[-1, 1]), 4)}',
          horizontalalignment='right', verticalalignment='center', fontdict={'size': 14})
axes.text(data[:, 0].max(), _data[-1, 1], f'Стаж={round(float(_data[0:1, 1]), 4)}',
          horizontalalignment='left', verticalalignment='center', fontdict={'size': 14})

axes.scatter(data[:, 0], _data[:, 2], c='g')
axes.text(data[:, 0].min(), _data[0:1, 2], f'Стоимость={round(float(_data[-1, 2]), 4)}',
          horizontalalignment='right', verticalalignment='center', fontdict={'size': 14})
axes.text(data[:, 0].max(), _data[-1, 2], f'Стоимость={round(float(_data[0:1, 2]), 4)}',
          horizontalalignment='left', verticalalignment='center', fontdict={'size': 14})


axes.scatter(data[:, 0], _data[:, 3], c='y')
axes.text(data[:, 0].min(), _data[0:1, 3], f'Квалификация={round(float(_data[-1, 3]), 4)}',
          horizontalalignment='right', verticalalignment='center', fontdict={'size': 14})
axes.text(data[:, 0].max(), _data[-1, 3], f'Квалификация={round(float(_data[0:1, 3]), 4)}',
          horizontalalignment='left', verticalalignment='center', fontdict={'size': 14})


plt.show()
# Ответ на периоде lambda от 0.001 до 0.01 зависимость фитч меняется слабо.

# 2. Можно ли к одному и тому же признаку применить сразу и нормализацию, и стандартизацию?
# Ответ: Можно, только нужно обязательно тогда нормализировать/стандартизировать все колонки данных.

# 3. *Напишите функцию наподобие eval_model_reg2, но для применения L1-регуляризации.
