"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Загружаем данные из файла
df = pd.read_csv("2015.csv")

# Выбираем столбец данных
selected_column = 'dbz_geo'
# Подбираем распределение
fit_distribution = stats.norm # Нормальное распределение (можно выбрать другое, если необходимо)
#params = fit_distribution.fit(df[selected_column])
print(fit_distribution)

# Рассчитываем вероятность превышения максимального значения
max_value = df[selected_column].max()
probability_exceeding = 1 - fit_distribution.cdf(max_value)
print(f"Вероятность превышения максимального значения: {probability_exceeding:.2f}")

# Строим гистограмму
plt.figure(figsize=(10, 6))
sns.histplot(df[selected_column], kde=True, color='skyblue', bins=20)
plt.title('Гистограмма выбранного столбца')
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.grid(True)
plt.show()
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import warnings

# Подавление предупреждений
warnings.filterwarnings("ignore")

# Функция для подбора наилучшего распределения
def get_best_distribution(data):
    distributions = [
        stats.norm, stats.expon, stats.weibull_min, stats.beta, stats.gamma,
        stats.t, stats.lognorm, stats.uniform, stats.pearson3, stats.triang,
        stats.logistic, stats.gumbel_r, stats.gumbel_l, stats.nakagami
    ]
    best_distribution = None
    best_ks_stat = np.inf

    for distribution in distributions:
        try:
            params = distribution.fit(data)
            # Проводим тест Колмогорова-Смирнова
            ks_stat, _ = stats.kstest(data, distribution.cdf, args=params)
            # Если статистика Колмогорова-Смирнова лучше, сохраняем распределение
            if ks_stat < best_ks_stat:
                best_distribution = distribution
                best_ks_stat = ks_stat
        except Exception as e:
            continue

    return best_distribution, best_ks_stat

# Загружаем данные из файла
df = pd.read_csv("2015.csv")

# Выбираем столбец данных
selected_column = 'dbn_geo'

# Удаляем пропущенные и нечисловые значения
data = df[selected_column].dropna()
data = data[np.isfinite(data)]

# Подбираем наилучшее распределение
best_distribution, best_ks_stat = get_best_distribution(data)

if best_distribution is not None:
    # Параметры наилучшего распределения
    params = best_distribution.fit(data)
    print(f"Наилучшее распределение: {best_distribution.name}")
    print("Параметры распределения:", params)

    # Строим гистограмму данных
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(data, kde=False, color='skyblue', bins=200, stat='density', label='Данные', ax=ax1)

    # Плотность наилучшего распределения
    x = np.linspace(data.min(), data.max(), 1000)
    pdf_fitted = best_distribution.pdf(x, *params)
    ax1.plot(x, pdf_fitted, 'r-', label=f'{best_distribution.name}')

    # Настройки оси для плотности
    ax1.set_xlabel('Значение')
    ax1.set_ylabel('Плотность')
    ax1.legend(loc='upper left')

    # Выводим значение статистики Колмогорова-Смирнова
    ax1.text(0.5, 0.9, f"Статистика Колмогорова-Смирнова: {best_ks_stat:.6f}",
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax1.transAxes, fontsize=12)

    # Вторая ось для функции выживаемости
    ax2 = ax1.twinx()
    survival_function = 1 - best_distribution.cdf(x, *params)
    ax2.plot(x, survival_function, 'b--', label='Функция выживаемости')
    ax2.set_ylabel('Вероятность выживания')
    ax2.legend(loc='upper right')

    plt.title('Гистограмма данных, плотность вероятности и функция выживаемости')
    ax1.grid(True)
    plt.show()
else:
    print("Не удалось найти подходящее распределение.")


