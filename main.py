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


