import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import jarque_bera, norm

df_combined = pd.read_csv('data/df_combined.csv', parse_dates=['DATE_MIX'])
df_combined.set_index('DATE_MIX', inplace=True)

plt.rcParams.update({
    'figure.figsize': (12, 9),
    'font.size': 12,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

reg2 = df_combined[df_combined.index >= '2024-05-17'].copy()

# Целевая переменная
reg2['RET_TARGET'] = ((reg2['HIGH_MIX'] + reg2['LOW_MIX']) / 2).pct_change()

# Факторы
reg2['RGBI_Ret'] = reg2['CLOSE_RGBI'].pct_change()
reg2['RET_LAG1'] = reg2['RET_TARGET'].shift(1)
reg2['D_NET_YUR'] = reg2['NET_YUR'].diff()
reg2['D_RVI'] = reg2['CLOSE_RVI'].diff()

# Список факторов для отчета
factors = ['RGBI_Ret', 'RET_LAG1', 'D_NET_YUR', 'D_RVI']
model_data = reg2[factors + ['RET_TARGET']].dropna()

Y = model_data['RET_TARGET']
X = sm.add_constant(model_data[factors])
results = sm.OLS(Y, X).fit(cov_type='HC0')

resids = results.resid
preds = results.fittedvalues

# СТАТИСТИЧЕСКИЕ ТЕСТЫ ---
dw_stat = durbin_watson(resids)
jb_stat, jb_p_val = jarque_bera(resids)


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Остатки во времени
axes[0, 0].plot(resids.index, resids.values, 'k-', linewidth=0.8, alpha=0.7)
axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1)
axes[0, 0].set_title('Временной ряд остатков (Режим 2)')
axes[0, 0].set_ylabel('Значение ошибки')

# Остатки vs Прогнозы
axes[0, 1].scatter(preds, resids, color='black', alpha=0.5, s=15)
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=1)
axes[0, 1].set_xlabel('Прогнозные значения')
axes[0, 1].set_ylabel('Остатки')
axes[0, 1].set_title('Гомоскедастичность: Остатки vs Прогнозы')
axes[0, 1].text(0.05, 0.9, f'DW = {dw_stat:.2f}', transform=axes[0,1].transAxes, bbox=dict(facecolor='white', alpha=0.8))

#  QQ-График
sm.qqplot(resids, line='45', fit=True, ax=axes[1, 0], marker='o', 
          markerfacecolor='none', markeredgecolor='black', alpha=0.5)
axes[1, 0].get_lines()[1].set_color('red')
axes[1, 0].set_title('График Q-Q (Проверка нормальности)')

# Распределение остатков
axes[1, 1].hist(resids, bins=40, density=True, color='lightgray', edgecolor='black', alpha=0.7)
x_axis = np.linspace(resids.min(), resids.max(), 100)
axes[1, 1].plot(x_axis, norm.pdf(x_axis, resids.mean(), resids.std()), 'r-', lw=2)
axes[1, 1].set_title('Гистограмма распределения ошибок')
axes[1, 1].text(0.05, 0.9, f'JB p-val = {jb_p_val:.3e}', transform=axes[1,1].transAxes, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout(pad=3.0)
plt.savefig('Диагностика_Режим2_ФИНАЛ.png', dpi=300)
plt.show()

#  ОТЧЕТ
print("="*65)
print(f"ИТОГОВЫЙ ОТЧЕТ: РЕЖИМ 2 (С 17.05.2024)")
print("="*65)
print(f"Коэффициент детерминации (R²): {results.rsquared:.4f}")
print(f"Скорректированный R²:         {results.rsquared_adj:.4f}")
print(f"Статистика Дарбина-Уотсона:  {dw_stat:.3f}")
print(f"Тест Жарка-Бера (p-value):    {jb_p_val:.4e}")
print("-"*65)
print("РЕЗУЛЬТАТЫ РЕГРЕССИИ (Коэффициенты):")
print(results.summary().tables[1])
print("="*65)
