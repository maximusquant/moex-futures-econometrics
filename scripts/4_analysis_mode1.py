import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import jarque_bera, norm


df_final = pd.read_csv('data/df_final.csv', parse_dates=['DATE_MIX'])
df_final.set_index('DATE_MIX', inplace=True)

plt.rcParams.update({
    'figure.figsize': (12, 9),
    'font.size': 12,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})


reg1 = df_final[df_final.index < '2024-05-17'].copy()

reg1['Доходность_MIX'] = ((reg1['HIGH_MIX'] + reg1['LOW_MIX']) + reg1['CLOSE_MX'] / 3).pct_change()
reg1['Юань_Лаг1'] = reg1['CLOSE_CNY'].pct_change().shift(1)
reg1['Нефть_Brent'] = reg1['CLOSE_Brent'].pct_change()
reg1['Изм_RVI'] = reg1['CLOSE_RVI'].diff()
reg1['Изм_RGBI'] = reg1['CLOSE_RGBI'].diff()
reg1['Базис_РТС'] = ((reg1['CLOSE_RI'] - (reg1['CLOSE_RTSI'] * 100)) / (reg1['CLOSE_RTSI'] * 100)) * 100
reg1['Базис_Лаг4'] = reg1['Базис_РТС'].shift(4)

# Очистка от пустых значений
факторы = ['Юань_Лаг1', 'Базис_Лаг4', 'Нефть_Brent', 'Изм_RVI', 'Изм_RGBI']
данные_модели = reg1[факторы + ['Доходность_MIX']].dropna()

Y = данные_модели['Доходность_MIX']
X = sm.add_constant(данные_модели[факторы])
результаты = sm.OLS(Y, X).fit(cov_type='HC0')

остатки = результаты.resid
прогнозы = результаты.fittedvalues
dw_стат = durbin_watson(остатки)
jb_стат, jb_p_val = jarque_bera(остатки)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(остатки.index, остатки.values, 'k-', linewidth=0.8, alpha=0.7)
axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1)
axes[0, 0].set_title('Временной ряд остатков')
axes[0, 0].set_ylabel('Значение ошибки')
axes[0, 0].text(0.05, 0.9, f'std = {остатки.std():.4f}', transform=axes[0,0].transAxes, bbox=dict(facecolor='white', alpha=0.8))

axes[0, 1].scatter(прогнозы, остатки, color='black', alpha=0.5, s=15)
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=1)
axes[0, 1].set_xlabel('Прогнозные значения')
axes[0, 1].set_ylabel('Остатки')
axes[0, 1].set_title('Остатки vs Прогнозы')
axes[0, 1].text(0.05, 0.9, f'DW = {dw_стат:.2f}', transform=axes[0,1].transAxes, bbox=dict(facecolor='white', alpha=0.8))

#QQ-График
sm.qqplot(остатки, line='45', fit=True, ax=axes[1, 0], marker='o', 
          markerfacecolor='none', markeredgecolor='black', alpha=0.5)
axes[1, 0].get_lines()[1].set_color('red')
axes[1, 0].set_title('График Q-Q (Квантиль-Квантиль)')
axes[1, 0].set_xlabel('Теоретические квантили')
axes[1, 0].set_ylabel('Выборочные квантили')

# Распределение остатков
axes[1, 1].hist(остатки, bins=40, density=True, color='lightgray', edgecolor='black', alpha=0.7)
ось_x = np.linspace(остатки.min(), остатки.max(), 100)
axes[1, 1].plot(ось_x, norm.pdf(ось_x, остатки.mean(), остатки.std()), 'r-', lw=2)
axes[1, 1].set_title('Распределение остатков')
axes[1, 1].set_xlabel('Ошибка')
axes[1, 1].text(0.05, 0.9, f'JB p-val = {jb_p_val:.3f}', transform=axes[1,1].transAxes, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout(pad=3.0)
plt.savefig('Диагностика_Режим1_РФ.png', dpi=300)
plt.show()

#  ИТОГОВЫЙ ОТЧЕТ ---
print("="*60)
print(f"ИТОГОВЫЙ ОТЧЕТ: РЕЖИМ 1 (ДО 17.05.2024)")
print("="*60)
print(f"Коэффициент детерминации (R²): {результаты.rsquared:.4f}")
print(f"Скорректированный R²:         {результаты.rsquared_adj:.4f}")
print(f"Статистика Дарбина-Уотсона:  {dw_стат:.3f}")
print(f"Тест Жарка-Бера (p-value):    {jb_p_val:.4e}")
print("-"*60)
print("РЕЗУЛЬТАТЫ РЕГРЕССИИ:")
print(результаты.summary().tables[1])
print("="*60)
