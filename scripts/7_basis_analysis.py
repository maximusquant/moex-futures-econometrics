import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# Загружаем df_final (все цены) из папки data/
df_final = pd.read_csv('data/df_final.csv', parse_dates=['DATE_MIX'])
df_final.set_index('DATE_MIX', inplace=True)

# Загружаем OI данные
df_oi = pd.read_csv('data/MIX_OI_daily_CORRECT.csv', parse_dates=['date'])
df_oi.columns = df_oi.columns.str.strip().str.lower()
df_oi.set_index('date', inplace=True)


# Берём нужные колонки из df_final
df_plot = df_final[['CLOSE_MX', 'CLOSE_IMOEX', 'CLOSE_RUSFAR', 'CLOSE_RGBI']].copy()

# Приклеиваем OI (short_yur и long_yur)
df_plot = df_plot.merge(df_oi[['short_yur', 'long_yur']], left_index=True, right_index=True, how='left')

# Заливаем пустоты вперёд
df_plot = df_plot.ffill()


#  Расчет базиса и Z-Score

q = 0.065  # Дивидентная доходность из формулы COST-of-CARRY 6.5 % средние цифры для Росс.рынка
expiration_dates = [
    '2022-09-15', '2022-12-15', '2023-03-16', '2023-06-15', 
    '2023-09-21', '2023-12-21', '2024-03-21', '2024-06-20', 
    '2024-09-19', '2024-12-19', '2025-03-20', '2025-06-19', 
    '2025-09-18', '2025-12-18', '2026-03-19'
]
exp_dates = pd.to_datetime(expiration_dates).sort_values()

def get_t(d):
    future = exp_dates[exp_dates >= d]
    return max((future[0] - d).days, 1) if not future.empty else np.nan

df_plot['T'] = df_plot.index.to_series().apply(get_t)
S_adj = df_plot['CLOSE_IMOEX'] * 100
r = df_plot['CLOSE_RUSFAR'] / 100

# Теоретическая цена и базис
df_plot['F_theory'] = S_adj * (1 + (r - q) * df_plot['T'] / 365)
df_plot['Abnormal_Basis'] = df_plot['CLOSE_MX'] - df_plot['F_theory']

# Считаем Z-score для масок аномалий
df_plot['basis_z'] = (df_plot['Abnormal_Basis'] - df_plot['Abnormal_Basis'].mean()) / df_plot['Abnormal_Basis'].std()

# 
#  Визуализация зон премии и дисконта
# 
threshold = 1.5

# Маски для аномалий
mask_premium = df_plot['basis_z'] > threshold      # Контанго (премия)
mask_discount = df_plot['basis_z'] < -threshold    # Бэквардация (дисконт)

# Группировка непрерывных зон
df_plot['group_premium'] = (mask_premium != mask_premium.shift()).cumsum()
df_plot['group_discount'] = (mask_discount != mask_discount.shift()).cumsum()

# ГРАФИК 1: ЗОНЫ ПРЕМИИ (КОНТАНГО) 
fig1, (ax1_price, ax1_short, ax1_rgbi) = plt.subplots(3, 1, figsize=(16, 14), sharex=True,
                                                        gridspec_kw={'height_ratios': [3, 1, 1.5]})

ax1_price.plot(df_plot.index, df_plot['CLOSE_MX'], color='black', lw=1.5, label='Цена MIX Futures')
ax1_short.plot(df_plot.index, df_plot['short_yur'], color='#C0392B', lw=1.2)
ax1_short.fill_between(df_plot.index, 0, df_plot['short_yur'], color='#C0392B', alpha=0.05)
ax1_short.set_ylabel('Short OI (контракты)', color='#C0392B')

ax1_rgbi.plot(df_plot.index, df_plot['CLOSE_RGBI'], color='darkblue', lw=1.5)
ax1_rgbi.set_ylabel('Индекс RGBI')
ax1_rgbi.grid(True, alpha=0.1)

for g in df_plot.loc[mask_premium, 'group_premium'].unique():
    block = df_plot[df_plot['group_premium'] == g]
    start_date = block.index[0]
    end_date = block.index[-1]
    
    price_min = block['CLOSE_MX'].min()
    price_max = block['CLOSE_MX'].max()
    
    ax1_price.axvspan(start_date, end_date, color='#27AE60', alpha=0.15, zorder=0)
    ax1_short.axvspan(start_date, end_date, color='#27AE60', alpha=0.15, zorder=0)
    ax1_rgbi.axvspan(start_date, end_date, color='#27AE60', alpha=0.15, zorder=0)
    
    x_start = (end_date - df_plot.index[0]) / (df_plot.index[-1] - df_plot.index[0])
    
    ax1_price.axhspan(price_min, price_max, xmin=x_start, xmax=1.0, color='#27AE60', alpha=0.25, zorder=1)
    ax1_price.axhline(price_min, xmin=x_start, xmax=1.0, color='#27AE60', linestyle='--', alpha=0.6, lw=0.8)
    ax1_price.axhline(price_max, xmin=x_start, xmax=1.0, color='#27AE60', linestyle='--', alpha=0.6, lw=0.8)
    ax1_price.scatter(block.index, block['CLOSE_MX'], color='#27AE60', s=40, alpha=0.7, zorder=3, marker='^')

ax1_price.set_title('Зоны экстремального КОНТАНГО (премия) → Проекция уровней ТОЛЬКО ВПРАВО', fontsize=14)
ax1_price.set_ylabel('Цена MIX')
ax1_price.grid(True, alpha=0.1)
ax1_price.legend(loc='upper left')
ax1_short.grid(True, alpha=0.1)
ax1_rgbi.set_xlabel('Дата')
plt.tight_layout()
plt.savefig('notebooks/images/basis_premium.png', dpi=300, bbox_inches='tight')
plt.show()

# --- ГРАФИК 2: ЗОНЫ ДИСКОНТА (БЭКВАРДАЦИЯ) ---
fig2, (ax2_price, ax2_long, ax2_rgbi) = plt.subplots(3, 1, figsize=(16, 14), sharex=True,
                                                      gridspec_kw={'height_ratios': [3, 1, 1.5]})

ax2_price.plot(df_plot.index, df_plot['CLOSE_MX'], color='black', lw=1.5, label='Цена MIX Futures')
ax2_long.plot(df_plot.index, df_plot['long_yur'], color='#2980B9', lw=1.2)
ax2_long.fill_between(df_plot.index, 0, df_plot['long_yur'], color='#2980B9', alpha=0.05)
ax2_long.set_ylabel('Long OI (контракты)', color='#2980B9')

ax2_rgbi.plot(df_plot.index, df_plot['CLOSE_RGBI'], color='darkblue', lw=1.5)
ax2_rgbi.set_ylabel('Индекс RGBI')
ax2_rgbi.grid(True, alpha=0.1)

for g in df_plot.loc[mask_discount, 'group_discount'].unique():
    block = df_plot[df_plot['group_discount'] == g]
    start_date = block.index[0]
    end_date = block.index[-1]
    
    price_min = block['CLOSE_MX'].min()
    price_max = block['CLOSE_MX'].max()
    
    ax2_price.axvspan(start_date, end_date, color='#E74C3C', alpha=0.15, zorder=0)
    ax2_long.axvspan(start_date, end_date, color='#E74C3C', alpha=0.15, zorder=0)
    ax2_rgbi.axvspan(start_date, end_date, color='#E74C3C', alpha=0.15, zorder=0)
    
    x_start = (end_date - df_plot.index[0]) / (df_plot.index[-1] - df_plot.index[0])
    
    ax2_price.axhspan(price_min, price_max, xmin=x_start, xmax=1.0, color='#E74C3C', alpha=0.25, zorder=1)
    ax2_price.axhline(price_min, xmin=x_start, xmax=1.0, color='#E74C3C', linestyle='--', alpha=0.6, lw=0.8)
    ax2_price.axhline(price_max, xmin=x_start, xmax=1.0, color='#E74C3C', linestyle='--', alpha=0.6, lw=0.8)
    ax2_price.scatter(block.index, block['CLOSE_MX'], color='#E74C3C', s=40, alpha=0.7, zorder=3, marker='v')

ax2_price.set_title('Зоны экстремальной БЭКВАРДАЦИИ (дисконт) → Проекция уровней ТОЛЬКО ВПРАВО', fontsize=14)
ax2_price.set_ylabel('Цена MIX')
ax2_price.grid(True, alpha=0.1)
ax2_price.legend(loc='upper left')
ax2_long.grid(True, alpha=0.1)
ax2_rgbi.set_xlabel('Дата')
plt.tight_layout()
plt.savefig('notebooks/images/basis_discount.png', dpi=300, bbox_inches='tight')
plt.show()


# ИТОГОВАЯ СТАТИСТИКА

print("\n" + "="*70)
print("ИТОГОВАЯ СТАТИСТИКА ПО ЗОНАМ")
print("="*70)

premium_zones = df_plot[df_plot['basis_z'] > threshold]
discount_zones = df_plot[df_plot['basis_z'] < -threshold]

print(f"\n📈 ЗОНЫ ПРЕМИИ (контанго):")
print(f"   Количество зон: {df_plot.loc[mask_premium, 'group_premium'].nunique()}")
print(f"   Всего дней: {len(premium_zones)} ({len(premium_zones)/len(df_plot)*100:.1f}%)")
print(f"   Средняя цена: {premium_zones['CLOSE_MX'].mean():.0f}")
print(f"   Средний Short OI: {premium_zones['short_yur'].mean():.0f}")

print(f"\n📉 ЗОНЫ ДИСКОНТА (бэквардация):")
print(f"   Количество зон: {df_plot.loc[mask_discount, 'group_discount'].nunique()}")
print(f"   Всего дней: {len(discount_zones)} ({len(discount_zones)/len(df_plot)*100:.1f}%)")
print(f"   Средняя цена: {discount_zones['CLOSE_MX'].mean():.0f}")
print(f"   Средний Long OI: {discount_zones['long_yur'].mean():.0f}")
print("="*70)
