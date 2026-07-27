import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
# Загрузка данных
df_combined = pd.read_csv('data/df_combined.csv', parse_dates=['DATE_MIX'])
df_combined.set_index('DATE_MIX', inplace=True)

window_size = 90 # Окно 3 месяца можно месяц или два
df_plot = df_combined[['CLOSE_MX', 'NET_YUR']].copy().dropna()
df_plot['rolling_corr'] = df_plot['CLOSE_MX'].rolling(window=window_size).corr(df_plot['NET_YUR'])


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True, 
                               gridspec_kw={'height_ratios': [2, 1]})

# Цена и Позиция ---
lns1 = ax1.plot(df_plot.index, df_plot['CLOSE_MX'], color='black', label='Цена MIX (лев. шкала)', alpha=0.7, lw=1.5)
ax1_tw = ax1.twinx()
lns2 = ax1_tw.plot(df_plot.index, df_plot['NET_YUR'], color='red', label='NET Юрики (прав. шкала)', alpha=0.8, lw=1.2)

# легенды
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left', frameon=True)

ax1.set_title(f'Стратегическая корреляция: Цена MIX vs Позиции Юриков (Окно: {window_size} дней)', fontsize=15, pad=20)
ax1.grid(True, alpha=0.2)

# продлеваем зоны на верхний график
high_corr_periods = df_plot['rolling_corr'] >= 0.5
low_corr_periods = df_plot['rolling_corr'] <= -0.5

def get_continuous_periods(mask):
    periods = []
    in_period = False
    start_idx = None
    
    for i, (idx, is_true) in enumerate(mask.items()):
        if is_true and not in_period:
            in_period = True
            start_idx = idx
        elif not is_true and in_period:
            in_period = False
            periods.append((start_idx, idx))
    
    if in_period:
        periods.append((start_idx, mask.index[-1]))
    
    return periods

high_periods = get_continuous_periods(high_corr_periods)
low_periods = get_continuous_periods(low_corr_periods)

for start, end in high_periods:
    ax1.axvspan(start, end, alpha=0.25, color='green', zorder=0)
    ax1_tw.axvspan(start, end, alpha=0.25, color='green', zorder=0)

for start, end in low_periods:
    ax1.axvspan(start, end, alpha=0.25, color='red', zorder=0)
    ax1_tw.axvspan(start, end, alpha=0.25, color='red', zorder=0)

# Скользящая корреляция
ax2.plot(df_plot.index, df_plot['rolling_corr'], color='blue', lw=2.5, label=f'Rolling Correlation ({window_size}d)')

for level, color, label in [(0.5, 'green', 'Высокая прямая'), (0, 'black', None), (-0.5, 'red', 'Высокая обратная')]:
    ls = '--' if level == 0 else '-.'
    alpha_line = 0.8 if level == 0 else 0.6
    linewidth = 1 if level == 0 else 1.5
    
    ax2.axhline(level, color=color, linestyle=ls, alpha=alpha_line, linewidth=linewidth, 
                label=label if label else None)
    
    if level != 0:
        if level > 0:
            condition = df_plot['rolling_corr'] >= level
            fill_color = 'lightgreen'
            edge_color = 'green'
        else:
            condition = df_plot['rolling_corr'] <= level
            fill_color = 'lightcoral'
            edge_color = 'red'
        
        ax2.fill_between(df_plot.index, df_plot['rolling_corr'], level, 
                         where=condition, color=fill_color, alpha=0.4, 
                         interpolate=True, linewidth=0)
        
        for start, end in (high_periods if level > 0 else low_periods):
            period_mask = (df_plot.index >= start) & (df_plot.index <= end) & condition
            if period_mask.any():
                subset = df_plot[period_mask]
                if len(subset) > 0:
                    ax2.fill_between(subset.index, subset['rolling_corr'], level,
                                     color=fill_color, alpha=0.6, edgecolor=edge_color,
                                     linewidth=1.5, linestyle='-', hatch='///')

handles, labels = ax2.get_legend_handles_labels()
filtered = [(h, l) for h, l in zip(handles, labels) if l is not None]
if filtered:
    handles, labels = zip(*filtered)
    ax2.legend(handles, labels, loc='lower left', fontsize=10, framealpha=0.9)
else:
    ax2.legend(loc='lower left')

ax2.set_ylabel('Коэффициент корреляции', fontsize=12)
ax2.set_ylim(-1.1, 1.1)
ax2.set_xlabel('Дата', fontsize=12)
ax2.grid(True, alpha=0.2)

for ax in [ax1, ax1_tw, ax2]:
    ax.axvline(pd.to_datetime('2024-05-17'), color='purple', linestyle='-', alpha=0.8, lw=2.5)
    
    if ax == ax1:
        y_pos = ax.get_ylim()[1] * 0.95
        ax.text(pd.to_datetime('2024-05-17'), y_pos, '  Смена режима', 
                color='purple', fontweight='bold', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.tight_layout()

# Сохраняем график
plt.savefig('notebooks/images/rolling_corr.png', dpi=300, bbox_inches='tight')
print("✅ График сохранён в notebooks/images/rolling_corr.png")

plt.show()

current_corr = df_plot['rolling_corr'].iloc[-1]
latest_date = df_plot.index[-1].date()
print(f"\nТекущая корреляция (на дату {latest_date}): {current_corr:.3f}")

print(f"\nСтатистика по периодам сильной корреляции:")
print(f"Периоды с высокой прямой корреляцией (≥0.5): {len(high_periods)}")
print(f"Периоды с высокой обратной корреляцией (≤-0.5): {len(low_periods)}")
