import pandas as pd
import os

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================

# Загружаем df_final (цены всех инструментов)
df_final = pd.read_csv('data/df_final.csv', parse_dates=['DATE_MIX'])
df_final.set_index('DATE_MIX', inplace=True)

# Загружаем OI данные (из data/)
df_oi = pd.read_csv('data/MIX_OI_daily_CORRECT.csv', parse_dates=['date'])
df_oi.set_index('date', inplace=True)

# ============================================
# 2. ОБЪЕДИНЕНИЕ
# ============================================
# Соединяем по дате
df_oi.index.name = 'DATE_MIX'   # выравниваем имя индекса перед объединением
df_combined = pd.merge(df_final, df_oi, left_index=True, right_index=True, how='left')

# Заполняем пропуски в OI (если есть)
df_combined = df_combined.ffill()

# ============================================
# 3. СОХРАНЕНИЕ
# ============================================
df_combined.to_csv('data/df_combined.csv')
print(f"✅ Сохранено {len(df_combined)} строк в data/df_combined.csv")
print("\nКолонки в df_combined:")
print(df_combined.columns.tolist())
print("\nПервые 5 строк:")
print(df_combined.head())
