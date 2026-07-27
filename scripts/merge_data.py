import pandas as pd
import numpy as np
import os

# --- 1. Настройка путей и загрузка ---
path = "data"
file_name = 'market_data.xlsx'
full_path = os.path.join(path, file_name)


df_raw = pd.read_excel(full_path)

# --- 2. Группировка столбцов по инструментам ---

instruments = {
    'USD': ['DATE_USD', 'OPEN_USD', 'HIGH_USD', 'LOW_USD', 'CLOSE_USD'],
    'CNY': ['DATE_CNY', 'OPEN_CNY', 'HIGH_CNY', 'LOW_CNY', 'CLOSE_CNY'],
    'Brent': ['DATE_Brent', 'OPEN_Brent', 'HIGH_Brent', 'LOW_Brent', 'CLOSE_Brent'],
    'RVI': ['DATE_RVI', 'OPEN_RVI', 'HIGH_RVI', 'LOW_RVI', 'CLOSE_RVI'],
    'BOND': ['DATE_BOND', 'BOND_Y0_25', 'BOND_Y0_5', 'BOND_Y0_75', 'BOND_Y1', 'BOND_Y2', 'BOND_Y5', 'BOND_Y10', 'BOND_Y15', 'BOND_Y20'],
    'IMOEX': ['DATE_IMOEX', 'OPEN_IMOEX', 'HIGH_IMOEX', 'LOW_IMOEX', 'CLOSE_IMOEX'],
    'RUONIA': ['DATE_RUONIA', 'Индекс_RUONIA', 'RUONIA_1m', 'RUONIA_3m', 'RUONIA_6m'],
    'RTSI': ['DATE_RTSI', 'OPEN_RTSI', 'HIGH_RTSI', 'LOW_RTSI', 'CLOSE_RTSI'],
    'RI': ['DATE_RI', 'OPEN_RI', 'HIGH_RI', 'LOW_RI', 'CLOSE_RI'],
    'RUSFAR_ON': ['DATE_RUSFAR', 'CLOSE_RUSFAR', 'OPEN_RUSFAR', 'HIGH_RUSFAR', 'LOW_RUSFAR', 'VALUE_RUSFAR'],
    'RUSFAR_1W': ['CLOSE_RUSFAR1W', 'OPEN_RUSFAR1W', 'HIGH_RUSFAR1W', 'LOW_RUSFAR1W', 'VALUE_1W'], # Обычно даты совпадают с ON
    'RUSFAR_1M': ['CLOSE_RUSFAR1M', 'OPEN_RUSFAR1M', 'HIGH_RUSFAR1M', 'LOW_RUSFAR1M', 'VALUE_RUSFAR1M'],
    'RUSFAR_3M': ['CLOSE_RUSFAR3M', 'OPEN_RUSFAR3M', 'HIGH_RUSFAR3M', 'LOW_RUSFAR3M', 'VALUE_RUSFAR3M'],
    'RGBI':['DATE_RGBI', 'OPEN_RGBI', 'HIGH_RGBI', 'LOW_RGBI','CLOSE_RGBI',	'VOL_RGBI'],
    'S&P500': ['DATE_S&P', 'CLOSE_S&P']

}

# --- 3. Создание базового датафрейма (Backbone) ---
# Начинаем с фьючерса MIX
df_final = df_raw[['DATE_MIX', 'OPEN_MIX', 'HIGH_MIX', 'LOW_MIX', 'CLOSE_MX', 'VOL_MIX']].copy()
df_final['DATE_MIX'] = pd.to_datetime(df_final['DATE_MIX'])
df_final = df_final.dropna(subset=['DATE_MIX'])
df_final = df_final.sort_values('DATE_MIX').drop_duplicates('DATE_MIX')

# --- 4. Последовательная "сшивка" ---
for inst_name, cols in instruments.items():
    date_col = [c for c in cols if 'DATE_' in c]
    
    if date_col:
        temp_df = df_raw[cols].copy()
        temp_df[date_col[0]] = pd.to_datetime(temp_df[date_col[0]])
        temp_df = temp_df.dropna(subset=[date_col[0]]).drop_duplicates(date_col[0])
        
        df_final = pd.merge(df_final, temp_df, left_on='DATE_MIX', right_on=date_col[0], how='left')
        df_final = df_final.drop(columns=date_col[0])
    else:
        df_final = pd.concat([df_final, df_raw[cols]], axis=1)

# --- 5. Финальная очистка ---
# Удаляем строки, где нет данных по самому фьючерсу
df_final = df_final.dropna(subset=['CLOSE_MX'])

# Заполняем пропуски в факторах
df_final = df_final.ffill()

# Индекс по дате для удобства
df_final = df_final.set_index('DATE_MIX')


print(df_final.head())
