"""
Скачивание открытого интереса (OI) по фьючерсу MIX (Индекс МосБиржи)
с 01.09.2022 по 18.04.2026.

При запросе чанка 20 дней API отдаёт только последние 3 дня из периода такое уж  ограничение бесплатного доступа к ISS MOEX.

РЕШЕНИЕ:
Запрашиваем по ОДНОМУ дню за раз. Берём запись с max seqnum
(последний снимок дня = итоговый OI на закрытие). Итоговый OI по факту самый главный накопленный на конец дня позции Юридически и физ.лиц
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# ===================== НАСТРОЙКИ =====================
TICKER = "MX"
START_DATE = "2022-09-01"
END_DATE   = "2026-04-18"
OUTPUT_FILE = "MIX_OI_daily_CORRECT.csv"
SLEEP_SEC   = 0.5   


def get_one_day_oi(security: str, date_str: str) -> pd.DataFrame:
    """
    Скачивает все snapshot-записи futoi за конкретную дату.
    """
    url = (
        f"https://iss.moex.com/iss/analyticalproducts/futoi/"
        f"securities/{security}.json"
    )
    params = {
        "from": date_str,
        "till": date_str,
        "iss.only": "futoi",
        "limit": 500,    
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        rows = data.get("futoi", {}).get("data", [])
        cols = data.get("futoi", {}).get("columns", [])
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        print(f"  [!] Ошибка для {date_str}: {e}")
        return pd.DataFrame()


def is_trading_day_candidate(d: datetime) -> bool:
    """Пропускаем выходные — суббота(5) и воскресенье(6)."""
    return d.weekday() < 5


def extract_eod_snapshot(df: pd.DataFrame, date_str: str) -> pd.Series | None:
    """
    Из всех внутридневных снимков берём последний (max seqnum) для каждой
    группы клиентов (FIZ / YUR), затем агрегируем в одну строку.

    Возвращает Series с колонками:
        date, OI_total, OI_FIZ, OI_YUR,
        NET_FIZ (pos_long_FIZ - pos_short_FIZ),
        NET_YUR (pos_long_YUR - pos_short_YUR),
        LONG_FIZ, SHORT_FIZ, LONG_YUR, SHORT_YUR
    """
    if df.empty:
        return None

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Фильтруем только валидные строки (без ERROR_MESSAGE)
    if "error_message" in df.columns:
        df = df[df["error_message"].isna() | (df["error_message"] == "")]

    if df.empty:
        return None

    df["seqnum"] = pd.to_numeric(df["seqnum"], errors="coerce")

    # Берём последний снимок по seqnum для каждой clgroup
    eod = (
        df.sort_values("seqnum")
          .groupby("clgroup", as_index=False)
          .last()
    )

    # Приводим числовые колонки
    for col in ["pos", "pos_long", "pos_short"]:
        eod[col] = pd.to_numeric(eod[col], errors="coerce")

    result = {"date": date_str}

    for grp in ["FIZ", "YUR"]:
        row = eod[eod["clgroup"] == grp]
        if row.empty:
            result[f"LONG_{grp}"]  = None
            result[f"SHORT_{grp}"] = None
            result[f"OI_{grp}"]    = None
            result[f"NET_{grp}"]   = None
        else:
            r = row.iloc[0]
            long_v  = abs(r["pos_long"])
            short_v = abs(r["pos_short"])
            result[f"LONG_{grp}"]  = long_v
            result[f"SHORT_{grp}"] = short_v
            result[f"OI_{grp}"]    = long_v + short_v   # открытый интерес группы
            result[f"NET_{grp}"]   = long_v - short_v   # чистая позиция 

    # Суммарный OI = (OI_FIZ + OI_YUR) / 2  (каждый контракт считается дважды) 
    oi_fiz = result.get("OI_FIZ") or 0
    oi_yur = result.get("OI_YUR") or 0
    result["OI_TOTAL"] = (oi_fiz + oi_yur) / 2

    return pd.Series(result)


# ===================== ОСНОВНОЙ ЦИКЛ =====================
print(f"Скачиваем OI для {TICKER} | {START_DATE} — {END_DATE}")
print(f"Метод: по одному дню за запрос\n")

current  = datetime.strptime(START_DATE, "%Y-%m-%d")
end_dt   = datetime.strptime(END_DATE,   "%Y-%m-%d")
records  = []
skipped  = 0
no_data  = 0

while current <= end_dt:
    date_str = current.strftime("%Y-%m-%d")

    if not is_trading_day_candidate(current):
        current += timedelta(days=1)
        continue

    df_day = get_one_day_oi(TICKER, date_str)

    if df_day.empty:
        no_data += 1
        print(f"  {date_str} — нет данных (праздник или нет торгов)")
        current += timedelta(days=1)
        time.sleep(SLEEP_SEC)
        continue

    row = extract_eod_snapshot(df_day, date_str)
    if row is not None:
        records.append(row)
        print(
            f"  {date_str}  OI_TOTAL={row.get('OI_TOTAL', 'n/a'):>8.0f}"
            f"  NET_FIZ={row.get('NET_FIZ', 'n/a'):>8.0f}"
            f"  NET_YUR={row.get('NET_YUR', 'n/a'):>8.0f}"
        )
    else:
        skipped += 1
        print(f"  {date_str} — данные есть, но не удалось разобрать")

    current += timedelta(days=1)
    time.sleep(SLEEP_SEC)

# ===================== СБОРКА И СОХРАНЕНИЕ =====================
if records:
    result_df = pd.DataFrame(records)
    result_df["date"] = pd.to_datetime(result_df["date"])
    result_df = result_df.sort_values("date").reset_index(drop=True)

    # Добавляем производные показатели
    result_df["DELTA_OI"]      = result_df["OI_TOTAL"].diff()           # изменение OI за день
    result_df["DELTA_NET_YUR"] = result_df["NET_YUR"].diff()            # изменение чистой позиции юриков
    result_df["DELTA_NET_FIZ"] = result_df["NET_FIZ"].diff()

    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"ГОТОВО!")
    print(f"Торговых дней скачано: {len(result_df)}")
    print(f"Без данных (праздники): {no_data}")
    print(f"Не удалось разобрать:   {skipped}")
    print(f"Файл сохранён: {OUTPUT_FILE}")
    print(f"\nКолонки:")
    for col in result_df.columns:
        print(f"  {col}")
    print(f"\nПервые строки:")
    print(result_df.head(5).to_string())
else:
    print("\nНичего не скачалось.")
