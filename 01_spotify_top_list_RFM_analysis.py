# Çalışma tanımı: 01/02/2022 tarihinde https://chartmasters.org/most-streamed-artists-ever-on-spotify/ adresinden alınan
# Spotify'in en çok dinlenen sanatçılarının RFM ile segmentlere ayrılması
# Project Description: RFM segmentation of Spotify's top-streamed artists, retrieved from https://chartmasters.org/most-streamed-artists-ever-on-spotify/ on 1/02/2022


import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', None)

df_ = pd.read_excel("datasets/w3/spotify.xlsx")
df = df_.copy()

# Recency değerleri için Last Update, frequency değerleri içinse Lead Streams seçilmiştir.
# Last Update was selected for recency values, and Lead Streams was selected for frequency values.

df.head()
df.dtypes
df['LAST_UPDATE_REG'] = pd.to_datetime(df['LAST_UPDATE_REG'])
df["LAST_UPDATE_REG"].max()
analysis_date = dt.datetime(2022, 2, 2)

# Recency ve frequency metriklerinin hesaplanması:
# Calculation of recency and frequency metrics:
rf = pd.DataFrame()
rf["artist_name"] = df["ARTIST NAME"]
rf["recency"] = (analysis_date - df["LAST_UPDATE_REG"]).dt.days
rf["frequency"] = df["LEAD STREAMS"]
rf.head()

# Recency ve frequency metriklerinin RF skoruna dönüştürülmesi:
# Conversion of recency and frequency metrics to RF score:
rf["recency_score"] = pd.qcut(rf['recency'], 5, labels=[5,4,3,2,1])
rf["frequency_score"] = pd.qcut(rf['frequency'], 5, labels=[1,2,3,4,5])

rf["RF_SCORE"] = (rf['recency_score'].astype(str) + rf['frequency_score'].astype(str))


# RF skorları için segmentlerin belirlenmesi:
# Determination of segments for RF scores:
# 11: hibernating
# 12: hibernating
# 13: at_risk
# 14: at_risk
# 15: at_risk
# 21: hibernating
# 22: hibernating
# 23: at_risk
# 24: at_risk
# 25: at_risk
# 31: about_to_sleep
# 32: about_to_sleep
# 33: need_attention
# 34: those_who_keep_their_seats
# 35: those_who_keep_their_seats
# 41: promising
# 42: potential_champions
# 43: potential_champions
# 44: those_who_keep_their_seats
# 45: those_who_keep_their_seats
# 51: fresh_ones
# 52: potential_champions
# 53: potential_champions
# 54: champions
# 55: champions

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-5]': 'at_risk',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'those_who_keep_their_seats',
    r'41': 'promising',
    r'51': 'fresh_ones',
    r'[4-5][2-3]': 'potential_champions',
    r'5[4-5]': 'champions'
}

rf['segment'] = rf['RF_SCORE'].replace(seg_map, regex=True)
rf[["segment", "recency", "frequency"]].groupby("segment").agg(["mean"])
