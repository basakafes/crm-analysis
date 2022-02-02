##########################################################
# Blood Transfusion Prediction with BG-NBD & Gamma-Gamma
##########################################################

# Çalışma Tanımı: "Knowledge discovery on RFM model using Bernoulli sequence" makalesi doğrultusunda 
# Tayvan'daki Hsin-Chu Şehrindeki Kan Transfüzyon Hizmet Merkezinin donör veri tabanına ait veriler kullanılmıştır. Veri edinme tarihi 2008'dir.
# Amaç: Geçmişte yapılan kan bağışlarını göz önünde bulundurarak verilerin edinildiği tarihten 12 ay sonrası için olası kan bağışı tahminlerini elde etmektir.
# Bu çalışmanın arka planda sağlık alanında herhangi bir bilimsel dayanağı yoktur.
# Project Description: In line with the article "Knowledge discovery on RFM model using Bernoulli sequence", data from the donor database of the 
# Blood Transfusion Service Center in Hsin-Chu City, Taiwan were used. Data acquisition date is 2008.
# Purpose: To obtain estimates of possible blood donations for 12 months from the date of data acquisition, taking into account past blood donations.
# This study has no scientific basis in the background of health.

# Kaynaklar / Sources:
# https://5cef0b7b43e7a42fd019406d30fe2e79ab6b0d1f.vetisonline.com/science/article/pii/S0957417408004508?via%3Dihub
# https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
# Yeh, I. C., Yang, K. J., & Ting, T. M. (2009). Knowledge discovery on RFM model using Bernoulli sequence. Expert Systems with Applications, 36(3), 5866-5871.
# https://www.sciencedirect.com/science/article/abs/pii/S0957417408004508
# https://www.openml.org/d/1464

# Veriler / Data

# V1 =  - months since last donation
# V2:  - total number of donation
# V3:  - total blood donated in c.c.
# V4:  - months since first donation
# and a binary variable representing whether he/she donated blood in March 2007 (1 stand for donating blood; 0 stands for not donating blood).

# Recency = V4 - V1
# Frequency = V2
# Monetary: V3
# T: V4

import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

df_ = pd.read_csv('datasets/w3/blood-transfusion.csv')
df = df_.copy()
df.head()
df.isnull().sum()
df.describe([.90,.99]).T



# Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonları:
# outlier_thresholds and replace_with_thresholds functions to suppress outliers:

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = (quartile3 + 1.5 * interquantile_range).round()
    low_limit = (quartile1 - 1.5 * interquantile_range).round()
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


replace_with_thresholds(df, "V3")
df.describe([.90,.99]).T

cltv_dataframe = pd.DataFrame()
cltv_dataframe["recency"] = (df["V4"] - df["V1"]) / 7
cltv_dataframe["frequency"] = df["V2"]
cltv_dataframe["monetary"] = df["V3"] / df["V2"]
cltv_dataframe["T"] = df["V4"] / 7

# BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV'nin Hesaplanması:
# Establishment of BG/NBD, Gamma-Gamma Models and Calculation of CLTV

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_dataframe['frequency'],
        cltv_dataframe['recency'],
        cltv_dataframe['T'])

cltv_dataframe["exp_donation_12_month"] = bgf.predict(48,
                                                       cltv_dataframe['frequency'],
                                                       cltv_dataframe['recency'],
                                                       cltv_dataframe['T']).sort_values(ascending=False)

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_dataframe['frequency'], cltv_dataframe['monetary'])
cltv_dataframe["expected_average_donation"] = ggf.conditional_expected_average_profit(cltv_dataframe['frequency'], cltv_dataframe['monetary'])

cltv_dataframe["cltv"] = ggf.customer_lifetime_value(bgf,
                                   cltv_dataframe['frequency'],
                                   cltv_dataframe['recency'],
                                   cltv_dataframe['T'],
                                   cltv_dataframe['monetary'],
                                   time=12,
                                   freq="W",
                                   discount_rate=0.01)

# CLTV Değerlerine Göre Segmentlerin Oluşturulması:
# Creating Segments According to CLTV Values:

cltv_dataframe["segment"] = pd.qcut(cltv_dataframe["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_dataframe.groupby("segment").agg({"count", "mean", "sum"})
