# Model pour l'API

# 1. Library imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
import joblib
import pickle


# 2. Classes which describes the client and the features
class Client(BaseModel):
    num_client: float


class Features(BaseModel):
    PAYMENT_RATE: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    DAYS_BIRTH: float
    AMT_ANNUITY: float
    BURO_DAYS_CREDIT_MAX: float
    BURO_DAYS_CREDIT_ENDDATE_MAX: float
    DAYS_EMPLOYED: float
    AMT_GOODS_PRICE: float
    DAYS_REGISTRATION: float
    ANNUITY_INCOME_PERC: float
    PREV_CNT_PAYMENT_MEAN: float
    INSTAL_DAYS_ENTRY_PAYMENT_MAX: float
    DAYS_ID_PUBLISH: float
    APPROVED_CNT_PAYMENT_MEAN: float
    BURO_AMT_CREDIT_SUM_SUM: float
    INSTAL_DPD_MEAN: float
    INSTAL_AMT_PAYMENT_MIN: float
    REGION_POPULATION_RELATIVE: float
    BURO_AMT_CREDIT_SUM_DEBT_MEAN: float
