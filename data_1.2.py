import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


"""读取数据"""
data = pd.read_csv("data_all.csv")
x = data.drop(labels='status', axis=1)
y = data['status']
x_train, x_test, y_train, y_test = train_test_split(x,  y, test_size=0.3, random_state=2018)


"""数据标准化"""
scaler = StandardScaler()
scaler.fit(x_train)
x_train_stand = scaler.transform(x_train)
x_test_stand = scaler.transform(x_test)

"""随机森林模型"""
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_score = rfc.score(x_test, y_test)
print("The score of RF：", rfc_score)

rfc1 = RandomForestClassifier()
rfc1.fit(x_train_stand, y_train)
rfc1_score = rfc1.score(x_test_stand, y_test)
print("The score of RF(with preprocessing)：", rfc1_score)


gbdt = GradientBoostingRegressor()
gbdt.fit(x_train, y_train)
gbdt_score = gbdt.score(x_test, y_test)
print("The score of GBDT：",gbdt_score)

xgb = xgb.XGBClassifier()
xgb.fit(x_train, y_train)
xgb_score = xgb.score(x_test, y_test)
print("The score of XGBoost：", xgb_score)

gbm = lgb.LGBMRegressor()
gbm.fit(x_train, y_train)
gbm_score = gbm.score(x_test, y_test)
print("The score of LightGBM：", gbdt_score)
print("ok")