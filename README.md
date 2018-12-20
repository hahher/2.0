# 金融贷款逾期的模型构建2——集成模型
##模型构建
构建随机森林、GBDT、XGBoost和LightGBM这4个模型，并对每一个模型进行评分，评分方式任意。

### 相关库安装
sklearn已经包含随机森林、GBDT

### LightGBM 安装 
```
pip install lightgbm
```
### XGBoost安装
[通过Whl安装xgboost(不需要本地编译)](https://www.lfd.uci.edu/~gohlke/pythonlibs/#xgboost)
下载对应的wheel文件后，在当前目录打开cmd输入以下命令
```
pip install xxx.whl
```

### 导入数据
```
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
```

### 读取数据
```
"""读取数据"""
data = pd.read_csv("data_all.csv")
x = data.drop(labels='status', axis=1)
y = data['status']
x_train, x_test, y_train, y_test = train_test_split(x,  y, test_size=0.3, random_state=2018)
```

### 随机森林模型
```
"""随机森林模型"""
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_score = rfc.score(x_test, y_test)
print("The score of RF：", rfc_score)

rfc1 = RandomForestClassifier()
rfc1.fit(x_train_stand, y_train)
rfc1_score = rfc1.score(x_test_stand, y_test)
print("The score of RF(with preprocessing)：", rfc1_score)
```
### GBDT模型 梯度下降树
```
gbdt = GradientBoostingRegressor()
gbdt.fit(x_train, y_train)
gbdt_score = gbdt.score(x_test, y_test)
print("The score of GBDT：",gbdt_score)
```
### XGBoost模型
```
xgb = xgb.XGBClassifier()
xgb.fit(x_train, y_train)
xgb_score = xgb.score(x_test, y_test)
print("The score of XGBoost：", xgb_score)
```
### lightGBM
```
gbm = lgb.LGBMRegressor()
gbm.fit(x_train, y_train)
gbm_score = gbm.score(x_test, y_test)
print("The score of LightGBM：", gbdt_score)
```
### 评分 准确率
The score of RF： 0.767344078486335

The score of RF(with preprocessing)： 0.76734407848633

The score of GBDT： 0.1816373798518226

The score of XGBoost： 0.7855641205325858

The score of LightGBM： 0.1816373798518226


