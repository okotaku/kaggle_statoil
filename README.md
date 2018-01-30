# Kaggle Statoil/C-CORE Iceberg Classifier Challenge

# Usage
inc_angleなし
```
python3 main.py
```
modelのコメントアウトの部分で使用するモデルを変えることができる。

inc_angleあり
```
python3 main_angle.py
```
modelのコメントアウトの部分で使用するモデルを変えることができる。

# Single Report
- VGG16 Fine tuning

5foldsで実験を行なった。

|freeze layer|folds number|val acc|val loss|
|:--|:--:|:--:|:--:|
|15|1|0.8542|0.4044|
|15|2|0.8847| 0.2671|
|15|3|0.9390|0.1866|
|15|4|0.8874|0.2736|
|15|5|0.8771|0.3208|

LB=0.2094

- VGG16

5foldsで実験を行なった。
Image Net 学習済みモデルを初期値として、freezeせずに学習(freeze_leyer=0)。

|clossval number|val acc|val loss|
|:--|:--:|:--:|
|1|0.8949|0.3053|
|2|0.9085| 0.2421|
|3|0.9017|0.2322|
|4|0.9078|0.2276|
|5|0.9113|0.2874|

LB=0.1843

- small cnn

10foldsで実験を行なった。

|clossval number|val acc|val loss|
|:--|:--:|:--:|
|1|0.9189 0.1681|
|2|0.8716|0.2823|
|3|0.9527|0.1158|
|4|0.9252|0.2066|
|5|0.9184|0.1633|
|6|0.9592|0.1001|
|7|0.9252|0.1764|
|8|0.9252|0.1766|
|9|0.8562|0.3545|
|10|0.8835|0.2502|

- vgg like

10foldsで実験を行なった。

|clossval number|val acc|val loss|
|:--|:--:|:--:|
|1|0.9256|0.1632|
|2|0.8649|0.2924|
|3|0.9392|0.1532|
|4|0.9184|0.2159|
|5|0.9116|0.1930|
|6|0.9456|0.1249|
|7|0.9388|0.1530|
|8|0.9320|0.1655|
|9|0.8562|0.3960|
|10|0.9041|0.2314|


- resnet like

5foldsで実験を行なった。

|clossval number|val acc|val loss|
|:--|:--:|:--:|
|1|0.9119|0.2168|
|2|0.9389| 0.1841|
|3|0.9288|0.1896|
|4|0.9215|0.2139|
|5|0.8908|0.2631|

LB=0.1804

- densenet like

5foldsで実験を行なった。

|clossval number|val acc|val loss|
|:--|:--:|:--:|
|1|0.9118|0.2148|
|2|0.9390|0.1897|
|3|0.9390|0.1388|
|4|0.9352|0.1794|
|5|0.8908|0.2620|

LB=0.1747

- small anglenet

5foldsで実験を行なった。

|clossval number|val acc|val loss|
|:--|:--:|:--:|
|1|0.8881|0.2301|
|2|0.9186|0.2029|
|3|0.9119|0.1711|
|4|0.9181|0.2094|
|5|0.9044|0.2713|

- jirkamodel

10foldsで実験を行なった。

|clossval number|val acc|val loss|
|:--|:--:|:--:|
|1|0.9257|0.1789|
|2|0.8919|0.3032|
|3|0.9324|0.1518|
|4|0.9116|0.1997|
|5|0.9184|0.1856|
|6|0.9320|0.1204|
|7|0.9388|0.2018|
|8|0.9456|0.1514|
|9|0.8699|0.3710|
|10|0.8973|0.2351|

# Ensemble Report
- - 7 models and https://www.kaggle.com/submarineering/submarineering-even-better-public-score-until-now ensemble
  - 0.1381(private 0.1470)

- 7 models and https://www.kaggle.com/submarineering/submarineering-even-better-public-score-until-now base ensemble and min max
  - 0.1308(private 0.1562)
