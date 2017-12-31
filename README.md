# Kaggle Statoil/C-CORE Iceberg Classifier Challenge

# Installation
```
pip install scikit-learn tensorflow Keras numpy pandas
```

# Usage
```
python3 main.py
```
modelのコメントアウトの部分を変えることでVGG16のfine tuningとSmallCNNを選択できる。

# Report
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

folds number|val acc|val loss|
|:--|:--:|:--:|:--:|
|1|0.8949|0.3053|
|2|0.9085| 0.2421|
|3|0.9017|0.2322|
|4|0.9078|0.2276|
|5|0.9113|0.2874|

LB=0.1843

- small cnn

5foldsで実験を行なった。

|clossval number|val acc|val loss|
|:--|:--:|:--:|
|1|0.9050|0.2123|
|2|0.9186|0.1860|
|3|0.9254|0.1668|
|4|0.9317|0.2016|
|5|0.8976|0.2896|

LB=0.1722

- vgg like

5foldsで実験を行なった。

|clossval number|val acc|val loss|
|:--|:--:|:--:|
|1|0.9186|0.2212|
|2|0.9322|0.1937|
|3|0.9389|0.2017|
|4|0.5119|0.6928|
|5|0.8634|0.3172|

LB=0.1750

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
