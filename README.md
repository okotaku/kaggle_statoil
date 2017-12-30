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

# TODO
- [ ]最適なパラメーター探索
- [ ]Xceptionのfine tuning
- [ ]SmallCNNの改良。
- [ ]VGG16の0からの学習
