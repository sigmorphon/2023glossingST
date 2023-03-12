# Automatic-IGT-Glossing

Train model:

```shell
python3 token_class_model.py train --lang ddo --track open
```

Make predictions:

```shell
python3 token_class_model.py predict --lang ddo --track open --pretrained_path ./output --data_path ../../data/Tsez/ddo-dev-track1-covered
```

Eval predictions:

```shell
python3 eval.py --pred ./predictions --gold ../../data/Tsez/ddo-train-track1-uncovered
```

## Model design

## Results

Trained models: [download here](https://o365coloradoedu-my.sharepoint.com/:f:/g/personal/migi8081_colorado_edu/EhzVMGQwS_5GuV4R1BZYbVIBJbj0zHi09t85zGRuAwEkbw?e=iEIIfH)

### Dev Performance
#### Closed Track
| Lang | Morpheme Acc| Word Acc | BLEU (Morpheme) | Stems | Grams |
| --- | --- | --- | --- | --- | --- |
| arp | Ovr: 43.2<br>Avg: 51.9 | Ovr: 70.8<br>Avg: 70.1 | 41.8 | P: 50.2<br>R: 48.0<br>F1: 49.1 | P: 63.4<br>R: 33.4<br>F1: 43.7 |
| ddo | Ovr: 47.5<br>Avg: 52.9 | Ovr: 71.8<br>Avg: 72.1 | 57.8 | P: 49.7<br>R: 49.2<br>F1: 49.4 | P: 50.7<br>R: 46.1<br>F1: 48.3 |
| git | Ovr: 13.6<br>Avg: 16.3 | Ovr: 26.5<br>Avg: 29.1 | 4.5 | P: 6.7<br>R: 5.8<br>F1: 6.2 | P: 22.2<br>R: 17.6<br>F1: 19.7 |]
| lez | Ovr: 48.1<br>Avg: 49.2 | Ovr: 56.9<br>Avg: 55.7 | 52.0 | P: 54.0<br>R: 51.3<br>F1: 52.6 | P: 53.5<br>R: 40.7<br>F1: 46.2 |
| nyb | Ovr: 77.1<br>Avg: 78.2 | Ovr: 83.9<br>Avg: 82.4 | 74.2 | P: 86.2<br>R: 78.7<br>F1: 82.3 | P: 78.6<br>R: 75.3<br>F1: 76.9 |
| usp | Ovr: 63.1<br>Avg: 65.5 | Ovr: 74.0<br>Avg: 70.3 | 53.8 | P: 72.0<br>R: 62.7<br>F1: 67.0 | P: 61.3<br>R: 63.7<br>F1: 62.4 |

#### Open Track
| Lang | Morpheme Acc| Word Acc | BLEU (Morpheme) | Stems | Grams |
| --- | --- | --- | --- | --- | --- |
| arp | Ovr: 91.1<br>Avg: 91.5 | Ovr: 85.4<br>Avg: 85.5 | 79.2 | P: 91.3<br>R: 89.2<br>F1: 90.2 | P: 91.2<br>R: 94.9<br>F1: 93.0 |
| ddo | Ovr: 85.0<br>Avg: 86.0 | Ovr: 74.2<br>Avg: 75.8 | 68.6 | P: 89.3<br>R: 86.6<br>F1: 87.9 | P: 82.2<br>R: 83.6<br>F1: 82.9 |
| git | Ovr: 30.0<br>Avg: 30.2 | Ovr: 25.0<br>Avg: 25.7 | 14.2 | P: 37.8<br>R: 15.0<br>F1: 21.5 | P: 41.8<br>R: 37.8<br>F1: 39.7 |
| lez | Ovr: 50.1<br>Avg: 52.5 | Ovr: 32.6<br>Avg: 39.4 | 42.0 | P: 61.2<br>R: 48.6<br>F1: 54.2 | P: 50.1<br>R: 53.5<br>F1: 51.8 |
| nyb | Ovr: 89.2<br>Avg: 88.5 | Ovr: 84.7<br>Avg: 83.6 | 78.4 | P: 92.5<br>R: 90.5<br>F1: 91.5 | P: 85.9<br>R: 87.6<br>F1: 86.8 |
| usp | Ovr: 81.3<br>Avg: 76.2 | Ovr: 75.9<br>Avg: 72.0 | 64.9 | P: 79.4<br>R: 74.0<br>F1: 76.6 | P: 83.7<br>R: 90.4<br>F1: 87.0 |

