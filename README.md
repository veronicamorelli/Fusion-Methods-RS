# Fusion-Methods-RS

* Experiment1 (**RGB1**): batch size 8. (feature channels: [96, 192, 384, 768], embed dim: 96)
* Experiment2 (**RGB2**): batch size 32. 8-->32 (feature channels: [96, 192, 384, 768], embed dim: 96)
* Experiment3 (**RGB3**): batch size 8. (feature channels: [192, 384, 768, 1536], embed dim: 192)
* Experiment4 (**RGB4**): batch size 10. 8 -->10 (embed dim: 96, attention_heads: [2, 2, 6, 2] --> [2, 2, 18, 2])
* Experiment5 (**RGB5**): batch size 4. 8 -->4 (embed dim: 192, attention_heads: [2, 2, 18, 2])
* Experiment6 (**RGB6**): batch size 4. 8 -->4 (embed dim: 192, attention_heads: [2, 2, 18, 2], ape=True)
* Experiment7 (**RGB7**): batch size 4. 8 -->4 (embed dim: 192, attention_heads: [2, 2, 18, 2], ape=True, patience=10-->5)
* Experiment8 (**RGB8**): batch size 4. 8 -->4 (embed dim: 192, attention_heads: [2, 2, 6, 2], ape=True, patience=10)
* Experiment9 (**RGB9**): batch size 4. 8 -->4 (embed dim: 192, attention_heads: [2, 2, 18, 2], ape=True, lr=0.0001-->0.00001)
* Experiment10 (**RGB10**): batch size 4. 8 -->4 (embed dim: 192, attention_heads: [2, 2, 18, 2], ape=True, attn_drop_rate=0.0 -->0.3, lr:0.00001)
* Experiment11 (**RGB11**): batch size 4. 8 -->4 (embed dim: 192, attention_heads: [2, 2, 18, 2], ape=True, wd=0.00001 -->  0.0001, lr:0.00001)
* Experiment12 (**RGB12**): batch size 4. 8 -->4 (embed dim: 192, attention_heads: [2, 2, 18, 2], ape=True, wd=0.00001 -->  0.0001, lr:0.0001)
* Experiment12 (**RGB12**): batch size 4. 8 -->4 (embed dim: 192, attention_heads: [2, 2, 18, 2], ape=True, wd=0.00001 -->  0.0001, lr:0.0001)

| Version | Accuray | Precision | mIoU  |
| ------- | ------- | --------- | ----- |
| RGB1    | 59.82   | 63.95     | 49.75 | 
| RGB2    | 61.14   | 65.48     | 51.04 |
| RGB3    | 62.17   | 66.03     | 51.84 |
| RGB4    | 63.09   | 65.36     | 52.11 |
| RGB5    | 62.94   | 66.35     | 52.49 |
| RGB6    | 63.70   | 66.96     | 53.25 | 
| RGB7    | 61.71   | 64.39     | 50.70 | 
| RGB8    | 63.36   | 67.17     | 53.01 | 
| RGB9    | 61.70   | 65.47     | 51.41 | 
| RGB10   | 61.26   | 65.47     | 50.24 | 
| RGB11   | 61.46   | 65.51     | 51.13 | 
| RGB12   | 63.02   | 65.67     | 52.12 | 


| Version | Accuracy | Precision | mIoU  |
| ------- | -------- | --------- | ----- |
| RGB1    | 68.45    | 73.37     | 57.71 |
| RGB2    | 69.71    | 74.91     | 58.34 |
| RGB3    | 70.74    | 75.43     | 59.55 |
| RGB4    | 71.57    | 74.86     | 59.83 |
| RGB5    | 71.43    | 76.00     | 60.57 |
| RGB6    | 72.97    | 77.57     | 61.93 |
| RGB7    | 69.42    | 71.23     | 56.00 |
| RGB8    | 72.00    | 76.57     | 61.14 |
| RGB9    | 69.74    | 74.05     | 58.41 |
| RGB10   | 68.91    | 74.05     | 56.34 |
| RGB11   | 70.26    | 74.61     | 58.50 |
| RGB12   | 71.91    | 75.45     | 59.67 |

* Experiment 1  (**LC1**): batch size: 8, feature channels: [144, 288, 576, 1152], embed dim: 48*3, lr: 0.0001, wd: 0.00001, depth: [2, 2, 6, 2], ape:False, drop_rath:0.0, attn_drop_path_rate:0.3

* Experiment 1  (**LC2**): batch size: 8, feature channels: [144, 288, 576, 1152], embed dim: 48*3, lr: 0.0001, wd: 0.00001, depth: [2, 2, 8, 2], ape:False, drop_rath:0.0, attn_drop_path_rate:0.3

| Version | Accuracy | Precision | mIoU  |
| ------- | -------- | --------- | ----- |
| LT1     | 61.28    | 63.78     | 50.18 |