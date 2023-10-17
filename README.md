# Fusion-Methods-RS

Experiment1 (**RGB1**): batch size 8. (feature channels: [96, 192, 384, 768], embed dim: 96)
Experiment2 (**RGB2**): batch size 32. 8-->32 (feature channels: [96, 192, 384, 768], embed dim: 96)
Experiment3 (**RGB3**): batch size 8. (feature channels: [192, 384, 768, 1536], embed dim: 192)
Experiment1 (**RGB4**): batch size 10. 8 -->10 (embed dim: 96, attention_heads: [2, 2, 6, 2] --> [2, 2, 18, 2])
Experiment1 (**RGB5**): batch size 4. 8 -->6 (embed dim: 96, attention_heads: [2, 2, 6, 2] --> [2, 2, 18, 2])


| Version | Accuray | Precision | mIoU |
| ------- | ------- | --------- | ---- |
| RGB1 | 59.82 | 63.95 | 49.75 | 
| RGB2 | 61.14 | 65.48 | 51.04 |
| RGB3 | 62.17 | 66.03 | 51.84 |
| RGB4 | 63.09 | 65.36 | 52.11 |