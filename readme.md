# A simple BiLSTM-CRF model for Chinese Named Entity Recognition

This repository includes the code for buliding a very simple **character-based BiLSTM-CRF sequence labeling model** for Chinese Named Entity Recognition task. Its goal is to recognize three types of Named Entity: PERSON, LOCATION and ORGANIZATION.

This code works on **Python 3.7 & Pytorch 1.1.0** and the following repositories or course

https://github.com/Determined22/zh-NER-TF

https://github.com/mtreviso/linear-chain-crf

http://web.stanford.edu/class/cs224n/

 gives me much help.



All of code are annotated well.

# How to run

```
python main.py
```

- After 230 epoch, the accuracy is 0.97+

- You can test your sentence,  e.g.

  ![](https://raw.githubusercontent.com/WallE-Chang/Picture/master/bilstm_crf.png)