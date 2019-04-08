#!/bin/bash

# pre-processed embedding
wget https://www.dropbox.com/s/95jsoo2qrfh801m/embedding.pkl?dl=0 -O ./data/embedding.pkl

# rnn model
wget https://www.dropbox.com/s/de53867rhb78hgb/rnn.model.best.pkl?dl=0 -O ./models/test_rnn/model.best.pkl

# attention model
wget https://www.dropbox.com/s/jx1zvbii4pnbr33/attention.model.best.pkl?dl=0 -O ./models/test_attention/model.best.pkl

# best models
wget https://www.dropbox.com/s/dy4z5j7gsbd8gom/best.0.model.best.pkl?dl=0 -O ./models/test_best/0/model.best.pkl
wget https://www.dropbox.com/s/wwyyeochkidalpo/best.1.model.best.pkl?dl=0 -O ./models/test_best/1/model.best.pkl
wget https://www.dropbox.com/s/o25l6k3fttk9ddv/best.2.model.best.pkl?dl=0 -O ./models/test_best/2/model.best.pkl