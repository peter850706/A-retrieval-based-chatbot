#!/bin/bash
TEST_JSON_PATH=$1
PREDICT_CSV_PATH=$2

# preprocess test.json
python3.7 ./src/make_testdataset.py ./data $TEST_JSON_PATH

# predict
python3.7 ./src/predict.py ./models/test_attention $PREDICT_CSV_PATH