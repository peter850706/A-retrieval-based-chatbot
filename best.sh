#!/bin/bash
TEST_JSON_PATH=$1
PREDICT_CSV_PATH=$2

# preprocess test.json
python3.7 ./src/make_testdataset.py ./data $TEST_JSON_PATH

# predict
python3.7 ./src/predict.py ./models/test_best/0 ./models/test_best/1 ./models/test_best/2 $PREDICT_CSV_PATH