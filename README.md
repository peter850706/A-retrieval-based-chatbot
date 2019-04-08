# How to test rnn, attention and best models

1. Run download.sh to download the pre-trained embedding (FastText is used here), the pre-processed embedding and the pre-trained models.
```
./data/embedding.pkl
./data/cc.en.300.bin
./models/test_rnn/model.best.pkl
./models/test_attention/model.best.pkl
./models/test_best/0/model.best.pkl
./models/test_best/1/model.best.pkl
./models/test_best/2/model.best.pkl
```

2. Run rnn.sh, attention.sh and best.sh respectively, then you will get the results.
```
bash ./rnn.sh /path/to/test.json /path/to/predict.csv
bash ./attention.sh /path/to/test.json /path/to/predict.csv
bash ./best.sh /path/to/test.json /path/to/predict.csv
```

# How to plot attention map

1. Prepare the validation dataset in `./data`:
```
./data/valid.json
```

2. Change directory to src/

3. Input the following command (device can be cpu or cuda:0, cuda:1, etc):
```
python3.7 plot_attention_map.py ../models/test_attention/ --device=0
```

# How to train

1. Prepare the dataset and pre-trained embeddings (FastText is used here) in `./data`:
```
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz -P ./data
gunzip ./data/cc.en.300.bin.gz
```
```
./data/train.json
./data/valid.json
./data/test.json
./data/cc.en.300.bin
```

2. Preprocess the data
```
cd src/
python3.7 make_dataset.py ../data/
```

3. Input the following command to train rnn(device can be cpu or cuda:0, cuda:1, etc):
```
python3.7 train.py ../models/train_rnn/exp --device=0
```

4. To train attention and best, first train the teacher models by using the following commands:
```
python3.7 train.py ../models/train_attention/teacher/exp --device=0
python3.7 train.py ../models/train_best/teacher1/exp --device=0
python3.7 train.py ../models/train_best/teacher2/exp --device=0
```

5. After getting the teacher model, using the following commands to get the final student models:
```
python3.7 train.py ../models/train_attention/student/exp --device=0
python3.7 train.py ../models/train_best/student1/exp --device=0
python3.7 train.py ../models/train_best/student2/exp --device=0
```

4. To predict, run the following commands (generally speaking, best needs multiple directories):
```
python3.7 predict.py ../models/train_rnn/exp /path/to/predict.csv
python3.7 predict.py ../models/train_attention/student/exp /path/to/predict.csv
python3.7 predict.py ../models/train_attention/student/exp ../models/train_best/student1/exp ../models/train_best/student2/exp /path/to/predict.csv
```