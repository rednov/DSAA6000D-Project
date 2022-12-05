# Knowledge-Graph-Completion-based-on-Neural-Network

## Dependencies
* numpy
* torch
* torch-scatter
* torch-sparse
* torch-cluster
* torch-geometric
* tensorflow
* ordered-set


## Dataset
Unzip data.zip
* FB15k
* WN18

## How to run
Install dependencies
```
      pip3 install requirements.txt
```
Run the program
```
      python ./main.py --dim 50 --batch 50 --data ./data/FB15k/ --eval_per 1 --worker 8 --eval_batch 50 --max_iter 1 --generator 5
```

## Results
Sample result on
```
      python ./main.py --dim 50 --batch 50 --data ./data/FB15k/ --eval_per 1 --worker 8 --eval_batch 50 --max_iter 1 --generator 1
```
```
all worker stopped.
[VALID] ITER 0 [HEAD PREDICTION] MEAN RANK: 1111.5 FILTERED MEAN RANK 935.2 HIT@10 0.134 FILTERED HIT@10 0.277
[VALID] ITER 0 [TAIL PREDICTION] MEAN RANK: 1164.9 FILTERED MEAN RANK 1061.4 HIT@10 0.182 FILTERED HIT@10 0.289
waiting for worker finishes their work
all worker stopped.
[TEST] ITER 0 [HEAD PREDICTION] MEAN RANK: 1104.8 FILTERED MEAN RANK 930.8 HIT@10 0.140 FILTERED HIT@10 0.279
[TEST] ITER 0 [TAIL PREDICTION] MEAN RANK: 1164.0 FILTERED MEAN RANK 1055.0 HIT@10 0.181 FILTERED HIT@10 0.290
```