# Run Min-Max + L2 Regularisation For all datasets and both longformer and roberta

python main.py --config config/soft_attention/min_max_reg/roberta-fce.json --train data/processed/fce/json/train.json  --eval data/processed/fce/json/dev.json
python main.py --config config/soft_attention/min_max_reg/roberta-bea2019.json --train data/processed/bea2019/json/train.json  --eval data/processed/bea2019/json/dev.json
python main.py --config config/soft_attention/min_max_reg/roberta-pos_imdb.json --train data/processed/imdb/json/pos_train.json  --eval data/processed/imdb/json/pos_dev.json
python main.py --config config/soft_attention/min_max_reg/roberta-neg_imdb.json --train data/processed/imdb/json/neg_train.json  --eval data/processed/imdb/json/neg_dev.json
