# Run Transformer training and the top-k attention eval

#python main.py --config config/document_classification/roberta-fce.json --train data/processed/fce/json/train.json  --eval data/processed/fce/json/dev.json --test data/processed/fce/json/test.json
#python main.py --config config/document_classification/roberta-bea2019.json --train data/processed/bea2019/json/train.json  --eval data/processed/bea2019/json/dev.json --test data/processed/bea2019/json/test.json
#python main.py --config config/document_classification/roberta-pos_imdb.json --train data/processed/imdb/json/pos_train.json  --eval data/processed/imdb/json/pos_dev.json --test data/processed/imdb/json/pos_test.json
#python main.py --config config/document_classification/roberta-neg_imdb.json --train data/processed/imdb/json/neg_train.json  --eval data/processed/imdb/json/neg_dev.json --test data/processed/imdb/json/neg_test.json

python main.py --config config/document_classification/longformer-fce.json --train data/processed/fce/json/train.json  --eval data/processed/fce/json/dev.json --test data/processed/fce/json/test.json
python main.py --config config/document_classification/longformer-bea2019.json --train data/processed/bea2019/json/train.json  --eval data/processed/bea2019/json/dev.json --test data/processed/bea2019/json/test.json
python main.py --config config/document_classification/longformer-pos_imdb.json --train data/processed/imdb/json/pos_train.json  --eval data/processed/imdb/json/pos_dev.json --test data/processed/imdb/json/pos_test.json
python main.py --config config/document_classification/longformer-neg_imdb.json --train data/processed/imdb/json/neg_train.json  --eval data/processed/imdb/json/neg_dev.json --test data/processed/imdb/json/neg_test.json