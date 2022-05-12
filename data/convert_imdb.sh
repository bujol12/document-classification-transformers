echo "Running dev"
python processed/imdb/convert_to_json.py -csv_file original/imdb/dev.txt -block_csv_file original/imdb/dev.txt.block.txt -pos_out processed/imdb/json/pos_dev.json -neg_out processed/imdb/json/neg_dev.json
echo "\nRunning train"
python processed/imdb/convert_to_json.py -csv_file original/imdb/train.txt -block_csv_file original/imdb/train.txt.block.txt -pos_out processed/imdb/json/pos_train.json -neg_out processed/imdb/json/neg_train.json
echo "\nRunning test"
python processed/imdb/convert_to_json.py -csv_file original/imdb/test.txt -block_csv_file original/imdb/test.txt.block.txt -pos_out processed/imdb/json/pos_test.json -neg_out processed/imdb/json/neg_test.json