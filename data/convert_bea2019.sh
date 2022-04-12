
# Run from original json to internal json format with token-level labels
python processed/convert_from_orig_gec_to_json.py -gold original/bea2019/json/A.dev.json -out processed/bea2019/json/A.dev.json
python processed/convert_from_orig_gec_to_json.py -gold original/bea2019/json/B.dev.json -out processed/bea2019/json/B.dev.json
python processed/convert_from_orig_gec_to_json.py -gold original/bea2019/json/C.dev.json -out processed/bea2019/json/C.dev.json
python processed/convert_from_orig_gec_to_json.py -gold original/bea2019/json/N.dev.json -out processed/bea2019/json/N.dev.json

python processed/convert_from_orig_gec_to_json.py -gold original/bea2019/json/A.train.json -out processed/bea2019/json/A.train.json
python processed/convert_from_orig_gec_to_json.py -gold original/bea2019/json/B.train.json -out processed/bea2019/json/B.train.json
python processed/convert_from_orig_gec_to_json.py -gold original/bea2019/json/C.train.json -out processed/bea2019/json/C.train.json

# Merge datasets of different proficiency & filter only for A/C proficiency as labels
python processed/bea2019/merge_datasets.py -i processed/bea2019/json/A.dev.json -i processed/bea2019/json/B.dev.json -i processed/bea2019/json/C.dev.json -i processed/bea2019/json/N.dev.json -out processed/bea2019/json/test.json
python processed/bea2019/merge_datasets.py -i processed/bea2019/json/A.train.json -i processed/bea2019/json/B.train.json -i processed/bea2019/json/C.train.json -out processed/bea2019/json/train_orig.json

# Sample 20% of the train dataset to get dev dataset + save the indices
python processed/bea2019/split_data.py -i processed/bea2019/json/train_orig.json  -out_train processed/bea2019/json/train.json  -out_dev processed/bea2019/json/dev.json -dev_idx_file_out processed/bea2019/json/dev_idx.json