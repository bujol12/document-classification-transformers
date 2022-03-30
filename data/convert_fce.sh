
# Run from original json to internal json format with token-level labels
python processed/convert_from_orig_gec_to_json.py -gold original/fce/json/fce.dev.json -out processed/fce/json/dev.json
python processed/convert_from_orig_gec_to_json.py -gold original/fce/json/fce.train.json -out processed/fce/json/train.json
python processed/convert_from_orig_gec_to_json.py -gold original/fce/json/fce.test.json -out processed/fce/json/test.json
