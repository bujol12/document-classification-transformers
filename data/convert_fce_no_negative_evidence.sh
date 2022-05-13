
# Run from original json to internal json format with token-level labels

echo "Running dev"
python processed/convert_from_orig_gec_to_json.py -gold original/fce/json/fce.dev.json -out processed/fce_no_neg_evidence/json/dev.json -remove_neg_doc_evidence

echo "Running train"
python processed/convert_from_orig_gec_to_json.py -gold original/fce/json/fce.train.json -out processed/fce_no_neg_evidence/json/train.json -remove_neg_doc_evidence

echo "Running train"
python processed/convert_from_orig_gec_to_json.py -gold original/fce/json/fce.test.json -out processed/fce_no_neg_evidence/json/test.json -remove_neg_doc_evidence
