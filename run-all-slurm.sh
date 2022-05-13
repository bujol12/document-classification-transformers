
# FCE (normal)
sbatch run_slurm.sh "config/soft_attention/min_max/longformer-fce.json" "data/processed/fce/json/train.json" "data/processed/fce/json/dev.json"
sbatch run_slurm.sh "config/soft_attention/min_max/roberta-fce.json" "data/processed/fce/json/train.json" "data/processed/fce/json/dev.json"

sbatch run_slurm.sh "config/soft_attention/min_max_mean/longformer-fce.json" "data/processed/fce/json/train.json" "data/processed/fce/json/dev.json"
sbatch run_slurm.sh "config/soft_attention/min_max_mean/roberta-fce.json" "data/processed/fce/json/train.json" "data/processed/fce/json/dev.json"

sbatch run_slurm.sh "config/soft_attention/top_k/longformer-fce.json" "data/processed/fce/json/train.json" "data/processed/fce/json/dev.json"
sbatch run_slurm.sh "config/soft_attention/top_k/roberta-fce.json" "data/processed/fce/json/train.json" "data/processed/fce/json/dev.json"

sbatch run_slurm.sh "config/soft_attention/top_k_bottom_k/longformer-fce.json" "data/processed/fce/json/train.json" "data/processed/fce/json/dev.json"
sbatch run_slurm.sh "config/soft_attention/top_k_bottom_k/roberta-fce.json" "data/processed/fce/json/train.json" "data/processed/fce/json/dev.json"

# FCE (no neg evidence)
sbatch run_slurm.sh "config/soft_attention/min_max/longformer-fce.json" "data/processed/fce_no_neg_evidence/json/train.json" "data/processed/fce_no_neg_evidence/json/dev.json"
sbatch run_slurm.sh "config/soft_attention/min_max/roberta-fce.json" "data/processed/fce_no_neg_evidence/json/train.json" "data/processed/fce_no_neg_evidence/json/dev.json"

sbatch run_slurm.sh "config/soft_attention/min_max_mean/longformer-fce.json" "data/processed/fce_no_neg_evidence/json/train.json" "data/processed/fce_no_neg_evidence/json/dev.json"
sbatch run_slurm.sh "config/soft_attention/min_max_mean/roberta-fce.json" "data/processed/fce_no_neg_evidence/json/train.json" "data/processed/fce_no_neg_evidence/json/dev.json"

sbatch run_slurm.sh "config/soft_attention/top_k/longformer-fce.json" "data/processed/fce_no_neg_evidence/json/train.json" "data/processed/fce_no_neg_evidence/json/dev.json"
sbatch run_slurm.sh "config/soft_attention/top_k/roberta-fce.json" "data/processed/fce_no_neg_evidence/json/train.json" "data/processed/fce_no_neg_evidence/json/dev.json"

sbatch run_slurm.sh "config/soft_attention/top_k_bottom_k/longformer-fce.json" "data/processed/fce_no_neg_evidence/json/train.json" "data/processed/fce_no_neg_evidence/json/dev.json"
sbatch run_slurm.sh "config/soft_attention/top_k_bottom_k/roberta-fce.json" "data/processed/fce_no_neg_evidence/json/train.json" "data/processed/fce_no_neg_evidence/json/dev.json"

# BEA (normal)
sbatch run_slurm.sh "config/soft_attention/min_max/longformer-bea2019.json" "data/processed/bea2019/json/train.json" "data/processed/bea2019/json/dev.json"
sbatch run_slurm.sh "config/soft_attention/min_max/roberta-bea2019.json" "data/processed/bea2019/json/train.json" "data/processed/bea2019/json/dev.json"

sbatch run_slurm.sh "config/soft_attention/min_max_mean/longformer-bea2019.json" "data/processed/bea2019/json/train.json" "data/processed/bea2019/json/dev.json"
sbatch run_slurm.sh "config/soft_attention/min_max_mean/roberta-bea2019.json" "data/processed/bea2019/json/train.json" "data/processed/bea2019/json/dev.json"

sbatch run_slurm.sh "config/soft_attention/top_k/longformer-bea2019.json" "data/processed/bea2019/json/train.json" "data/processed/bea2019/json/dev.json"
sbatch run_slurm.sh "config/soft_attention/top_k/roberta-bea2019.json" "data/processed/bea2019/json/train.json" "data/processed/bea2019/json/dev.json"

sbatch run_slurm.sh "config/soft_attention/top_k_bottom_k/longformer-bea2019.json" "data/processed/bea2019/json/train.json" "data/processed/bea2019/json/dev.json"
sbatch run_slurm.sh "config/soft_attention/top_k_bottom_k/roberta-bea2019.json" "data/processed/bea2019/json/train.json" "data/processed/bea2019/json/dev.json"

# BEA (no neg evidence)
sbatch run_slurm.sh "config/soft_attention/min_max/longformer-bea2019.json" "data/processed/bea2019_no_neg_evidence/json/train.json" "data/processed/bea2019_no_neg_evidence/json/dev.json"
sbatch run_slurm.sh "config/soft_attention/min_max/roberta-bea2019.json" "data/processed/bea2019_no_neg_evidence/json/train.json" "data/processed/bea2019_no_neg_evidence/json/dev.json"

sbatch run_slurm.sh "config/soft_attention/min_max_mean/longformer-bea2019.json" "data/processed/bea2019_no_neg_evidence/json/train.json" "data/processed/bea2019_no_neg_evidence/json/dev.json"
sbatch run_slurm.sh "config/soft_attention/min_max_mean/roberta-bea2019.json" "data/processed/bea2019_no_neg_evidence/json/train.json" "data/processed/bea2019_no_neg_evidence/json/dev.json"

sbatch run_slurm.sh "config/soft_attention/top_k/longformer-bea2019.json" "data/processed/bea2019_no_neg_evidence/json/train.json" "data/processed/bea2019_no_neg_evidence/json/dev.json"
sbatch run_slurm.sh "config/soft_attention/top_k/roberta-bea2019.json" "data/processed/bea2019_no_neg_evidence/json/train.json" "data/processed/bea2019_no_neg_evidence/json/dev.json"

sbatch run_slurm.sh "config/soft_attention/top_k_bottom_k/longformer-bea2019.json" "data/processed/bea2019_no_neg_evidence/json/train.json" "data/processed/bea2019_no_neg_evidence/json/dev.json"
sbatch run_slurm.sh "config/soft_attention/top_k_bottom_k/roberta-bea2019.json" "data/processed/bea2019_no_neg_evidence/json/train.json" "data/processed/bea2019_no_neg_evidence/json/dev.json"

# Imdb pos
sbatch run_slurm.sh "config/soft_attention/min_max/longformer-pos_imdb.json" "data/processed/imdb/json/pos_train.json" "data/processed/imdb/json/pos_dev.json"
sbatch run_slurm.sh "config/soft_attention/min_max/roberta-pos_imdb.json" "data/processed/imdb/json/pos_train.json" "data/processed/imdb/json/pos_dev.json"

sbatch run_slurm.sh "config/soft_attention/min_max_mean/longformer-pos_imdb.json" "data/processed/imdb/json/pos_train.json" "data/processed/imdb/json/pos_dev.json"
sbatch run_slurm.sh "config/soft_attention/min_max_mean/roberta-pos_imdb.json" "data/processed/imdb/json/pos_train.json" "data/processed/imdb/json/pos_dev.json"

sbatch run_slurm.sh "config/soft_attention/top_k/longformer-pos_imdb.json" "data/processed/imdb/json/pos_train.json" "data/processed/imdb/json/pos_dev.json"
sbatch run_slurm.sh "config/soft_attention/top_k/roberta-pos_imdb.json" "data/processed/imdb/json/pos_train.json" "data/processed/imdb/json/pos_dev.json"

sbatch run_slurm.sh "config/soft_attention/top_k_bottom_k/longformer-pos_imdb.json" "data/processed/imdb/json/pos_train.json" "data/processed/imdb/json/pos_dev.json"
sbatch run_slurm.sh "config/soft_attention/top_k_bottom_k/roberta-pos_imdb.json" "data/processed/imdb/json/pos_train.json" "data/processed/imdb/json/pos_dev.json"

# Imdb neg
sbatch run_slurm.sh "config/soft_attention/min_max/longformer-neg_imdb.json" "data/processed/imdb/json/neg_train.json" "data/processed/imdb/json/neg_dev.json"
sbatch run_slurm.sh "config/soft_attention/min_max/roberta-neg_imdb.json" "data/processed/imdb/json/neg_train.json" "data/processed/imdb/json/neg_dev.json"

sbatch run_slurm.sh "config/soft_attention/min_max_mean/longformer-neg_imdb.json" "data/processed/imdb/json/neg_train.json" "data/processed/imdb/json/neg_dev.json"
sbatch run_slurm.sh "config/soft_attention/min_max_mean/roberta-neg_imdb.json" "data/processed/imdb/json/neg_train.json" "data/processed/imdb/json/neg_dev.json"

sbatch run_slurm.sh "config/soft_attention/top_k/longformer-neg_imdb.json" "data/processed/imdb/json/neg_train.json" "data/processed/imdb/json/neg_dev.json"
sbatch run_slurm.sh "config/soft_attention/top_k/roberta-neg_imdb.json" "data/processed/imdb/json/neg_train.json" "data/processed/imdb/json/neg_dev.json"

sbatch run_slurm.sh "config/soft_attention/top_k_bottom_k/longformer-neg_imdb.json" "data/processed/imdb/json/neg_train.json" "data/processed/imdb/json/neg_dev.json"
sbatch run_slurm.sh "config/soft_attention/top_k_bottom_k/roberta-neg_imdb.json" "data/processed/imdb/json/neg_train.json" "data/processed/imdb/json/neg_dev.json"
