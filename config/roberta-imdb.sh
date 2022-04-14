{
  "transformers_model_name_or_path": "roberta-base",
  "infer_labels": true,
  "seed": 20,
  "max_transformer_input_len": 512,
  "compose_sentence_representations": false,
  "train_batch_size": 8,
  "eval_batch_size": 16,
  "num_labels": 2,
  "epochs": 20,
  "gradient_accumulation_steps": 8,
  "stop_if_no_improvement_n_epochs": 5,
  "dataset_name": "imdb"
}