import argparse
import logging

from src.experiment import Experiment

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Run a Document-classification model')

    parser.add_argument('--config', dest='config', type=str, default=None, required=True,
                        help='Path to the JSON config file')
    parser.add_argument('--train', dest='train_dataset', type=str, default=None,
                        help='Path to the training dataset')
    parser.add_argument('--eval', dest='eval_dataset', type=str, default=None,
                        help='Path to the eval dataset')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    experiment = Experiment(config_filepath=args.config, train_data_filepath=args.train_dataset,
                            eval_data_filepath=args.eval_dataset)

    experiment.train()

    results = experiment.eval()

    print()
    logger.info("----------Final Eval Performance---------")
    logger.info(results.to_json())

    logger.info(f"Saving the model & results to: {experiment.experiment_folder}")
    experiment.save_results(results, "eval_results")
    experiment.save_model("final_model.torch")
