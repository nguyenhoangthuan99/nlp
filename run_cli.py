from classification.learners.classification_learner import ClassificationLearner
import json
import os
import shutil
import argparse
import pandas as pd
from utils.learner_util import evaluate


def train(config, checkpoint=False):
    serialization_dir = config['leaner']['serialization_dir']
    if checkpoint is False:
        if os.path.exists(serialization_dir):
            shutil.rmtree(serialization_dir)
    learner = ClassificationLearner(**config['leaner'])
    print(learner.vocab)
    print(learner.model)
    learner.train(**config['training'])


def eval(config=None, test_path=None, result_path='results', serialization_dir=''):
    if config is None and serialization_dir == '':
        raise Exception('config or serialization_dir must be not None')
    if serialization_dir == '':
        serialization_dir = config['leaner']['serialization_dir']
    if test_path is None:
        test_path = config['leaner']['test_path']

    test_df = pd.read_csv(test_path)
    learner = ClassificationLearner.from_serialization(serialization_dir=serialization_dir)
    evaluate(learner, test_df, result_path)


def infer(config=None, text: str = 'infer', serialization_dir=''):
    if config is None and serialization_dir == '':
        raise Exception('config or serialization_dir must be not None')
    if serialization_dir == '':
        serialization_dir = config['leaner']['serialization_dir']
    learner = ClassificationLearner.from_serialization(serialization_dir=serialization_dir)
    prediction = learner.predict(text)
    text = prediction["text"]
    label = prediction["prediction"]
    confidence = prediction["confidence"]
    probs = prediction["prob"]

    print(f'text      : {text}')
    print(f'prediction: {label}')
    print(f'confidence: {confidence}')
    print(f'probs     : {probs}')

    if "att_weight" in prediction:
        att_weight = prediction["att_weight"]
        print(f'att_weight: {att_weight}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--config_path', type=str, default='configs/sentiment_config.json')
    parser.add_argument('--serialization_dir', type=str, default='')
    parser.add_argument('--checkpoint', type=bool, default=False)
    parser.add_argument('--text', type=str, default='thầy giáo giảng rất nhiệt tình')
    parser.add_argument('--result_path', type=str, default='result_sent')
    parser.add_argument('--test_path', type=str, default='data/processed/test.csv')
    args = parser.parse_args()

    if args.mode is None:
        raise ValueError("argument --mode must be not None")
    if args.mode == 'train':
        with open(args.config_path, 'r') as pf:
            config = json.load(pf)
        train(config, args.checkpoint)
    elif args.mode == 'eval':
        if args.serialization_dir != '':
            eval(test_path=args.test_path, result_path=args.result_path, serialization_dir=args.serialization_dir)
        else:
            with open(args.config_path, 'r') as pf:
                config = json.load(pf)
            eval(config=config, test_path=args.test_path, result_path=args.result_path)
    elif args.mode == 'infer':
        if args.serialization_dir != '':
            infer(text=args.text, serialization_dir=args.serialization_dir)
        else:
            with open(args.config_path, 'r') as pf:
                config = json.load(pf)
            infer(config=config, text=args.text, serialization_dir=args.serialization_dir)



