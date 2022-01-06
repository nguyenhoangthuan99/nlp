from allennlp.data import Vocabulary
from classification.learners.classification_learner import ClassificationLearner
from classification.learners.joint_learner import JointLearner
import json
import os
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from utils.utils import plot_confusion_matrix


def evaluate(learner, test_df, result_path='results'):
    if os.path.exists(result_path) is False:
        os.makedirs(result_path)
    learner.model.eval()
    label_col = learner.config['label_col']
    vocab = learner.vocab
    texts = test_df['text'].tolist()
    label_true = test_df[label_col].tolist()
    predictions = learner.predict_on_texts(texts)
    label_pred = [prediction['prediction'] for prediction in predictions]
    y_true = [vocab.get_token_index(label, namespace='labels') for label in label_true]
    y_pred = [vocab.get_token_index(label, namespace='labels') for label in label_pred]

    # get metric scores
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    labels = list(vocab.get_token_to_index_vocabulary('labels').keys())
    report = classification_report(label_true, label_pred, labels=labels)
    print(f'Evaluate {label_col}')
    print('Accuracy :', acc)
    print('Precision:', precision)
    print('Recall   :', recall)
    print('F1-Score :', f1)

    with open(f'{result_path}/{label_col}_score.txt', 'w') as pf:
        pf.write(f'Accuracy score :{acc}\n')
        pf.write(f'Precision score:{precision}\n')
        pf.write(f'Recall score   :{recall}\n')
        pf.write(f'F1 score       :{f1}\n')
        pf.write('Detail:\n')
        pf.write(report)

    cm = confusion_matrix(label_true, label_pred, labels=labels)
    plot_confusion_matrix(cm, target_names=labels, title=f'Confusion Matrix ({label_col})',
                          normalize=False, save_dir=result_path)
    plot_confusion_matrix(cm, target_names=labels, title=f'Confusion Matrix Normalize ({label_col})',
                          normalize=True, save_dir=result_path)

    test_df[f'{label_col}_predict'] = label_pred
    false_pred_df = test_df[test_df[label_col] != test_df[f'{label_col}_predict']]
    false_pred_df.to_csv(f'{result_path}/false_{label_col}_predictions.csv', index=False)
    true_pred_df = test_df[test_df[label_col] == test_df[f'{label_col}_predict']]
    true_pred_df.to_csv(f'{result_path}/true_{label_col}_predictions.csv', index=False)


def load_joint_learner(config_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(config_path, 'r') as pf:
        config = json.load(pf)
    serialization_dir = config['serialization_dir']
    vocab = Vocabulary.from_files(serialization_dir + '/vocabulary')
    learner = JointLearner(
        vocab=vocab,
        text_col=config['text_col'],
        sent_col=config['sent_col'],
        topic_col=config['topic_col'],
        max_tokens=config['max_tokens'],
        embedding_dim=config['embedding_dim'],
        char_embedding_dim=config['char_embedding_dim'],
        ngram_filter_sizes=config['ngram_filter_sizes'],
        num_filters=config['num_filters'],
        hidden_size=config['hidden_size'],
        dropout=config['dropout'],
        num_layers=config['num_layers'],
    )
    weight_path = config['weight_path']
    if weight_path is None:
        weight_path = serialization_dir + '/best.th'
    learner.load_weight(weight_path)
    return learner


def load_learner(config_path):
    with open(config_path, 'r') as pf:
        config = json.load(pf)
    serialization_dir = config['serialization_dir']
    vocab = Vocabulary.from_files(serialization_dir + '/vocabulary')
    learner = ClassificationLearner(
        vocab=vocab,
        label_col=config['label_col'],
        max_tokens=config['max_tokens'],
        embedding_dim=config['embedding_dim'],
        char_embedding_dim=config['char_embedding_dim'],
        ngram_filter_sizes=config['ngram_filter_sizes'],
        token_characters=config['token_characters'],
        num_filters=config['num_filters'],
        hidden_size=config['hidden_size'],
        dropout=config['dropout'],
        num_layers=config['num_layers']
    )
    weight_path = config['weight_path']
    if weight_path is None:
        weight_path = serialization_dir + '/best.th'
    learner.load_weight(weight_path)
    return learner


def predict_on_text(sentiment_learner, topic_learner, text):
    sent_prediction = sentiment_learner.predict(text)
    topic_prediction = topic_learner.predict(text)
    output = {
        'text': text,
        'sentiment': sent_prediction['prediction'],
        'sentiment_confidence': sent_prediction['confidence'],
        'topic': topic_prediction['prediction'],
        'topic_confidence': topic_prediction['confidence'],
        'model_sentiment': sentiment_learner.model.__class__.__name__,
        'model_topic': sentiment_learner.model.__class__.__name__
    }

    if 'att_weight' in sent_prediction:
        output['sentiment_att_weight'] = sent_prediction['att_weight']

    if 'att_weight' in topic_prediction:
        output['topic_att_weight'] = topic_prediction['att_weight']

    return output