import torch
from typing import Dict
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.nn import util


class JointClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_feild_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self._text_feild_embedder = text_feild_embedder
        self._encoder = encoder
        num_sent = vocab.get_vocab_size("sentiment")
        num_topic = vocab.get_vocab_size("topic")
        self.sent_clf = torch.nn.Linear(encoder.get_output_dim(), num_sent)
        self.topic_clf = torch.nn.Linear(encoder.get_output_dim(), num_topic)
        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(1)

    def forward(
            self,
            tokens,
            sentiment: torch.Tensor = None,
            topic: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self._text_feild_embedder(tokens)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(tokens)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self._encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        sent_logits = self.sent_clf(encoded_text)
        topic_logits = self.topic_clf(encoded_text)
        # Shape: (batch_size, num_labels)
        sent_probs = torch.nn.functional.softmax(sent_logits)
        topic_probs = torch.nn.functional.softmax(topic_logits)
        # Shape: (1,)
        output = {
            'sent_probs': sent_probs,
            'sent_logits': sent_logits,
            'topic_logits': topic_logits,
            'topic_probs': topic_probs
        }
        if sentiment is not None:
            loss = torch.nn.functional.cross_entropy(sent_probs, sentiment)
            output['loss'] = loss
            self.accuracy(sent_probs, sentiment)
            self.f1_measure(sent_probs, sentiment)

        return output

    def get_metrics(self, reset: bool = False):
        metrics = {}
        metrics['sent_accuracy'] = self.accuracy.get_metric(reset)
        return metrics
