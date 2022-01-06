import torch
from typing import Dict
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, Seq2SeqEncoder
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.nn import util
from allennlp.modules.attention.attention import Attention
from allennlp.modules.attention.linear_attention import LinearAttention
from allennlp.modules.attention.dot_product_attention import DotProductAttention
from allennlp.modules.attention.bilinear_attention import BilinearAttention
from allennlp.modules.attention.additive_attention import AdditiveAttention
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder
from .transformer import TransformerModel
import numpy as np

class TextClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super(TextClassifier, self).__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(1)
        self.max_len = 100
    def forward(
            self,
            tokens,
            label: torch.Tensor=None,
    ) -> Dict[str, torch.Tensor]:
        n = self.max_len - tokens["tokens"]["tokens"].size(1)
       # print(tokens)
        if n>0:
            pad = torch.zeros(tokens["tokens"]["tokens"].size(0),n,dtype =torch.long)
            tokens["tokens"]["tokens"] = torch.cat([tokens["tokens"]["tokens"],pad],dim=-1)
            try:
                pad2 = torch.zeros(tokens["token_characters"]["token_characters"].size(0),n,tokens["token_characters"]["token_characters"].size(2),dtype =torch.long)
                tokens["token_characters"]["token_characters"] = torch.cat([tokens["token_characters"]["token_characters"],pad2],dim=1)
            except:
                pass
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(tokens)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(tokens)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        output = {
            'probs': probs,
            'logits': logits
        }
        if label is not None:
            loss = torch.nn.functional.cross_entropy(logits, label)
            output['loss'] = loss
            self.accuracy(logits, label)
            self.f1_measure(logits, label)

        return output
    
    def get_metrics(self, reset: bool = False):

        metrics_result = self.f1_measure.get_metric(reset)
        metrics_result['accuracy'] = self.accuracy.get_metric(reset)
        
        return metrics_result


class AttentionClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 dropout=0.4, max_len = 100,
                 ):
        super(AttentionClassifier, self).__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        self.bidirectional = self.encoder.is_bidirectional()
        if self.bidirectional:
            self.hidden_size = int(self.encoder.get_output_dim() / 2)
        else:
            self.hidden_size = self.encoder.get_output_dim()
        print(self.hidden_size,"hidden")
        self.num_labels = vocab.get_vocab_size("labels")
        self.attention_layer = AdditiveAttention(vector_dim=encoder.get_output_dim(),
                                                 matrix_dim=encoder.get_output_dim(),
                                                 normalize=True)
        self.classifier = torch.nn.Linear(self.encoder.get_output_dim()*2, self.num_labels)
        self.accuracy = CategoricalAccuracy()
        self.max_len = max_len
        self.init_weight()

        self.f1_measure = F1Measure(1)

    def init_weight(self):
        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        n = self.max_len - tokens["tokens"]["tokens"].size(1)
       # print(tokens)
        if n>0:
            pad = torch.zeros(tokens["tokens"]["tokens"].size(0),n,dtype =torch.long)
            tokens["tokens"]["tokens"] = torch.cat([tokens["tokens"]["tokens"],pad],dim=-1)
            try:
                pad2 = torch.zeros(tokens["token_characters"]["token_characters"].size(0),n,tokens["token_characters"]["token_characters"].size(2),dtype =torch.long)
                tokens["token_characters"]["token_characters"] = torch.cat([tokens["token_characters"]["token_characters"],pad2],dim=1)
            except:
                pass
       # print(tokens["token_characters"]["token_characters"][0].size())
        embedded_text = self.embedder(tokens)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(tokens)
        
        #print(tokens["tokens"]["tokens"][0].size())
        #print("__________________")
        # Shape: (batch_size, encoding_dim) if use seq2vec
        # Shape: (batch_size, seq_len, encoding_dim) if use seq2seq
        encoded_text = self.encoder(embedded_text, mask)
        #print(encoded_text[0],mask)
        if self.bidirectional:
            last_forward = encoded_text[:, -1, :self.hidden_size]
            first_backward = encoded_text[:, 0, self.hidden_size:]
            context_vector = torch.cat([last_forward, first_backward], dim=-1)
        else:
            context_vector = encoded_text[:, -1, :]

        # Shape: (batch_size, seq_len)
        att_weight = self.attention_layer(context_vector, encoded_text)

        # Shape: (batch_size, hidden_size)
        att_hidden = torch.bmm(encoded_text.transpose(-2, -1), att_weight.unsqueeze(-1)).squeeze(-1)
        context_att = torch.cat([context_vector, att_hidden], dim=-1)
        logits = self.classifier(context_att)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        output = {
            'probs': probs,
            'logits': logits,
            'att_weight': att_weight,
        }
        if label is not None:
            loss = torch.nn.functional.cross_entropy(logits, label,weight =torch.from_numpy(np.array([1,1,1],dtype=np.float32)))
            output['loss'] = loss
            self.accuracy(logits, label)
            self.f1_measure(logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_result = self.f1_measure.get_metric(reset)
        metrics_result['accuracy'] = self.accuracy.get_metric(reset)

        return metrics_result

class TransformerClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 dropout=0.4, max_len = 100,
                 ):
        super(TransformerClassifier, self).__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        self.bidirectional = self.encoder.is_bidirectional()
        if self.bidirectional:
            self.hidden_size = int(self.encoder.get_output_dim() / 2)
        else:
            self.hidden_size = self.encoder.get_output_dim()
        print(self.hidden_size,"hidden")
        self.num_labels = vocab.get_vocab_size("labels")
        self.attention_layer = AdditiveAttention(vector_dim=encoder.get_output_dim(),
                                                 matrix_dim=encoder.get_output_dim(),
                                                 normalize=True)
        self.classifier = torch.nn.Linear(self.encoder.get_output_dim(), self.num_labels)
        self.self_att = torch.nn.Linear(self.encoder.get_output_dim(), 1)
        self.accuracy = CategoricalAccuracy()
        self.max_len = max_len
        self.init_weight()

        self.f1_measure = F1Measure(1)

    def init_weight(self):
        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        n = self.max_len - tokens["tokens"]["tokens"].size(1)
       # print(tokens)
        if n>0:
            pad = torch.zeros(tokens["tokens"]["tokens"].size(0),n,dtype =torch.long)
            tokens["tokens"]["tokens"] = torch.cat([tokens["tokens"]["tokens"],pad],dim=-1)
            try:
                pad2 = torch.zeros(tokens["token_characters"]["token_characters"].size(0),n,tokens["token_characters"]["token_characters"].size(2),dtype =torch.long)
                tokens["token_characters"]["token_characters"] = torch.cat([tokens["token_characters"]["token_characters"],pad2],dim=1)
            except:
                pass
       # print(tokens["token_characters"]["token_characters"][0].size())
        embedded_text = self.embedder(tokens)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(tokens)
        
        #print(tokens["tokens"]["tokens"][0].size())
        #print("__________________")
        # Shape: (batch_size, encoding_dim) if use seq2vec
        # Shape: (batch_size, seq_len, encoding_dim) if use seq2seq
        encoded_text = self.encoder(embedded_text, mask)
        #print(encoded_text[0],mask)
        if self.bidirectional:
            last_forward = encoded_text[:, -1, :self.hidden_size]
            first_backward = encoded_text[:, 0, self.hidden_size:]
            context_vector = torch.cat([last_forward, first_backward], dim=-1)
        else:
            context_vector = encoded_text[:, -1, :]
        #print("context shape", context_vector.shape)
        # Shape: (batch_size, seq_len)
        att_weight = torch.nn.Softmax(dim=1)(self.self_att(encoded_text)) #self.attention_layer(context_vector, encoded_text)
        #print(att_weight)
        # Shape: (batch_size, hidden_size)*att_weight
        att_hidden =  torch.mean(encoded_text,dim=1) #torch.bmm(encoded_text.transpose(-2, -1), att_weight.unsqueeze(-1)).squeeze(-1)
        #print(att_hidden.size())
        context_att = att_hidden #torch.cat([context_vector, att_hidden], dim=-1)
        logits = self.classifier(context_att)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        output = {
            'probs': probs,
            'logits': logits,
            'att_weight': att_weight,
        }
        if label is not None:
            loss = torch.nn.functional.cross_entropy(logits, label)
            output['loss'] = loss
            self.accuracy(logits, label)
            self.f1_measure(logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_result = self.f1_measure.get_metric(reset)
        metrics_result['accuracy'] = self.accuracy.get_metric(reset)

        return metrics_result