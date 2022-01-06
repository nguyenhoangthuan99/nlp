import torch
from typing import Dict, List
from allennlp.data import Vocabulary
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders import TokenCharactersEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.trainer import GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.util import evaluate
from allennlp.modules.seq2vec_encoders import CnnEncoder, LstmSeq2VecEncoder
from classification.model.joint_model import JointClassifier
from classification.data_reader.joint_reader import JointDataReader
from utils.utils import load_vocab
from allennlp.data import Token, Instance
from allennlp.data.fields.text_field import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from utils.utils import plot_confusion_matrix
import numpy as np


class JointLearner:
    def __init__(
            self,
            train_path: str = None,
            sent_col: str = 'sentiment',
            topic_col: str = 'topic',
            text_col: str = 'text',
            vocab: Vocabulary = None,
            vocab_path: Dict[str, str] = None,
            test_path: str = None,
            val_path: str = None,
            max_tokens: int = 80,
            token_indexers=None,
            min_count=None,
            extend_vocab=False,
            embedding_dim=100,
            hidden_size=128,
            char_embedding_dim=34,
            ngram_filter_sizes=(3,),
            num_filters=64,
            dropout=0.4,
            num_layers=2,
            pretrained_w2v_path=None,
            pretrained_c2v_path=None,
            char_encoder_path=None,
            seq_encoder_path=None,
            device: str = None,
            serialization_dir=None
    ):
        if token_indexers is not None:
            self._token_indexers = token_indexers
        else:
            self._token_indexers = {
                    "tokens": SingleIdTokenIndexer(namespace="tokens"),
                    "token_characters": TokenCharactersIndexer(
                        namespace="token_characters",
                        min_padding_length=3
                    )
                }
        # create data reader
        self.data_reader = JointDataReader(
            max_tokens=max_tokens,
            token_indexers=self._token_indexers,
            text_col=text_col,
            sent_col=sent_col,
            topic_col=topic_col
        )

        # create dataset
        if train_path is not None:
            self.train_data = list(self.data_reader.read(train_path))

        if val_path is not None:
            self.val_data = list(self.data_reader.read(val_path))
        else:
            self.val_data = None

        if test_path is not None:
            self.test_data = list(self.data_reader.read(test_path))
        else:
            self.test_data = None

        # create vocab
        if vocab is None:
            if vocab_path is not None:
                self.vocab = load_vocab(vocab_path, min_count)
            else:
                if self.val_data is not None:
                    self.vocab = Vocabulary.from_instances(self.train_data + self.val_data)
                else:
                    self.vocab = Vocabulary.from_instances(self.train_data)
        else:
            self.vocab = vocab

        if extend_vocab:
            if self.val_data is not None:
                self.vocab.extend_from_instances(self.train_data + self.val_data)
            else:
                self.vocab.extend_from_instances(self.train_data)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda':
            cuda_device = 0
        else:
            cuda_device = -1

        self.device = device
        self.cuda_device = cuda_device

        self.serialization_dir = serialization_dir

        # build model
        # token embedding
        embedding = Embedding(embedding_dim=embedding_dim,
                              vocab_namespace='tokens',
                              vocab=self.vocab,
                              pretrained_file=pretrained_w2v_path)
            
        # char embedding with cnn_encoder
        character_embedding = Embedding(embedding_dim=char_embedding_dim,
                                        vocab_namespace='token_characters',
                                        vocab=self.vocab,
                                        pretrained_file=pretrained_c2v_path)
        cnn_encoder = CnnEncoder(embedding_dim=char_embedding_dim,
                                 num_filters=num_filters,
                                 ngram_filter_sizes=ngram_filter_sizes)

        if char_encoder_path is not None:
            try:
                cnn_encoder.load_state_dict(torch.load(char_encoder_path))
                print('Load char encoder done......')
            except:
                print("Load char encoder failxxxxxx")

        token_encoder = TokenCharactersEncoder(character_embedding, cnn_encoder)

        text_feild_embedder = BasicTextFieldEmbedder(
            {
                "tokens": embedding,
                "token_characters": token_encoder
            }
        )
            
        encoder = LstmSeq2VecEncoder(input_size=text_feild_embedder.get_output_dim(),
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     bidirectional=True,
                                     dropout=dropout)

        if seq_encoder_path is not None:
            try:
                encoder.load_state_dict(torch.load(seq_encoder_path))
                print('Load sequence encoder done ......')
            except:
                print("Load sequence encoder fail xxxxxx")

        self.model = JointClassifier(
            vocab=self.vocab,
            text_feild_embedder=text_feild_embedder,
            encoder=encoder
        )
        self.model.to(self.device)

    def train(
        self,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=64,
        num_epochs=50,
        grad_clipping=5
    ):

        self.vocab.save_to_files(self.serialization_dir + '/vocabulary')
        train_loader = SimpleDataLoader(self.train_data, batch_size, shuffle=True)
        train_loader.index_with(self.vocab)

        if self.val_data is not None:
            val_loader = SimpleDataLoader(self.val_data, batch_size, shuffle=False)
            val_loader.index_with(self.vocab)
        else:
            val_loader = None

        parameters = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer = AdamOptimizer(parameters, lr=lr, weight_decay=weight_decay)

        trainer = GradientDescentTrainer(
            model=self.model,
            serialization_dir=self.serialization_dir,
            data_loader=train_loader,
            validation_data_loader=val_loader,
            num_epochs=num_epochs,
            optimizer=optimizer,
            grad_clipping=grad_clipping,
            cuda_device=self.cuda_device
        )

        trainer.train()

        self.evaluate()

    def evaluate(self, test_path=None, batch_size=64):
        if test_path is not None:
            test_data = list(self.data_reader.read(test_path))
        elif self.test_data is not None:
            test_data = self.test_data
        else:
            raise Exception("Don't have test data please pass test_path arguments")
        
        test_loader = SimpleDataLoader(test_data, batch_size=batch_size, shuffle=False)
        test_loader.index_with(self.vocab)
        self.model.eval()
        results = evaluate(self.model.to('cpu'), test_loader)
        print(results)

    def predict(self, text: str):
        tokens = [Token(token) for token in text.split(' ')]
        text_field = TextField(tokens, self._token_indexers)
        instance = Instance({
            "tokens": text_field
        })
        output = self.model.forward_on_instance(instance)
        y_sent = np.argmax(output['sent_probs'], axis=-1)
        sent_probs = np.max(output['sent_probs'], axis=-1)
        y_topic = np.argmax(output['topic_probs'], axis=-1)
        topic_probs = np.max(output['topic_probs'], axis=-1)
        sent_prediction = self.vocab.get_token_from_index(y_sent, namespace='sentiment')
        topic_prediction = self.vocab.get_token_from_index(y_topic, namespace='topic')
        return {
            "sentiment": [sent_prediction, sent_probs],
            "topic": [topic_prediction, topic_probs]
        }

    def predict_on_texts(self, texts: List[str]):
        predictions = [self.predict(text) for text in texts]
        return predictions

    def load_weight(self, weight_path):
        print(self.device)
        self.model.load_state_dict(torch.load(weight_path, map_location=torch.device(self.device)))

