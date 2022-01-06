import json
import os
import torch
import numpy as np
from typing import Dict, List
from allennlp.data import Vocabulary
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders import TokenCharactersEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.trainer import GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer, SgdOptimizer
from allennlp.training.util import evaluate
from allennlp.modules.seq2vec_encoders import CnnEncoder, LstmSeq2VecEncoder
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder,PytorchTransformer
from classification.model.classification_model import TextClassifier, AttentionClassifier,TransformerClassifier
from classification.data_reader.classification_reader import ClassificationDataReader
from utils.utils import load_vocab
from allennlp.data import Token, Instance
from allennlp.data.fields.text_field import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from utils.utils import set_seed
from ..model.transformer import TransformerModel

class ClassificationLearner:
    def __init__(
            self,
            train_path: str = None,
            label_col: str = 'sentiment',
            text_col: str = 'text',
            vocab: Vocabulary = None,
            vocab_count_path: Dict[str, str] = None,
            vocabulary_dir: str = None,
            test_path: str = None,
            val_path: str = None,
            max_tokens: int = 80,
            token_indexers=None,
            token_characters=False,
            min_count=None,
            extend_vocab=False,
            model_name='TextClassifier',
            embedding_dim=100,
            bidirectional=True,
            hidden_size=128,
            char_embedding_dim=34,
            ngram_filter_sizes=(3,),
            num_filters=64,
            dropout=0.2,
            num_layers=2,
            pretrained_w2v_path=None,
            pretrained_c2v_path=None,
            char_encoder_path=None,
            seq_encoder_path=None,
            device: str = "cpu",
            serialization_dir=None,
            model_weight_path=None,
            phase = "val",
            seed=42
    ):
        set_seed(seed)

        if token_indexers is not None:
            self._token_indexers = token_indexers
        else:
            if token_characters:
                self._token_indexers = {
                        "tokens": SingleIdTokenIndexer(namespace="tokens"),
                        "token_characters": TokenCharactersIndexer(
                            namespace="token_characters",
                            min_padding_length=3
                        )
                    }
            else:
                self._token_indexers = {
                    "tokens": SingleIdTokenIndexer(namespace="tokens")
                }

        # create data reader
        self.data_reader = ClassificationDataReader(
            max_tokens=max_tokens,
            token_indexers=self._token_indexers,
            text_col=text_col,
            label_col=label_col,phase = phase
        )

        # create dataset
        if train_path is not None:
            self.train_data = list(self.data_reader.read(train_path))
        
        
        if val_path is not None:
            self.val_data = list(self.data_reader.read(val_path))
        else:
            self.val_data = None
        self.data_reader = ClassificationDataReader(
            max_tokens=max_tokens,
            token_indexers=self._token_indexers,
            text_col=text_col,
            label_col=label_col)
        if test_path is not None:
            self.test_data = list(self.data_reader.read(test_path))
        else:
            self.test_data = None

        # create vocab
        if vocab is None:
            if vocabulary_dir:
                self.vocab.from_files(vocabulary_dir)
            elif vocab_count_path is not None:
                self.vocab = load_vocab(vocab_count_path, min_count)
            else:
                if self.val_data is not None:
                    self.vocab = Vocabulary.from_instances(self.train_data + self.val_data)
                else:
                    self.vocab = Vocabulary.from_instances(self.train_data)
        else:
            self.vocab = vocab

        if extend_vocab:
            if self.val_data is not None:
                self.vocab.extend_from_instances(self.train_data + self.val_data+self.test_data)
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
        if token_characters:
            embedder = BasicTextFieldEmbedder(
                {
                    "tokens": embedding,
                    "token_characters": token_encoder
                }
            )
        else:
            embedder = BasicTextFieldEmbedder(
                {
                    "tokens": embedding
                }
            )

        if model_name == 'TextClassifier':
            encoder = LstmSeq2VecEncoder(input_size=embedder.get_output_dim(),
                                         hidden_size=hidden_size,
                                         num_layers=num_layers,
                                         bidirectional=bidirectional,
                                         dropout=dropout)

            if seq_encoder_path is not None:
                try:
                    encoder.load_state_dict(torch.load(seq_encoder_path))
                    print('Load sequence encoder done ......')
                except:
                    print("Load sequence encoder fail xxxxxx")

            self.model = TextClassifier(
                vocab=self.vocab,
                embedder=embedder,
                encoder=encoder
            )
        elif model_name == 'AttentionClassifier':
            encoder = LstmSeq2SeqEncoder(input_size=embedder.get_output_dim(),
                                         hidden_size=hidden_size,
                                         num_layers=num_layers,
                                         bidirectional=bidirectional,
                                         dropout=dropout)

            if seq_encoder_path is not None:
                try:
                    encoder.load_state_dict(torch.load(seq_encoder_path))
                    print('Load sequence encoder done ......')
                except:
                    print("Load sequence encoder fail xxxxxx")
            self.model = AttentionClassifier(
                vocab=self.vocab,
                embedder=embedder,
                encoder=encoder,
                dropout=dropout
            )
        elif model_name == 'TransformerClassifier':
            #encoder = TransformerModel(ntoken=hidden_size, d_model=embedder.get_output_dim(), nhead=1, d_hid=hidden_size, nlayers=4, dropout=dropout)
            encoder = PytorchTransformer(
                input_dim= embedder.get_output_dim(),
                num_layers=2,
                feedforward_hidden_dim= 128,
                num_attention_heads= 2,
                positional_encoding= "sinusoidal",
                positional_embedding_size= 128,
                dropout_prob = 0.4,
            )

            if seq_encoder_path is not None:
                try:
                    encoder.load_state_dict(torch.load(seq_encoder_path))
                    print('Load sequence encoder done ......')
                except:
                    print("Load sequence encoder fail xxxxxx")
            self.model = TransformerClassifier(
                vocab=self.vocab,
                embedder=embedder,
                encoder=encoder,
                dropout=dropout,max_len=max_tokens
            )
            # self.model = TextClassifier(
            #     vocab=self.vocab,
            #     embedder=embedder,
            #     encoder=encoder
            # )

        if model_weight_path:
            print(model_weight_path)
            self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))
        self.model.to(self.device)

        self.config = {}
        self.config['text_col'] = text_col
        self.config['label_col'] = label_col
        self.config['max_tokens'] = max_tokens
        self.config['embedding_dim'] = embedding_dim
        self.config['char_embedding_dim'] = char_embedding_dim
        self.config['ngram_filter_sizes'] = ngram_filter_sizes
        self.config['num_filters'] = num_filters
        self.config['hidden_size'] = hidden_size
        self.config['dropout'] = dropout
        self.config['num_layers'] = num_layers
        self.config['bidirectional'] = bidirectional
        self.config['serialization_dir'] = serialization_dir
        self.config['token_characters'] = token_characters
        self.config['model_name'] = model_name
        self.config['seed'] = seed

        if os.path.exists(self.serialization_dir + '/vocabulary') is False:
            self.vocab.save_to_files(self.serialization_dir + '/vocabulary')

    def train(
        self,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=64,
        num_epochs=50,
        grad_clipping=5
    ):

        train_loader = SimpleDataLoader(self.train_data, batch_size, shuffle=True)
        train_loader.index_with(self.vocab)

        if self.val_data is not None:
            val_loader = SimpleDataLoader(self.val_data, batch_size, shuffle=False)
            val_loader.index_with(self.vocab)
        else:
            val_loader = None

        parameters = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        #optimizer = 
        #lr = 5.0  # learning rate
       # optimizer = SgdOptimizer(parameters, lr=lr, weight_decay=weight_decay)
        optimizer =  AdamOptimizer(parameters, lr=lr, weight_decay=weight_decay)

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
        self.save_config()

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
        y_prediction = np.argmax(output['probs'], axis=-1)
        y_probs = np.max(output['probs'], axis=-1)
        prediction_label = self.vocab.get_token_from_index(y_prediction, namespace='labels')
        probs = {}
        
        for i in range(len(output['probs'])):
            probs[self.vocab.get_token_from_index(i, namespace='labels')] = float(output['probs'][i])
        print(probs)
        prediction = {
            "prediction": prediction_label,
            "confidence": float(y_probs.item()),
            "text": text,
            "prob":probs
        }
        if 'att_weight' in output:
            prediction['att_weight'] = output['att_weight'].astype(np.float32).tolist()
        return prediction

    def predict_on_texts(self, texts: List[str]):
        predictions = [self.predict(text) for text in texts]
        return predictions

    def load_weight(self, weight_path):
        self.model.load_state_dict(torch.load(weight_path, map_location=torch.device(self.device)))

    def save_config(self, config_path=None):
        if config_path is None:
            serialization_dir = self.serialization_dir
            config_path = serialization_dir + '/config.json'
        with open(config_path, 'w') as pf:
            json.dump(self.config, pf)

    @classmethod
    def from_serialization(
            cls,
            serialization_dir,
            train_path=None,
            val_path=None,
            test_path=None
    ):
        config_path = serialization_dir + '/config.json'
        with open(config_path, 'r') as pf:
            config = json.load(pf)

        vocabulary_dir = serialization_dir + '/vocabulary'
        vocab = Vocabulary.from_files(vocabulary_dir)

        model_weight_path = serialization_dir + '/best.th'
        print(model_weight_path)
        return cls(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            vocab=vocab,
            label_col=config['label_col'],
            text_col=config['text_col'],
            model_name=config['model_name'],
            token_characters=config['token_characters'],
            max_tokens=config['max_tokens'],
            embedding_dim=config['embedding_dim'],
            char_embedding_dim=config['char_embedding_dim'],
            ngram_filter_sizes=config['ngram_filter_sizes'],
            num_filters=config['num_filters'],
            hidden_size=config['hidden_size'],
            dropout=config['dropout'],
            num_layers=config['num_layers'],
            bidirectional=config['bidirectional'],
            serialization_dir=config['serialization_dir'],
            seed=config['seed'],
            model_weight_path=model_weight_path
        )
