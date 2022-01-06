from typing import Dict, Iterable, List

import torch
import pandas as pd
from allennlp.data import DatasetReader, Instance, Vocabulary, TextFieldTensors
from allennlp.data.fields import LabelField, TextField
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer, CharacterTokenizer


class JointDataReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        char_tokenizer: CharacterTokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        sent_col: str = 'sentiment',
        topic_col: str = 'topic',
        text_col: str = 'text',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.char_tokenizer = char_tokenizer or CharacterTokenizer()
        if token_indexers is None:
            self.token_indexers = {
                    "tokens": SingleIdTokenIndexer(namespace="tokens"),
                    "token_characters": TokenCharactersIndexer(
                        namespace="token_characters",
                        min_padding_length=3
                    )
                }
        else:
            self.token_indexers = token_indexers
        self.max_tokens = max_tokens
        self.sent_col = sent_col
        self.topic_col = topic_col
        self.text_col = text_col

    def text_to_instance(self, text, sentiment=None, topic=None):
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
                tokens = tokens[: self.max_tokens]

        text_field = TextField(tokens, self.token_indexers)
        fields = {
            "tokens": text_field
        }
        if sentiment:
            fields['sentiment'] = LabelField(sentiment, label_namespace='sentiment')
        if topic:
            fields['topic'] = LabelField(topic, label_namespace='topic')
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            text = row[self.text_col]
            sentiment = row[self.sent_col]
            topic = row[self.topic_col]
            yield self.text_to_instance(text, sentiment, topic)
