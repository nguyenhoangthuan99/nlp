import multiprocessing
from gensim.models.word2vec import LineSentence, PathLineSentences
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
import os
import argparse

cores = multiprocessing.cpu_count()
print('Num cores: ', cores)


class LossLogger(CallbackAny2Vec):
    """
    Output loss at each epoch
    """
    def __init__(self):
        self.epoch = 1
        self.losses = []

    def on_epoch_begin(self, model):
        print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        print(f'  Loss: {loss}')
        self.epoch += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, default='data/viki2021', help='')
    parser.add_argument('--model_dir', type=str, default='models/viki', help='')
    parser.add_argument('--name', type=str, default='viki', help='')
    parser.add_argument('--vector_size', type=int, default=100, help='')
    parser.add_argument('--window', type=int, default=8, help='')
    parser.add_argument('--n_epochs', type=int, default=10, help='')
    parser.add_argument('--min_count', type=int, default=30, help='')
    parser.add_argument('--max_final_vocab', type=int, default=None, help='')

    args = parser.parse_args()

    if os.path.exists('models') is False:
        os.mkdir('models')

    if os.path.exists(args.model_dir) is False:
        os.mkdir(args.model_dir)

    loss_logger = LossLogger()

    corpus = PathLineSentences(args.corpus_path)

    w2v_model = Word2Vec(
        sentences=corpus,
        vector_size=args.vector_size,
        workers=cores,
        compute_loss=True,
        window=args.window,
        max_final_vocab=args.max_final_vocab,
        min_count=args.min_count,
        callbacks=[loss_logger],
        epochs=args.n_epochs
    )

    if args.name == '':
        name = 'w2v'
    else:
        name = f'{args.name}_w2v'

    w2v_path = args.model_dir + f'/{name}.txt'
    vocab_path = args.model_dir + f'/{name}_vocab.txt'
    model_path = args.model_dir + f'/{name}.bin'

    w2v_model.wv.save_word2vec_format(fname=w2v_path, fvocab=vocab_path)
    w2v_model.save(model_path)



