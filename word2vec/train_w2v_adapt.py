import multiprocessing
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
import argparse
import os


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
    parser.add_argument('--corpus_path', type=str, default='data/sentence.txt', help='')
    parser.add_argument('--model_path', type=str, default='pretrained/viki/viki_w2v.bin', help='')
    parser.add_argument('--model_dir', type=str, default='pretrained/viki_adapt', help='')
    parser.add_argument('--name', type=str, default='viki_adapt', help='')
    parser.add_argument('--n_epochs', type=int, default=20, help='')
    parser.add_argument('--min_count', type=int, default=0, help='')

    args = parser.parse_args()

    loss_logger = LossLogger()

    cores = multiprocessing.cpu_count()
    print('Num cores: ', cores)

    if args.name == '':
        name = 'w2v'
    else:
        name = f'{args.name}_w2v'

    if os.path.exists(args.model_dir) is False:
        os.mkdir(args.model_dir)

    w2v_adap_path = args.model_dir + f'/{name}.txt'
    vocab_adap_path = args.model_dir + f'/{name}_vocab.txt'
    model_adap_path = args.model_dir + f'/{name}.bin'

    w2v_model = Word2Vec.load(fname=args.model_path)

    print('sg', w2v_model.sg)
    sentence = LineSentence(args.corpus_path)
    sentence = list(sentence)
    w2v_model.min_count = args.min_count
    w2v_model.build_vocab(sentence, update=True, min_count=0, keep_raw_vocab=False)
    w2v_model.train(sentence, total_examples=len(sentence),
                    epochs=args.n_epochs, callbacks=[loss_logger],
                    compute_loss=True)

    w2v_model.wv.save_word2vec_format(fname=w2v_adap_path, fvocab=vocab_adap_path)

    try:
        w2v_model.save(model_adap_path)
    except:
        print("save model false")