from typing import Dict
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
from allennlp.data import Vocabulary


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_vocab(vocab_path: Dict[str, str], min_count: Dict[str, int]):
    counter = {}
    for namespace in vocab_path:
        with open(vocab_path[namespace], 'r') as pf:
            lines = pf.readlines()
            lines = [line.split(' ') for line in lines]
            namespace_counter = {token: int(count) for token, count in lines}
            counter[namespace] = namespace_counter

    vocab = Vocabulary(
        counter=counter,
        min_count=min_count
    )
    return vocab


def load_w2v(w2v_path: str, token2idx, embed_size):

    print('Load word embedding pretrained from', w2v_path)
    w2v = {}
    with open(w2v_path, 'r') as pf:
        lines = pf.readlines()
        lines = [line.replace('\n', '').split(' ') for line in lines]
    for l in lines:
        w2v[l[0]] = torch.tensor([float(t) for t in l[1:]], dtype=torch.float32)
    weight = torch.randn(len(token2idx), embed_size)
    for w in token2idx:
        if w in w2v:
            weight[token2idx[w]] = w2v[w]
    return weight


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def make_train_state(early_stop_max_epochs, model_dir, learning_rate):
    return {
        'stop_early': False,
        'early_stop_num_epoch': 0,
        'early_stop_max_epochs': early_stop_max_epochs,
        'early_stop_best_val_loss': 1e8,
        'epoch_index': 0,
        'model_dir': model_dir,
        'learning_rate': learning_rate,
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }


def update_train_state(model, train_state, save_all_checkpoint=False):
    if save_all_checkpoint:
            torch.save(model.state_dict(),
                train_state['model_dir'] + '/checkpoint{}.pth'.format(train_state['epoch_index']))

    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_dir'] + '/last_checkpoint.pth')
    else:
        torch.save(model.state_dict(), train_state['model_dir'] + '/last_checkpoint.pth')
        loss_t = train_state['val_loss'][-1]
        if loss_t < train_state['early_stop_best_val_loss']:
            print('SAVE BEST MODEL AT', train_state['model_dir'] + '/best_model.pth')
            torch.save(model.state_dict(), train_state['model_dir'] + '/best_model.pth')
            train_state['early_stop_num_epoch'] = 0
            train_state['early_stop_best_val_loss'] = loss_t
        else:
            train_state['early_stop_num_epoch'] += 1

        if train_state['early_stop_num_epoch'] >= train_state['early_stop_max_epochs']:
            train_state['stop_early'] = True

    return train_state


def plot_confusion_matrix(
        cm,
        target_names,
        title='Confusion matrix',
        cmap=None,
        normalize=True,
        save_dir='results'):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    width = int(10/7*len(target_names))

    height = int(8/7*len(target_names))

    # plt.figure(figsize=(width, height))
    plt.imshow(cm, cmap=cmap)
    # plt.title(title)
    plt.colorbar()

    if target_names:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    try:
        print(f"Save confusion-matrix...")
        plt.savefig((save_dir + '/{}.png'.format(title)))
    except IOError:
        print(f"Could not save file in directory: {save_dir}")

    plt.show()
