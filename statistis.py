import pandas as pd
import matplotlib.pyplot as plt


def statistic(data_df: pd.DataFrame):
    print('Thống kê sentiment')
    print(len(data_df))
    n = len(data_df)
    sent_count = data_df['sentiment'].value_counts()
    print('Số lượng sample theo các nhãn sentiment')
    for v, c in sent_count.items():
        print(v, c)
    print('-' * 30)
    print('Tỉ lệ sample theo các nhãn sentiment')
    for v, c in sent_count.items():
        print(v, c/n)
    print('+-'*30)
    print('Thống kê topic')
    print('Số lượng sample theo các nhãn topic')
    topic_count = data_df['topic'].value_counts()
    for v, c in topic_count.items():
        print(v, c)
    print('-'*30)
    print('Tỉ lệ sample theo các nhãn topic')
    topic_count = data_df['topic'].value_counts()
    for v, c in topic_count.items():
        print(v, c/n)


if __name__ == '__main__':
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    test_df = pd.read_csv('data/processed/test.csv')

    print('-----------------------Train dataset------------------')
    statistic(train_df)
    print('-----------------------Valid dataset------------------')
    statistic(val_df)
    print('-----------------------Test  dataset------------------')
    statistic(test_df)

    # Thống kê chiều dài của mỗi từ
    train_df['seq_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
    val_df['seq_len'] = val_df['text'].apply(lambda x: len(x.split(' ')))
    test_df['seq_len'] = test_df['text'].apply(lambda x: len(x.split(' ')))

    print('++'*40)
    print('---------------Train dataset-------------')
    print(train_df['seq_len'].describe())
    train_df['seq_len'].hist()
    plt.show()
    print('---------------Val dataset-------------')
    print(val_df['seq_len'].describe())
    val_df['seq_len'].hist()
    plt.show()
    print('---------------Test dataset-------------')
    print(test_df['seq_len'].describe())
    test_df['seq_len'].hist()
    plt.show()
