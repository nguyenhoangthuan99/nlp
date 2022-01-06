# BTL_NLP

## Set up
#### Cài đặt môi trường
- Cài đặt môi trường anaconda theo hướng dẫn tại [đây](https://docs.anaconda.com/anaconda/install/)
- Sau khi cài đặt tạo môi trường mới:
```
conda create -n name_env python==3.6.9
conda activate name_env
```
- Cài đặt các package và thư viện liên quan
```bash
pip install -r requirement.txt
```

## Train Word2vec

### 1 Training word embedding
#### Để huấn luyện word embedding từ đầu chạy lệnh sau:
```
python word2vec/train_w2v.py \
--corpus_path='data/viki' \
--model_dir='pretrained/viki' \
--name='viki' \
--vector_size=100 \
--window=10 \
--n_epochs=10 \
--min_count=30 \
--max_final_vocab=20000
```
- corpus_path: đường dẫn tới corpus. corpus được lưu thành các file trong một thư mục.
- model_dir: đường dẫn tới mơi lưu model sau khi train.
- name: tên của file model sẽ được lưu.
- vector_size: số chiều của vector word embedding
- window: kích thước của một cửa sổ trượt
- n_epochs: số chu kỳ huấn luyện
- min_count: số lần xuất hiện nhỏ nhất của một từ để được thêm vào bộ từ điển (vocabulary)
- max_final_vocab=20000: số lượng lớn nhất có thể của bộ từ điển sau khi xây dựng.

#### Để huấn luyện thích ứng word embedding với bộ dữ liệu riêng chạy
```
python word2vec/train_w2v_adapt.py \
--corpus_path='data/sentence.txt' \
--model_path='pretrained/viki/wiki_w2v.bin' \
--model_dir='models/adapt_w2v \
--name='viki_adapt' \
--n_epochs=20 \
```
- corpus_path: đường dẫn tới corpus. corpus được lưu thành các file trong một thư mục.
- model_path: đường dẫn tới model word embedding đã được huấn luận trước đó
- model_dir: đường dẫn tới mơi lưu model mới sau khi train.
- name: tên của file model mới sẽ được lưu.
- n_epochs: số chu kỳ huấn luyện


## Experiment model
Để huấn luyện, đánh giá mô hình dự đoán, và dự đoán sử dụng command line interface sau:

#### Huấn luyện mô hình
```
python run_cli.py --mode=train --config_path=configs/sentiment_config.json --checkpoint=False
```

#### Đánh giá mô hình
```
python run_cli.py --mode=eval --config_path=configs/sentiment_config.json --result_path=results
python run_cli.py --mode=eval --serialization_dir=model_done/bilstm/sentiment --result_path=results
```

#### Dự đoán
```
python run_cli.py --mode=infer --config_path=configs/sentiment_config.json --text="Cô giáo nhiệt tình thân thiện"
python run_cli.py --mode=infer --serialization_dir=model_done/bilstm/sentiment --text="Cô giáo nhiệt tình thân thiện"
```
với command line interface có 3 chế độ mode:
- train: huấn luyện mô hình
- eval: đánh giá mô hình
- infer: dự đoán một câu đầu vào

Khi đánh giá dự hoặc dự đoán, có thể sử dụng 2 lựa chọn truyền config_path hoặc serialization_dir.<br>
Nếu truyền serialization_dir hệ thống sẽ load trực tiếp mô hình từ thư mục serialization_dir..<br>
Nếu truyền config_path thì hệ thống sẽ sử dụng thư mục serialization_dir được khai báo trong file config.


## Deploy model via server api
chạy app với lệnh sau:
```
python3 app.py 
```
service dự đoán sẽ được chạy qua giao thức http: `http://127.0.0.1:8000/infer` <br>
body gói tin: 
```json
{
  "text": "Thầy giáo giảng bày nhiệt tình"
}
```




