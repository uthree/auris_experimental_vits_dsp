# インストール方法
## 事前に用意するもの
- Python 3.10.6 or later
- torch 2.0.1 or later, cuda等GPUが使用可能な環境

## 手順
1. このリポジトリをクローンし、ディレクトリ内に移動する
```sh
git clone https://github.com/uthree/auris
cd auris
```

2. 依存関係をインストール
```sh
pip3 install -r requirements.txt
```

3. monotonic_arginをビルドする(任意)   
(ビルドしない場合はnumba実装が使われる。ビルドしたほうがパフォーマンスが良い。)
```sh
cd module/model_components/monotonic_align/
python3 setup.py build_ext --inplace
```