# 学習方法
このドキュメントでは、モデルの学習方法について記す。

## 前処理
いくつかのデータセットのための前処理スクリプトを用意しておいた。  
データセットをダウンロードし、これらのスクリプトを実行するだけで前処理が完了する。  
１つ以上のデータセットに対して前処理を行えば学習自体は可能なので、すべてのデータセットを用意する必要はない。  

### JVSコーパス
[JVSコーパス](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)の前処理は以下のコマンドで行う。
```sh
python3 preprocess.py jvs jvs_ver1/ -c config/base.json 
```

### MoeSpeech
準備中

### 自作データセット
データセットを自作する場合(準備中)

## 学習を実行
```sh
python3 train.py -c config/base.json
```

## 学習を再開する
`models/vits.ckpt`を自動的に読み込んで再開してくれる。
読み込みに失敗した場合はファイルが壊れているので、`lightning_logs/`内にある最新のckptを`vits.ckpt`に名前を変更して`models/`に配置することで復旧できる。

## 学習の状態を確認
tensorboardというライブラリを使って学習進捗を可視化することができる。
```sh
tensorboard --logdir lightning_logs
```
をscreen等を用いてバックグラウンドで実行する。  
これが実行されている間はtensorboardのサーバーが動いているので、ブラウザで`http://localhost:6006`にアクセスすると進捗を見ることができる。

## FAQ
- `models/metadata.json`は何のファイルですか？  
    話者名と話者IDを関連付けるためのメタデータが含まれるJSONファイルです。  
    データセットの前処理を行うと自動的に生成されます。