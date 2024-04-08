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
準備中

## 学習の状態を確認
tensorboardを使用する(準備中)