# 学習方法
このドキュメントでは、モデルの学習方法について記す。

## 前処理
いくつかのデータセットのための前処理スクリプトを用意しておいた。  
データセットをダウンロードし、これらのスクリプトを実行するだけで前処理が完了する。  
ただし、複数回実行すると話者IDで不具合が発生するので今のところはどれか１種類だけの前処理にするべき。(TODO: 修正)
`-c`, `--config` オプションで使用するコンフィグファイルを指定できる。

### JVSコーパス
[JVSコーパス](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)の前処理は以下のコマンドで行う。
```sh
python3 preprocess.py jvs jvs_ver1/ -c config/base.json 
```


### 自作データセット
データセットを自作する場合、まず以下の構成のディレクトリを用意する。  
`root_dir`の名前、`speaker001`等の名前は何でもよいが、半角英数字で命名することを推奨。  
書き起こしの言語が日本語でない場合は、 `preprocess/wave_and_text.py`の`LANGUAGE`を書き換える。
```
root_dir/
	- speaker001/
		- speech001.wav
		- speech001.txt
		- speech002.wav
		- speech002.txt
		...
	- speaker02/
		- speech001.wav
		- speech001.txt
		- speech002.wav
		- speech002.txt
		...
	...
```
wavファイルと同じファイル名のテキストファイルに、その書き起こしが入る形にする。  
データセットが用意できたら、前処理を実行する。
```sh
python3 preprocess.py wav-txt root_dir/ -c config/base.json 
```


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
- `models/config.json`は何のファイルですか？
    推論時にデフォルトでロードするコンフィグです。前処理時に`-c`, `--config`オプションで指定したコンフィグが複製されます。