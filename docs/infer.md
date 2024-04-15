# 推論方法
このドキュメントでは、推論方法について記す

## 音声再構築タスク
入力された音声を再構築するタスク。VAEの性能確認用。  

1. 音声ファイルが複数入ったディレクトリを用意する
```sh
mkdir audio_inputs # ここに入力音声ファイルを入れる
```

2. 推論する。  
`-s 話者名`で話者を指定する必要がある。 
```sh
python3 infer.py -c config/base.json -i audio_inputs -t recon -s jvs001
```

3. `outputs/`内に出力ファイルが生成されるので、確認する。

## 音声読み上げ(TTS)
テキストを読み上げるタスク。

1. 台本フォルダを用意する。  
台本の内容は`example/text_inputs/`に例があるので、それを参考に作成する。
```sh
mkdir text_inputs # ここに台本を入れる
```

2. 推論する。
```sh
python3 infer.py -c config/base.json -i text_inputs -t tts
```

