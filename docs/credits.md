# クレジット表記
MITライセンスのリポジトリからソースコードをコピペしたりしている都合上、その旨を明記しなければいけないため、ここに記す。

## 引用したソースコード
- [monotonic_align](../module/model_components/monotonic_align/) : [ESPNet](https://github.com/espnet/espnet) から引用。エラーメッセージを一部改変。
- [duration_predictors.py](../module/model_components/duration_predictors.py/) : [VITS2](https://github.com/daniilrobnikov/vits2/blob/main/model/duration_predictors.py) から引用、改変。
- [transforms.py](../module/model_components/transforms.py) : [VITS2](https://github.com/daniilrobnikov/vits2/blob/main/utils/transforms.py)から引用。
- [transformer.py](../module/model_components/transformer.py) : [VITS2](https://github.com/daniilrobnikov/vits2/blob/main/model/transformer.py) から引用、改変。
- [normalization.py](../module/model_components/normalization.py) : [VITS2](https://github.com/daniilrobnikov/vits2/blob/main/model/normalization.py) から一部引用。

## 参考文献

### 参考にした記事
参考にしたブログなど記事たち。気づき次第随時追加していく。
- [【機械学習】VITSでアニメ声へ変換できるボイスチェンジャー&読み上げ器を作った話](https://qiita.com/zassou65535/items/00d7d5562711b89689a8) : zassou氏によるVITS解説記事。理論から実装、損失関数の導出など非常にわかりやすく書かれている。本プロジェクトはこの記事に出合えなかったら完成するのは不可能だろう。

### 参考にしたリポジトリ
- [zassou65535/VITS](https://github.com/zassou65535/VITS) : zassou氏によるVITS実装。 posterior_encoder.pyをはじめとする実装や、ディレクトリ構成などを参考にした。
- [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) : いわゆるRVC, HnNSF-HifiGANによるピッチ制御可能なデコーダーや、特徴量ベクトルを検索するという案は、RVCから着想を得ている。
- [fishaudio/Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) : VITSのテキストエンコーダー部分にBERTの特徴量を付与し、感情や文脈などもエンコードできるようにするという案は、Bert-VITS2から着想を得ている。
- [uthree/tinyvc](https://github.com/uthree/tinyvc) 自分のリポジトリを参考にするとはどういうことだ、と言われそうだが、TinyVCのデコーダーをそのままスケールアップして採用している。

### 論文
参考にした論文。正直たくさんありすぎてすべて書くだけで疲れる。