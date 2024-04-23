# 技術的な詳細
このドキュメントは、本リポジトリの技術的な詳細を記す。

## 開発の動機
- オープンソースのTTS(Text-To-Speech), SVS(Singing-Voice-Synthesis), SVC(Singing-Voice-Conversion)ができるモデルが欲しい。  
- なるべくライセンスが緩いものがいい。(MITライセンスなど)
- 日本語の事前学習モデルが欲しい。

## モデルアーキテクチャ
VITSをベースに改造するという形になる。  
具体的には、
- DecoderをDSP+HnNSF-HiFiGANに変更。DSP機能を取り入れることで、外部からピッチを制御可能に。これにより歌声合成ができる。
- Text Encoderに言語モデルの特徴量を参照する機能をつけ、感情や文脈などを読み取れるように。
- Feature Retrievalを追加し話者の再現性を向上させる
- VITS2のDuration Discriminatorを追加
- Discriminatorのうち、MultiScaleDiscriminatorをMultiResolutionalDiscriminatorに変更。スペクトログラムのオーバースムージングを回避する。

等の改造があげられる。

### 全体の構造
![](./images/auris_architecture.png)
VITSの構造とほぼ同じ。

### デコーダー
![](./images/auris_decoder.png)
DDSPのような加算シンセサイザによる音声合成の後にHiFi-GANに似た構造のフィルターをかけるハイブリッド構造。  

## 学習時の工夫
KL ダイバージェンス損失が大きすぎると音素と音声の対応がうまくいかず、小さすぎると学習が正常に進まないことを考慮し、ReLU関数を用いたヒンジ損失を導入し解決を図る。  
[ソースコード](../module/vits/prior_encoder.py) `134` 行目。