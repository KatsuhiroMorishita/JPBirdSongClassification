## <div align="center">JPBirdSongClassification</div>
JPBirdSongClassificationはVGG16をベースの野鳥の鳴き声を識別するプログラムです。

### 目的
このプロジェクトでは、環境音から野鳥を識別することを目的としています。
環境音も区別することで、野鳥の状態を鳴き声からモニタリングすることを可能とします。

### 識別方法
本プログラムは、画像識別の技術を利用して、音源を一定時間ごとに切り出し、そのスペクトログラム画像を学習モデルに識別させます。
物体検出とは異なり、バウンダリーボックス（BBox）は作成しません。

### その他
- 現時点でこのリポジトリで公開中のモデルが主な識別ターゲットとしているものは、ヤンバルクイナ、ウグイス、モズ、ヤイロチョウです。
- その他の種の識別精度が特に悪い訳ではないと思いますが、ご了承ください。
- 種によって最適な尤度の閾値が異なりますので、いくつかの音源で試してから閾値を決定してください。
- 今後は周辺ツールも公開するかもしれません。


## <div align="center">識別可能な野鳥などの種</div>
最新モデルで識別できる野鳥の種類は以下の通りです。
野鳥は48種ですのでまだ多くはありません。  

<details open>
<summary>野鳥</summary>
ヤンバルクイナ、ウグイス、ハシブトガラス、ハシボソガラス、カケス、リュウキュウコノハズク、フクロウ、アオバズク、アカショウビン、アカヒゲ、ヒヨドリ、モズ、ヒガラ、ヤマガラ、スズメ、カワラヒワ、ヤイロチョウ、ジョウビタキ、ルリビタキ、キビタキ、ホオジロ、ホトトギス、カッコウ、ツツドリ、ジュウイチ、シロハラ、ツグミ、クロツグミ、トラツグミ、オオトラツグミ、アオゲラ、ヒバリ、ニワトリ、メジロ、ヨタカ、アリスイ、ソウシチョウ、コジュケイ、サンショウクイ、キジバト、アオバト、イカル、クロジ、アオジ、エナガ、ヤブサメ、トビ、ミソサザイ、その他の鳥  
  
＊特定の囀りのみ、地鳴きだけの種を含む。いくつかの種でデータの少ない亜種を統合した。  
＊ヤンバルクイナ、ウグイス、モズ、ヤイロチョウは精度が出るように頑張っています。  
＊ツグミ・アリスイ・オオトラツグミは教師データが不足しています。  
</details>

<details>
<summary>昆虫</summary>
リュウキュウサワマツムシ、マダラコオロギ、ケラ、リュウキュウカネタタキ、タンボオカメコオロギ？、クチキコオロギ、エンマコオロギ、カマドコオロギ、ヤチスズ、タイワンウマオイ、タイワンクツワムシ、ネッタイシバスズ？、ヒメギス、キンヒバリ、オオシマゼミ、クロイワツクツク、ニイニイゼミ、クマゼミ、ヒグラシ、アブラゼミ、リュウキュウアブラゼミ、ヒメハルゼミ、ツクツクボウシ、その他沖縄の虫4種、その他の虫  
</details>

<details>
<summary>カエル</summary>
ニホンアマガエル、ヌマガエル、アオガエル系、カジカガエル、リュウキュウカジカガエル、イシカワガエル、その他のカエル  
</details>

<details>
<summary>その他</summary>
ニホンジカ、ネコ、オヒキコウモリ、アマミノクロウサギ、人の声、車の各種音、救急車、踏切、雨、風、小枝の折れる音、エアコンの室外機、小川、静寂、金属製の門扉、換気扇、航空機、テーピングの音、電子音、マイクノイズ、放電音、音楽  
  
＊音楽は2023-12-14モデルのみ  
＊アマミノクロウサギは教師データが不足しています。  
</details>


また、今後の整備方針は以下の通りです。

<details>
<summary>精度向上や、亜種との分離作業中</summary>
オオトラツグミ、アマミノクロウサギ  
サンショウクイ、リュウキュウサンショウクイ  
ツグミ、クロツグミ  
アオバズク  
トビ  
カワラヒワ  
クロジs・アオジs,c  
カッコウ  
アリスイ  
アカヒゲ、リュウキュウキビタキ、ズアカアオバト、ルリカケス  
</details>


<details>
<summary>今後追加予定の種</summary>
ムクドリ、セキレイ類  
シジュウカラ、オオルリ、センダイムシクイ、イカルチドリ、カシラダカ、コゲラ、オオアカゲラ  
ガビチョウ  
ツバメ類  
バン、オオバン、ヒクイナ、カイツブリ、ヒドリガモ、カルガモ、マガモ、コガモ、オナガガモ、カワセミ  
オオヨシキリ、セッカ  
サシバ、ノグチゲラ、リュウキュウオオコノハズク、カラスバト、シロガシラ  
</details>





## <div align="center">実行環境の作り方</div>
### ローカル環境で動かす場合
<details>
<summary>Pythonのインストール</summary>
https://www.python.org/  
からPython Python 3.10をインストールします。
（TensorflowはPython 3.10で動く。ただし、3.11以上は未対応なので注意。）
just meモードが良いでしょう。
</details>
  

<details>
<summary>ffmpegのインストール</summary>
下記の記事を参考に、ffmpegをインストールしてください。
なお、Ubutntuではaptコマンドでインストール可能です。  

参考記事：  
https://torisky.com/ffmpeg%E3%81%AE%E3%83%80%E3%82%A6%E3%83%B3%E3%83%AD%E3%83%BC%E3%83%89%E3%81%A8%E4%BD%BF%E3%81%84%E6%96%B9%EF%BC%882021%E5%B9%B41%E6%9C%88%EF%BC%89/
</details>


<details>
<summary>CUDAのインストール</summary>
NVIDIA社製のGPUを搭載したマシンでは、GPUを学習と予測に利用可能です。
計算にGPUを利用するには、CUDA toolkitと、cuDNNが必要です。
下記の記事を参考に、CUDA toolkitとcuDNNをインストールしてください。
（注意：たまに、CUDAに対応していないGPUボードがあります）  

参考記事：  
https://qiita.com/8128/items/2e884998cd1193f73e2f

なお、インストールできるバージョンにはtensorflowにより制限ががあります。
TensorFlow公式ページのGPU対応に関するソフトウェア要件に合致するバージョンを選択してください。
WindowsでGPUを使うには、Tensorflow 2.10までとなります。
（Windows上でWSLを使ってUbuntuを動かして、その上で最新のTensorflowを使うこともできますが、動作速度が遅いのでお勧めできません）  

TensorFlowのソフトウェア要件：  
https://www.tensorflow.org/install/gpu?hl=ja
</details>


<details>
<summary>Gitのインストール</summary>
Gitはバージョン管理ツールの一種です。下記のサイトからダウンロードして、インストールしてください。
基本的に設定はいじらなくても大丈夫です。

https://git-scm.com/
</details>


<details>
<summary>プログラムのダウンロードとPythonのライブラリのインストール</summary>
GitHubのCodeボタンから選べる「Download ZIP」でもプログラムをダウンロードはできますが、モデルファイルが入っていません。
モデルファイルのサイズが大きく、LFSという別の管理になっているためです。
下記のコマンドで、完全なダウンロード～Pythonへのライブラリのインストールができます。

```bash
$ git clone https://github.com/KatsuhiroMorishita/JPBirdSongClassification.git
$ cd JPBirdSongClassification
$ pip install -r requirements.txt  
```

ただし、Pythonにパスが通っていない場合は、下記のように実行せねばなりません。
```bash 
$ py -m pip install -r requirements.txt  
```

なお、requirements.txtにはPythonに必要なライブラリが記載されています。このファイル内でtensorflowのバージョンに2.10を指定していますが、これは素のWindows用の設定です。WindowsでもWSLを使って動かす場合や、UbuntuやMacOS上で動かす場合はこの制限を外しても大丈夫です。ただし、GPUを使う場合は、インストールするtensorflowのバージョンとCUDAのバージョンに整合性が必要ですのでご注意ください。  
</details>



### Google Colaboratoryで動かす場合
<details>
<summary>Google Colaboratoryで動かす場合</summary>
Googleアカウントをお持ちであれば、Google Colaboratoryで動かすことも可能です。
Googleアカウントにログインした状態でColaboratoryの新規ノートブックを作成してください。

参考：  
https://atmarkit.itmedia.co.jp/ait/articles/1812/10/news145.html

ノートブックを作成した後、セルに以下のコマンドを入力して実行します。
これでColaboratory上のカレントディレクトリにプログラムがダウンロードされます。
```bash
$ !git clone https://github.com/KatsuhiroMorishita/JPBirdSongClassification.git
```

後はローカルと同様に実行できます。
ただし、コマンドの先頭にエクスクラメーション・マーク「!」が必要です。

</details>


## <div align="center">実行方法</div>
### 学習
画像の識別を学習し、学習モデルを作成します。

<details>
<summary>学習の実行</summary>
設定をtrain_setting.yamlに記述する。
記法は予測の設定と同様です。  

学習処理は、下記のコマンドで実行します。  
```bash
$ pyton train.py  
```

または、  
```bash
$ py train.py  
```

学習結果は./run/train\*として連番で保存されます。
</details>

### 予測
学習後に保存されるモデルファイルを用いて、画像もしくは音声ファイルを用いて予測することができます。

<details open>
<summary>予測処理の実行</summary>
設定をpredict_setting.yamlに記述する。
書式はYAML記法です。
適当なテキストエディタ―で編集して下さい。
エディターは色分けしてくれるVSCodeやSublime Textがお勧めです。
デフォルトの設定ファイルを開けば、恐らく使い方は分かります。

予測処理は、下記のコマンドで実行します。  
```bash
$ pyton predict.py  
```

または、  
```bash
$ py predict.py  
```

予測結果はrunsの中に連番で保存されます。
</details>


<details>
<summary>予測結果の見方</summary>
予測結果は、2つのファイルに分けて保存されます。
「prediction_likelihoods*.csv」は一定時間ごとの尤度を記録しており、もう1つの「prediction_result*.csv」は一定時間ごとの識別結果を記録しています。
「prediction_result*.csv」の識別結果は、設定ファイルで指定した尤度以上の種の名前が記載されています。
Excelで開くと、左から音源のパス、切り出し開始時間\[秒\]、切り出し幅\[秒\]、尤度もしくは種名、の順で並んでいます。


表 「prediction_likelihoods\*.csv」の例

| fname |   s  |  w   | class0 | class1 |
| ----  | ---- | ---- | ----   |   ---- |
|  ファイルのパス1  |  0  |  5  | 0.1 | 0.5 |
|  ファイルのパス2  |  5  |  5  | 0.2 | 0.3 |
|  ファイルのパス3  | 10  |  5  | 0.7 | 0.1 |

表 「prediction_result\*.csv」の例

| fname |   s  |  w   | class0 | class1 |
| ----  | ---- | ---- | ----   |   ---- |
|  ファイルのパス1  |  0  |  5  | ND     |        |
|  ファイルのパス2  |  5  |  5  | uguisu |        |
|  ファイルのパス3  | 10  |  5  | uguisu | karasu |

</details>



### 評価
評価結果がどの程度の性能を示しているのか、[PR-AUC、ROC-AUC、F値](https://tech.ledge.co.jp/entry/metrics)により評価します。
評価には尤度を保存したファイル「prediction_likelihoods\*.csv」が必要です。

<details>
<summary>評価処理の実行</summary>
設定をevaluate_setting.yamlに記述する。
記法は予測の設定と同様です。


学習処理は、下記のコマンドで実行します。  
```bash
$ pyton evaluate.py  
```

または、  
```bash
$ py evaluate.py  
```

処理結果は./runs/evaluate\*に保存されます。

</details>

</details>


## <div align="center">リクエスト・お問い合わせ</div>
JPBirdSongClassificationのバグや将来へのリクエストがあれば、[GitHub issues](https://github.com/KatsuhiroMorishita/JPBirdSongClassification/issues)に書き込みをお願いします。対応して欲しい外来種等の情報でも構いません。  


## <div align="center">ライセンス（jp）</div>
学術用途に限り、無償で利用可能です。
引用された場合は、連絡は不要ですが参考文献にご掲載ください。

所属先への寄付金大歓迎です。
[e-mail](morimori.ynct@gmail.com)でご連絡ください。