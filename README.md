## <div align="center">JPBirdSongClassification</div>
JPBirdSongClassificationはVGG16をベースの野鳥の鳴き声を識別するプログラムです。現時点では予測用のモデルとプログラムしかアップしていません。

### 目的
このプロジェクトでは、環境音から野鳥を識別することを目的としています。
環境音も区別することで、野鳥の状態を鳴き声からモニタリングすることを可能とします。

### 識別方法
本プログラムは、画像識別の技術を利用して、音源を一定時間ごとに切り出し、そのスペクトログラム画像を学習モデルに識別させます。
物体認識とは異なり、バウンダリーボックス（BBox）は作成しません。

### その他
このリポジトリで公開中のモデルが識別ターゲットとしているものは基本的にヤンバルクイナです。
今後は周辺ツールも公開するかもしれません。


## <div align="center">識別可能な野鳥などの種</div>
2021年に作成したモデルで識別できる野鳥の種類は以下の通りです。
まだ多くありません。

- ヤンバルクイナ
- ウグイス
- カラス（ハシボソ・ハシブトの違いは未学習）
- ヒヨドリ
- オオシマゼミ
- リュウキュウサワマツムシ
- マダラコウロギ
- 人の声
- カエル類（カエルの区別はしない）
- 車の走行音
- 雨
- 小川
- 静寂

また、以下の種は対応作業中、または対応予定の種です。
- リュウキュウコノハズク
- アカヒゲ
- アカショウビン
- モズ
- ネコ

ヤンバルクイナは精度が出るように頑張っています。





## <div align="center">実行環境の作り方</div>
### ローカル環境で動かす場合
<details>
<summary>Pythonのインストール</summary>
https://www.python.org/  
からPython 3.8～Python 3.9をインストールします。
readmeモードが良いでしょう。
</details>
  

<details>
<summary>ffmpegのインストール</summary>
下記の記事を参考に、ffmpegをインストールしてください。

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
GitHubのCodeボタンから選べる「Download ZIP」でもダウンロードはできますが、モデルファイルが入っていません。
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
</details>



### Google Colaboratoryで動かす場合
<details>
<summary>Google Colaboratoryで動かす場合</summary>
Googleアカウントをお持ちであれば、Google Colaboratoryで動かすことも可能です。
Googleアカウントにログインした状態でColaboratoryの新規ノートブックを作成してください。

参考：  
https://atmarkit.itmedia.co.jp/ait/articles/1812/10/news145.html

ノートブックを作成した後、カレントディレクトリにプログラムをcloneコマンドでダウンロードしてください。
```bash
$ !git clone https://github.com/KatsuhiroMorishita/JPBirdSongClassification.git
```

後はローカルと同様に実行できます。

</details>


## <div align="center">実行方法</div>
### 学習
半年以内に追記予定

### 予測
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
1つは一定時間ごとの尤度で、もう1つは一定時間ごとの識別結果です。
識別結果のファイルには設定ファイルで指定した尤度以上の種の名前が記載されています。
Excelで開くと、左から音源のパス、切り出し開始時間[秒]、切り出し幅[秒]、尤度もしくは種名、の順で並んでいます。
</details>


## <div align="center">お問い合わせ</div>
JPBirdSongClassificationのバグや将来へのリクエストがあれば、[GitHub issues](https://github.com/KatsuhiroMorishita/JPBirdSongClassification/issues)に書き込みをお願いします。


## <div align="center">ライセンス（jp）</div>
学術用途に限り、無償で利用可能です。
引用された場合は、連絡は不要ですが参考文献にご掲載ください。
商用での利用には有償（所属先への寄付金）で対応いたします。
[e-mail](morimori.ynct@gmail.com)でご連絡ください。