学習はTensorflowのkerasで行っています。
学習済みのモデルをSavedModel形式とonnx形式の2つ保存していますが、中身は同じです。
onnxの方はPython以外のプログラミング言語でご利用下さい。



■ predict_setting.yamlの記述例
=============================================================

# 学習モデルを動かすための設定
GPU: True       # GPUを使うならTrue
batch_size: 100  # 1度に予測する画像枚数。GPU使用で100とかだと若干だが時短になる

# 識別対象の設定（パターンにマッチするファイルを全て処理できます。）
file_names: glob.glob(r'./yourdir/*.mp3')


# その他の設定
mode: "sound"   # image or sound
size: [200, 120, 3]  # 識別機用の画像のサイズ. (width, height, channel)
models: "last train"   # モデルの選択
model_format: "SavedModel"   # 保存するモデルの形式. "SavedModel" or ".hdf5"
th: 0.5          # 尤度で識別した、と判断する閾値
loss: "local.focal_loss"    # 損失関数


# 音源からスペクトログラムを作るための設定（学習に使用した画像の生成条件と揃えること）
imagegen_params:    # スペクトログラムを作る基本関数へ渡すパラメータ
  sr: 44100         # 読み込み後のサンプリングレート。音源と異なる場合はリサンプリングされて時間がかかる。
  n_mels: 120       # スペクトログラムを作成する際の周波数のビン数。縦方向のピクセル数に関係する。
  fmax: 22000       # 画像上端の周波数
  n_fft: 2048       # フーリエ変換に使うポイント数
  top_remove: 0     # 内部で画像生成後に削る上端のピクセル数。縦のピクセル数はn_mels-top_removeとなる。
  raw: True         # スペクトログラムに生の音声の振幅情報を埋め込むかどうか。Trueで埋め込む。
  #cut_band: [[15000, 22000, "lower"]]

load_mode: "kaiser_fast"   # 読み込む方法。mp3で影響大。kaiser_bestの方が綺麗ではある。
hop: 0.0251       # 1 pixel幅に相当する時間幅[秒]
window_size: 5    # 1枚の画像にする時間幅[秒]
shift_rate: 1.0     # 次の画像生成のためにどの程度ずらすかを決める。1だとオーバーラップなしで次のwindow_size秒後から作り始める。1以上でもOK。


# 前処理の設定
preprocess_chain: ["preprocessing", "preprocessing3"]

=============================================================




■ ラベル名と種名の関係
=============================================================
学習モデルの出力は整数ですが、それをラベル名に変換した上で保存しています。
ただし、ラベル名はkarasu1等となっており、具体的にはわかりません。
下記を参考にして確認してください。

なお、yaml形式で記述していますので、yamlファイルとして保存した後にプログラムで読み込めば、辞書（連想配列）になります。


# for model 20231209-20231216

name_dict:
  OutdoorUnit: エアコン室外機
  airplane: 航空機
  akahige1: アカヒゲS
  akahige2: アカヒゲC
  akashoubin1: アカショウビンS
  akashoubin2: アカショウビンC
  ambulance: サイレン
  aobato1: アオバト
  aobazuku1: アオバズク
  aogera1: アオゲラ
  aoji1: アオジ
  arisui: アリスイ
  bird: その他鳥
  car: 車の走行音
  carSign1: 車のサイン1
  carSign2: 車のサイン2
  cat: ネコ
  chattering: 振動
  coronaDischarge: 放電
  door: ドア
  enaga: エナガ
  fan: 換気扇
  frog: その他カエル
  frog1: アマガエル
  frog2: ヌマガエル
  frog3: アオガエル
  frog4: カジカガエル
  frog5: リュウキュウカジカガエル
  hibari: ヒバリ
  higara: ヒガラ
  hiyodori: ヒヨドリ
  hohjiro: ホオジロ
  hototogisu: ホトトギス
  hukurou: フクロウ
  human: ヒト
  ikaru1: イカル
  insect: その他虫
  insect10: エンマコオロギ
  insect11: カマドコオロギ
  insect12: insect12
  insect13: ヤチスズ
  insect14: タイワンウマオイ
  insect15: タイワンクツワムシ
  insect16: insect16
  insect17: insect17
  insect18: ネッタイシバスズ？
  insect19: タイワンエンマコオロギ
  insect2: insect2
  insect20: クマゼミ
  insect21: ニイニイゼミ
  insect22: ヒグラシ
  insect23: アブラゼミ
  insect24: クロイワツクツク
  insect25: リュウキュウアブラゼミ
  insect26: オオシマゼミ
  insect27: ヒメハルゼミ
  insect28: ツクツクボウシ
  insect3: リュウキュウサワマツムシ
  insect4: insect4
  insect5: マダラコオロギ
  insect50: ヒメギス
  insect51: キンヒバリ
  insect6: ケラ
  insect7: リュウキュウカネタタキ
  insect8: タンボオカメコオロギ？
  insect9: クチキコオロギ
  joubitaki2: ジョウビタキC
  juichi: ジュウイチ
  kakesu: カケス
  kakkou: カッコウ
  karasu1: ハシブトガラス
  karasu2: ハシボソガラス
  kawarahiwa: カワラヒワ
  kibitaki1: キビタキS
  kibitaki2: キビタキC
  kijibato: キジバト
  kojukei: コジュケイ
  kuina: ヤンバルクイナ
  kuroji2: クロジC
  kurotsugumi: クロツグミ
  mejiro1: メジロS
  mejiro2: メジロC
  micnoise: マイクノイズ
  misosazai1: ミソサザイ
  mozu1: モズS
  mozu2: モズC
  mozu3: モズのキョン鳴き
  music: 音楽
  niwatori: ニワトリ
  ohikikoumori: オヒキコウモリ
  rail: 踏切
  rain: 雨
  ruribitaki2: ルリビタキC
  ryukyukonohazuku: リュウキュウコノハズク
  sanshokui: サンショウクイ
  shika: ニホンジカ1
  shika2: ニホンジカ2
  shika3: ニホンジカの悲鳴
  shirohara: シロハラ
  signal: 電子音
  silence: 静寂
  soshicho1: ソウシチョウS
  soshicho2: ソウシチョウC
  suzume: スズメ
  taping: テーピング
  tobi: トビ
  toratsugumi: トラツグミ
  tsugumi: ツグミ
  tsutsudori: ツツドリ
  twig: 小枝の折れる音
  uguisu1: ウグイスS
  uguisu2: ウグイスの谷渡り
  water: 水音
  wind: 風
  yabusame1: ヤブサメ
  yairocho: ヤイロチョウ
  yamagara1: ヤマガラS
  yamagara2: ヤマガラC
  yotaka: ヨタカ
  ootoratsugumi: オオトラツグミ
  amaminokurousagi: アマミノクロウサギ
  frog6: イシカワガエル


=============================================================