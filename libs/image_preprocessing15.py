# purpose: 画像の前処理用モジュール
# main()では、画像の読み込みと保存を行います。
# author: Katsuhiro MORISHITA（森下功啓）
# history: 
#  2020-04-02, scratchを追加。ver.2のまま。
#  2020-04-23 ver.3. マルチラベルの画像読み込みに対応した。関数の仕様が変わったのでバージョン番号を増やした。 
#  2021-03-02 ver.4. filter_mask()を追加した。
#  2021-03-18 ver.5. randam_erasing()で最大輝度255にしていたのを1.0に訂正。横線の作成方法をちょっと変更。横縦シフト、背景強度、輝度値範囲の制限を追加。
#                    輝度を変える処理の安全性を確保
#  2021-03-27 ver.6  画像読み込み時に保存する対象を選択できる様にした。
#                    指定されたフォルダ内の画像から、画像の入っているフォルダの一覧とその属するクラスの一覧を辞書形式で返す関数search_image_dir()を追加した。
#  2021-04-03 ver.7  mixupで特定のラベルを無視できるようにした。合成しないラベルを選べるようになった。
#  2021-10-19 ver.8  保存先のフォルダを指定できるように変更
#  2021-10-30        printによる表示方法を少し見やすく変更
#  2022-01-06 ver.9  教師データを学習データと検証データに分ける方法を変更（split2を追加）
#  2022-01-06 ver.10 GPUが働いている間に画像を生成できる様に変更した。RAMが膨大にないと利用できないが。
#  2022-01-06 ver.11 背景画像に一様勾配の輝度を埋め込む前処理を実装。channels_lastの誤字を修正
#  2022-11-06 ver.12 画像の横方向の変動を画像の右端に埋め込む関数を実装
#                    RGB画像を前提として、BとGにに一様輝度勾配の画像を埋め込む関数をImageDataGeneratorで利用できるように整備した。
#                    これで、読み込み時の前処理で埋め込んだ輝度勾配がAugmentationでおかしくならない。効果はあるはず。
#  2022-12-20 ver.13 バッチごとに作成する画像において、全クラスの一部のみを抽出する処理を実装（引数use_classを使用）
#  2022-12-23 ver.14 ラベルスムージングを導入。また、map2の更新頻度を指定できるように変更した。
#  2023-05-23 ver.15 コメント追記。エラーをスローする際のメッセージをわかりやすく変更。
#  2023-07-12        to_categorical()にて、最大のID番号を指定できるように変更
#  2023-12-09        read_image()でファイルのリサイズに失敗することがったので、try文を追加した。再現しないのでよくわからん。
# created: 2018-08-20 @ver.1
import sys, os, re, glob, pickle, pprint, threading, time
from matplotlib import pylab as plt
from PIL import Image
from skimage.transform import rotate   # scipyのrotateは推奨されていない@2018-08ので、skimageを使う。こっちの方が使い勝手が良い
from skimage.transform import resize
from tensorflow.keras.utils import Sequence
from scipy.stats import norm
import numpy as np
import pandas as pd




def print2(*args):
    """ いい感じにstrやdictやlistを表示する
    """
    for arg in args:
        if isinstance(arg, dict) or isinstance(arg, list):
            pprint.pprint(arg)
        else:
            print(arg)


def read_name_dict(fname, skiprows=[], key_valule=[0, 1], delimiter=","):
    """ ファイル名とクラスIDなどが入ったファイルから、特定の列を辞書に加工して返す
    fname: str, ファイル名
    skiprows: list<int>, 読み飛ばす行番号を格納したリスト
    key_valule: list<int>, keyとvalueのそれぞれの列番号を格納したリスト
    """
    df = pd.read_csv(fname, delimiter, header=None, skiprows=skiprows)
    
    name_dict = {}
    for i in range(len(df)):
        name_dict[df.iloc[i, key_valule[0]]] = df.iloc[i, key_valule[1]]
    return name_dict




def get_class_names(dir_name, class_sets, ignore_list=[], exchange_dict={}):
    """ フォルダのパスを加工して、クラス名の文字列を返す
    "kuina_uguisu"などの名前をクラス名として,で結合した文字列に変換する。
    dir_name: str, 確認したいフォルダのパス
    """
    base_dir = os.path.basename(dir_name)   # 最下層のフォルダ名だけにする

    # 無視リストにある文字列が入っていたら無視する    
    for ig in ignore_list:
        if ig in base_dir:
            return ""

    # 補足説明は削除
    if "（" in base_dir:
        base_dir = re.sub("（.*）", "", base_dir)

    # クラス名置換
    for name in exchange_dict:
        base_dir = base_dir.replace(name, exchange_dict[name])

    # クラス名を求める
    splited = set(base_dir.split("_"))
    local_classes = splited & class_sets   # 積集合でリスト内のクラス名を取得
    local_classes = sorted(list(local_classes))  # 順番をそろえるために、リストにしてソート

    return ",".join(local_classes)


def search_image_dir(path, class_sets, ignore_list=[], exchange_dict={}):
    """ 指定されたフォルダ内の画像を格納したフォルダを探し、クラス名を付けた辞書を返す
    指定されたフォルダの中にさらにフォルダで小分けになっていることが前提です。
    path: str, 走査するフォルダのパス。相対パスでもよい。カレントディレクトリの場合は"."とすること
    """
    fnames = glob.glob(path + "/**/*.bmp", recursive=True) + \
             glob.glob(path + "/**/*.png", recursive=True) + \
             glob.glob(path + "/**/*.jpg", recursive=True)     # 再帰的に画像のパスを取得
    #print(path, path + "/**/*.jpg", fnames)
    dir_names = set([os.path.dirname(fpath) for fpath in fnames])      # フォルダのパスを取得
    dir_dict = {}

    print("\n\n\n\n< get image path and find class name from dir name >")
    for name in dir_names:
        classes = get_class_names(name, class_sets, ignore_list, exchange_dict)

        print("--------")   # 表示（不要ならコメントアウト）
        print(name)
        print(classes)
        if len(classes) == 0:
            print("len of class names is zero. check this dir. #################")

        if classes != "":
            if classes not in dir_dict:
                dir_dict[classes] = []
            
            dir_dict[classes].append(name)


    # 結果を表示（不要ならコメントアウト）
    print("\n\n< directories per class >")
    keys = sorted(dir_dict.keys())
    for key in keys:
        print(key, dir_dict[key])

    return dir_dict




def read_image(param):
    """ 指定されたフォルダ内の画像をリストとして返す
    読み込みはディレクトリ単位で行う。
    param: dict, 計算に必要なパラメータを収めた辞書
    """
    dir_name = param["dir_name"]            # dir_name: str, フォルダ名、又はフォルダへのパス
    data_format = param["data_format"]      # data_format: str, データ構造を指定
    size = param["size"]                    # size: tuple<int, int>, 読み込んだ画像のリサイズ後のサイズ。 width, heightの順。
    mode = param["mode"]                    # mode: str, 読み込んだ後の画像の変換モード
    resize_filter = param["resize_filter"]  # resize_filter: int, Image.NEARESTなど、リサイズに使うフィルターの種類。処理速度が早いやつは粗い
    image_list = []                  # 読み込んだ画像データを格納するlist
    name_list = []                   # 読み込んだファイルのファイル名
    files = os.listdir(dir_name)     # ディレクトリ内部のファイル一覧を取得
    print("--dir--: ", dir_name)
    for name in files[:5]:
        print(">> ", name)

    try:
        for file in files:
            root, ext = os.path.splitext(file)  # 拡張子を取得
            if ext != ".jpg" and ext != ".bmp" and ext != ".png":
                continue
            fname = os.path.basename(file)
            if fname[0] == ".":          # Macが自動的に生成するファイルを除外
                continue

            path = os.path.join(dir_name, file)             # ディレクトリ名とファイル名を結合して、パスを作成
            image = Image.open(path)                        # 画像の読み込み
            if "preprocess_each_image_func1" in param:       # 必要なら前処理.この時点ではimageはPillowの画像オブジェクトであることに注意。
                func = param["preprocess_each_image_func1"]
                image = func(image, param)
                if image is None:
                    print(path, "-- preprocessing function1 returned None object.")
                    continue
            image = image.resize(size, resample=resize_filter)     # 画像のリサイズ
            image = image.convert(mode)                     # 画像のモード変換。　mode=="LA":透過チャンネルも考慮して、グレースケール化, mode=="RGB":Aを無視
            image = np.array(image)                         # ndarray型の多次元配列に変換
            image = image.astype(np.float16)                # 型の変換（整数型のままだと、0-1にスケーリングした際に、0や1になるのでfloatに変換）
            if image.ndim == 2:                             # グレースケール画像だと2次元のはずなので、チャンネルの次元を増やす
                image = image[:, :, np.newaxis]
            if data_format == "channels_first":
                image = image.transpose(2, 0, 1)            # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
            if "preprocess_each_image_func2" in param:       # 必要なら前処理
                func = param["preprocess_each_image_func2"]
                image = func(image, param)
                if image is None:
                    print(path, "-- preprocessing function2 returned None object.")
                    continue
            image_list.append(image)                        # 出来上がった配列をimage_listに追加  
            name_list.append(file)
    except:
        print("!!!!!File read error!!!!!:  ", file)   # 追加はしたものの、機能するかは未確認
        import traceback
        traceback.print_exc()
    
    return image_list, name_list



def split(arr1, arr2, rate, names=None):
    """ 引数で受け取ったlistをrateの割合でランダムに抽出・分割する（副作用に注意）
    基本的には、学習データと検証データに分けることを想定している。
    arr1, arr2: list<ndarray or list or int>, 画像や整数が入ったリストを期待している
    rate: float, 抽出率。0.0~1.0
    names: list<str>, 画像のファイル名を想定している。別に番号でもいいと思う。
    """
    if len(arr1) != len(arr2):
        raise ValueError("length of arr1 and arr2 is not equal.")

    arr1_1, arr2_1 = list(arr1), list(arr2)  # arr1, arr2を代入すると、副作用覚悟で使用メモリを少し減らす。副作用が嫌いなら、→　list(arr1), list(arr2)　を代入
    arr1_2, arr2_2 = [], []       # 抽出したものを格納する
    names_copy = list(names)

    times = int(rate * len(arr1_1))
    pop_list = []
    for _ in range(times):
        i = np.random.randint(0, len(arr1_1))  # 乱数で抽出する要素番号を作成
        arr1_2.append(arr1_1.pop(i))
        arr2_2.append(arr2_1.pop(i))
        if names is not None:
            pop_list.append(names_copy.pop(i))

    if names is None:
        return np.array(arr1_1), np.array(arr2_1), np.array(arr1_2), np.array(arr2_2)
    else:
        return np.array(arr1_1), np.array(arr2_1), np.array(arr1_2), np.array(arr2_2), pop_list

# 関数の動作テスト
"""
a = [1,2,3,4,5,6,7,8,9,10]
b = [11,12,13,14,15,16,17,18,19,20]
print(split(a, b, 0.2))
exit()
"""


def split2(x, y, rate, names=None):
    """ クラス毎に、rateの割合でランダムに分割する
    学習データと検証データに分けることを想定している。
    クラス毎に分割するので、クラス内の画像数に差があっても、検証データ内に抽出されないということが起きない。
    x: list<ndarray or list or int>, 画像や整数が入ったリストを期待している
    y: ndarray, 0. or 1.を成分とするベクトルを格納した配列（2次元配列）。正解ラベルを想定
    rate: float, 検証に対する抽出率。0.0~1.0
    names: list<str>, 画像のファイル名を想定している。別に番号でもいいと思う。
    """
    # 変数の準備
    X1, X2, Y1, Y2, N1, N2 = [], [], [], [], [], []

    # データのシャッフル
    p = np.random.permutation(len(y))
    x_, y_ = np.array(x), np.array(y)    # 型をndarrayに変換
    x_, y_ = x_[p], y_[p]                # ランダムな順に並べ変え
    if names is not None:
        n_ = np.array(names)
        n_ = n_[p]

    # クラス毎に、ラベルのインデックスをリストにまとめる
    unique_labels = np.unique(y_, axis=0)      # ユニークなラベルの配列を作成
    class_num = y_.shape[1]                    # クラス数（== 列数）を数える
    index_each_class = [[] for _ in range(class_num)]   # これにラベルをクラス毎に格納する
    #print("unique_labels: ", unique_labels)

    # ラベルの種類ごとに、分割
    for label in unique_labels:
        match = np.all(y_ == label, axis=1)       # labelに一致する行がTrueとなる配列を作成
        indexes = np.where(match)[0]              # Trueとなっている行のインデックス（要素番号）を取得

        class_indexes = np.where(label)[0]         # labelの中で、1となっている要素番号の配列を作成
        size = int(np.ceil(len(indexes) * rate))   # 最低でも1枚を検証に回す
        i1 = indexes[size:]
        i2 = indexes[:size]

        X1 += list(x_[i1])
        X2 += list(x_[i2])
        Y1 += list(y_[i1])
        Y2 += list(y_[i2])
        if names is not None:
            N1 += list(x_[i1])
            N2 += list(x_[i2])

    # 結果を返す
    if names is None:
        return np.array(X1), np.array(Y1), np.array(X2), np.array(Y2)
    else:
        return np.array(X1), np.array(Y1), np.array(X2), np.array(Y2), N2





def to_categorical(array_1d, max_id=None):
    """ 整数で表現されたカテゴリを、ニューラルネットワークの出力層のユニットに合わせてベクトルに変換する
    kerasが無くても動作した方が良い気がして、自前で実装した。
    IDは0番から振られていること。
    array_1d: ndarray or list, 1次元配列で整数が格納されていることを期待している
    """
    # IDの最大値（クラス数-1）をセット
    _max = max_id
    if _max is None:
        _max = np.max(array_1d)

    ans = []
    for val in array_1d:    # 整数の番号に応じた[0,0,0,1]の様にone-hotな配列に変換する。なお、0番は[1, 0, 0, ・・・]となる。
        vector = [0] * (_max + 1)
        vector[val] = 1.          # mixupを考えると、浮動小数点が良い
        ans.append(vector)
    
    return np.array(ans)


def one_hotencoding(data=[]):
    """ one-hotencodingを行う
    （2018-08-12: クラスの数の割にクラス毎のサンプル数が少ないことが原因でdata内の各要素におけるクラスIDの欠落が生じないように、ロジックを書き換えた）
    data: list<ndarray>, 1次元のndarrayを格納したリスト
    """
    fusion = []   # 一旦、全部結合させる
    for mem in data:
        fusion += list(mem)
    fusion_onehot = to_categorical(fusion)  # 全部を一緒にしてからone-hotencoding

    ans = []    # fusion_onehotを個々に切り出す
    s = 0
    for length in [len(mem) for mem in data]:
        ans.append(fusion_onehot[s:s + length])
        s += length
    return ans




def read_images1(param):
    """ 辞書で指定されたフォルダ内にある画像を読み込んで、リストとして返す
    クラス毎にフォルダ名又はフォルダへのパスを格納した辞書が、param内に"dir_names_dict"をキーとして保存されていることを期待している。
    フォルダ名がそのままクラス名でも、この関数で処理すること。
    param: dict, 計算に必要なパラメータを収めた辞書
    """
    dir_names_dict = param["dir_names_dict"]    # dict<str:list<str>>, クラス毎にフォルダ名又はフォルダへのパスを格納した辞書。例：{"A":["dir_A1", "dir_A2"], "B":["dir_B"]}
    x, y = [], []    # 読み込んだ画像データと正解ラベル（整数）を格納するリスト
    file_names = []  # 読み込んだ画像のファイル名のリスト
    size_dict = {}   # データの数をクラス毎に格納する辞書

    class_name_list = sorted(dir_names_dict.keys())  # この時点ではstr。ソートすることで、local_id（プログラム中で割り振るクラス番号）とクラス名がずれない
    label_dict = {i:class_name_list[i]  for i in range(len(class_name_list))}   # local_idからクラス名を引くための辞書。ここでのlocal_idはこの学習内で通用するローカルなID。（予測段階で役に立つ）
    label_dict_inv = {class_name_list[i]:i  for i in range(len(class_name_list))}   # 逆に、クラス名から番号を引く辞書
    output_dim = len(label_dict)          # 出力層に必要なユニット数（出力の次元数）
    label_dict[len(label_dict)] = "ND"    # 分類不能に備えて、NDを追加


    print("\n\n\n\n< read images >  (print head 5)")
    for class_name in class_name_list:    # 貰った辞書内のクラス数だけループを回す
        for dir_name in dir_names_dict[class_name]:   # クラス毎に、フォルダ名が格納されたリストから1つずつフォルダ名を取り出してループ
            param["dir_name"] = dir_name              # 読み込み先のディレクトリを指定
            imgs, _file_names = read_image(param)     # 画像の読み込み
            if len(imgs) == 0:
                continue

            local_id = label_dict_inv[class_name]   # local_idはint型
            label_local = [local_id] * len(imgs)    # フォルダ内の画像は全て同じクラスに属するものとして処理
            x += imgs
            y += label_local
            file_names += _file_names
            if local_id in size_dict:    # クラス毎にその数をカウント
                size_dict[local_id] += len(imgs)
            else:
                size_dict[local_id] = len(imgs)

    # クラスごとの重みの計算と、重みの辞書の作成（教師データ数の偏りを是正する）
    size_keys = sorted(size_dict.keys())
    size_list = [size_dict[k] for k in size_keys]
    print2("\n\n< size_dict >", size_dict)
    print2("\n\n< size list >", size_list)
    weights = np.array(size_list)
    weights = np.max(weights) / weights
    weights_dict = {i:weights[i] for i in size_keys}

    # # 正解ラベルをone-hotencoding
    yv = one_hotencoding(data=[y])[0]   # 引数も返り値もlistなので、1つ渡して1つ返してもらうには[0]が必要

    return x, yv, weights_dict, label_dict, output_dim, file_names



def read_images2(param):
    """ リストで指定されたフォルダ内にある画像を読み込んで、リストとして返す
    ファイル名とクラス名（整数か文字列）を紐づけた辞書が、param内に"name_dict"をキーとして保存されていることを期待している。
    param: dict, 計算に必要なパラメータを収めた辞書
    """
    dir_names_list = param["dir_names_list"]    # list<str>, フォルダ名又はフォルダへのパスを格納したリスト
    name_dict = param["name_dict"]              # dict<key: str, value: int or str>, ファイル名をクラスIDに変換する辞書
    x, y = [], []    # 読み込んだ画像データと正解ラベル（整数）を格納するリスト
    file_names = []
    size_dict = {}   # データの数をクラス毎に格納する辞書

    class_name_list = sorted(list(set(name_dict.values())))      # この時点ではintかstrのどちらか。ソートすることで、local_id（プログラム中で割り振るクラス番号）とクラス名がずれない
    label_dict = {i:class_name_list[i]  for i in range(len(class_name_list))}   # local_idからクラス名を引くための辞書。ここでのlocal_idはこの学習内で通用するローカルなID。（予測段階で役に立つ）
    label_dict_inv = {class_name_list[i]:i  for i in range(len(class_name_list))}   # 逆に、クラス名から番号を引く辞書
    output_dim = len(label_dict)        # 出力層に必要なユニット数（出力の次元数）
    label_dict[len(label_dict)] = "ND"  # 分類不能に備えて、NDを追加

    print("\n\n\n\n< read images >  (print head 5)")
    for dir_name in dir_names_list:    # 貰ったフォルダ名の数だけループを回す
        param["dir_name"] = dir_name             # 読み込み先のディレクトリを指定
        imgs, _file_names = read_image(param)    # 画像の読み込み
        if len(imgs) == 0:
            continue

        label_raw = [name_dict[name]  for name in _file_names]         # ファイル名からラベルのリスト（クラス名のlist）を作る
        label_local = [label_dict_inv[raw_id] for raw_id in label_raw]  # 学習に使うlocal_idに置換
        print("--label--", label_local[:20])
        x += imgs
        y += label_local
        file_names += _file_names
        for local_id in label_local:    # クラス毎にその数をカウント
            if local_id in size_dict:
                size_dict[local_id] += 1
            else:
                size_dict[local_id] = 1

    # クラスごとの重みの計算と、重みの辞書の作成（教師データ数の偏りを是正する）
    size_keys = sorted(size_dict.keys())
    size_list = [size_dict[k] for k in size_keys]
    print2("\n\n< size_dict >", size_dict)
    print2("\n\n< size list >", size_list)
    weights = np.array(size_list)
    weights = np.max(weights) / weights
    weights_dict = {i:weights[i] for i in size_keys}

    # 正解ラベルをone-hotencoding
    yv = one_hotencoding(data=[y])[0]   # 引数も返り値もlistなので、1つ渡して1つ返してもらうには[0]が必要

    return x, yv, weights_dict, label_dict, output_dim, file_names




def split_class_name(class_name):
    """ クラス名を文字列から切り出す
    文字列内に含まれるクラス数が1つでも3つでもn個でもOK。
    """
    field = class_name.split(",")              # "foo, bar"の様な記載がある場合に分離させる
    field = [name.strip() for name in field]   # 余分な文字列を削除
    unique_names = list(set(field))            # "hoge,hoge"と書いていても、"hoge"に統一する

    return unique_names



def encode_class_name(class_name, class_list):
    """ クラス名をone-hot encodingする
    注意：to_categorical()と整合性を取ること。
    class_name: str, クラス名。"hoge"や"hoge,fuga"という形式であること。
    class_list: list<st>, クラス名を格納したリスト。ただし、要素に重複のないこととし、毎回順番が変わらないものとする。
    """
    names = split_class_name(class_name)
    I = np.identity(len(class_list))      # 単位行列を作る

    out = np.zeros(len(class_list))
    for name in names:                    # 複数のクラスでもOK
        out += I[class_list.index(name)]

    return out


def read_images3(param):
    """ 辞書で指定されたフォルダ内にある画像を読み込んで、リストとして返す　マルチラベル分類へ対応
    クラス毎にフォルダ名又はフォルダへのパスを格納した辞書が、param内に"dir_names_dict"をキーとして保存されていることを期待している。
    read_images1を代替すると思うが、念のため残しておく。
    param: dict, 計算に必要なパラメータを収めた辞書
    """
    dir_names_dict = param["dir_names_dict"]    # dict<str:list<str>>, クラス毎にフォルダ名又はフォルダへのパスを格納した辞書。例：{"A":["dir_A1", "dir_A2"], "B":["dir_B"]}
    x, y = [], []    # 読み込んだ画像データと正解ラベル（one-hotなndarray型）を格納するリスト
    file_names = []  # 読み込んだ画像のファイル名のリスト
    size_dict = {}   # データの数をクラス毎に格納する辞書

    # クラス名のリストを作成
    class_names = dir_names_dict.keys()  # この時点ではstr。
    name_list = []
    for class_name in class_names:
        name_list += split_class_name(class_name)
    class_name_list = sorted(list(set(name_list)))   # クラス名を取得（重複があったら潰す）

    # その他
    label_dict = {i:class_name_list[i]  for i in range(len(class_name_list))}   # 最尤値を出力したユニットの番号からクラス名を引くための辞書。（予測段階で役に立つ）
    output_dim = len(class_name_list)     # 出力層に必要なユニット数（出力の次元数）
    label_dict[len(label_dict)] = "ND"    # 分類不能に備えて、NDを追加

    # 画像の読み込みとラベルの作成
    print("\n\n\n\n< read images >  (print head 5)")
    for class_name in class_names:    # 貰った辞書内のクラス数だけループを回す
        for dir_name in dir_names_dict[class_name]:   # クラス毎に、フォルダ名が格納されたリストから1つずつフォルダ名を取り出してループ
            param["dir_name"] = dir_name              # 読み込み先のディレクトリを指定
            imgs, _file_names = read_image(param)     # 画像の読み込み
            if len(imgs) == 0:
                continue

            local_id = encode_class_name(class_name, class_name_list)   # local_idはone-hotなndarray型
            label_local = [local_id] * len(imgs)    # フォルダ内の画像は全て同じクラスに属するものとして処理
            x += imgs                               # 読み込んだ画像を記憶
            y += label_local                        # 正解ラベルを記憶
            file_names += _file_names               # ファイル名を記憶
            for name in split_class_name(class_name):    # クラス毎にその数をカウント
                if name in size_dict:
                    size_dict[name] += len(imgs)
                else:
                    size_dict[name] = len(imgs)

    # クラスごとの重みの計算と、重みの辞書の作成（教師データ数の偏りを是正する）
    size_list = [size_dict[name] for name in class_name_list]
    print2("\n\n< size_dict >", size_dict)
    print2("\n\n< size list >", size_list)
    weights = np.array(size_list)          # ndarrayに変換
    weights = np.max(weights) / weights    # 最大値をそれぞれの要素で割る（最小値が1となる）
    weights_dict = {i:weights[i] for i in range(len(class_name_list))}

    return x, y, weights_dict, label_dict, output_dim, file_names






def preprocessing1(imgs):
    """ 画像の前処理
    必要なら呼び出して下さい。
    （処理時間が長い・・・）
    
    imgs: ndarray or list<ndarray>, 画像が複数入っている多次元配列
    """
    image_list = []
    
    for img in imgs:
        _img = img.astype(np.float32)    # float16のままではnp.mean()がオーバーフローする
        img2 = (_img - np.mean(_img)) / np.std(_img) / 4 + 0.5   # 平均0.5, 標準偏差を0.25にする
        img2[img2 > 0.98] = 0.98  # 0-1からはみ出た部分が存在するとImageDataGeneratorに怒られるので、調整
        img2[img2 < 0.0] = 0.0
        img2 = img2.astype(np.float16)
        image_list.append(img2)
        
    return np.array(image_list)


def preprocessing2(imgs):
    """ 画像の前処理
    必要なら呼び出して下さい。
    
    imgs: ndarray, 画像が複数入っている多次元配列
    """
    return imgs / 256    # 255で割ると、numpyの関数処理後に1.0を超える事があり、エラーが出る・・・



def preprocessing3(img, params):
    """ RGB画像であることを前提に、BとGに一様勾配の画像を挿入する。
    img: ndarray type image
    """
    shape = img.shape
    #print(shape)
    
    # 画像サイズの取得
    if len(shape) == 3:
        channel = params["data_format"]
        if channel == "channels_first":
            z, h, w = shape
        else:
            h, w, z = shape
    elif len(shape) == 2:   # 2次元だったら
        h, w = shape
        z = 3

        # チャンネル数の調整（channels_lastの構造となった3チャンネル画像が生成される）
        img = np.dstack([img] * 3)  # imgが2次元配列の画像（チャンネルがない）ことを前提に、チャンネル分重ねる
        channel = "channels_last"
    #print(channel)

    # 輝度最大値を決める
    max_ = np.max(img)
    if max_ > 1:
        a = 255
    else:
        a = 1
    #print(max_)

    # 単純な輝度勾配の2次元画像を作成
    x = np.linspace(0, a, h)     # 等差数列を作成
    m1 = np.tile(x, (w, 1))       # 2次元配列に加工
    m1 = m1.T    # 横に倒す

    x = np.linspace(0, a, w)     # 等差数列を作成
    m2 = np.tile(x, (h, 1))       # 2次元配列に加工

    # channels_firstに変換
    if channel == "channels_last":
        img = img.transpose(2, 0, 1)
        #print("hoge")
    #print(img.shape)


    # GとBのレイヤーごと、差し替え
    img[1] = m1
    img[2] = m2
    img = np.array(img)

    # channels_lastに変換
    if channel == "channels_last":
        img = img.transpose(1, 2, 0)


    return img


# test of preprocessing3
"""
im = Image.new("RGB", (300, 500), (10, 100, 200))
im = np.array(im)
im = preprocessing3(im, {"data_format": "channels_last"})
im = Image.fromarray(im)
im.save("hoge.png")
exit()
"""


##  ImageDataGenerator start  ################################################

def random_crop(image, width_ratio=1.0):
    """ 一部を切り出す
    ref: https://www.kumilog.net/entry/numpy-data-augmentation
    """
    h, w, _ = image.shape    # 元のサイズに戻すために、サイズを覚えておく
    r = 0.7 * np.random.random() + 0.3
    h_crop = int(h * r)      # 切り取るサイズ
    w_crop = int(w * r * width_ratio)

    # 画像のtop, leftを決める
    top = np.random.randint(0, h - h_crop)
    left = np.random.randint(0, w - w_crop)

    # top, leftにcropサイズを足して、bottomとrightを決める
    bottom = top + h_crop   
    right = left + w_crop

    # 決めたtop, bottom, left, rightを使って画像を抜き出す
    img = image[top:bottom, left:right, :] 
    img = resize(img, (h, w))
    return img



def random_erasing(image, s=(0.02, 0.4), r=(0.3, 3)):
    """ ランダムに一部にマスクをかける
    ref: https://www.kumilog.net/entry/numpy-data-augmentation
    """
    # マスクする画素値をランダムで決める
    mask_value = np.random.rand()

    h, w, _ = image.shape
    # マスクのサイズを元画像のs倍の範囲からランダムに決める
    mask_area = np.random.randint(h * w * s[0], h * w * s[1])

    # マスクのアスペクト比をrの範囲からランダムに決める
    mask_aspect_ratio = np.random.rand() * r[1] + r[0]   # (r[1] - r[0])が正しいな？

    # マスクのサイズとアスペクト比からマスクの高さと幅を決める
    # 算出した高さと幅(のどちらか)が元画像より大きくなることがあるので修正する
    mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
    if mask_height > h - 1:
        mask_height = h - 1
    mask_width = int(mask_aspect_ratio * mask_height)
    if mask_width > w - 1:
        mask_width = w - 1

    top = np.random.randint(0, h - mask_height)
    left = np.random.randint(0, w - mask_width)
    bottom = top + mask_height
    right = left + mask_width

    #image[top:bottom, left:right, :].fill(mask_value)
    image[top:bottom, left:right, :] = mask_value
    return image



def bright_line_img(shape=(120, 200, 3), mean=0.5, std=0.1, amp=0.5, axis=2, channel="channels_last"):
    """ 幅のある輝線の画像を作成して返す
    """
    # 画像サイズの取得
    if len(shape) == 3:
        if channel == "channels_first":
            z, h, w = shape
        else:
            h, w, z = shape
    elif len(shape) == 2:
        h, w = shape
        z = 1

    # パラメータ作成
    a, b = h, w
    if axis == 1:      # 縦方向に伸ばす場合は、値を交換
        a, b = w, h

    # まずは2次元の画像を作成
    x = np.linspace(0, 1, a)     # 等差数列を作成
    y = norm.pdf(x, mean, std)   # ガウス分布を計算。array, mean, std
    m = np.tile(y, (b, 1))       # 2次元配列に加工
    if axis == 2:                # 横に伸びる線なら、転置する
        m = m.T
    
    # 輝度の調整
    max_ = np.max(m)
    if max_ > 0:             # 0のことがある
        m = m / max_ * amp   # 最大amp倍
    m[m > 1] = 1.0           # 最大値は1とする


    # チャンネル調整
    #print(m.shape)
    if z > 1:
        m = np.tile(m, (z, 1, 1))   # channels_firstになる
        if channel == "channels_last":
            m = m.transpose(1, 2, 0)

    #print(m.shape)
    #plt.imshow(m)
    #plt.show()
    return m



def scratch_img(image, amp=0.5, channel="channels_last"):
    """ 幅1 pixelの縦線を画像に付与する
    """
    # 画像サイズの取得
    if len(image.shape) == 3:
        if channel == "channels_first":
            z, h, w = image.shape
        else:
            h, w, z = image.shape
    elif len(shape) == 2:
        h, w = image.shape
        z = 1

    # パラメータ作成
    a = np.random.randint(0, h)
    b = np.random.randint(a, h)
    c = np.random.randint(0, w)

    # 縦の線を入れる
    image[a:b, c:c+1] += amp

    return image





def filter_mask(shape=(120, 200, 3), center=0.05, width=30, floor=0.05, filter_kind="high_pass", channel="channels_last"):
    """ スペクトログラムに周波数フィルターを掛けた時と同等の画像を作成するために、フィルターを作って返す
    shape: tuple<int, int, int>, 画像サイズ。
    center: float, 0-1.0の範囲。輝度変化を起こす中心位置。0で端、1で下端。
    width: int, 0から1.0に変化するまでに何ピクセル使うかを表す。
    floor: float, 0-1.0の範囲。最小の倍率。
    """
    # 引数の調整
    if width < 3:   # 0だと除算エラーが出るので最低1だが、幅を見て3とする
        width = 3

    # 画像サイズの取得
    if len(shape) == 3:
        if channel == "channels_first":
            z, h, w = shape
        else:
            h, w, z = shape
    elif len(shape) == 2:
        h, w = shape
        z = 1

    # まずは輝度変化を表すyを作成
    bias = int(h * center)
    scale = width / 10      # 0～1.0に変化するのにどの程度の幅を使うかを決める係数
    x = np.linspace(0, h / scale, h)     # 等差数列を作成
    if filter_kind == "high_pass":
        bias = h - bias
    y = 1 / (1 + np.exp(-(x - bias / scale)))
    if filter_kind == "high_pass":
        y = y[::-1]            # 反転

    # 最高値・最低値の調整
    y = y * (1-floor) + floor
    y[y < 0] = 0.0
    y[y > 1] = 1.0

    # 2次元の画像に加工
    m = np.tile(y, (w, 1))       # 2次元配列に加工
    m = m.T    # 横に倒す

    # チャンネル調整
    #print(m.shape)
    if z > 1:
        m = np.tile(m, (z, 1, 1))   # channels_firstになる
        if channel == "channels_last":
            m = m.transpose(1, 2, 0)

    #print(m.shape)
    #plt.imshow(m)
    #plt.show()
    return m



def horizontal_shift(img, shift=0.5, channel="channels_last"):
    """ 横方向に画像をずらす
    img: ndarray, 2D or 3D dnarray. ndarray形式の画像データ
    shift: float, ずらす量。1だと上下に最大
    """
    shape = img.shape

    # 画像サイズの取得
    if len(shape) == 3:
        if channel == "channels_first":
            z, h, w = shape
        else:
            h, w, z = shape
    elif len(shape) == 2:
        h, w = shape
        z = 1

    shift_w = int(w * shift * np.random.rand())
    img_scroll = np.roll(img, (0, shift_w), axis=(0, 1))

    return img_scroll



def vertical_shift(img, shift=0.5, channel="channels_last", top_ignore=True):
    """ 縦方向に画像をずらす
    img: ndarray, 2D or 3D dnarray. ndarray形式の画像データ
    shift: float, ずらす量。1だと上下に最大
    top_ignore: bool, Trueだと、一番上のセルを無視する
    """
    shape = img.shape

    # 画像サイズの取得
    if len(shape) == 3:
        if channel == "channels_first":
            z, h, w = shape
        else:
            h, w, z = shape
    elif len(shape) == 2:
        h, w = shape
        z = 1

    shift_h = int(h * shift * 2 * (np.random.rand() - 0.5))
    img_scroll = np.roll(img, (shift_h, 0), axis=(0, 1))


    if shift_h < 0:   # 上に持ち上がった場合
        #print(img[-1].shape)
        i = np.abs(shift_h)
        m = np.tile(img[-1], (i, 1, 1))       # 2次元配列に加工
        img_scroll[-i:] = m

    elif shift_h > 0:    # 下に下がった場合
        i = shift_h
        j = 0
        if top_ignore:
            i += 1
            j = 1
        m = np.tile(img[j], (i, 1, 1))       # 2次元配列に加工
        img_scroll[:i] = m
    
    # 上端の処理
    if top_ignore:
        img_scroll[0] = img[0]

    return img_scroll




def insert_std(image):
    """ 画像の右端に、輝度の変動の強さを1ピクセル幅で表す
    """
    m = np.std(image, axis=1)  # 画像の横方向に標準偏差を求める
    image[:, -1] = m           # 画像の右端に埋め込む
    return image




def insert_gradient(img, channel="channels_last"):
    """ RGB画像であることを前提に、BとGに一様勾配の画像を挿入する。
    img: ndarray type image
    """
    shape = img.shape
    #print(shape)
    
    # 画像サイズの取得
    if len(shape) == 3:
        if channel == "channels_first":
            z, h, w = shape
        else:
            h, w, z = shape
    elif len(shape) == 2:   # 2次元だったら
        h, w = shape
        z = 3

        # チャンネル数の調整（channels_lastの構造となった3チャンネル画像が生成される）
        img = np.dstack([img] * 3)  # imgが2次元配列の画像（チャンネルがない）ことを前提に、チャンネル分重ねる
        channel = "channels_last"
    #print(channel)

    # 輝度最大値を決める
    a = 1
    
    # 単純な輝度勾配の2次元画像を作成
    x = np.linspace(0, a, h)     # 等差数列を作成
    m1 = np.tile(x, (w, 1))       # 2次元配列に加工
    m1 = m1.T    # 横に倒す

    x = np.linspace(0, a, w)     # 等差数列を作成
    m2 = np.tile(x, (h, 1))       # 2次元配列に加工

    # channels_firstに変換
    if channel == "channels_last":
        img = img.transpose(2, 0, 1)
        #print("hoge")
    #print(img.shape)


    # GとBのレイヤーごと、差し替え
    img[1] = m1
    img[2] = m2
    img = np.array(img)

    # channels_lastに変換
    if channel == "channels_last":
        img = img.transpose(1, 2, 0)

    return img




# kerasのImageDataGeneratorと同じような使い方ができる様にした
# 2022-01-10 マルチコア対応のために作ったが、マルチコアの場合はRAMを死ぬほど食うので、とりあえず放置。
class MyImageDataGenerator2(Sequence):
    def __init__(self, rotation_range=0, zoom_range=0, horizontal_flip=False, vertical_flip=False, width_shift_range=0.0, height_shift_range=(0.0, False), 
                  crop=False, random_erasing=None, mixup=(0.0, 1, None), return_type="ndarray", shape=(100, 100), brightness_shift_range=0.0, 
                  bright_line=None, scratch=None, noise_std=0.0, freq_filter=None, back_ground=None, channel="channels_last"):
        """
        shape: tuple<int, int>, 最終的に出力する画像のサイズ。height, widthの順で格納すること
        bright_line: list<tuple<amp, axis, th>>, 輝線のパラメータ。強度、方向（縦・横）、確率
        scratch: tuple<amp, times, th>, 引っかき傷（1本の縦線）のパラメータ。強度、線の最大数、確率
        freq_filter: list<tuple<center, width, floor, kind, zitter, th, row-index>>, 周波数フィルタのパラメータ。位置0-1、ピクセル幅、最小値0-1、種類、変動の大きさ、確率、無視する行のインデックス
        """
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip 
        self.vertical_flip = vertical_flip
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.crop = crop
        self.random_erasing = random_erasing
        self.mixup = mixup
        self.return_type = return_type
        self.shape = shape
        self.brightness_shift_range = brightness_shift_range
        self.bright_line = bright_line
        self.scratch = scratch
        self.noise_std = noise_std
        self.freq_filter = freq_filter
        self.channel = channel
        self.back_ground = back_ground

    
    def _get_indexes_for_class_balance(self, y):
        """ クラス間の学習数を調整するための各クラス毎のインデックスのリストを返す
        y: ndarray, 0. or 1.を成分とするベクトルを格納した配列（2次元配列）。正解ラベルを想定
        """
        # クラス毎に、ラベルのインデックスをリストにまとめる
        unique_labels = np.unique(y, axis=0)      # ユニークな要素をカウント
        class_num = y.shape[1]                    # クラス数（== 列数）を数える
        index_each_class = [[] for _ in range(class_num)]   # これにラベルをクラス毎に格納する
        #print("unique_labels: ", unique_labels)

        for label in unique_labels:
            match = np.all(y == label, axis=1)        # labelに一致する行がTrueとなる配列を作成
            indexes = np.where(match)[0]              # Trueとなっている行のインデックス（要素番号）を取得

            class_indexes = np.where(label)[0]         # labelの中で、1となっている要素番号の配列を作成
            size = len(indexes) // len(class_indexes)  # 1クラス当たりの分割サイズ

            # 所属クラス毎に、ラベルを分配（複数のクラスに属するラベルの場合、若干のロスが出るがここでは無視）
            for i, val in enumerate(class_indexes):    # 要素番号と要素を取り出す
                s = size * i
                e = size * (i + 1)
                index_each_class[val] += list(indexes[s : e])  # 複数クラスに属するデータはそれぞれに分割する
        #        if len(class_indexes) >= 2:
        #            print("len(indexes), size, s, e: ", len(indexes), size, s, e)
        #print("index_each_class: ", index_each_class)

        # 要素数の調整
        min_amount = len(y) // class_num       # それぞれのクラスが必要な最低限の要素数
        for i in range(len(index_each_class)):
            indexes = index_each_class[i]
            if len(indexes) < min_amount:      # 配列の要素数がeach_access_timesよりも小さい場合
                coe = int(np.ceil(min_amount / len(indexes)))   # 配列のサイズを増やす係数
                index_each_class[i] = list(indexes) * coe       # 配列のサイズを増やす（中身を繰り返す）

        return index_each_class


    def _get_address_map_for_balance(self, index_each_class, size):
        """ クラス間のバランスを取ったアドレスマップを返す
        画像の少ないクラスに対してはオーバーサンプリングすることになる。画像の多いクラスに対してはアンダーサンプリングとなる。
        この関数を適宜呼び出して、アンダーサンプリングとなっているクラスの画像を適宜入れ替えてください。
        index_each_class: 2d-ndarray or list<list>, クラス毎の画像のインデックスが入ったリスト（配列）
        size: int, 教師画像の全体数
        """
        class_num = len(index_each_class)       # クラス数を数える
        each_access_times = size // class_num   # それぞれのクラスに1epochでアクセスする回数

        # アクセスマップを作成
        address_map = []                        # アドレスマップ
        for indexes in index_each_class:
            np.random.shuffle(indexes)          # 同じアクセス順とならないようにシャッフル
            address_map += list(indexes[:each_access_times])

        # 要素数が足りない分を付け足す
        if len(address_map) < size:
            diff = size - len(address_map)
            numbers = np.arange(0, size)   # 0からsize-1までの連番を作成
            np.random.shuffle(numbers)     # シャッフル
            address_map += list(numbers[:diff])   # 適当に付け足す

        # この関数の返すアドレスマップ単体で使われた場合に、バッチ内の画像が偏るのを防止するために、シャッフル
        np.random.shuffle(address_map)

        return address_map


    def set(self, x, y, save_to_dir=None, save_format=None, batch_size=10, shuffle=True, balancing=False):
        """
        x: ndarray, 0-1.0に正規化された画像が複数入っていることを想定
        y: ndarray, 0. or 1.を成分とするベクトルを格納した配列（2次元配列）。正解ラベルを想定
        shuffle: bool, Trueのとき、1 epoch分の画像を生成したら、画像の順番をランダムに変更する。
        balancing: bool, Trueのとき、クラス毎の画像生成確率を均一とする。Trueのとき、実質的にshuffle==Trueと同じである。
        """
        self.x = x
        self.y = y
        self.save_to_dir = save_to_dir
        self.save_format = save_format
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.balancing = balancing

        # 第1のアドレスマップの作成（アクセス順をシャッフルする際に役立つ）
        address_map = np.arange(0, len(x))   # アクセス先のインデックスを管理する配列
        if shuffle:
            np.random.shuffle(address_map)   # アクセス先をシャッフル

        # 第2のアドレスマップの作成（クラス間のバランス（学習回数を等しくするの）に役立つ）
        address_map2 = np.copy(address_map)
        if balancing:
            index_each_class = self._get_indexes_for_class_balance(y)
            address_map2 = self._get_address_map_for_balance(index_each_class, len(y))

        self.address_map = address_map
        self.address_map2 = address_map2


    def __getitem__(self, idx):
        x = self.x
        y = self.y
        save_to_dir = self.save_to_dir
        save_format = self.save_format
        batch_size = self.batch_size
        shuffle = self.shuffle
        balancing = self.balancing
        address_map = self.address_map
        address_map2 = self.address_map2


        def get_img(index):
            """ 指定された画像のシャローコピーを返す
            高速化優先で、コピーは作らない。
            index: int, 取得する画像の要素番号
            """
            n = address_map[index]
            m = address_map2[n]
            img = x[m]
            label_vect = y[m]
            return img, label_vect

        i =  idx * batch_size
        x_ = []
        y_ = []
        for k in range(batch_size):   # バッチサイズ分、ループ
            j = (i + k) % len(x)      # 取得する画像のインデックス（要素番号を作成）
            img_origin, label_origin = get_img(j)  # j番目の画像を取得（実際にはxのj番目とは限らない）
            img1 = img_origin.astype(np.float64)   # 単なるnumpyの行列計算なら32bitが速いが、64bitでの計算がより速い
            flag_label = False
            
            # --画像の加工--
            # 回転
            if self.rotation_range != 0 and np.random.rand() < 0.5:
                theta = np.random.randint(-self.rotation_range, self.rotation_range)
                img1 = rotate(img1, theta)     # skimageのライブラリを使った場合、サイズは変わらないがfloat64になる。scipyのrotateはサイズが変わる
            
            # 水平方向に反転
            if self.horizontal_flip and np.random.rand() < 0.5:
                img1 = np.fliplr(img1)
            
            # 垂直方向に反転
            if self.vertical_flip and np.random.rand() < 0.5:
                img1 = np.flipud(img1)

            # 横方向にshift
            if self.width_shift_range:  # Noneじゃない
                img1 = horizontal_shift(img1, self.width_shift_range, self.channel)

            # 縦方向にshift
            if self.height_shift_range:  # Noneじゃない
                shift, ignore = self.height_shift_range
                img1 = vertical_shift(img1, shift, self.channel, ignore)
            
            # 他の画像と合成
            mixup_th, mix_amount, invalid_label = self.mixup
            if not(isinstance(invalid_label, np.ndarray) and np.inner(label_origin, invalid_label)) and mixup_th > np.random.rand() and mix_amount >= 2:
                if mix_amount > 2:
                    mix_amount = np.random.randint(2, mix_amount + 1)  # 3以上が指定されていた場合は、実際に合成する数を乱数で決める
                ratios = np.random.rand(mix_amount)   # 合成比をつくる
                ratios /= np.sum(ratios)              # 全部足して1.0になるように調整
                
                img_ = img1 * ratios[0]               # 元画像に合成比をかける
                label_vect1 = label_origin * ratios[0]  # ラベルにも合成比をかける
                
                j, k = 1, 0
                while j < mix_amount:
                    k += 1
                    r = ratios[j]
                    n = np.random.randint(0, len(x))   # 合成する画像のインデックス
                    img_mix, label_mix = get_img(n)    # 合成する画像を1枚とりだす
                    
                    if isinstance(invalid_label, np.ndarray) and np.inner(label_mix, invalid_label):  # mixupを指示されていないクラスなら、無視する
                        if k > 1000:
                            print("--mixup loop warning--, too many loop times...")
                            break
                        continue

                    img_ = img_ + img_mix * r   # 合成
                    label_vect1 = label_vect1 + label_mix * r

                    j += 1

                # 合成できていれば、反映させる
                if j == mix_amount:
                    flag_label = True
                    img1 = img_

            # 切り出し（トリミング）
            if self.crop and np.random.rand() < 0.5:
                img1 = random_crop(img1)

            # 輝度を変える
            if self.brightness_shift_range != 0:
                r = 1.0 + (2 * np.random.rand() - 1.0) * self.brightness_shift_range
                if r < 0.1:   # 負や極端に小さい値にならない様にする
                    r = 0.1
                img1 = img1 * r
            
            # 輝線を加える（縦か横方向のみ）
            if self.bright_line is not None:
                for amp, axis, th in self.bright_line:
                    if np.random.rand() > th:
                        continue
                    mean = np.random.rand()
                    std = (np.random.rand() + 0.01) / 10    # 標準偏差（線の太さを決める指標）は最大でもsigma==0.101。最小でも0.001。
                    amp *= np.random.rand() + 0.05
                    img_ = bright_line_img(shape=img1.shape, mean=mean, std=std, amp=amp, axis=axis, channel=self.channel)
                    img1 = img1 + img_

            # 引っかき傷のような縦線を加える
            if self.scratch is not None:
                amp, times, th = self.scratch
                if np.random.rand() < th:
                    times_ = 1
                    if times > 1:                     # 2本以上が指定されていた場合は、線の数を乱数で決める
                        times_ = np.random.randint(1, times + 1)
                    for _ in range(times_):
                        amp_ = amp * np.random.rand()
                        img1 = scratch_img(img1, amp=amp_, channel=self.channel)

            # ノイズを画像に加える
            if self.noise_std > 0 and np.random.rand() < 0.5:
                noise_img = np.random.normal(0, self.noise_std, img1.shape)
                img1 = img1 + noise_img   # ノイズを加える

            # 背景の輝度を調整する
            if self.back_ground:
                th, amp = self.back_ground
                if np.random.rand() < th:
                    p = np.random.rand() * amp
                    img1[img1 < p] = p


            # 周波数フィルター（画像の上端や下端を滑らかに0にする）
            if self.freq_filter is not None:
                for center, width, floor, kind, zitter, th, ignore_row in self.freq_filter:
                    if np.random.rand() > th:
                        continue
                    center *= 1 + np.random.randn() * zitter
                    width = int(width * (1 + np.random.randn() * zitter))
                    floor *= 1 + np.random.randn() * zitter
                    mask = filter_mask(shape=img1.shape, center=center, width=width, floor=floor, filter_kind=kind, channel=self.channel)
                    img_ = img1 * mask
                    if isinstance(ignore_row, int):  # 無視したい行があれば、元の値を書き込む
                        img_[ignore_row] = img1[ignore_row]
                    img1 = img_

            # ランダムに一部にマスクをかける
            if self.random_erasing:
                th, box_scale = self.random_erasing
                if np.random.rand() < th:
                    img1 = random_erasing(img1, box_scale)

            # ラベルに対して何の処理も行われなかった場合の対応
            if flag_label == False:
                label_vect1 = label_origin.copy()

            # 値の制限
            img1 /= np.max(img1)  # 最大を1とする
            img1[img1 < 0] = 0.0  # 最小は0

            # リサイズ
            if x[0].shape[:2] != self.shape[:2]:    # channel lastが前提で書いている
                img1 = resize(img1, self.shape[:2])

            # 保存
            if save_to_dir is not None and save_format is not None:
                img3 = img1 * 255   # 別の変数として保存しないと、x_に影響する
                img3 = img3.astype(np.uint8)
                pilImg = Image.fromarray(np.uint8(img3))
                pilImg.save("{0}/{1}_{2}_hoge.{3}".format(save_to_dir, i, k, save_format))

            # 返り値のリストに格納
            img1 = img1.astype(np.float16)    # メモリ節約のため、型を小さくする
            x_.append(img1)
            y_.append(label_vect1)


        # 次のアクセスに備えた処理
        i_backup = i
        i = (i + batch_size) % len(x)
        if i_backup > i:       # 1順を検知
            # アドレスマップの更新
            if shuffle:        # シャッフルが指定されていたら
                np.random.shuffle(self.address_map)   # アクセス先をシャッフル

            # クラス間のバランス取り用のアドレスも更新（balancing==Trueなのにそのままだと、学習されない画像がでる）
            if balancing:
                self.address_map2 = self._get_address_map_for_balance(index_each_class, len(y))
                
                # 確認用（普段はコメントアウト）
                #print(address_map2)
                #import matplotlib.pyplot as plt
                #plt.hist(address_map2)
                #plt.show()
                #plt.plot(address_map2)
                #plt.show()


        # 処理結果を返す
        if self.return_type == "ndarray":
            x_, y_ = np.array(x_), np.array(y_)
            x_ = x_.astype(np.float16)
            y_ = y_.astype(np.float16)
            return x_, y_
        else:
            return x_, y_

            

    def __len__(self):
        return int(np.ceil(len(self.x) // self.batch_size))   # 整数を返さないと、ifでの評価でエラーが出る。


    #def __bool__(self):    # ifでこのクラスオブジェクトが評価された際に、Trueを返す
    #    return True



# 使用例
"""
    # 学習
    datagen_train.set(x_train, y_train, batch_size=batch_size, shuffle=True, balancing=True)
    datagen_validation.set(x_test, y_test, batch_size=batch_size, shuffle=True, balancing=True)


    history = model.fit(   # ImageDataGeneratorを使った学習
        datagen_train,  # シャッフルは順序によらない学習のために重要
        epochs=setting["epochs"],
        steps_per_epoch=steps_per_epoch,
        verbose=1,
        #class_weight=weights_dict,   # 与えない方がまし
        callbacks=[cb_save],
        validation_data=datagen_validation,  # マルチプロセスを利用しない場合、ここにジェネレータを渡すことも出来る
        initial_epoch=setting["initial_epoch"],
        validation_steps=len(x_test) // batch_size,

        # マルチプロセスを利用する設定（不要ならコメントアウト. 実用的には、最低でも128 GBのRAMが必要だと思う。）
        ## generatorは使えない。
        workers=2,
        max_queue_size=2,
        use_multiprocessing=True,
        ) # 返り値には、学習中のlossやaccなどが格納される（metricsに指定する必要がある）

"""







# kerasのImageDataGeneratorと同じような使い方ができる様にした
class MyImageDataGenerator:
    def __init__(self, rotation_range=0, zoom_range=0, horizontal_flip=False, vertical_flip=False, width_shift_range=0.0, height_shift_range=(0.0, False), 
                  crop=False, random_erasing=None, mixup=(0.0, 1, None), return_type="ndarray", shape=(100, 100), brightness_shift_range=0.0, 
                  bright_line=None, scratch=None, noise_std=0.0, freq_filter=None, back_ground=None, insert_std=False, 
                  insert_gradient=False, channel="channels_last", use_class=1, label_smoothing_e=0.005, 
                  balancing_refresh_iteration=None):
        """
        shape: tuple<int, int>, 最終的に出力する画像のサイズ。height, widthの順で格納すること
        bright_line: list<tuple<amp, axis, th>>, 輝線のパラメータ。強度、方向（縦・横）、確率
        scratch: tuple<amp, times, th>, 引っかき傷（1本の縦線）のパラメータ。強度、線の最大数、確率
        freq_filter: list<tuple<center, width, floor, kind, zitter, th, row-index>>, 周波数フィルタのパラメータ。位置0-1、ピクセル幅、最小値0-1、種類、変動の大きさ、確率、無視する行のインデックス
        """
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip 
        self.vertical_flip = vertical_flip
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.crop = crop
        self.random_erasing = random_erasing
        self.mixup = mixup
        self.return_type = return_type
        self.shape = shape
        self.brightness_shift_range = brightness_shift_range
        self.bright_line = bright_line
        self.scratch = scratch
        self.noise_std = noise_std
        self.freq_filter = freq_filter
        self.back_ground = back_ground
        self.insert_std = insert_std
        self.insert_gradient = insert_gradient
        self.channel = channel
        self.use_class = use_class
        self.label_smoothing_e = label_smoothing_e
        self.balancing_refresh_iteration = balancing_refresh_iteration


    def _get_indexes_for_class_balance(self, y):
        """ クラス間の学習数を調整するための各クラス毎のインデックスのリストを返す
        y: ndarray, 0. or 1.を成分とするベクトルを格納した配列（2次元配列）。正解ラベルを想定
        """
        # クラス毎に、ラベルのインデックスをリストにまとめる
        unique_labels = np.unique(y, axis=0)      # ユニークな要素をカウント
        class_num = y.shape[1]                    # クラス数（== 列数）を数える
        index_each_class = [[] for _ in range(class_num)]   # これにラベルをクラス毎に格納する
        #print("unique_labels: ", unique_labels)

        for label in unique_labels:
            match = np.all(y == label, axis=1)        # labelに一致する行がTrueとなる配列を作成
            indexes = np.where(match)[0]              # Trueとなっている行のインデックス（要素番号）を取得

            class_indexes = np.where(label)[0]         # labelの中で、1となっている要素番号の配列を作成
            size = len(indexes) // len(class_indexes)  # 1クラス当たりの分割サイズ

            # 所属クラス毎に、ラベルを分配（複数のクラスに属するラベルの場合、若干のロスが出るがここでは無視）
            for i, val in enumerate(class_indexes):    # 要素番号と要素を取り出す
                s = size * i
                e = size * (i + 1)
                index_each_class[val] += list(indexes[s : e])  # 複数クラスに属するデータはそれぞれに分割する
        #        if len(class_indexes) >= 2:
        #            print("len(indexes), size, s, e: ", len(indexes), size, s, e)
        #print("index_each_class: ", index_each_class)

        # 要素数の調整
        min_amount = len(y) // class_num       # それぞれのクラスが必要な最低限の要素数
        for i in range(len(index_each_class)):
            indexes = index_each_class[i]
            if len(indexes) < min_amount:      # 配列の要素数がeach_access_timesよりも小さい場合
                coe = int(np.ceil(min_amount / len(indexes)))   # 配列のサイズを増やす係数
                index_each_class[i] = list(indexes) * coe       # 配列のサイズを増やす（中身を繰り返す）

        return index_each_class


    def _get_address_map_for_balance(self, index_each_class, size, use_class_rate=1.0):
        """ クラス間のバランスを取ったアドレスマップを返す
        画像の少ないクラスに対してはオーバーサンプリングすることになる。画像の多いクラスに対してはアンダーサンプリングとなる。
        この関数を適宜呼び出して、アンダーサンプリングとなっているクラスの画像を適宜入れ替えてください。
        index_each_class: 2d-ndarray or list<list>, クラス毎の画像のインデックスが入ったリスト（配列）
        size: int, 教師画像の全体数
        use_class_rate: float, 学習に使う画像に含ませるクラスの割合
        """

        class_num = len(index_each_class)       # クラス数を数える
        count = 0
        while True: 
            use = np.random.rand(class_num) <= use_class_rate    # 抽出するクラスを選ぶ. useは[True, False, True]の様な、バイナリの配列となる
            if np.sum(use) >= 2:                 # 最低でも２クラスが含まれるようにする
                break
            if count > 10:
                raise ValueError(f"value of use_class_rate={use_class_rate} is too small.")
            count += 1
        each_access_times = size // np.sum(use)   # 返すリストに含まれる1クラスあたりの画像数

        # アクセスマップを作成
        address_map = []                        # アドレスマップ
        for i in np.where(use)[0]:              # 抽出されたクラスの要素番号を取り出しつつ処理
            indexes = index_each_class[int(i)]  # i番目のクラスのインデックスリストを取り出す
            np.random.shuffle(indexes)          # 同じアクセス順とならないようにシャッフル
            scale = each_access_times // len(indexes) + 1
            indexes_ = list(indexes) * scale    # 足りない分は繰り返す
            address_map += list(indexes_[:each_access_times])

        # 要素数が足りない分を付け足す
        if len(address_map) < size:
            diff = size - len(address_map)
            numbers = np.arange(0, size)   # 0からsize-1までの連番を作成
            np.random.shuffle(numbers)     # シャッフル
            address_map += list(numbers[:diff])   # 適当に付け足す

        # この関数の返すアドレスマップ単体で使われた場合に、バッチ内の画像が偏るのを防止するために、シャッフル
        np.random.shuffle(address_map)

        return address_map



    def flow(self, x, y, save_to_dir=None, save_format=None, batch_size=10, shuffle=True, balancing=False):
        """
        x: ndarray, 0-1.0に正規化された画像が複数入っていることを想定
        y: ndarray, 0. or 1.を成分とするベクトルを格納した配列（2次元配列）。正解ラベルを想定
        shuffle: bool, Trueのとき、1 epoch分の画像を生成したら、画像の順番をランダムに変更する。
        balancing: bool, Trueのとき、クラス毎の画像生成確率を均一とする。Trueのとき、実質的にshuffle==Trueと同じである。
        """

        # map2の更新頻度を決める
        if self.balancing_refresh_iteration is None:
            self.balancing_refresh_iteration = len(x) // batch_size    # 1 epoch分の学習を行うと更新するようにセット

        # ラベルスムージング
        if 0 < self.label_smoothing_e < 1:
            y = y * (1 - 2 * self.label_smoothing_e) + self.label_smoothing_e

        # 第1のアドレスマップの作成（アクセス順をシャッフルする際に役立つ）
        address_map = np.arange(0, len(x))   # アクセス先のインデックスを管理する配列
        if shuffle:
            np.random.shuffle(address_map)   # アクセス先をシャッフル

        # 第2のアドレスマップの作成（クラス間のバランス（学習回数を等しくするの）に役立つ）
        address_map2 = np.copy(address_map)
        if balancing:
            index_each_class = self._get_indexes_for_class_balance(y)
            address_map2 = self._get_address_map_for_balance(index_each_class, len(y), self.use_class)


        def get_img(index):
            """ 指定された画像のシャローコピーを返す
            高速化優先で、コピーは作らない。
            index: int, 取得する画像の要素番号
            """
            n = address_map[index]
            m = address_map2[n]
            img = x[m]
            label_vect = y[m]
            return img, label_vect

        i = 0
        i_backup = 0
        count_for_refresh = 0

        while True:
            x_ = []
            y_ = []
            for k in range(batch_size):   # バッチサイズ分、ループ
                j = (i + k) % len(x)      # 取得する画像のインデックス（要素番号を作成）
                img_origin, label_origin = get_img(j)  # j番目の画像を取得（実際にはxのj番目とは限らない）
                img1 = img_origin.astype(np.float64)   # 単なるnumpyの行列計算なら32bitが速いが、64bitでの計算がより速い
                flag_label = False
                
                # --画像の加工--
                # 回転
                if self.rotation_range != 0 and np.random.rand() < 0.5:
                    theta = np.random.randint(-self.rotation_range, self.rotation_range)
                    img1 = rotate(img1, theta)     # skimageのライブラリを使った場合、サイズは変わらないがfloat64になる。scipyのrotateはサイズが変わる
                
                # 水平方向に反転
                if self.horizontal_flip and np.random.rand() < 0.5:
                    img1 = np.fliplr(img1)
                
                # 垂直方向に反転
                if self.vertical_flip and np.random.rand() < 0.5:
                    img1 = np.flipud(img1)

                # 横方向にshift
                if self.width_shift_range:  # Noneじゃない
                    img1 = horizontal_shift(img1, self.width_shift_range, self.channel)

                # 縦方向にshift
                if self.height_shift_range:  # Noneじゃない
                    shift, ignore = self.height_shift_range
                    img1 = vertical_shift(img1, shift, self.channel, ignore)
                
                # 他の画像と合成
                mixup_th, mix_amount, invalid_label = self.mixup
                if not(isinstance(invalid_label, np.ndarray) and np.inner(label_origin, invalid_label)) and mixup_th > np.random.rand() and mix_amount >= 2:
                    if mix_amount > 2:
                        mix_amount = np.random.randint(2, mix_amount + 1)  # 3以上が指定されていた場合は、実際に合成する数を乱数で決める
                    ratios = np.random.rand(mix_amount)   # 合成比をつくる
                    ratios /= np.sum(ratios)              # 全部足して1.0になるように調整
                    
                    img_ = img1 * ratios[0]               # 元画像に合成比をかける
                    label_vect1 = label_origin * ratios[0]  # ラベルにも合成比をかける
                    
                    j, k = 1, 0
                    while j < mix_amount:
                        k += 1
                        r = ratios[j]
                        n = np.random.randint(0, len(x))   # 合成する画像のインデックス
                        img_mix, label_mix = get_img(n)    # 合成する画像を1枚とりだす
                        
                        if isinstance(invalid_label, np.ndarray) and np.inner(label_mix, invalid_label):  # mixupを指示されていないクラスなら、無視する
                            if k > 1000:
                                print("--mixup loop warning--, too many loop times...")
                                break
                            continue

                        img_ = img_ + img_mix * r   # 合成
                        label_vect1 = label_vect1 + label_mix * r

                        j += 1

                    # 合成できていれば、反映させる
                    if j == mix_amount:
                        flag_label = True
                        img1 = img_

                # 切り出し（トリミング）
                if self.crop and np.random.rand() < 0.5:
                    img1 = random_crop(img1)

                # 輝度を変える
                if self.brightness_shift_range != 0:
                    r = 1.0 + (2 * np.random.rand() - 1.0) * self.brightness_shift_range
                    if r < 0.1:   # 負や極端に小さい値にならない様にする
                        r = 0.1
                    img1 = img1 * r
                
                # 輝線を加える（縦か横方向のみ）
                if self.bright_line is not None:
                    for amp, axis, th in self.bright_line:
                        if np.random.rand() > th:
                            continue
                        mean = np.random.rand()
                        std = (np.random.rand() + 0.01) / 10    # 標準偏差（線の太さを決める指標）は最大でもsigma==0.101。最小でも0.001。
                        amp *= np.random.rand() + 0.05
                        img_ = bright_line_img(shape=img1.shape, mean=mean, std=std, amp=amp, axis=axis, channel=self.channel)
                        img1 = img1 + img_

                # 引っかき傷のような縦線を加える
                if self.scratch is not None:
                    amp, times, th = self.scratch
                    if np.random.rand() < th:
                        times_ = 1
                        if times > 1:                     # 2本以上が指定されていた場合は、線の数を乱数で決める
                            times_ = np.random.randint(1, times + 1)
                        for _ in range(times_):
                            amp_ = amp * np.random.rand()
                            img1 = scratch_img(img1, amp=amp_, channel=self.channel)

                # ノイズを画像に加える
                if self.noise_std > 0 and np.random.rand() < 0.5:
                    noise_img = np.random.normal(0, self.noise_std, img1.shape)
                    img1 = img1 + noise_img   # ノイズを加える

                # 背景の輝度を調整する
                if self.back_ground:
                    th, amp = self.back_ground
                    if np.random.rand() < th:
                        p = np.random.rand() * amp
                        img1[img1 < p] = p


                # 周波数フィルター（画像の上端や下端を滑らかに0にする）
                if self.freq_filter is not None:
                    for center, width, floor, kind, zitter, th, ignore_row in self.freq_filter:
                        if np.random.rand() > th:
                            continue
                        center *= 1 + np.random.randn() * zitter
                        width = int(width * (1 + np.random.randn() * zitter))
                        floor *= 1 + np.random.randn() * zitter
                        mask = filter_mask(shape=img1.shape, center=center, width=width, floor=floor, filter_kind=kind, channel=self.channel)
                        img_ = img1 * mask
                        if isinstance(ignore_row, int):  # 無視したい行があれば、元の値を書き込む
                            img_[ignore_row] = img1[ignore_row]
                        img1 = img_

                # ランダムに一部にマスクをかける
                if self.random_erasing:
                    th, box_scale = self.random_erasing
                    if np.random.rand() < th:
                        img1 = random_erasing(img1, box_scale)

                # RGB画像を前提として、BとGにに一様輝度勾配の画像を埋め込む
                if self.insert_gradient == True:
                    img1 = insert_gradient(img1)

                # 輝度の変動を右端に埋め込む
                if self.insert_std == True:
                    img1 = insert_std(img1)

                # ラベルに対して何の処理も行われなかった場合の対応
                if flag_label == False:
                    label_vect1 = label_origin.copy()

                # 値の制限
                img1 /= np.max(img1)  # 最大を1とする
                img1[img1 < 0] = 0.0  # 最小は0

                # リサイズ
                if x[0].shape[:2] != self.shape[:2]:    # channel lastが前提で書いている
                    img1 = resize(img1, self.shape[:2])

                # 保存
                if save_to_dir is not None and save_format is not None:
                    img3 = img1 * 255   # 別の変数として保存しないと、x_に影響する
                    img3 = img3.astype(np.uint8)
                    pilImg = Image.fromarray(np.uint8(img3))
                    pilImg.save("{0}/{1}_{2}_hoge.{3}".format(save_to_dir, i, k, save_format))

                # 返り値のリストに格納
                img1 = img1.astype(np.float16)    # メモリ節約のため、型を小さくする
                x_.append(img1)
                y_.append(label_vect1)


            # 処理結果を返す
            if self.return_type == "ndarray":
                yield np.array(x_), np.array(y_)
            else:
                yield x_, y_


            # 次のアクセスに備えた処理
            i += batch_size
            if shuffle and (i // len(x)) > 0:       # シャッフルが指定されていて、更に1順を検知した場合
                # アドレスマップの更新
                np.random.shuffle(address_map)

                # init
                i = i % len(x)


            # クラス間のバランス取り用のアドレスも更新（balancing==Trueなのにそのままだと、学習されない画像がでる）
            count_for_refresh += 1
            if balancing and count_for_refresh >= self.balancing_refresh_iteration:
                address_map2 = self._get_address_map_for_balance(index_each_class, len(y), self.use_class)
                count_for_refresh = 0
                    
                # 確認用（普段はコメントアウト）
                #print(address_map2)
                #import matplotlib.pyplot as plt
                #plt.hist(address_map2)
                #plt.show()
                #plt.plot(address_map2)
                #plt.show()




##  ImageDataGenerator end  ################################################




def load_save_images(read_func, param, validation_rate=0.1, save_dir="."):
    """ 画像の読み込みと教師データの作成と保存を行う
    read_func: function, 画像を読み込む関数
    param: dict<str: obj>, read_funcに渡すパラメータ 
    validation_rate: float, 検証に使うデータの割合
    load_dir: str, 保存するフォルダ
    """
    # 画像を読み込む
    x, yv, weights_dict, label_dict, output_dim, file_names = read_func(param)
    x_train, y_train, x_test, y_test, test_file_names = split2(x, yv, validation_rate, file_names)  # データを学習用と検証用に分割

    if "preprocess_func" in param:   # 必要なら前処理
        preprocess_func = param["preprocess_func"]  # func, 前処理を行う関数
        x_train = preprocess_func(x_train)
        x_test = preprocess_func(x_test)

    # 保存設定（pramに入っていなければ、全部保存する様に設定）
    if "save_flags" not in param: # 設定がなかったら、デフォルトの設定を行う
        flags =  {"x_train": True, "y_train": True, "x_test": True, "y_test": True, 
                  "weights": True, "label": True, "param": True, "test_names": True}
    else:
        flags = param["save_flags"]
    
    # 再利用のために、ファイルに保存しておく
    ## フォルダの準備
    os.makedirs(save_dir, exist_ok=True)

    ## 保存
    if flags["x_train"] and len(x_train) > 0:   # 学習用の画像
        np.save(os.path.join(save_dir, 'x_train.npy'), x_train)    # 保存をsavez_compressed()に変えて、拡張子をnpyからnpzにすると圧縮するが、超遅い。
    if flags["y_train"] and len(y_train) > 0:   # 学習用画像の正解ラベル
        np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    if flags["x_test"] and len(x_test) > 0:     # 検証用画像
        np.save(os.path.join(save_dir, 'x_test.npy'), x_test)
    if flags["y_test"] and len(y_test) > 0:     # 検証用画像の正解ラベル
        np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
    if flags["weights"]:                        # クラス毎の重み（画像数の逆数的数）
        with open(os.path.join(save_dir, 'weights_dict.pickle'), 'wb') as f:  
            pickle.dump(weights_dict, f)
    if flags["label"]:                          # ラベル辞書（番号とクラス名の辞書）
        with open(os.path.join(save_dir, 'label_dict.pickle'), 'wb') as f:
            pickle.dump(label_dict, f)
    if flags["param"]:                          # paramのバイナリを保存
        with open(os.path.join(save_dir, 'param.pickle'), 'wb') as f:
            pickle.dump(param, f)
    if flags["test_names"]:                     # 検証画像のファイル名をバイナリで保存
        with open(os.path.join(save_dir, 'test_names.pickle'), 'wb') as f:
            pickle.dump(test_file_names, f)

    return x_train, x_test, weights_dict, label_dict, y_train, y_test, output_dim, test_file_names



def main():
    data_format = "channels_last"

    # pattern 1, flower
    dir_names_dict = {"yellow":["flower_sample/1"], 
                      "white":["flower_sample/2"],
                      "yellow, white":["flower_sample/3"],   # マルチラベルな画像
                      } 

    param = {"dir_names_dict":dir_names_dict, 
             "data_format":data_format, 
             "size":(32, 32), 
             "mode":"RGB", 
             "resize_filter":Image.NEAREST, 
             "preprocess_func":preprocessing2,
             "save_flags": {"x_train": True, "y_train": True, "x_test": True, "y_test": True, "weights": True, "label": True, "param": True, "test_names": True},
             }
    x_train, x_test, weights_dict, label_dict, y_train, y_test, output_dim, test_file_names = load_save_images(read_images3, param, validation_rate=0.2)

    # 確認
    print("test_file_names: ", test_file_names)
    print("y_test", y_test)

    # pattern 1, animal
    #dir_names_dict = {"cat":["sample_image_animal/cat"], 
    #                  "dog":["sample_image_animal/dog"]} 
    #param = {"dir_names_dict":dir_names_dict, "data_format":data_format, "size":(32, 32), "mode":"RGB", "resize_filter":Image.NEAREST, "preprocess_func":preprocessing}
    #x_train, x_test, weights_dict, label_dict, y_train, y_test, output_dim =load_save_images(read_images1, param, validation_rate=0.2)

    # pattern 2, animal
    #dir_names_list = ["sample_image_animal/cat", "sample_image_animal/dog"]
    #name_dict = read_name_dict("sample_image_animal/file_list.csv")
    #param = {"dir_names_list":dir_names_list, "name_dict":name_dict, "data_format":data_format, "size":(32, 32), "mode":"RGB", "resize_filter":Image.NEAREST, "preprocess_func":preprocessing}
    #x_train, x_test, weights_dict, label_dict, y_train, y_test, output_dim =load_save_images(read_images2, param, validation_rate=0.2)
    

    # 教師データを無限に用意するオブジェクトを作成
    datagen = MyImageDataGenerator(
        random_erasing=(0.2, (0.02, 0.25)),   # 確率、箱の大きさの範囲
        #mixup=(0.2, 3, None),                # 確率、合成数、無視するラベル
        mixup=(0.5, 3, np.array([0, 0])),
        width_shift_range=0.3,                # 横方向のシフト率
        height_shift_range=(0.05, False),     # 縦方向のシフト率、上端のセルを無視するかどうか
        shape=(32, 32, 3),
        brightness_shift_range=0.1,           # 輝度の変化
        bright_line=[(0.8, 2, 0.3)],          # 輝線のパラメータ. 強度、方向（縦・横）、確率
        scratch=(0.8, 3, 0.3),                # 引っかき傷（1本の縦線）のパラメータ。強度、線の最大数、確率
        freq_filter=[(0.9, 4, 0.05, "high_pass", 0.05, 0.3, None), (0.1, 3, 0.05, "low_pass", 0.05, 0.3, 0)],   # 周波数フィルタのパラメータ。位置0-1、ピクセル幅、最小値0-1、種類、変動の大きさ、確率
        insert_gradient=True,                 # RGB画像を前提として、BとGにに一様輝度勾配の画像を埋め込む
        insert_std = True,                    # 横方向の輝度の変動を埋め込む
        )

    gen = datagen.flow(x_train, y_train, batch_size=10, shuffle=True, balancing=True)  # シャッフルは順序によらない学習のために重要
    for _ in range(1):
        images, labels = gen.__next__()
        for img, label in zip(images, labels):
            print("y: ", label)
            img *= 255
            img[img > 255] = 255
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.show()



if __name__ == "__main__":
    main()


