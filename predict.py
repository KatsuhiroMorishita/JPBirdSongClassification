#!/usr/bin/python
# -*- coding:utf-8 -*-
# writen for python 3
# purpose: mp3などの音源や、画像を識別する。
# history:
#  2020-12-20 ver.2  保存するファイルの先頭に項目名を入れることとした
#                    sound_image5.pyのrawオプションにも対応
#                    コメント内容も修正・追加
#                    predict_classes()の仕様をpredict_classes2()に合わせて変更。
#                    preprocessing()の実装方法を汎用性のために変更
#  2021-03-06 ver.3  sound_image6に差し替え。他の影響はなし。
#  2021-03-12 ver.4  バッチサイズを指定して、複数の画像をいっぺんに予測するように変更した。GPUを使うと若干高速化した。
#  2021-03-15 ver.5  画像ファイルを識別する機能を付けた。カラー画像には対応していないが、研究の範囲では問題ない。
#  2021-03-22 ver.6  GPUを使うか、CPUを使うかを設定ファイルで制御できるようにした。
#  2021-04-03 ver.7  beta版リリース。複数のモデルを指定できるようにした。現時点では非効率処理となっている。
#  2021-04-04        複数のモデルを使って効率的に処理できるように構造を変えた。課題はあるが、正式リリースとする。
#  2021-04-05 ver.8  ラベル違いのモデルでも、フォルダに小分けにしておけば自動的に読み込まれる様に変更した。
#  2021-07-18 ver.9  kerasをtensorlowの中のものを使うように変更  
#  2021-07-27        軽微なデバッグ（CPU版は動作確認）。変数名の意味が合わないので、overlap_rateをshift_rateに変更した。 
#  2021-10-19 ver.10 yamlによる設定ファイルの読み込みに変更し、少し安全になった。
#  2021-10-28        設定を保存するようにした。また、多数のモデルを指定した場合のバグを修正・フォルダを分けて保存することにも対応した。
#                    前回の学習結果を使って予測するモードも付けたので、PowerShellなどで学習から予測までを連続的に実施できるようになった。
#                    （以前からやれていたけど、いちいちモデルのコピーとかしなくていいのでスクリプトがシンプルになるはず）
#  2021-11-04        GPUの設定方法が変わったようなので、一旦コメントアウト
#  2021-12-11 ver.11 CPUを使いたいことがあるので、予測処理にCPUを使えるように変更。スクリプトに対して引数を指定できるように変更した。
#  2021-12-15        音源のファイルフォーマットが不正な場合にエラーで止まらないように、try構文を追加した。ログも保存する。
#  2022-01-06        sound_image8に変更。 yamlの保存をunicode形式に変更。
#  2022-11-06        sound_image9に変更
#                    前処理を複数の関数を連続して呼び出すことで実現できるように変更し、image_preprocessing12.pyで
#                    実装した新しいAugmentationにも対応した。
#  2022-12-14        既存のフォルダ探索に使う正規表現でsearchからmatchに変更した。これで文字列先頭の一致をチェックするようにした。
#  2022-12-20        スペクトログラム作成において、lrやcut_band等の新しい設定に対応し、デバッグ用の画像保存処理をデバッグ
#  2022-12-29        独自の損失関数を指定できるようにした。ついでに、カスタムオブジェクトにも対応。
#  2023-05-20        SavedModel形式の保存方式にも対応。独自の損失関数を使った場合の学習で実際に使った関数を用意しなくても済むように変更した。
#  2023-08-30        予測フォルダの整理のために、フォルダ名に任意の文字列をつけれるようにした。
#  2023-10-02        保存フォルダに名前を付ける機能に対応して、last_dirnumber(), last_dirpath(), next_dirpath()を修正。
# author: Katsuhiro Morishita
# created: 2019-10-29
import sys, os, re, glob, copy, time, pickle, pprint, argparse, traceback
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import librosa
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import yaml

import libs.sound_image9 as si





def print2(*args):
    """ いい感じにstrやdictやlistを表示する
    """
    for arg in args:
        if isinstance(arg, dict) or isinstance(arg, list):
            pprint.pprint(arg)
        else:
            print(arg)


def last_dirnumber(pattern="train", root=r'./runs'):
    """ 既に保存されている学習・予測フォルダ等の最大の番号を取得
    root: str, 探索するディレクトリへのパス
    """
    path_dir = ""
    dirs = os.listdir(path=root)

    max_ = 0
    for d in dirs:
        path_dir_ = os.path.join(root, d)
        if os.path.isdir(path_dir_):
            m = re.match(r"{}(?P<num>\d+)".format(pattern), d)
            if m:
                num = int(m.group("num"))
                if max_ < num:
                    max_ = num
                    path_dir = path_dir_
                
    return max_, root, path_dir


def last_dirpath(pattern="train", root=r'./runs'):
    """ 最大の番号を持つ既に保存されている学習・予測フォルダ等のパスを返す
    root: str, 探索するディレクトリへのパス
    """
    max_, path_root, path_dir = last_dirnumber(pattern, root)
    return path_dir


def next_dirpath(pattern="train", root=r'./runs'):
    """ 学習・予測フォルダ等を新規で作成する場合のパスを返す
    root: str, 探索するディレクトリへのパス
    """
    max_, path_root, path_dir = last_dirnumber(pattern, root)
    return os.path.join(path_root, pattern + str(max_ + 1)) 



class Discriminator:
    """ 画像に対する識別を行うクラス
    kerasによるニューラルネットワークでの識別を行います。
    """
    def __init__(self, model_path="model.hdf5", label_path="label_dict.pickle", img_size=(32, 32, 3), th=0.2, save_dir=".", custom_objects={}):
        """ 
        model_path: str, 結合係数も一緒に保存されたモデルへのパス
        label_path: str, 識別結果（整数）をラベル（名称）に変換する辞書へのパス
        img_size: tuple, 識別機用の画像のサイズ. (width, height, channel)
        """
        self.model = load_model(model_path, custom_objects=custom_objects)  # model
        self.model.summary()     # モデル構造の表示（モデルの状態を表示させないと入出力画像フォーマットが不明なことがある）
        self.th = th             # 識別時の尤度の閾値

        # 識別結果を保存するファイル名
        if os.path.isdir(model_path):
            name = os.path.basename(model_path)
        else:
            dir_path, fname = os.path.split(model_path)
            name, ext = os.path.splitext(fname)    # モデルのファイル名から拡張子を除いたものを取得
        self.fname_result = os.path.join(save_dir, "prediction_result_{}.csv".format(name))
        self.fname_likelihoods = os.path.join(save_dir, "prediction_likelihoods_{}.csv".format(name))

        # 保存先のフォルダを作成
        if save_dir != "." and save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        # その他
        self.fw_result = None
        self.fw_likeliboods = None

        # ラベルを変換する辞書の読み込み
        self.label_dict = None
        with open(label_path, 'rb') as f:
            self.label_dict = pickle.load(f)  # オブジェクト復元. dict

            # NDというラベルがなければ、追加する
            values = self.label_dict.values()
            if "ND" not in values:
                self.label_dict[len(self.label_dict)] = "ND"
            print2("label: ", self.label_dict)
        
        # 辞書を基に、配列も作成しておく
        label_arr_ = []
        key_values = list(self.label_dict.keys())
        for i in np.arange(0, np.max(key_values) + 1):
            if i in self.label_dict:
                label_arr_.append(self.label_dict[i])
            else:
                label_arr_.append("unknown")
        self.label_arr = np.array(label_arr_)

        # GPUは初回の動作が遅いので、ダミー画像を予め処理させておく
        w, h, c = img_size
        self.dummy_img = np.zeros((1, h, w, c))  # batch, height, width, channel
        self.predict_dummy()
        self.predict_dummy()


    def init_save_files(self):
        """ ファイルの初期化（項目名を入れる）
        """
        c_names = self.class_names
        with open(self.fname_result, "w", encoding="utf-8-sig") as fw:  # 尤度を閾値で二値化した識別結果を保存するファイル
            names = ["class{}".format(i) for i in range(len(c_names) - 1)]    # ラベルのNDを無視するために、-1
            txt = ",".join(names)
            fw.write("{},{},{},{}\n".format("fname", "s", "w", txt))

        with open(self.fname_likelihoods, "w", encoding="utf-8-sig") as fw:    # 尤度のファイル
            names = c_names[:-1]    # 最後の名前はNDなので、削る
            txt = ",".join(names)
            fw.write("{},{},{},{}\n".format("fname", "s", "w", txt))


    def predict_classes(self, x):
        """ 最大尤度となったクラスラベルと尤度のリストを返す
        注意：それぞれの画像に対してラベルは1つだが、predict_classes2()との整合性を保つために、predicted_classesは2次元配列となっている。
        x: ndarray, 識別したい画像のリスト
        """
        result_raws = self.model.predict(x, batch_size=len(x), verbose=0) # クラス毎の尤度を取得。 尤度の配列がレコードの数だけ取得される
        result_list = [len(arr) if np.max(arr) < self.th else arr.argmax() for arr in result_raws]  # 最大尤度を持つインデックスのlistを作る。ただし、最大尤度<thの場合は、"ND"扱いとする
        predicted_classes = np.array([[self.label_dict[class_id]] for class_id in result_list])   # 予測されたclass_local_idをラベルに変換

        return predicted_classes, result_raws


    def predict_classes2(self, x):
        """ 尤度が閾値を超えたクラスラベルと尤度のリストを返す
        x: ndarray, 識別したい画像のリスト
        """
        result_raws = self.model.predict(x, batch_size=len(x), verbose=0) # クラス毎の尤度を取得。 尤度の配列がレコードの数だけ取得される
        result_list = [arr >= self.th for arr in result_raws]             # 尤度が閾値を超えた要素をTrueとしたlistを作る。
        result_list2 = [list(y) + [True] if np.sum(y) == 0 else list(y) + [False] for y in result_list]  # NDの分の処理
        predicted_classes = [self.label_arr[class_ids] for class_ids in result_list2]     # 予測されたclass_idをラベルに変換

        return predicted_classes, result_raws


    def predict_classes2_with_save(self, x: np.ndarray, tags: list):
        """ 予測とその結果の保存を行う。返値はない。
        x: ndarray, 識別したい画像のリスト
        tags: list<str>, ファイルに結果とともに書き込む文字列
        """
        # ファイルの準備
        if self.fw_result is None:
            self.fw_result = open(self.fname_result, "a", encoding="utf-8-sig")

        if self.fw_likeliboods is None:
            self.fw_likeliboods = open(self.fname_likelihoods, "a", encoding="utf-8-sig")

        # 識別
        results, likelihoods = self.predict_classes2(x)
        #print(results)

        # 結果の保存
        for tag, result in zip(tags, results):
            num = len(result)
            txt = "," * (len(self.class_names) - num - 1)  # 足りないラベルの分、カンマを作る。ただし、NDがあるので、１引く。
            labels = ",".join(result) + txt
            self.fw_result.write("{},{}\n".format(tag, labels))

        for tag, likelihood in zip(tags, likelihoods):
            likelihood = [str(x)  for x in likelihood]
            likelihood_txt = ",".join(likelihood)
            self.fw_likeliboods.write("{},{}\n".format(tag, likelihood_txt))

        return


    def close_file(self):
        """ 開いていたファイルを閉じる
        可能であれば、エラー処理でも呼び出してほしい。
        """
        if self.fw_result:
            self.fw_result.close()
            self.fw_result = None

        if self.fw_likeliboods:
            self.fw_likeliboods.close()
            self.fw_likeliboods = None



    def predict_dummy(self):
        """ ダミーの画像を処理
        たまに動かさないとGPUの反応が鈍いので作成した。
        """
        #print("dummy test")
        self.predict_classes(self.dummy_img)


    @property
    def class_names(self):
        """ クラスの名前一覧を返す。
        名前の順番はpredict_classes()が返す尤度の順番でもある。
        """
        return [self.label_dict[key_] for key_ in sorted(self.label_dict.keys())]




def read_image(fpath):
    """ 画像ファイルをグレースケールで読み込んで返す（返す画像は2次元配列）
    fpath: str, ファイルのパス
    """
    # ファイルの簡易チェック
    root, ext = os.path.splitext(fpath)  # 拡張子を取得
    if ext != ".jpg" and ext != ".bmp" and ext != ".png":
        return
    fname = os.path.basename(fpath)
    if fname[0] == ".":          # Macが自動的に生成するファイルを除外
        return

    # 画像の読み込み
    img = Image.open(fpath)     # 画像の読み込み
    img = img.convert("L")      # 画像のモード変換。　mode=="LA":透過チャンネルも考慮して、グレースケール化, mode=="RGB":Aを無視
    img = np.array(img)         # ndarray型の多次元配列に変換
    img = img.astype(np.uint8)  # 型の変換

    return img





def preprocessing(img, params):
    """ 画像の前処理（リサイズなど）を行い、前処理済みの画像を格納した配列をndarray型で返す
    img: ndarray, 画像1枚分のndarray型オブジェクト. 輝度値が0-255であること。画像は2次元配列でチャンネルがないことを前提とする。
    
    """
    size = params["size"]     # size: tuple<int, int, int>, 画像のサイズ(Width, Height, channel)
    w, h, c = size

    # リサイズ
    pil_img = Image.fromarray(img)    # リサイズできるように、pillowのImageに変換
    img2 = pil_img.resize((w, h))  # リサイズ。リサイズにはチャンネルの情報は不要
    #img2.save('img2.png')            # for debug
    img3 = np.asarray(img2)
    
    # チャンネル数の調整（これでできるのはchannels_lastの構造となった画像）
    img4 = np.dstack([img3] * c)  # img3が2次元配列の画像（チャンネルがない）ことを前提に、チャンネル分重ねる

    # 画素の輝度値を最大1.0とする
    return img4 / 255




# image_preprocessing11.pyより移植・ちょっと改編
def preprocessing2(img, params):
    """ RGB画像であることを前提に、BとGに一様勾配の画像を挿入する。
    img: ndarray type image
    """
    channel = "channels_last"
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

        # チャンネル数の調整（これでできるのはchannels_lastの構造となった画像）
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


# image_preprocessing12.pyより移植・ちょっと改編
def preprocessing3(img, params):
    """ 画像の右端に、輝度の変動の強さを1ピクセル幅で表す
    """
    m = np.std(img, axis=1)  # 画像の横方向に標準偏差を求める
    img[:, -1] = m           # 画像の右端に埋め込む
    return img




def predict_images(fnames, setting, discriminators):
    """ 画像ファイルを識別し、その結果を保存する
    fnames: str, 画像ファイルのパスが入ったリスト
    setting: dict, 設定を格納した辞書
    discriminators: list<Discriminator>, 識別器のリスト
    """
    start = time.perf_counter()        # 処理時間計測用  （不要ならコメントアウト）

    # 変数の取り出し
    batch_size = setting["batch_size"]

    # 識別
    i, i_max = 0, len(fnames)     # 処理番号とその最大値（以下ではiが最大値に達してはならない）

    while i < i_max:
        print(i, "/", len(fnames))
        imgs = []
        tags = []
        
        # 読み込み
        # batch_size分の画像を読み込む（ただし、i_maxの方が小さい場合はi_max分）
        end_index = min([i_max, i + batch_size])
        for j in range(i, end_index):
            fname = fnames[j]
            img = read_image(fname)

            if img is not None:   # 並列化しても、ここは順序を守った方がいい
                fpath = os.path.abspath(fname)
                imgs.append(img)
                tags.append("{},{},{}".format(fpath, float("nan"), float("nan")))

        i = end_index

        if len(imgs) != 0:
            # 画像の前処理
            for f in setting["preprocess_chain"]:
                imgs = [f(img, setting) for img in imgs]    # 前処理
            imgs2 = np.array(imgs)  # ndarrayに型を変える

            # 取得した画像を識別・結果の保存
            for dis in discriminators:
                dis.predict_classes2_with_save(imgs2, tags)
        else:
            print("return None")
            break

        # kill.txtを見つけたら、終了する。安全に終了させるため。単独で動かしている場合は不要かもだが。
        if os.path.exists("kill.txt"):
            break

    elapsed_time = time.perf_counter() - start                 # 処理時間計測用  （不要ならコメントアウト）
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")   # 処理時間計測用  （不要ならコメントアウト）




def predict_sound(fname, setting, discriminators):
    """ 音源ファイルを識別し、その結果を保存する
    fname: str, 音源ファイルのパス
    setting: dict, 設定を格納した辞書
    discriminators: list<Discriminator>, 識別器のリスト
    """
    start = time.perf_counter()        # 処理時間計測用  （不要ならコメントアウト）

    # 変数の取り出しなど
    shift_rate = setting["shift_rate"]
    w = setting["window_size"]
    sw = w * shift_rate     # スライドさせる時間幅[s]
    batch_size = setting["batch_size"]
    params = setting["imagegen_params"]   # スペクトログラム作成上のパラメータ
    sr = params["sr"]
    params["hop_length"] = int(setting["hop"] * sr)   # SFFTのスライドポイント数を計算

    # 音源ファイルの読み込み
    print("--read file--", fname)
    setting["sr"] = sr
    wav, sr, _ = si.load_sound(fname, setting)
    #print("wav len and sampling rate: ", len(wav), sr)   # for debug


    # 少しずつ切り出して、画像化して識別
    i, i_max = 0, int(len(wav) / sr / sw)     # 処理番号とその最大値（以下ではiが最大値に達してはならない）
    #print(f"i_max:{i_max}")   # for debug

    while i < i_max:
        imgs = []
        tags = []

        # batch_size分の画像を作成する（ただし、i_maxの方が小さい場合はi_max分）
        end_index = min([i_max, i + batch_size])
        for j in range(i, end_index):
            s, e = j * sw, j * sw + w       # 切り出し区間[s]
            ss, ee = int(s*sr), int(e*sr)   # 切り出すインデックス
            b = wav[ss : ee]                # 音源の切り出し
            #print("b len, ", len(b), ss, ee)   # for debug
            params["data"] = b
            img = si.get_melspectrogram_image(**params)  # スペクトログラム画像を作成

            if img is not None:   # 並列化しても、ここは順序を守る必要がある
                fpath = os.path.abspath(fname)
                imgs.append(img)
                tags.append("{},{},{}".format(fpath, s, e-s))

        i = end_index
        #print(f"imgs length is {len(imgs)}.")

        if len(imgs) != 0:
            # 画像の前処理
            for f in setting["preprocess_chain"]:
                imgs = [f(img, setting) for img in imgs]    # 前処理
            imgs2 = np.array(imgs)  # ndarrayに型を変える

            # 取得した画像を識別・結果の保存
            for dis in discriminators:
                dis.predict_classes2_with_save(imgs2, tags)

            # 画像を保存する（デバッグ用）
            if setting["save_img"]:
                os.makedirs("save_img", exist_ok=True)
                for k, img in enumerate(imgs):
                    img = img - np.min(img)   # 輝度の正規化
                    img = img / np.max(img)
                    img = img * 255
                    img[img < 0] = 0             # 輝度値の下限制限
                    img[img > 255] = 255         # 輝度値の上限制限
                    img = img.astype(np.uint8)    # 画像として扱えるように、型を変える
                    pil_img = Image.fromarray(img)        # 保存できるように、pillowのImageに変換  
                    pil_img.save(f'save_img/save_{os.path.basename(fname)}_{i}_{k}_test.png')    # 画像ファイルとして保存  
        else:
            print("return None")
            break

        # kill.txtを見つけたら、終了する。安全に終了させるため。単独で動かしている場合は不要かもだが。
        if os.path.exists("kill.txt"):
            break


    elapsed_time = time.perf_counter() - start                 # 処理時間計測用  （不要ならコメントアウト）
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")   # 処理時間計測用  （不要ならコメントアウト）




def read_models(dir_path, img_size, th, model_format=".hdf5", label_pattern="label*.pickle", save_dir=".", custom_objects={}):
    """ 指定されたフォルダ内のモデルを読み込む。返値は識別機オブジェクトの配列。
    1フォルダにつき有効なラベルは1つだけ。
    """
    # ラベルの辞書のパスを取得
    labels = glob.glob(os.path.join(dir_path, label_pattern))
    if len(labels) == 0:
        return []
    label_path = labels[0]   # 複数のラベルがあっても、1つに絞る（原則、1つのフォルダに1つのラベルを保存すること）

    # モデルのパスを取得（複数OK）
    model_pattern = "*" + model_format
    models = glob.glob(os.path.join(dir_path, model_pattern))
    print("--------------", dir_path, models, model_pattern)
    if len(models) == 0:
        return []

    # 識別器をモデルの数だけ作成
    discriminators_ = [Discriminator(model_path=model_path, 
                                     label_path=label_path, 
                                     img_size=img_size, 
                                     th=th, 
                                     save_dir=os.path.join(save_dir, os.path.basename(os.path.dirname(model_path))),
                                     custom_objects=custom_objects,
                                     ) 
                        for model_path in models]

    return discriminators_





def arg_parse(params):
    """ 引数の解析と、設定の上書き
    """
    parser = argparse.ArgumentParser(description="略")

    # 引数の追加
    parser.add_argument("-f", "--files")
    parser.add_argument("-g", "--gpu")
    parser.add_argument("-t", "--tag")

    # 引数を解析
    args = parser.parse_args()

    # 設定に反映
    if args.gpu: params["GPU"] = bool(args.gpu)
    if args.files:    # 処理対象ファイルの指定への対応（現時点ではglobにのみ対応）。スクリプトの引数内では、\は\\としてエスケープすること。
        value = args.files
        print("***", value, "***")
        if "glob.glob" == value[:9]:
            index = value.find(")")
            if index > 0:
                order = value[:index + 1]
                v = eval(order)
                params["file_names"] = sorted(v)
    if args.tag: params["tag"] = str(args.tag)

    return params






def set_default_setting():
    """ デフォルトの設定をセットして返す
    """
    params = {}
    params["file_names"] = []      # 処理対象の音源・画像のパスのリスト
    params["targets"] = []         # 処理対象の音源・画像のフォルダやファイルパスパターンのパスのリスト（ディレクトリでの指定や、複数指定はこちらを使う）
    params["models"] = "all"       # 予測に使用するモデルのあるディレクトリ名の リスト or "all" or "last train"。"all"だと、カレントディレクトリとそのサブディレクトリ直下を探す。
    params["model_format"] = ".hdf5"  # モデルの形式。.hdf5, .h5, SavedModel
    params["loss"] = "binary_crossentropy"  # 損失関数（独自の定義関数がなければ、無視してよい）
    params["custom_objects"] = {}      # 独自の活性化関数や損失関数を格納するカスタムオブジェクト
    params["label_pattern"] = "label*.pickle"  # ラベルの名前パターン
    params["mode"] = "sound"       # 処理モード（画像imageか、音源soundか）
    params["GPU"] = True           # TrueだとGPUを使用する
    params["batch_size"] = 1       # バッチサイズ
    params["size"] = (32, 32, 1)   # 予測にかける画像のサイズ。最後の1はチャンネル。設定ファイルではlist型として記述すること。
    params["window_size"] = 5      # 音声の切り出し幅[s]
    params["hop"] = 0.025          #: int, 時間分解能[s]
    params["load_mode"] = "kaiser_fast",    #: str, librosa.load()でres_typeに代入する文字列。読み込み速度が変わる。kaiser_fastとkaiser_bestではkaiser_fastの方が速い。
    params["shift_rate"] = 1.0     # 音源のスライド量。0.5だと、半分重ねる。1だとw分ずらす。2だと2w分ずらす（処理量は半分）。
    params["imagegen_params"] = {  # スペクトログラムを作成する関数への指示パラメータ
            "sr": 44100,           #: float, 音源を読み込む際のリサンプリング周波数[Hz]
            "fmax": 10000,         #: int, スペクトログラムの最高周波数[Hz]。fmax < sr / 2でないと、警告がでる。
            "top_remove": 0,               #: int, 作成したスペクトログラムの上部（周波数の上端）から削除するピクセル数。フィルタの影響を小さくするため。
            "n_mels": 128,         #: int, 周波数方向の分解能（画像の縦方向のピクセル数）
            "n_fft": 2048,         #: int, フーリエ変換に使うポイント数
            "raw": False,          # Trueだと音圧情報をスペクトログラムの一番上のセルに埋め込む
            }
    params["th"] = 0.9             # 判定に用いる尤度
    params["preprocess_chain"] = [preprocessing]
    params["save_img"] = False     # デバッグ用に、画像を作成
    params["lr"] = ""              # "left" or "right" or other
    params["tag"] = ""
    return params



def read_setting(fname):
    """ 設定を読み込む。返り値は辞書で返す
    eval()を使っているので、不正なコードを実行しないように、気をつけてください。
    """
    param = set_default_setting()

    with open(fname, "r", encoding="utf-8-sig") as fr:
        obj = yaml.safe_load(fr)
        
        for key, value in obj.items():
            # globを使ったファイル検索命令への対応
            if isinstance(value, str) and "glob.glob" == value[:9]:
                index = value.find(")")
                if index > 0:
                    order = value[:index + 1]
                    v = eval(order)
                    obj[key] = v

            # 単なるlistがあれば、tupleにしておく
            #if isinstance(value, list):
            #    obj[key] = tuple(value)
        param.update(**obj)   # 辞書の結合 （Python 3.9以降なら記述を簡単にできる）

    # このモジュール特有の処理
    ## 指定されたフォルダ内で識別可能なファイルを探す。
    files = []
    for target in param["targets"]:
        if os.path.isdir(target):      # ディレクトリが指定されていた場合
            if param["mode"] == "image":
                exts = [".jpeg", ".jpg", ".png", ".bmp"]
            elif param["mode"] == "sound":
                exts = [".mp3", ".wav"]
            for ext in exts:
                p = target + "/**/*" + ext
                files += glob.glob(p)     # サブディレクトリも探す
                p = target + "/*" + ext
                files += glob.glob(p)     # 指定されたディレクトリ直下も探す
        elif "*" in target:               # *が含まれる文字列だったら。
            files += glob.glob(target)    # globを使ったパターン検索とみなす

    #print("============================")
    #print(files)

    param["file_names"] += files
    param["file_names"] = sorted(list(set(param["file_names"])))   # 重複排除


    ## 前処理関数の設定
    funcs = []
    for func_name in param["preprocess_chain"]:
        if func_name == "preprocessing":
            funcs.append(preprocessing)
        elif func_name == "preprocessing2":
            funcs.append(preprocessing2)
        elif func_name == "preprocessing3":
            funcs.append(preprocessing3)

    if len(funcs) > 0:
        param["preprocess_chain"] = funcs


    # lossの処理（学習に使用した損失関数の関数名は必要。学習はしないので中身は適当でよい。）
    if "local." == param["loss"][:6]:
         func_name = param["loss"][6:]    # 先頭文字列を削る
         func_name = func_name[:30]       # 文字数制限

         # カスタムオブジェクトとして登録しておく
         param["custom_objects"][func_name] = None


    return param





def main():
    # 設定を読み込み
    setting = read_setting("predict_setting.yaml")

    # 引数のチェック
    setting = arg_parse(setting)
    print2("\n\n< setting >", setting)

    # 対象ファイルのチェック
    if len(setting["file_names"]) == 0:
        print("-- error: file is not found. check file path. --\n")
        exit()

    # 読み込むファイルの拡張子をチェック
    for f in setting["file_names"]:
        root, ext = os.path.splitext(f)
        ext = ext.lower()
        mode = setting["mode"] 
        if mode == "sound" and ("jpg" in ext or "jpeg" in ext or "bmp" in ext or "png" in ext):
            print(f, "is not match to mode. The mode is {}".format(mode))
            exit()
        if mode == "image" and ("mp3" in ext or "wav" in ext):
            print(f, "is not match to mode. The mode is {}.".format(mode))
            exit()

    # GPUを使用するか、決める
    if setting["GPU"]:
        pass   # GPUがあれば、勝手に使用される
    else:
        # CPUを使う場合
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # 識別器の準備
    discriminators = []
    models = setting["models"]
    if models == "all":   # カレントディレクトリ内の全ディレクトリ直下の場合
        dirs = os.listdir() + ["."]
    elif models == "last train":         # 最後の学習フォルダを指定された場合
        dirs = [last_dirpath("train")]   # 最後に保存されている学習結果の保存フォルダ内のモデルを探させる
    else:
        dirs = models   # ディレクトリパスのリストの場合
    
    # 保存先のフォルダ名の準備
    save_dir = next_dirpath("predict")   # 結果の保存先
    if setting["tag"] != "":
        save_dir += f"_{setting['tag']}"

    for dir_path in dirs:
        dis_ = read_models(dir_path, setting["size"], setting["th"], 
                        model_format=setting["model_format"], 
                        label_pattern=setting["label_pattern"], 
                        save_dir=save_dir,
                        custom_objects=setting["custom_objects"])
        discriminators += dis_


    # 設定の保存（後でパラメータを追えるように）
    now_ = dt.now().strftime('%Y%m%d%H%M')
    fname = os.path.join(save_dir, "predict_setting_{}.yaml".format(now_))
    with open(fname, 'w', encoding="utf-8-sig") as fw:
        yaml.dump(setting, fw, encoding='utf8', allow_unicode=True)

    
    # 識別器があれば、処理開始
    if len(discriminators) != 0:
        # 保存用のファイルを用意
        for dis in discriminators:
            dis.init_save_files()

        # 予測
        fnames = sorted(setting["file_names"])    # 処理対象ファイルをソート
        print(fnames[:10])

        # ファイルを処理
        if setting["mode"] == "sound":       # 音声ファイルを予測
            for fname in fnames:             # 音声ファイルはサイズが大きいので、音声ファイルごとに処理
                try:             # たまにエラーが発生するので、その対応
                    predict_sound(fname, setting, discriminators)  # 予測の実行
                except Exception as e:
                    print("### error ###")
                    # エラー状況をファイルに残す
                    with open("error_predict.log", "a", encoding="utf-8-sig") as fw:
                        fw.write("### {} ###\n".format(str(dt.now())))
                        fw.write(f"{fname}\n")
                        fw.write(f"cause: {e}\n")
                        fw.write("{}\n".format(traceback.format_exc()))
        elif setting["mode"] == "image":  # 画像を予測
            predict_images(fnames, setting, discriminators)
        


        # ファイルを閉じる
        for dis in discriminators:
            dis.close_file()
    else:
        print("There is no discriminator.")

    # 修了処理
    tf.keras.backend.clear_session()
    print("proccess is finished.")


if __name__ == "__main__":
    main()

