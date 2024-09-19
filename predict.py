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
#  2024-03-14        今年の1月に作成したrealtime_prediction202312c.pyから、Discriminatorを移植した。predict_classes_topN()が増加している。
#                    Grad-CAMによる識別結果の可視化機能も追加した。
#  2024-03-16        listでフォルダやファイルのパスを複数並べるのをやりやすく変更した。
#                    libs.sound_imageをバージョン10に切り替えて、スペクトログラムを作る前に生波形にノイズを加えることが可能になった。
#  2024-03-17        predict_images()内の関数も対応した。この関数については動作未検証。
#                    軽微なバグを修正した。
#  2024-03-19        軽微なバグを修正した。
#                    引数で処理対象のファイルをリスト形式で指定できるようにして、更にCAMの有効・無効も制御できるようにした。
#  2024-04-01        引数処理のバグを修正
#  2024-04-11        複数の音源ファイルを予測する場合に、残りの処理時間を表示するようにした。
#  2024-07-05        CPUしか使えない場合に処理速度を上げるために、並列処理が可能なように変更
#                    以下、試験結果
#   Test result =======================
#    Test env. CPU: Core i7-11700K, GPU: RTX3060
#     CPUのみで処理した場合
#      並列処理を明示せず 186 sec
#      4スレッド　143 sec
#     GPUを利用して処理した場合
#      2スレッド 176 sec
#      4スレッド 172 sec
#      並列処理を明示せず 1回目　119 sec
#      並列処理を明示せず 2回目　115 sec
#   ===================================
#                   テストに使った音源には極端に大きいものもあったので、CPUのみの並列処理時間は実際にはもう少し早いと思う。
#                   結論としては、GPUが使えるなら並列処理にしないほうが速い。これ以上の速度が必要なら、計算機を複数使用したほうがいい。
#                   １台のマシンでGPUが１つしかないなら、プロセスは分割しないほうがいい。
#  2024-08-20       音源の読み込みに失敗した場合にエラーではなくレポートを出力するように変更した。
#  2024-09-19       runsフォルダが無い場合に作るように変更
# author: Katsuhiro Morishita
# created: 2019-10-29
import sys, os, re, glob, copy, time, pickle, pprint, argparse, traceback, ast, time, random
from multiprocessing import Process, Pipe
from datetime import datetime as dt
from datetime import timedelta as td
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import librosa
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import yaml
import unicodedata, hashlib

import libs.sound_image10 as si





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



# Grad-CAMによる識別結果の可視化（判定に強く利用された画像野領域を可視化）
# https://keras.io/examples/vision/grad_cam/
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()



# ヒートマップを重ねて保存する。
# ref: https://keras.io/examples/vision/grad_cam/
def superimpose_gradcam(img, heatmap, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    return superimposed_img



class Discriminator:
    """ 画像に対する識別を行うクラス
    kerasによるニューラルネットワークでの識別を行います。
    """
    def __init__(self, model_path="model.hdf5", label_path="label_dict.pickle", 
                       img_size=(32, 32, 3), th=0.2, save_dir=".", custom_objects={},
                       CAM=None):
        """ 
        model_path: str, 結合係数も一緒に保存されたモデルへのパス
        label_path: str, 識別結果（整数）をラベル（名称）に変換する辞書へのパス
        img_size: tuple, 識別機用の画像のサイズ. (width, height, channel)
        CAM: dict, 最期の畳み込み層の名前。CAMによるヒートマップを保存する場合に指定すること。
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
        self.model_name = name.replace(".", "").replace("_", "")   # ドット.とアンダーバー_は無視する
        self.fname_result = ""
        self.fname_likelihoods = ""

        # 保存先のフォルダを作成
        if save_dir != "." and save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        # その他
        self.fw_result = None
        self.fw_likeliboods = None

        # ラベルを変換する辞書（index番号からクラス名を調べる辞書）の読み込み
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

        # 識別結果の可視化の準備（判定に強く利用された画像野領域を可視化）
        self.CAM_ready = False
        if CAM is not None and CAM["enable"] == True:
            label_dict_swap = {v: k for k, v in self.label_dict.items()}  # 番号から名前を引く辞書から、名前から番号を引く辞書をつくる
            self.CAM_index = label_dict_swap[CAM["class_name"]]           # 名前に紐づいた番号
            self.CAM_last_conv_layer_name = CAM["last_conv_layer_name"]   # 最期の畳み込み層の名前
            #self.CAM_model = tf.keras.models.clone_model(self.model)      # モデルをコピー
            #self.CAM_model.layers[-1].activation = None                   # 最期の層の活性化関数を削除　（必要なの？）
            self.CAM_model = self.model
            self.CAM_savedir = os.path.join(save_dir, "CAM")              # 保存先のフォルダ
            os.makedirs(self.CAM_savedir, exist_ok=True)                  # 保存先のフォルダを作っておく
            self.CAM_ready = True                                         # CAMの保存の準備ができたことをフラグで示す

        # GPUは初回の動作が遅いので、ダミー画像を予め処理させておく
        w, h, c = img_size
        self.dummy_img = np.zeros((1, h, w, c))  # batch, height, width, channel
        self.predict_dummy()
        self.predict_dummy()


    def init_save_files(self, identifer=""):
        """ ファイルの初期化（項目名を入れる）
        """
        self.fname_result = os.path.join(self.save_dir, f"prediction_result_{self.model_name}{identifer}.csv")
        self.fname_likelihoods = os.path.join(self.save_dir, f"prediction_likelihoods_{self.model_name}{identifer}.csv")

        c_names = self.class_names
        with open(self.fname_result, "w", encoding="utf-8-sig") as fw:  # 尤度を閾値で二値化した識別結果を保存するファイル
            names = ["class{}".format(i) for i in range(len(c_names) - 1)]    # ラベルのNDを無視するために、-1
            txt = ",".join(names)
            fw.write("{},{},{},{}\n".format("fname", "s", "w", txt))

        with open(self.fname_likelihoods, "w", encoding="utf-8-sig") as fw:    # 尤度のファイル
            names = c_names[:-1]    # 最後の名前はNDなので、削る
            txt = ",".join(names)
            fw.write("{},{},{},{}\n".format("fname", "s", "w", txt))


    def predict_classes(self, x: np.ndarray, tags=None, save=False, image_name_lambda=None):
        """ 最大尤度となったクラスラベルと尤度のリストを返す
        注意：それぞれの画像に対してラベルは1つだが、predict_classes2()との整合性を保つために、predicted_classesは2次元配列となっている。
        x: ndarray, 識別したい画像のリスト
        save: bool, 識別結果をcsvで保存したい場合はTrue
        image_name_lambda: lambda, CAMを保存する際につける名前を作る関数
        """
        result_raws = self.model.predict(x, batch_size=len(x), verbose=0) # クラス毎の尤度を取得。 尤度の配列がレコードの数だけ取得される
        result_list = [len(arr) if np.max(arr) < self.th else arr.argmax() for arr in result_raws]  # 最大尤度を持つインデックスのlistを作る。ただし、最大尤度<thの場合は、"ND"扱いとする
        predicted_classes = np.array([[self.label_dict[class_id]] for class_id in result_list])   # 予測されたclass_local_idをラベルに変換

        # 識別結果の保存を指示されていた場合
        if tags is not None and save == True:
            self.__save(tags, results=predicted_classes, likelihoods=result_raws)

        # 識別の根拠となった部分を画像として保存する
        if self.CAM_ready:
            result_list_ = np.array(result_list)
            target_result_over_th = result_list_ != (len(self.label_arr) - 1)    # 尤度が閾値を超えた画像をTrueとする配列を作成
            target_result_likelihoods = [np.max(arr) for arr in result_raws]    # 最大尤度値の配列を作成

            # save class activation heatmap
            self.__save_CAM(x, target_result_over_th, target_result_likelihoods, predicted_classes, image_name_lambda)


        return predicted_classes, result_raws


    def predict_classes2(self, x: np.ndarray, tags=None, save=False, image_name_lambda=None):
        """ 尤度が閾値を超えたクラスラベルと尤度のリストを返す
        x: ndarray, 識別したい画像のリスト
        tags: list<str>, ファイルに結果とともに書き込む文字列
        save: bool, 識別結果をcsvで保存したい場合はTrue
        image_name_lambda: lambda, CAMを保存する際につける名前を作る関数
        """
        result_raws = self.model.predict(x, batch_size=len(x), verbose=0) # クラス毎の尤度を取得。 尤度の配列がレコードの数だけ取得される
        result_list = [arr >= self.th for arr in result_raws]             # 尤度が閾値を超えた要素をTrueとしたlistを作る。
        result_list2 = [list(y) + [True] if np.sum(y) == 0 else list(y) + [False] for y in result_list]  # NDの分の処理
        predicted_classes = [self.label_arr[class_ids] for class_ids in result_list2]     # 予測されたclass_idをラベルに変換


        # 識別結果の保存を指示されていた場合
        if tags is not None and save == True:
            self.__save(tags, results=predicted_classes, likelihoods=result_raws)

        # 識別の根拠となった部分を画像として保存する
        if self.CAM_ready:
            result_list_ = np.array(result_list)
            target_result_over_th = result_list_[:, self.CAM_index]   # 論理反転もそのうち必要かも
            target_result_likelihoods = result_raws[:, self.CAM_index]
            labels = [self.label_dict[self.CAM_index]] * len(x)       # ラベルの配列

            # save class activation heatmap
            self.__save_CAM(x, target_result_over_th, target_result_likelihoods, labels, image_name_lambda)

        return predicted_classes, result_raws


    def predict_classes_topN(self, x: np.ndarray, N: int, tags=None, save=False):
        """ 尤度順にラベルと尤度をN個格納した配列を返す
        x: ndarray, 識別したい画像のリスト
        tags: list<str>, ファイルに結果とともに書き込む文字列
        save: bool, 結果を保存するならTrue
        """
        predicted_classes, result_raws = self.predict_classes2(x, tags, save) # クラス毎の尤度を取得。 尤度の配列がレコードの数だけ取得される
        

        # top Nを集計
        result = []

        for arr in result_raws:
            # 尤度が大きい順に並べる
            index_ = np.argsort(arr)[::-1][:N]    # 大きい順に並べたときのインデックス（要素番号）を取得
            likelihoods = np.sort(arr)[::-1][:N]  # 大きい順に並べる

            #print(index_)
            #print(likelihoods)

            # 基準値以上のものを抽出
            n = np.sum(likelihoods > self.th)     
            if n > 0:
                index_ = index_[:n]
                likelihoods = likelihoods[:n]
            else:
                index_ = [len(arr)]
                likelihoods = [0.0]

            classes = [self.label_dict[class_id] for class_id in index_]   # 予測されたclass_local_idをラベルに変換

            result.append([(x, y)   for x, y in zip(classes, likelihoods)])

        return result


    def __save(self, tags, results=None, likelihoods=None):
        # ファイルの準備
        if self.fw_result is None and self.fname_result != "":
            self.fw_result = open(self.fname_result, "a", encoding="utf-8-sig")

        if self.fw_likeliboods is None and self.fname_likelihoods != "":
            self.fw_likeliboods = open(self.fname_likelihoods, "a", encoding="utf-8-sig")


        # 結果の保存
        if results is not None and len(results) > 0:
            for tag, result in zip(tags, results):
                num = len(result)
                txt = "," * (len(self.class_names) - num - 1)  # 足りないラベルの分、カンマを作る。ただし、NDがあるので、１引く。
                labels = ",".join(result) + txt
                self.fw_result.write("{},{}\n".format(tag, labels))

        #print(likelihoods)
        if likelihoods is not None and len(likelihoods) > 0:
            for tag, likelihood in zip(tags, likelihoods):
                likelihood = [str(x)  for x in likelihood]
                likelihood_txt = ",".join(likelihood)
                self.fw_likeliboods.write("{},{}\n".format(tag, likelihood_txt))



    def __save_CAM(self, x, over_th, likelihoods, labels, fname_generator):
        """ CAM画像を保存する
        """
        for i, x_ in enumerate(x):
            if over_th[i]:
                # ヒートマップと、それを重ねた画像を作成
                heatmap = make_gradcam_heatmap(np.array([x_]), self.CAM_model, self.CAM_last_conv_layer_name)
                x_ = (x_ * 255).astype(np.uint8)
                img_ = superimpose_gradcam(x_, heatmap)

                # ファイル名を作成
                if fname_generator is None:
                    save_name = f"{labels[i]}_{i:04d}_th{p:.2f}.png"
                else:
                    p = likelihoods[i]
                    save_name = fname_generator(i, p, labels[i])

                # 画像を保存
                img_.save(os.path.join(self.CAM_savedir, save_name))




    def close_file(self):
        """ 開いていたファイルを閉じる
        可能であれば、エラー処理でも呼び出してほしい。
        """
        if self.fw_result:
            self.fw_result.close()

        if self.fw_likeliboods:
            self.fw_likeliboods.close()

        self.fw_result = None
        self.fw_likeliboods = None
        self.fname_result = ""
        self.fname_likelihoods = ""



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




# sound_image10.pyより移植
def get_location_ID(file_path):
    """ 音源ファイルの親フォルダ名から、録音場所等を識別するための略称を作って返す
    location_Aなどは、2017年以降に沖縄で録音した場所を基準として命名したものである。
    単純なので画像ファイルの区別には便利だが、単純すぎたのでIDだけでは録音年度を特定できないので注意して欲しい。
    """
    file_path = os.path.abspath(file_path)
    location = "unknown"
    
    # location_Aやlocation_B1などを判別
    p = r"location_(?P<loc>[A-Z]+\d*)"
    m = re.search(p, file_path)
    if m:
        location = m.group("loc")


    # CDの音源ファイルを区別するために、locationをフォルダ名から作る
    if location == "unknown":
        field = re.split(r"\\|¥|/", file_path)    # NASの中のフォルダ名はos.basename()では得られなかったので、区切り文字で切断
                                                  # dir_name = os.path.basename(os.path.dirname(fpath))てな感じでも良かったかも
        if len(field) >= 2:       # 指定されたファイルがフォルダ内だった場合
            dir_name = field[-2]
        else:                     # 分割できなかった場合
            dir_name = field[0]

        dir_name = unicodedata.normalize("NFC", dir_name)  # MacはNFD形式でファイル名を扱うので、Windowsと同じNFC形式に変換
        hs = hashlib.md5(dir_name.encode()).hexdigest()
        location = str(len(dir_name)) + hs[:4]

    return location




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

        if len(imgs) != 0:
            # 画像の前処理
            for f in setting["preprocess_chain"]:
                imgs = [f(img, setting) for img in imgs]    # 前処理
            imgs2 = np.array(imgs)  # ndarrayに型を変える

            # 取得した画像を識別・結果の保存
            for dis in discriminators:
                dis.predict_classes2(imgs2, 
                                     tags, 
                                     save=True, 
                                     image_name_lambda=lambda a, b, c=0.0: f"{dis.model_name}_{c}_{fnames[a + i]}_LF{b:.2f}.png")

        else:
            print("return None")
            break

        i = end_index

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

    # 読み込みに失敗した場合の処理
    if wav is None:
        msg = f"wav data of '{fname}' is None."
        print(msg)
        return msg

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

        #print(f"imgs length is {len(imgs)}.")

        if len(imgs) != 0:
            # 画像の前処理
            for f in setting["preprocess_chain"]:
                imgs = [f(img, setting) for img in imgs]    # 前処理
            imgs2 = np.array(imgs)  # ndarrayに型を変える

            # 保存するファイル名に入れる文字列を作成
            name, ext = os.path.splitext(os.path.basename(fname))    # ファイル名と拡張子を分割
            loc = get_location_ID(fname)
            file_id = f"{loc}__{name}"

            # 取得した画像を識別・結果の保存
            for dis in discriminators:
                dis.predict_classes2(imgs2, 
                                     tags, 
                                     save=True, 
                                     image_name_lambda=lambda a, b, c=0.0: f"{dis.model_name}__{file_id}__{(a + i) * sw:.1f}_{w:.1f}_{c}_LF{b:.2f}.png")

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
                    pil_img.save(f'save_img/save_{file_id}_{(k + i) * sw:.1f}_test.png')    # 画像ファイルとして保存  

        else:
            print("return None")
            break

        i = end_index

        # kill.txtを見つけたら、終了する。安全に終了させるため。単独で動かしている場合は不要かもだが。
        if os.path.exists("kill.txt"):
            break


    elapsed_time = time.perf_counter() - start                 # 処理時間計測用  （不要ならコメントアウト）
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")   # 処理時間計測用  （不要ならコメントアウト）


    return ""




def read_models(dir_path, img_size, th, model_format=".hdf5", label_pattern="label*.pickle", 
                save_dir=".", custom_objects={}, CAM=None):
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
                                     CAM=CAM,
                                     ) 
                        for model_path in models]

    return discriminators_






def predict_sub(fnames, setting, conn1=None, conn2=None, id_=0):
    """ 予測処理を行う
    """
    # 識別器の準備
    models = setting["models"]
    if models == "all":   # カレントディレクトリ内の全ディレクトリ直下の場合
        dirs = os.listdir() + ["."]
    elif models == "last train":         # 最後の学習フォルダを指定された場合
        dirs = [last_dirpath("train")]   # 最後に保存されている学習結果の保存フォルダ内のモデルを探させる
    else:
        dirs = models   # ディレクトリパスのリストの場合

    discriminators = []
    for dir_path in dirs:
        dis_ = read_models(dir_path, setting["size"], setting["th"], 
                        model_format=setting["model_format"], 
                        label_pattern=setting["label_pattern"], 
                        save_dir=setting["save_dir"],
                        custom_objects=setting["custom_objects"],
                        CAM=setting["CAM"]
                        )
        discriminators += dis_

    if len(discriminators) == 0:
        print(f"ID {id_}: 0 models or 0 label dictionary founded. Check your setting file.")
    else:
        print(f"ID {id_}: {len(discriminators)} models founded.")


    
    # 識別器があれば、予測処理開始
    if len(discriminators) != 0:
        # 保存用のファイルを用意
        for dis in discriminators:
            identifer = ""
            if id_ >= 0:
                identifer = f"_ID{id_}"
            dis.init_save_files(identifer=identifer)

        # ファイルを処理
        ts = time.time()
        if setting["mode"] == "sound":            # 音声ファイルを予測
            for i, fname in enumerate(fnames):    # 音声ファイルはサイズが大きいので、音声ファイルごとに処理
                # 残りの処理時間を表示
                if i >= 1:
                    tn = time.time()
                    last_time = td(seconds=((tn - ts) / i * (len(fnames) - i)))
                    print(f">>>{i + 1}/{len(fnames)}, pass:{int(tn - ts)} sec., last:{last_time}.")

                try:             # たまにエラーが発生するので、その対応
                    res = predict_sound(fname, setting, discriminators)  # 予測の実行
                    if res != "":
                        with open(f"report_predict_ID{id_}.log", "a", encoding="utf-8-sig") as fw:
                            fw.write("### {} ###\n".format(str(dt.now())))
                            fw.write(f"{fname}\n")
                            fw.write(f"{res}\n")
                except Exception as e:
                    print("### error ###")
                    # エラー状況をファイルに残す
                    with open(f"error_predict_ID{id_}.log", "a", encoding="utf-8-sig") as fw:
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


    # （可能なら）親プロセスに終了をお知らせ
    if conn1 is not None:
        conn1.send(f"id:{id_} is fin.")




def predict_main(setting):
    """ 並列処理による予測処理を行う
    プロセスの作成には時間がかかるので、1プロセスしか実行しない場合はこの関数ではなく、predict_sub()を呼び出した方が良い。
    """
    # 処理対象音源の分割
    process_num = setting["process_num"]     # 並列にする数
    if process_num < 0:
        process_num = 1
    fnames = sorted(setting["file_names"])    # 処理対象ファイルをソート
    print(fnames[:10])

    size = len(fnames) // process_num
    fnames_list = [fnames[i*size : (i+1)*size] for i in range(process_num)]
    if size * len(fnames_list) < len(fnames):
        fnames_list[-1] += fnames[size * len(fnames_list):]  # 余った分は最期のリストに加える

    # 子プロセスの作成と実行
    child_process = []
    for i in range(process_num):
        parent_conn, child_conn = Pipe(True)    # 子プロセスとの間の通信に使うオブジェクトの作成
        child = Process(target=predict_sub, args=(fnames_list[i], setting, parent_conn, child_conn, i))
        child_process.append((parent_conn, child_conn, child))
        child.start()

    # 子プロセスの終了を待つ
    fin_count = 0
    while fin_count < process_num:
        for i in range(process_num):
            conn1, conn2, c = child_process[i]
            #print(conn1.poll(), conn2.poll())

            if conn2.poll():
                recv = conn2.recv()

                print("子プロセスからの受信:{}\n".format(recv))
                c.join()
                print(f"process {i} is fin.\n")

                fin_count += 1



def fusion_results(save_dir, pattern):
    """ 予測結果のファイルの統廃合
    """
    # 処理対象のファイルを探す
    fnames_all = glob.glob(os.path.join(save_dir, "*/" + pattern))
    if len(fnames_all) == 0:
        return

    # フォルダの一覧を作成
    dirs = [os.path.basename(os.path.dirname(fpath)) for fpath in fnames_all]
    dirs = natsorted(set(dirs))

    for dir_ in dirs:
        fnames = natsorted([name for name in fnames_all if dir_ in name])
        if len(fnames) == 0:
            continue

        # 複数ファイルのテキストを結合
        lines = []
        head = ""
        for fname in fnames:
            with open(fname, "r", encoding="utf-8-sig") as fr:
                lines_ = fr.readlines()
                lines_ = [x.rstrip() for x in lines_]   # 残っている余計な改行コードを削除
                head = lines_[0]
                lines += lines_[1:]
        lines = [head] + lines

        # 保存するファイル名を作成
        basename = os.path.basename(fnames[0])
        fname_new = basename.split("_ID")[0] + ".csv"
        fpath_new = os.path.join(save_dir, dir_, fname_new)

        # 保存
        with open(fpath_new, "w", encoding="utf-8-sig") as fw:
            fw.write("\n".join(lines))

        # 元ファイルを削除
        for fpath in fnames:
            os.remove(fpath)







# https://note.nkmk.me/python-bool-true-false-usage/#bool-bool
def strtobool(val):
    """Convert a string representation of truth to true or false.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value {!r}".format(val))



def arg_parse(params):
    """ 引数の解析と、設定の上書き
    """
    parser = argparse.ArgumentParser(description="略")

    # 引数の追加
    parser.add_argument("-f", "--files")
    parser.add_argument("-g", "--gpu")
    parser.add_argument("-t", "--tag")
    parser.add_argument("-ce", "--cam_enable")
    parser.add_argument("-ct", "--cam_target")

    # 引数を解析
    args = parser.parse_args()
    
    # 設定に反映
    if args.gpu: params["GPU"] = strtobool(args.gpu)
    if args.files:    # 処理対象ファイルの指定への対応（現時点ではglobにのみ対応）。スクリプトの引数内では、\は\\としてエスケープすること。
        value = args.files
        print("***", value, "***")
        if "glob.glob" == value[:9]:
            index = value.find(")")
            if index > 0:
                order = value[:index + 1]
                v = eval(order)
                params["file_names"] = sorted(v)
        elif value[0] == "[" and value[-1] == "]":
            index = value.find("]")
            if index > 0:
                order = value[:index + 1]
                v = ast.literal_eval(order)      # -f "[r'./a/fuga.mp3', r'E:\fuga.wav', 'c']"
                params["file_names"] = sorted(v)

    if args.tag: params["tag"] = str(args.tag)
    if args.cam_enable: params["CAM"]["enable"] = strtobool(args.cam_enable)
    if args.cam_target: params["CAM"]["class_name"] = args.cam_target

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
            "noise": 0.0,          # 音源の波形に加えるノイズの大きさ。波形の標準偏差が基準。
            }
    params["th"] = 0.9             # 判定に用いる尤度
    params["preprocess_chain"] = [preprocessing]
    params["save_img"] = False     # デバッグ用に、画像を作成
    params["lr"] = ""              # "left" or "right" or other
    params["tag"] = ""               # フォルダに付ける名前
    params["CAM"] = {"enable":False,    # 識別結果の根拠を可視化する場合はTrue
                     "class_name": "", 
                     "last_conv_layer_name":""}  # 最期の畳み込み層の名前。CAMによるヒートマップを保存する場合に指定すること。

    params["process_num"] = 1       # 予測演算に使用するプロセスの数。1なら並列演算処理を実行しない。
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

            # listで指定されたファイルやフォルダのパスの場合
            if isinstance(value, list):
                value_ = []
                for x in value:
                    if isinstance(x, str) and re.match(r"r('|\")(\w|:|/|\\|\.|_|\-|\+|。)+('|\")", x):
                        x = x[2:-1]   # 先頭のrとダブるコートまたはシングルコートをとる
                        x = x.replace("\\", "/")  # バックスラッシュをスラッシュに置換
                        value_.append(x)
                    else:
                        value_.append(x)
                obj[key] = value_

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

    
    # 保存先の親フォルダを作成
    os.makedirs("runs", exist_ok=True)  # 保存先のフォルダを作成

    # 保存先のフォルダ名の準備
    save_dir = next_dirpath("predict")   # 結果の保存先
    if setting["tag"] != "":
        save_dir += f"_{setting['tag']}"
    setting["save_dir"] = save_dir

    # 保存先のフォルダを作成
    if save_dir != "." and save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    # 設定の保存（後でパラメータを追えるように）
    now_ = dt.now().strftime('%Y%m%d%H%M')
    fname = os.path.join(save_dir, "predict_setting_{}.yaml".format(now_))
    with open(fname, 'w', encoding="utf-8-sig") as fw:
        yaml.dump(setting, fw, encoding='utf8', allow_unicode=True)


    # 予測処理を実行
    if setting["process_num"] > 1:
        predict_main(setting)

        # 予測結果のファイルの統廃合
        fusion_results(save_dir, "prediction_result_*ID*.csv")
        fusion_results(save_dir, "prediction_likelihoods_*ID*.csv")
    else:
        predict_sub(sorted(setting["file_names"]), setting, id_=-1)


    # 修了処理
    tf.keras.backend.clear_session()
    print("proccess is finished.")


if __name__ == "__main__":
    main()

