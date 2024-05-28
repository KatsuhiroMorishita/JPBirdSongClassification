# purpose: 画像の学習サンプルプログラム
# history:
#  2021-10-20 yamlによる設定ファイルの読み込みに対応し、利用しやすくなった。
#  2021-10-22 良く変わるパラメータをコマンドライン引数で指定できるようにした。内容は設定ファイルよりも優先される。
#  2021-10-30 学習結果の保存で、retryで上書きされるファイルはサブディレクトリに保存するように変更
#  2021-11-01 check pointでモデルを保存する先をフォルダに変更した。これで学習後の予測処理がやりやすくなった。
#  2022-01-06 mlcore11とimage_preprocessing9に変更。これで教師数が少ないクラスがあってもエラーにはならない。yamlの保存をunicode形式に変更。
#  2022-01-09 警告が出ていたので、ModelCheckpoint()の引数periodをsave_freqに変更。
#             fit()に渡すImageDataGeneratorをマルチコア対応のものに差し替えた。メモリが不足して使えないが。
#  2022-11-06 mlcore14, image_preprocessing12に切り替え
#  2022-12-20 mlcore15, image_preprocessing13に切り替え
#  2022-12-23 mlcore16, image_preprocessing14に切り替え
#  2023-05-23 SavedModel形式の読み込みに対応にともない、mlcore17, image_preprocessing15に切り替え
#  2023-08-05 設定ファイルの読み込み時の対応を少し変えた。中途半端かも。
#  2023-08-30 教師データを格納したフォルダのパス確認を追加
#  2023-10-02 mlcore18.pyのplot_history()とsave_history()の修正に対応. Ubuntu対応のために、パスの区切り文字を/に変更。
#             predict.pyの保存フォルダに名前を付ける機能に対応して、last_dirnumber(), last_dirpath(), next_dirpath()を修正。
#  2023-10-11 build_model_local()内でのモデル作成方法を変更（試し）
#  2023-11-18 モデル作成方法がうまくいっていなかったので、再修正（うまくいった）
#  2023-12-20 tensorflowjs形式でも保存するようにしてみた。未検証なので上手くいくか不明。パッケージの競合もあるらしく、上手くいかない可能性もある。
#  2024-03-28 カスタム損失関数でクラスごとに学習の重みを変えることができるようにしてみた。
#  2024-03-29 1回の学習でnpyファイルが30 GB以上作られる様になったので、学習後に不要になったファイルを削除できるようにした。mlcoreを18から19に変更した。
#             これで学習条件を変えながら複数回の連続学習が可能になる。
#  2024-04-01 引数処理のバグを修正
#             引数でlossを渡せるようにした。
#  2024-04-11 複数の条件での学習を試すために、初回の学習に使用した学習データやモデルを流用できる機能を追加した。
# author: Katsuhiro MORISHITA　森下功啓
# created: 2021-10-18
import sys, os, re, glob, copy, time, pickle, pprint, argparse, ast, shutil
from datetime import datetime as dt
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from PIL import Image
import numpy as np
import gc
import cv2
import yaml

from libs.mlcore19 import *
import libs.image_preprocessing15 as ip



## 学習上の条件
np.random.seed(seed=1)



def print2(*args):
    """ いい感じにstrやdictやlistを表示する
    """
    for arg in args:
        if isinstance(arg, dict) or isinstance(arg, list):
            pprint.pprint(arg)
        else:
            print(arg)


def get_dir_name(pattern="train", root=r'./runs', index=0):
    """ 既に保存されている学習・予測フォルダ等の最大の番号を取得
    root: str, 探索するディレクトリへのパス
    """
    dirs = os.listdir(path=root)
    dirs_read = []
    
    # パターンに一致するフォルダを探す
    for d in dirs:
        path_dir_ = os.path.join(root, d)
        if os.path.isdir(path_dir_):
            m = re.match(r"{}(?P<num>\d+)".format(pattern), d)
            if m:
                num = int(m.group("num"))
                dirs_read.append((num, path_dir_))

    dirs_read = sorted(dirs_read, key=lambda number_dirname_pair: number_dirname_pair[0])  # 番号で小さい順にソート

    if len(dirs_read) > 0:
        number, path_dir = dirs_read[index]
        return number, root, path_dir
    else:
        return 0, root, ""



def first_dirpath(pattern="train", root=r'./runs'):
    """ 最小の番号を持つ既に保存されている学習・予測フォルダ等のパスを返す
    root: str, 探索するディレクトリへのパス
    """
    min_, path_root, path_dir = get_dir_name(pattern, root, 0)
    return path_dir


def last_dirpath(pattern="train", root=r'./runs'):
    """ 最大の番号を持つ既に保存されている学習・予測フォルダ等のパスを返す
    root: str, 探索するディレクトリへのパス
    """
    max_, path_root, path_dir = get_dir_name(pattern, root, -1)
    return path_dir


def next_dirpath(pattern="train", root=r'./runs'):
    """ 学習・予測フォルダ等を新規で作成する場合のパスを返す
    root: str, 探索するディレクトリへのパス
    """
    max_, path_root, path_dir = get_dir_name(pattern, root, -1)
    return os.path.join(path_root, pattern + str(max_ + 1)) 



def build_model_local(input_shape, output_dim, data_format, loss='binary_crossentropy'):
    """ 機械学習のモデルを作成する
    入力は画像、出力はラベルという構造を想定しています。
    loss: 損失関数。多クラス多ラベル分類問題の場合、通常はbinary_crossentropyを使う
    """
    print("input_shape", input_shape)
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    top_model = Sequential()
    top_model.add(Flatten())
    top_model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(output_dim))    # 出力層のユニット数はoutput_dim個
    top_model.add(Activation('sigmoid'))

    #fix weights of base_model
    for layer in base_model.layers:
        layer.trainable = False    # Falseで更新しない

    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
    opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08) # 最適化器のセット。lrは学習係数. , decay=0.0003はtensorflow2.12では不要
    #opt = SGD(learning_rate=0.00001, momentum=0.9)
    model.compile(optimizer=opt,           # コンパイル
        loss=loss,   # 損失関数
        metrics=['accuracy'])
    print(model.summary())
    return model




def build_model_local2(input_shape, output_dim, data_format, loss='binary_crossentropy'):
    """ 機械学習のモデルを作成する
    入力は画像、出力はラベルという構造を想定しています。
    build_model_localに対して、モデルの作り方が異なります。
    恐らく出力層1つ前の出力を取り出すならこちらが簡単だと思う。
    loss: 損失関数。多クラス多ラベル分類問題の場合、通常はbinary_crossentropyを使う
    """
    np.random.seed(seed=1)      # 学習条件をそろえるために乱数をリセット
    
    print("input_shape", input_shape)
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    x = base_model.output
    x = Flatten()(x)
    x = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(output_dim)(x)    # 出力層のユニット数はoutput_dim個
    x = Activation('sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)

    #fix weights of base_model
    for layer in base_model.layers:
        layer.trainable = False    # Falseで更新しない

    opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08) # 最適化器のセット。lrは学習係数. , decay=0.0003はtensorflow2.12では不要
    #opt = SGD(learning_rate=0.00001, momentum=0.9)
    model.compile(optimizer=opt,           # コンパイル
        loss=loss,   # 損失関数
        metrics=['accuracy'])
    print(model.summary())
    return model




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
    parser.add_argument("--lr")
    parser.add_argument("-r", "--retry")
    parser.add_argument("-e", "--epochs")
    parser.add_argument("-i", "--initial_epoch")
    parser.add_argument("-l", "--leranig_layerid")
    parser.add_argument("-rat", "--remove_after_train")
    parser.add_argument("--loss")
    parser.add_argument("--reuse")


    # 引数を解析
    args = parser.parse_args()

    # 設定に反映
    if args.lr: params["lr"] = float(args.lr)
    if args.retry: params["retry"] = strtobool(args.retry)
    if args.epochs: params["epochs"] = int(args.epochs)
    if args.initial_epoch: params["initial_epoch"] = int(args.initial_epoch)
    if args.leranig_layerid: params["leranig_layerid"] = int(args.leranig_layerid)
    if args.remove_after_train: params["remove_after_train"]["enable"] = strtobool(args.remove_after_train)
    if args.loss: params["loss"] = str(args.loss)
    if args.reuse: params["reuse"] = strtobool(args.reuse)


    # lossの処理
    if "local." == params["loss"][:6]:
        # "local.hogehoge_func,{a=10}"の様な文字列を想定
        lossName_lossParam = params["loss"][6:]            # 先頭文字列を削る
        func_name, loss_param_str = lossName_lossParam.split(",", 1)   # 分割
        func_name = func_name.strip()
        loss_param_str = loss_param_str.strip()
        func_name = func_name[:30]                       # 関数名の文字数制限
        loss_param = ast.literal_eval(loss_param_str)    # 辞書オブジェクトに変換

        # 重みの処理
        w = [1.0] * len(params["class_sets"])
        if "weights" in loss_param:
            weights = loss_param["weights"]   # 例："{"kuina": 1.5}"

            labels = sorted(list(params["class_sets"]))
            for label in weights:
                w[labels.index(label)] = weights[label]
        loss_param["weights"] = np.array(w)

        loss_function = gen_custom_loss(func_name, loss_param)
        params["loss"] = loss_function
        #print("------------------------", func_name, params["loss"])

        # カスタムオブジェクトとしても登録しておく
        params["custom_objects"][func_name] = loss_function

    return params



def set_default_setting():
    """ デフォルトの設定をセットして返す
    """
    params = {}
    params["image_root"] = r"./data/images/flower_sample"   # 学習したい画像の入っているフォルダ（相対パス）
    params["image_shape_wh"] = (50, 50)           # 画像のサイズ。横と縦。
    params["data_format"] = "channels_last"       # チャンネル配置。確か、"channels_last"前提で書いちゃった部分がある。
    params["class_list"] = ["white", "yellow"]    # クラス名の一覧
    params["ignore_list"] = ["ignore", "double"]  # ファイルのパスに含まれていたら無視する文字列のリスト。
    params["exchange_dict"] = {}                  # クラス名を置換したい場合に辞書で指定。例：{"gray": "white"}
    params["model_format"] = ".hdf5"   # 保存するモデルの形式. "SavedModel" or ".hdf5"
    params["epochs"] = 20              # 1つの画像当たりの学習回数
    params["batch_size"] = 10          # 1回の学習係数の修正に利用する画像数
    params["initial_epoch"] = 0        # 初期エポック数（学習を再開させる場合は0以外とする）
    params["validation_rate"] = 0.2    # 画像が大量にあるなら0.05程度でも良い。
    params["lr"] = 0.001               # 学習係数
    params["validation_th"] = 0.5      # 検証データに対する識別精度検証に使う尤度の閾値
    params["cp_period"] = 5            # 学習途中にモデルを保存する周期。5なら5 epochごとに保存する。
    params["leranig_layerid"] = 16     # このレイヤー以降の結合係数を更新する。初回の学習は大きく、２回目は6が推奨。
    params["loss"] = "binary_crossentropy"  # 損失関数
    params["datagen_params"] = {}           # image data generatorへの指示パラメータ
    params["mixup_ignores"] = ["silence"]   # mixupで無視するクラス名
    params["retry"] = False                 # 前回の学習から続ける場合はTrue
    params["reuse"] = False                 # 以前（初回）の学習に使用したデータとモデルを流用する場合はTrue
    params["custom_objects"] = {}      # 独自の活性化関数や損失関数を格納するカスタムオブジェクト
    params["onnx_output"] = True       # onnx形式でのモデル保存を実行する場合、True
    params["tensorflowjs_output"] = False       # tensorflowjs形式でのモデル保存を実行する場合、True. ライブラリのバージョンでコンフリクトの恐れあり。
    params["remove_after_train"] = {"enable":False, "pattern":""}    # 学習後にファイルを削除する際に使用する
    
    return params


def read_setting(fname):
    """ 設定を読み込む。返り値は辞書で返す
    eval()を使っているので、不正なコードを実行しないように、気をつけてください。
    """
    param = set_default_setting()

    with open(fname, "r", encoding="utf-8-sig") as fr:
        obj = yaml.safe_load(fr)
        
        if obj is not None:
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
    ## クラスの重複対応
    param["class_sets"] = set(param["class_list"])  # もし重複があっても排除


    # lossの処理
    if "local." == param["loss"][:6]:
        # "local.hogehoge_func,{a=10}"の様な文字列を想定
        lossName_lossParam = param["loss"][6:]            # 先頭文字列を削る
        func_name, loss_param_str = lossName_lossParam.split(",", 1)   # 分割
        func_name = func_name.strip()
        loss_param_str = loss_param_str.strip()
        func_name = func_name[:30]                       # 関数名の文字数制限
        loss_param = ast.literal_eval(loss_param_str)    # 辞書オブジェクトに変換

        # 重みの処理
        w = [1.0] * len(param["class_sets"])
        if "weights" in loss_param:
            weights = loss_param["weights"]   # 例："{"kuina": 1.5}"

            labels = sorted(list(param["class_sets"]))
            for label in weights:
                w[labels.index(label)] = weights[label]
        loss_param["weights"] = np.array(w)

        loss_function = gen_custom_loss(func_name, loss_param)
        param["loss"] = loss_function
        #print("------------------------", func_name, param["loss"])

        # カスタムオブジェクトとしても登録しておく
        param["custom_objects"][func_name] = loss_function

    return param





def main():
    start = time.time()

    # 設定を読み込み
    setting = read_setting("train_setting.yaml")

    # 引数のチェック
    setting = arg_parse(setting)
    print2("\n\n< setting >", setting)

    ## 矛盾チェック
    if setting["epochs"] <= setting["initial_epoch"]:  # 矛盾があればエラー
        raise ValueError("epochs <= initial_epoch")


    # 教師データの存在チェック
    if not os.path.exists(setting["image_root"]):
        raise ValueError("Teacher image data directory is not exists. Please check your setting.")


    # 保存先の親フォルダを作成
    os.makedirs("runs", exist_ok=True)  # 保存先のフォルダを作成


    # 教師データの読み込みと、モデルの構築
    ## 学習を再開させる場合
    if setting["retry"] and setting["reuse"] != True:
        # 学習を再開させる場合
        save_dir = last_dirpath("train")   # （前回の、そして今回の）結果の保存先
        x_train, y_train, x_test, y_test, weights_dict, label_dict, model = reload(load_dir=save_dir, custom_objects=setting["custom_objects"], model_format=setting["model_format"])
        test_file_names = restore(['test_names.pickle'], load_dir=save_dir)[0]   # 1つしか無いがlistで返ってくるので[0]で取り出す
    
    # 学習用のデータを流用する場合
    elif setting["reuse"]:
        save_dir = next_dirpath("train")      # 結果の保存先
        if setting["retry"]:
            save_dir = last_dirpath("train")   # （前回の、そして今回の）結果の保存先
        os.makedirs(save_dir, exist_ok=True)  # 保存先のフォルダを作成
        
        load_dir = first_dirpath("train")      # 保存済みの学習データ等が保存されているフォルダ
        x_train, y_train, x_test, y_test, weights_dict, label_dict, model = reload(load_dir=load_dir, custom_objects=setting["custom_objects"], model_format=setting["model_format"])
        test_file_names = restore(['test_names.pickle'], load_dir=load_dir)[0]   # 1つしか無いがlistで返ってくるので[0]で取り出す

        # モデルの対応
        if setting["retry"]:
            # 追加の学習の場合は、学習済みのモデルを読み込む
            model = reload(load_names=[], with_model=True, load_dir=save_dir, custom_objects=setting["custom_objects"], model_format=setting["model_format"])[0]
        else:
            # 最初から学習を始める場合は、モデルを再構築
            model = build_model_local2(input_shape=x_train.shape[1:], 
                                       output_dim=y_train.shape[1], 
                                       data_format=setting["data_format"])   # モデルの作成

        # ラベル名の辞書は予測処理での利用に備えてコピーしておく
        if not setting["retry"]:
            shutil.copy2(os.path.join(load_dir, "label_dict.pickle"), save_dir)

    # 教師データの読み込みと、モデルの構築。必要なら、callbackで保存していた結合係数を読み込む
    else:
        last_dir = last_dirpath("train")      # 前回の結果の保存先
        save_dir = next_dirpath("train")      # 今回の結果の保存先
        os.makedirs(save_dir, exist_ok=True)  # 保存先のフォルダを作成
        data_format = setting["data_format"]
        dir_names_dict = ip.search_image_dir(setting["image_root"],  # 画像フォルダのクラス名とパスの一覧を辞書で取得
                                             setting["class_sets"], 
                                             setting["ignore_list"], 
                                             setting["exchange_dict"] )
        # 画像読み込みに必要なパラメータの設定
        param = {"dir_names_dict": dir_names_dict, 
                 "data_format": data_format, 
                 "size": setting["image_shape_wh"], 
                 "mode":"RGB", 
                 "resize_filter": Image.NEAREST, 
                 "preprocess_func": ip.preprocessing2,
                 }       
        x_train, x_test, weights_dict, label_dict, y_train, y_test, output_dim, test_file_names = \
                                       ip.load_save_images(ip.read_images3, 
                                                           param, 
                                                           validation_rate=setting["validation_rate"], 
                                                           save_dir=save_dir)
        
        if "--load_model" in sys.argv:
            # 前回の学習結果から、モデルだけを読み込む（保存されたモデルに対し、クラスが変化（増えたり減ったり入れ替わったり）なければ動く）
            # 機会はあまりないけど、追加で学習を進める場合に利用する
            model = reload(load_names=[], with_model=True, load_dir=last_dir, custom_objects=setting["custom_objects"], model_format=setting["model_format"])[0]
        else:
            # モデルの新規作成
            model = build_model_local2(input_shape=x_train.shape[1:], 
                                      output_dim=output_dim, 
                                      data_format=data_format)   # モデルの作成
    

    # 諸々を確認のために表示
    print2("shape of x_train, y_train: ", x_train.shape, y_train.shape)
    print2("shape of x_test, y_test: ", x_test.shape, y_test.shape)
    print2("weights_dict: ", weights_dict)
    print2("label_dict: ", label_dict)
    print2("y_train, y_test: ", y_train, y_test)
    print2("layer size: ", len(model.layers))



    # モデルの調整（最適化器や、学習係数や、結合係数の調整など）
    for i, layer in enumerate(model.layers):  # 結合係数を更新させるか、させないか調整
        if i >= setting["leranig_layerid"]:
            layer.trainable = True     # Trueで更新して、Falseで更新しない
        else:
            layer.trainable = False

    # re-compile
    model.compile(optimizer=SGD(learning_rate=setting["lr"], momentum=0.9),    # コンパイル
            loss=setting["loss"],   # 損失関数
            metrics=['accuracy'])



    # ImageDataGeneratorの準備
    w, h = setting["image_shape_wh"]
    shape = (h, w, 3)

    # 教師データを無限に用意するオブジェクト用のパラメータを作成
    dg_param = setting["datagen_params"]
    dg_param["shape"] = shape

    # mixupで無視するラベルを作る（例えば、静音は無視したい）
    label_dict_swap = {v: k for k, v in label_dict.items()}
    invalid_label = np.zeros(len(label_dict) - 1)   # NDの分を引く
    mixup_flag = False
    for label in setting["mixup_ignores"]:
        if label in label_dict_swap:
            invalid_label[label_dict_swap[label]] = 1
            mixup_flag = True
    if mixup_flag:     # 該当ラベルがあれば、mixup用のパラメータを修正する
        dg_param["mixup"][2] = invalid_label
    
    datagen_train = ip.MyImageDataGenerator(**dg_param)   # 学習用
    dg_param["label_smoothing_e"] = 0.0                         # 検証データではラベルスムージングは無し
    datagen_validation = ip.MyImageDataGenerator(**dg_param)    # 検証用


    # 設定の保存（後でパラメータを追えるように）
    now_ = dt.now().strftime('%Y%m%d%H%M')
    fname = os.path.join(save_dir, "train_setting_{}.yaml".format(now_))
    with open(fname, 'w', encoding="utf-8-sig") as fw:
        yaml.dump(setting, fw, encoding='utf8', allow_unicode=True)


    # 途中でモデルを保存したりする設定
    batch_size = setting["batch_size"]
    steps_per_epoch = x_train.shape[0] // batch_size
    cp_dir = os.path.join(save_dir, "cp_models")   # 保存先のパスを作成
    os.makedirs(cp_dir, exist_ok=True)             # 保存先のフォルダを作成
    cp_name = os.path.join(cp_dir, "cp.ckpt_{epoch:04d}" + setting["model_format"])
    #cb_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='auto')  # 学習を適当なタイミングで止める仕掛け
    cb_save = tf.keras.callbacks.ModelCheckpoint(cp_name, 
                                                 monitor='val_loss', 
                                                 verbose=1, 
                                                 save_best_only=False, 
                                                 save_weights_only=False, 
                                                 mode='auto', 
                                                 save_freq=setting["cp_period"] * steps_per_epoch,   # 学習中に定期的に保存
                                                 )

    # 学習
    np.random.seed(seed=1)      # 学習条件をそろえるために乱数をリセット
    history = model.fit(        # ImageDataGeneratorを使った学習
        datagen_train.flow(x_train, y_train, batch_size=batch_size, shuffle=True, balancing=True),  # シャッフルは順序によらない学習のために重要
        epochs=setting["epochs"],
        steps_per_epoch=steps_per_epoch,
        verbose=1,
        #class_weight=weights_dict,   # 与えない方がまし
        callbacks=[cb_save],
        validation_data=datagen_validation.flow(x_test, y_test, batch_size=batch_size, shuffle=True, balancing=True),  # マルチプロセスを利用しない場合、ここにジェネレータを渡すことも出来る
        initial_epoch=setting["initial_epoch"],
        validation_steps=len(x_test) // batch_size,

        # マルチプロセスを利用する設定（不要ならコメントアウト. 実用的には、最低でも128 GBのRAMが必要だと思う。）
        ## generatorは使えない。
        #workers=2,
        #max_queue_size=2,
        #use_multiprocessing=True,
        ) # 返り値には、学習中のlossやaccなどが格納される（metricsに指定する必要がある）


    # 学習結果を保存
    print(model.summary())      # レイヤー情報を表示(上で表示させると流れるので)
    model.save(os.path.join(save_dir, "model{}".format(setting["model_format"])))    # 獲得した結合係数を保存
    report_path = next_dirpath("report", root=save_dir)   # 細かいレポートはサブフォルダに保存（retryへの対応）
    os.makedirs(report_path, exist_ok=True)  # 保存先のフォルダを作成
    plot_history(history, show_flag=False, save_name=os.path.join(report_path, "history.png"))       # lossの変化をグラフで表示
    save_history(os.path.join(report_path, "history.csv"), history, mode="a")

    # onnx形式での保存 (https://qiita.com/studio_haneya/items/be9bc7c56af44b7c1e0a)
    if setting["onnx_output"]:
        import tf2onnx, onnx
        model_path = os.path.join(save_dir, "model.onnx")
        input_signature = [tf.TensorSpec([None] + list(x_train.shape[1:]), tf.float32, name='x')]
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature)  # 引数のopsetは2023-05時点では15がデフォルトらしい。ここでは省略する。
        onnx.save(onnx_model, model_path)


    # tensorflowjs形式での保存 (https://note.com/npaka/n/n143ba7e60176)
    if setting["tensorflowjs_output"]:
        import tensorflowjs as tfjs
        model_path = os.path.join(save_dir, "model_tfjs")
        os.makedirs(model_path, exist_ok=True)  # 保存先のフォルダを作成
        tfjs.converters.save_keras_model(model, model_path)



    # 学習成果のチェックとして、検証データに対して分割表を作成し、正解・不正解のリストをもらう
    checked_list = check_validation(0.15, model, x_test, y_test, label_dict, 
                                    batch_size=30, 
                                    mode="multi_label", 
                                    save_dir=report_path)

    # validationに使われたファイルに対する識別結果を保存
    with open(os.path.join(report_path, "test_result.csv"), "w") as fw:
        for file_name, result in zip(test_file_names, checked_list):
            fw.write("{},{}\n".format(file_name, result))



    # メモリの後始末（必要か不明）
    tf.keras.backend.clear_session()
    gc.collect()


    # ファイルの後始末
    if setting["remove_after_train"]["enable"]:
        print("--- file removing start... ---")
        pattern = os.path.join(save_dir, setting["remove_after_train"]["pattern"])
        for p in glob.glob(pattern):
            if os.path.isfile(p):
                os.remove(p)
                print(f"remove: {p}")


    print("経過時間[s]：", time.time() - start)

if __name__ == "__main__":
    main()
