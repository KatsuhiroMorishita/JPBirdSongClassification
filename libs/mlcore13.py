# purpose: kerasによる画像識別の関数をまとめた
# main()に利用方法のサンプルを置いているので、参考にしてください。
# author: Katsuhiro MORISHITA　森下功啓
# history: 
#  2018-08-12 ver.1.
#  2020-04-23 ver.2. image_preprocessing3.pyに合わせてマルチラベルの画像読み込みに対応
#  2021-03-04 ver.5  image_preprocessing4.pyに変更
#  2021-03-18 ver.6  image_preprocessing5.pyに変更
#  2021-03-29 ver.7  image_preprocessing6.pyに変更
#  2021-04-03 ver.8  image_preprocessing7.pyに変更
#  2021-06-09 ver.9  kerasを単独のものからtensorflow内のものに変更
#  2021-10-19 ver.10 保存先、読み込み元のフォルダを指定できるように変更
#  2022-01-06 ver.11 image_preprocessing9.pyに変更
#  2022-01-10 ver.12 image_preprocessing10.pyに変更
# created: 2020-04-23
import sys, os, pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from matplotlib import pylab as plt
from PIL import Image
import numpy as np
import pandas as pd


from . import image_preprocessing11 as ip



def build_model(input_shape, output_dim, data_format):
    """ 転移学習を利用したモデルを作成する
    入力は画像、出力はラベルという構造を想定しています。
    """
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    #base_model = Xception(include_top=False, weights='imagenet', input_shape=input_shape)

    top_model = Sequential()   # 追加する層
    top_model.add(Conv2D(32, (3, 3), padding="same", input_shape=base_model.output_shape[1:]))
    top_model.add(Conv2D(32, (3, 3), padding="same"))
    top_model.add(Conv2D(32, (3, 3), padding="same"))
    top_model.add(MaxPooling2D(pool_size=(3, 3)))
    top_model.add(Flatten())
    top_model.add(Dense(100, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(output_dim))    # 出力層のユニット数はoutput_dim個
    top_model.add(Activation('sigmoid'))
    top_model.add(Activation('softmax'))

    # fix weights of base_model
    for layer in base_model.layers:
        layer.trainable = False    # Falseで更新しない

    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),    # コンパイル
        loss='categorical_crossentropy',   # 損失関数は、判別問題なのでcategorical_crossentropyを使う
        metrics=['accuracy'])
    print(model.summary())
    return model



def build_model_simple(input_shape, output_dim, data_format):
    """ 転移学習を利用しないモデルを作成する
    入力は画像、出力はラベルという構造を想定しています。
    """
    # モデルの作成
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", data_format=data_format, input_shape=input_shape))  # カーネル数32, カーネルサイズ(3,3), input_shapeは1層目なので必要。https://keras.io/ja/layers/convolutional/#conv2d
    model.add(Activation('relu'))
    model.add(Conv2D(24, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim))                      # 出力層のユニット数は2
    model.add(Activation('sigmoid'))
    model.add(Activation('softmax'))
    opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0003) # 最適化器のセット。lrは学習係数
    model.compile(optimizer=opt,             # コンパイル
          loss='categorical_crossentropy',   # 損失関数は、判別問題なのでcategorical_crossentropyを使う
          metrics=['accuracy'])
    print(model.summary())

    return model


def plot_history(history, show_flag=True, save_name="", log_scale=True):
    """ 損失の履歴を図示する
    from http://www.procrasist.com/entry/2017/01/07/154441
    """
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15)  # 左側のラベルが見切れないようにする
    
    x = history.epoch
    y1 = history.history['loss']
    y2 = history.history['val_loss']
    plt.plot(x, y1, "o-", label="loss")
    plt.plot(x, y2, "^-", label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.grid()
    
    if log_scale:
        plt.yscale("log") # y軸を対数軸とする
    if save_name != "":
        plt.savefig(save_name)
    if show_flag:
        plt.show()


def save_history(fname, history, mode="a"):
    """ 損失の履歴をcsvファイルに保存する
    mode: str, "a"なら追記する。そうでなければ新規作成・上書きする。
    """   
    # 保存するデータをpandasのDataFrameに格納
    df_ = pd.DataFrame()
    df_["epoch"] = history.epoch
    df_["loss"] = history.history['loss']
    df_["val_loss"] = history.history['val_loss']

    # すでに保存しているものがあれば、読み込む
    if mode == "a" and os.path.exists(fname):
        df = pd.read_csv(fname)
    else:
        # 無ければ、空のDataFrameを作る
        df = pd.DataFrame()
        df["epoch"] = []
        df["loss"] = []
        df["val_loss"] = []

    # 結合と保存
    df_concat = pd.concat([df, df_])
    df_concat.to_csv(fname, index=False)



def save_validation_table(predicted_class, correct_class, save_dir="."):
    """ 学習に使わなかった検証データに対する予測と正解ラベルを使って、スレットスコアの表的なものを作って保存する
    predicted_class: list or ndarray, 1次元配列を想定。予測されたラベルが格納されている事を想定
    correct_class: list or ndrray, 1又は2次元配列を想定。正解ラベルが格納されている事を想定
    save_dir: str, 保存先のフォルダのパス
    """
    
    df = pd.DataFrame()
    df["correct"] = correct_class
    df["predicted"] = predicted_class
    df1 = pd.crosstab(df["correct"], df["predicted"])               # クロス集計表の作成

    print("--件数でカウントした分割表--")
    print(df1)
    df1.to_csv(os.path.join(save_dir, "validation_table1.csv"))

    # 正解ラベルを使って正規化する
    df2 = df1.copy()
    #print(correct_class, df2.index)
    amounts = [len(np.where(correct_class==x)[0]) for x in df2.index]  # 正解ラベルをカウント. correct_classはndarray型でないと機能しない
    #print(amounts)

    for i in range(len(df2)):
        #print("i: ", i)
        if amounts[i] != 0:
            df2.iloc[i, :] = df2.iloc[i, :] / amounts[i]   # 列単位で割る
        else:
            df2.iloc[:,i] = np.float("inf")
    print("--割合で表した分割表--")
    print(df2)
    df2.to_csv(os.path.join(save_dir, "validation_table2.csv"))



def check_validation(th, model, x_test, y_test, label_dict, batch_size=None, mode="uni_label", save_dir="."):
    """ 学習成果のチェックとして、検証データに対して分割表を作成・保存し、正解状況をndarrayのbool配列で返す
    th: float, 尤度の閾値
    model: 学習済みのモデル
    x_test: ndarray, 検証用データ
    y_test: ndarray, 検証用データの正解データ
    label_dict: dict, keyがlocal_IDでvalueがラベルの辞書
    mode: str, "uni_label"の場合、最尤値を持つクラスを推定結果とします。"multi_label"の場合は、閾値を超えたクラス全てを識別結果とします
    save_dir: str, 保存先のフォルダのパス
    """
    if batch_size is None:
        batch_size = len(x_test)

    result_raw = model.predict(x_test, batch_size=batch_size, verbose=0) # クラス毎の尤度を取得。 尤度の配列がレコードの数だけ取得される
    representative_value = np.percentile(result_raw.flatten(), q=[0, 25, 50, 75, 100])
    print("representative value of result : ", representative_value)  # 閾値決定の参考用

    if mode == "uni_label":    # 正解のクラスは1つしかないので、最尤推定とする
        result_list = [len(arr) if np.max(arr) < th else arr.argmax() for arr in result_raw]  # 最大尤度を持つインデックスのlistを作る。ただし、最大尤度<thの場合は、"ND"扱いとする
        predicted_class = np.array([label_dict[class_id] for class_id in result_list])   # 予測されたclass_local_idをラベルに変換
        print("test result: ", predicted_class)
        correct_class = np.array([label_dict[label.argmax()] for label in y_test])  # 正解class_idをラベルに変換
        save_validation_table(predicted_class, correct_class, save_dir=save_dir)
    
    elif mode == "multi_label":
        result_list = [arr > th for arr in result_raw]
        predicted_class = []
        for label in result_list:
            class_ids = np.where(label)[0]     # labelの中で、1となっている要素番号の配列を作成
            label_name = []
            for id_ in class_ids:              # 検出されたクラスの名をリストに入れる
                label_name.append(label_dict[id_])
            label_name = ",".join(label_name)  # ","でクラス名を結合した文字列を生成
            if label_name == "":
                label_name = "ND"
            predicted_class.append(label_name)
        predicted_class = np.array(predicted_class)  # ブロードキャストが使える様に、ndarray型に変換する

        correct_class = []
        for label in y_test:
            class_ids = np.where(label)[0]
            label_name = []
            for id_ in class_ids:
                label_name.append(label_dict[id_])
            label_name = ",".join(label_name)
            if label_name == "":     # 正解ラベルでどこにも所属しないクラスはあり得ないと思うが、念のため記述しておく
                label_name = "ND"
            correct_class.append(label_name)
        correct_class = np.array(correct_class)

        save_validation_table(predicted_class, correct_class, save_dir=save_dir)

    return correct_class == predicted_class



def restore(files, load_dir="."):
    """ 保存されているファイルを読み込んでリストで返す
    files: list<str>, ファイル名がリストに格納されている事を想定
    load_dir: str, 読み込むファイルのあるフォルダのパス
    """
    # ファイルの存在確認
    for fname in files:
        fpath = os.path.join(load_dir, fname)
        if os.path.exists(fpath) == False:
            return

    ans = []
    for fname in files:
        print("--load--: ", fname)
        fpath = os.path.join(load_dir, fname)
        
        if "npy" in fname or "npz" in fname:
            ans.append(np.load(fpath))
        elif "pickle" in fname:
            with open(fpath, 'rb') as f:
                ans.append(pickle.load(f))
    return ans



def reload(load_names=['x_train.npy', 'y_train.npy', 'x_test.npy', 'y_test.npy', 'weights_dict.pickle', 'label_dict.pickle'], 
           with_model=True, 
           custom_objects={},
           load_dir="."):
    """ 保存済みの画像やモデルを読み込む
    custom_objects: dict, load_modelに渡す辞書
    load_dir: str, 読み込むファイルのあるフォルダのパス
    """
    obj = restore(load_names, load_dir)
    if with_model:
        model_path = os.path.join(load_dir, "model.hdf5")
        if os.path.exists(model_path) and obj is not None:
            # モデルを再構築
            print("--load 'model.hdf5'--")
            model = load_model(model_path, custom_objects=custom_objects)
            obj.append(model)

            return obj 
        else:
            print("--failure for restore--")
            exit()
    else:
        print("--no model load--")
        return obj





def main():
    # 調整することの多いパラメータを集めた
    image_shape = (32, 32)   # 画像サイズ
    epochs = 5               # 1つのデータ当たりの学習回数
    batch_size = 10          # 学習係数を更新するために使う教師データ数
    initial_epoch = 0        # 再開時のエポック数。途中から学習を再開する場合は、0以外を指定しないとhistryのグラフの横軸が0空になる
    if epochs <= initial_epoch:  # 矛盾があればエラー
        raise ValueError("epochs <= initial_epoch")

    # 教師データを無限に用意するオブジェクトを作成
    """
    datagen = ImageDataGenerator(       # kerasのImageDataGenerator
        #samplewise_center = True,              # 平均をサンプル毎に0
        #samplewise_std_normalization = True,   # 標準偏差をサンプル毎に1に正規化
        #zca_whitening = True,                  # 計算に最も時間がかかる。普段はコメントアウトで良いかも
        rotation_range = 30,                    # 回転角度[degree]
        zoom_range=0.5,                         # 拡大縮小率、[1-zoom_range, 1+zoom_range]
        #fill_mode='nearest',                    # 引き伸ばしたときの外側の埋め方
        horizontal_flip=True,                   # 水平方向への反転
        vertical_flip=True,                     # 垂直方向での反転
        #rescale=1,                              # fit()などで引数xに、更に掛ける係数があれば1以外を設定
        width_shift_range=0.2,                  # 横方向のシフト率
        height_shift_range=0.2)                 # 縦方向のシフト率
    #datagen.fit(x_train)                        # zca用に、教師データの統計量を内部的に求める
    """

    #"""
    datagen = ip.MyImageDataGenerator(       # 自作のImageDataGenerator
        rotation_range = 45,                    # 回転角度[degree]
        horizontal_flip=True,                   # 水平方向への反転
        vertical_flip=True,                     # 垂直方向での反転
        #crop=True,
        #random_erasing=True,
        mixup = (0.2, 2, None),                 # 画像の混合確率と最大混合数
        shape=image_shape)                      # 出力する画像のサイズ
    #"""
    
    # 教師データの読み込みと、モデルの構築。必要なら、callbackで保存していた結合係数を読み込む
    if len(sys.argv) > 1 and sys.argv[1] == "retry":
        x_train, y_train, x_test, y_test, weights_dict, label_dict, model = reload()
    else:
        data_format = "channels_last"
        
        # pattern 1, flower
        dir_names_dict = {"yellow":["flower_sample/1"], 
                          "white":["flower_sample/2"],
                          "white,yellow":["flower_sample/3"],
                          } 
        param = {"dir_names_dict":dir_names_dict, 
                 "data_format":data_format, 
                 "size":image_shape, 
                 "mode":"RGB", 
                 "resize_filter":Image.NEAREST, 
                 "preprocess_func":ip.preprocessing2, 
                 }
        x_train, x_test, weights_dict, label_dict, y_train, y_test, output_dim, test_file_names = ip.load_save_images(ip.read_images3, param, validation_rate=0.2)
        model = build_model_simple(input_shape=x_train.shape[1:], output_dim=output_dim, data_format=data_format)   # モデルの作成
        
        # pattern 1, animal
        #dir_names_dict = {"cat":["sample_image_animal/cat"], 
        #                  "dog":["sample_image_animal/dog"]} 
        #param = {"dir_names_dict":dir_names_dict, "data_format":data_format, "size":image_shape, "mode":"RGB", "resize_filter":Image.NEAREST, "preprocess_func":preprocessing2}
        #x_train, x_test, weights_dict, label_dict, y_train, y_test, output_dim = ip.load_save_images(ip.read_images1, param, validation_rate=0.2)
        #model = build_model_simple(input_shape=x_train.shape[1:], output_dim=output_dim, data_format=data_format)   # モデルの作成

        # pattern 2, animal
        #dir_names_list = ["sample_image_animal/cat", "sample_image_animal/dog"]
        #name_dict = read_name_dict("sample_image_animal/file_list.csv")
        #param = {"dir_names_list":dir_names_list, "name_dict":name_dict, "data_format":data_format, "size":image_shape, "mode":"RGB", "resize_filter":Image.NEAREST, "preprocess_func":preprocessing2}
        #x_train, x_test, weights_dict, label_dict, y_train, y_test, output_dim = ip.load_save_images(ip.read_images2, param, validation_rate=0.2)
        #model = build_model_simple(input_shape=x_train.shape[1:], output_dim=output_dim, data_format=data_format)   # モデルの作成
    

    # 諸々を確認のために表示
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print(weights_dict)
    print(label_dict)
    print(y_train, y_test)

    # 学習
    cb_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='auto')  # 学習を適当なタイミングで止める仕掛け
    cb_save = tf.keras.callbacks.ModelCheckpoint("cb_model.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq=5)  # 学習中に最高の成績が出るたびに保存

    #"""
    history = model.fit(   # ImageDataGeneratorを使った学習
        datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),  # シャッフルは順序によらない学習のために重要
        epochs=epochs,
        steps_per_epoch=int(x_train.shape[0] / batch_size),
        verbose=1,
        class_weight=weights_dict,
        callbacks=[cb_stop, cb_save],
        validation_data=(x_test, y_test),  # ここにジェネレータを渡すことも出来る
        initial_epoch=initial_epoch
        ) # 返り値には、学習中のlossやaccなどが格納される（metricsに指定する必要がある）
    #"""
    
    """
    # Generatorを使わないパターン
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        verbose=1,
        class_weight=weights_dict,
        validation_data=(x_test, y_test),
        callbacks=[cb_stop, cb_save],
        shuffle=True,
        batch_size=batch_size
        ) # 返り値には、学習中のlossやaccなどが格納される（metricsに指定する必要がある）
    #"""

    # 学習成果のチェックとして、検証データに対して分割表を作成する
    check_validation(0.7, model, x_test, y_test, label_dict, mode="multi_label")

    # 学習結果を保存
    print(model.summary())      # レイヤー情報を表示(上で表示させると流れるので)
    model.save('model.hdf5')    # 獲得した結合係数を保存
    plot_history(history)       # lossの変化をグラフで表示


if __name__ == "__main__":
    main()


