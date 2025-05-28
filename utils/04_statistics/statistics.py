# writen for python 3
# purpose: 識別結果の尤度の表や、画像の一覧から、検出数の時系列グラフや、音源毎の時間別検出数、通算日と検出時刻の散布図を作製する。
# history: 
#   2024-07-08   今までのスクリプトを整理して作製開始
#   2025-03-19   Python 3.13の警告に対応。状況に合わせてエラーを出すように変更した。
#   2025-03-30   mutagenとcv2を使ってファイルの長さを取得し、タイムスタンプが記録終了時刻であっても対応できるものとした。
# author: Katsuhiro Morishita
# created: 2024-07-10
# license: MIT. If you use this program for your study, you should write Acknowledgement in your paper.
import os, glob, re, copy, unicodedata, hashlib
from datetime import datetime as dt
from datetime import timedelta as td
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from dateutil.relativedelta import relativedelta
from natsort import natsorted
import yaml
import ephem
import Levenshtein
import librosa

import cv2
from mutagen.mp4 import MP4
from mutagen.flac import FLAC 
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from mutagen.asf import ASF



# label_from_likelifoods.pyよりコピー
def read_likelihoods(likelihood_file):
    """ 予測で出力された尤度のリストを読み込む
    likelihood_file: str, 尤度が格納されたテキストファイルのパス
    """
    df = pd.read_csv(likelihood_file, encoding="utf-8-sig")
    print(df.head())

    return df



def read_images(setting):
    """ 画像ファイルを読み込んで、尤度の記載された識別結果と同様のDataFrameを作製して返す
    setting: dir, 諸々の設定が書き込まれた辞書
    """
    # 画像のパスを取得
    image_files = glob.glob(os.path.join(setting["image_source_dir"], "*.png"))
    image_files = sorted(image_files)

    if len(image_files) == 0:
        print("warning: 0 images founded. Check your env. or setting.")
        return pd.DataFrame(), []

    # "likelihoods_result_akahige1_th0.7to1.01"の様なフォルダ名であることを前提に整理
    # 自作のスクリプトで生成された画像フォルダであることが前提。これ以外への対応が必要なら要改造。
    dirs = [os.path.basename(os.path.dirname(fpath)) for fpath in image_files]
    
    for i in range(len(dirs)):
        dir_ = dirs[i]
        separated_name = dir_.split("_")
        if len(separated_name) >= 4:
            dirs[i] = separated_name[2]    
    dir_lists = sorted(set(dirs))     # クラス名（≒種名）になっているはず

    fnames = [x.split("__")[1] + "__" + x.split("__")[2] for x in image_files]
    time_info = [x.split("__")[3] for x in image_files]

    #print(fnames)
    #print(time_info)
    if time_info[0][-4:] == ".png":
        time_info = [ti[:-4] for ti in time_info]
    s = [float(ti.split("_")[0]) for ti in time_info]
    w = [float(ti.split("_")[1]) for ti in time_info]

    #print(image_files[:5])
    #print(time_info[:5])
    #print(s[:5])

    df_ = pd.DataFrame()
    #df_["image_path"] = image_files
    df_["fname"] = fnames
    df_["s"] = s
    df_["w"] = w
    df_["dir"] = dirs
    df_["file_full_path"] = image_files
    df_["audio_path"] = fnames
    #df_.to_csv("df_.csv")   # for debug
    locations = [x.split("__")[0] for x in df_["fname"].values]
    df_["dir_name"] = locations
    fnames_unique = sorted(set(fnames))


    # 音源ファイルとの突合せ
    audio_length = {}
    if setting["audio_source"] != "":
        audio_files = glob.glob(setting["audio_source"])    # パターンに合致するものを検索
        print(f"{len(audio_files)} audio files were founded. ")
        if len(audio_files) == 0 and setting["datetime_method"] == "timestamp": 
            raise ValueError("Not found any audio files. Check param of 'audio_source' in statistics_setting.yaml.")

        if len(audio_files) > 0:
            am = audio_matcher(audio_files)
            if am.ready:
                for i in range(len(df_)):
                    path_img = df_.at[i, "file_full_path"]
                    audio_path, ts, te, tw = am.match(path_img)    # 合致するものを探す

                    if audio_path != "":
                        p = Path(audio_path)
                        audio_path_abs = str(p.resolve())   # 絶対パスに変換
                        df_.at[i, "audio_path"] = audio_path_abs

                        if audio_path_abs in audio_length:
                            audio_length[audio_path_abs] = librosa.get_duration(filename=audio_path)  # 長さ[s]を取得
    elif setting["datetime_method"] == "timestamp": 
        raise ValueError("Set param of 'audio_source' in statistics_setting.yaml.")


    # 検出結果を格納する基本的な表を作製
    df = pd.DataFrame()
    dfs = []
    for fn in fnames_unique:
        df_b = pd.DataFrame()
        df_c = df_.loc[df_["fname"] == fn]
        df_c.reset_index(inplace=True, drop=True)
        s_max = df_c["s"].max()     # 検出された最も遅い時刻（単位は秒で、音源の先頭からの経過時間を指す）
        w_min = df_c["w"].min()
        ap = df_c.at[0, "audio_path"]
        if ap in audio_length:
            s_max = (int(audio_length[ap]) // w_min + 1) * w_min
        df_b["s"] = np.arange(0, s_max+1, w_min)
        df_b["w"] = w_min
        df_b["fname"] = fn
        df_b["file_full_path"] = df_c.at[0, "file_full_path"]
        df_b["audio_path"] = df_c.at[0, "audio_path"]
        df_b["dir_name"] = df_c.at[0, "dir_name"]
        dfs.append(df_b)
    df = pd.concat(dfs)
    print(df.head())
    df = df[["fname", "s", "w", "file_full_path", "audio_path", "dir_name"]]  # 列の並べ替え
    df.reset_index(inplace=True, drop=True)

    # 検出結果を反映
    for dir_ in dir_lists:   # 種ごとにフォルダ分けされている前提
        df_a = df_.loc[df_["dir"] == dir_]
        for fn in df_a["fname"].unique():
            df_c = df_a[df_a["fname"] == fn]
            index = (df["fname"] == fn) & df["s"].isin(df_c["s"])
            df.loc[index, dir_] = 1
    df["fname"] = df["audio_path"]

    # 列の順番入れ替え
    cnames = list(df.columns)
    for c in ["dir_name", "file_full_path", "audio_path"]:
        cnames.remove(c)
    df = df[cnames + ["dir_name"]]

    # 該当の無かった部分を0で埋める
    df = df.fillna(0)
    
    return df, list(set(locations))





# restoration_timelist_from_image6.pyより移植
def get_location_ID(file_path, id_size=4):
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
        location = str(len(dir_name)) + hs[:id_size]

    return location


# restoration_timelist_from_image6.pyより移植
def split_fname(path_image, ext):
    """ スペクトログラム画像のファイル名のパスから、画像ファイル名単体と元音源ファイルのファイル名とIDと抜出時間のタプルを返す
    path_image: str, 画像ファイルのパス名
    ext: str, 音源の拡張子。例：".mp3"
    """
    fname_image = os.path.basename(path_image)         # ファイル名取り出し
    dummy, loc, name_and_times = re.split(r'_{2,4}', fname_image, maxsplit=2)  # __で分割

    # ファイル名と時間区間の取得（ファイル名の中に__があっても対応）
    name_and_times_mirror = name_and_times[::-1]
    time_mirror, name_mirror = re.split(r'_{2,4}', name_and_times_mirror, maxsplit=1)
    name = name_mirror[::-1]
    times = time_mirror[::-1]                    # この時点では拡張子が含まれる
    time_pair, ext2 = os.path.splitext(times)    # 拡張子を分離
    #print(">>>> ", time_pair)
    t_start, t_width = time_pair.split("_")[:2]      # _で分離（たまにファイルのコピーしたやつがあるので、2つだけ取り出す

    # 音源ファイルのフルパスを作成
    fname_music = name + ext.lower()

    return (fname_image, fname_music, loc, float(t_start), float(t_width))




# restoration_timelist_from_image6.pyより移植
class audio_matcher:
    """ スクリプト画像ファイルがどの音源ファイルから作成されたのか判定するクラス
    """
    def __init__(self, audio_paths=[]):
        self.audio_files = copy.deepcopy(audio_paths)
        self.ready = len(self.audio_files) > 0

        # 音源ファイルと画像ファイルをマッチさせるための辞書を作成  
        # 旧仕様では親フォルダ名からのハッシュ値を2桁利用していたが、今は4桁なので、2種類の辞書を作っている。
        #print(audio_files)       
        self.id_dict1 = {}
        for fpath in self.audio_files:
            locaction_ID = get_location_ID(fpath, 2)    # 親フォルダ名からIDを作る
            name, ext = os.path.splitext(fpath)         # パスを名前と拡張子に分ける
            name = self.__clean_fname(name)             # ファイル名の振れを取る
            temp_path = name + ext.lower()              # 拡張子を小文字に加工
            key = (locaction_ID, os.path.basename(temp_path))   # 親フォルダとファイル名の情報をペアにしたタプルを作る

            self.id_dict1[key] = fpath        # 辞書にパスを格納
            print("founded audio file: ", key, fpath)
            
            
        self.id_dict2 = {}
        for fpath in self.audio_files:
            locaction_ID = get_location_ID(fpath, 4)    # 親フォルダ名からIDを作る
            name, ext = os.path.splitext(fpath)         # パスを名前と拡張子に分ける
            name = self.__clean_fname(name)             # ファイル名の振れを取る
            temp_path = name + ext.lower()              # 拡張子を小文字に加工
            key = (locaction_ID, os.path.basename(temp_path))   # 親フォルダとファイル名の情報をペアにしたタプルを作る

            self.id_dict2[key] = fpath        # 辞書にパスを格納
            #print("founded audio file: ", key, fpath)

        # 音源のファイル名と音源のパスを組みとした辞書を作成（ユニーク性はない）
        self.fname_dict = {}
        for fpath in self.audio_files:
            name, ext = os.path.splitext(os.path.basename(fpath))   # ファイル名を名前と拡張子に分ける
            temp_name = name + ext.lower()            # 拡張子を小文字に加工
            self.fname_dict[temp_name] = fpath        # 辞書にパスを格納

        # 短縮した音源ファイル名の一覧を作成
        self.fname_trimeds = [os.path.basename(x)[:17] for x in self.audio_files]

        # 
        self.cache = {}


    def update_audio(self, audio_path_pattern):
        """ 音源ファイルの一覧をパターンマッチングで探す
        """
        self.audio_files = glob.glob(audio_path_pattern)
        self.ready = len(self.audio_files) > 0


    def clear_cache(self):
        """ キャッシュをクリアする
        """
        self.cache = {}


    def __clean_fname(self, fname):
        """ ファイル名の最後に_があれば削る（命名規則でミスったので対応）
        """
        while True:
            if fname[-1] != "_":
                break
                
            fname = fname[:-1]
                
        return fname


    def __match(self, image_file_path, ext=".mp3"):
        """ 引数で渡された画像ファイル（のパス）に合致する音源を探し、その結果を返す
        """
        if self.ready == False:
            return "", "no audio", 0, 0, 0

        # 検索に必要な情報を作成
        fname_img, fname_music, loc, ts, tw = split_fname(image_file_path, ext)   # 画像ファイルのファル名を分解
        name, ext = os.path.splitext(fname_music)
        fname_music = self.__clean_fname(name) + ext
        key = (loc, fname_music)   # 辞書から音源ファイルのパスを取り出すためのキーを作成
        te = ts + tw

        # 検索
        if key in self.id_dict2:     # 辞書内に完全に合致する物があれば、それを返す。音源の親フォルダをその名前のハッシュ値4桁で区別するのでほぼ間違いない。
            #print("this file is matched.", fname_img)
            path_music = self.id_dict2[key]
            return path_music, "best", ts, te, tw
        
        elif key in self.id_dict1:     # 辞書内に合致する物があれば、それを返す。音源の親フォルダをその名前のハッシュ値2桁で区別するので、ファイル名が同じ場合にたまに間違える。
            #print("this file is matched.", fname_img)
            path_music = self.id_dict1[key]
            return path_music, "best", ts, te, tw

        else:
            # 類似したファイルを探す
            ## 一致するファイル名があるかどうかだけで検査（複数のフォルダに同じファイル名の音源があると一致しない事の方が多いだろうから注意）
            if fname_music in self.fname_dict:      
                path_music = self.fname_dict[fname_music]
                return path_music, "weak", ts, te, tw

            else:
                ## ファイル名の類似性で突き合わせる（音源のファイル名を一部変更していた場合に役に立つと思う）
                name_head = fname_music[:17]
                scores = np.array([Levenshtein.ratio(name_head, x) for x in self.fname_trimeds])
                i = np.argmax(scores)
                score = scores[i]
                if score > 0.7:
                    path_music = self.audio_files[i]
                    return path_music, "weak", ts, te, tw

        return "", "no match", ts, te, tw


    def match(self, image_file_path):
        """
        辞書でマッチするものがあれば（best）、それを返す。
        同名で拡張子だけが違う音源があれば、.wavを優先して返す。
        """
        image_file_path_temp = os.path.join(os.path.basename(os.path.dirname(image_file_path)), os.path.basename(image_file_path))

        # キャッシュを確認し、あればそれを返す
        image_file_path_temp_mirror = image_file_path_temp[::-1]   # 文字列反転
        time_mirror, name_mirror = re.split(r'_{2,4}', image_file_path_temp_mirror, maxsplit=1)
        key = name_mirror[::-1]

        if key in self.cache:
            return self.cache[key]


        # キャッシュがない場合は、マッチするものを探す
        audio_path1, condition1, ts, te, tw = self.__match(image_file_path, ".wav")
        audio_path2, condition2, ts, te, tw = self.__match(image_file_path, ".mp3")

        # 最も妥当な検索結果を返す
        if condition1 == "best":
            print(f"  Best match file: {image_file_path_temp} --is-- {audio_path1}　.")
            ans = (audio_path1, ts, te, tw)
            self.cache[key] = ans
            return ans
        elif condition2 == "best":
            print(f"  Best match file: {image_file_path_temp} --is-- {audio_path2}　.")
            ans = (audio_path2, ts, te, tw)
            self.cache[key] = ans
            return ans
        elif condition1 == "weak":
            print(f"  ★Weak match file: {image_file_path_temp} --is-- {audio_path1}　かもしれない。")
            ans = (audio_path1, ts, te, tw)
            self.cache[key] = ans
            return ans
        elif condition2 == "weak":
            print(f"  ★Weak match file: {image_file_path_temp} --is-- {audio_path2}　かもしれない。")
            ans = (audio_path2, ts, te, tw)
            self.cache[key] = ans
            return ans
        else:
            print(f"  No match file: {image_file_path_temp}.")
            ans = ("", ts, te, tw)
            self.cache[key] = ans
            return ans







# label_from_likelifoods.pyよりコピー＆改編
def to_binary(df, tags, th, positive=True, ignore=""):
    """ 識別結果に対して、尤度以上となった部分を1とする
    df: pandas.DataFrame, 
    tags: list<str>, 処理対象のクラス
    th: float, 尤度に対する閾値
    positive: bool, Trueなら、target_kindの結果を読み込む。Falseならtarget_kind以外の結果を読み込む。
    ignore: str, 無視する列
    """
    df_ = df.copy()
    df_ = df_[["fname", "s", "w"] + tags]  # 抽出
    
    for c in tags:
        vals = df_[c].values
        results = (vals >= th)
        if positive == False:
            results = ~results

        df_[c] = results

    df_[ignore] = df[ignore].values
        
    return df_.copy()




def get_length(file_path):
    """ メタ情報から音源や動画の長さを確認して返す
    """
    name, ext = os.path.splitext(file_path)
    ext = ext[1:].lower()

    length = 100 * 3600  # 48 hours
    audio_info = None

    if ext in ["mp3", "aac", "m4a", "flac", "wav", "wma"]:    # 音声データへの対応
        if ext == "mp3":
            audio_info = MP3(file_path).info
        elif ext in ["aac", "m4a"]:
            audio_info = MP4(file_path).info
        elif ext == "flac":
            audio_info = FLAC(file_path).info
        elif ext == "wav":
            audio_info = WAVE(file_path).info
        elif ext == "wma":
            audio_info = ASF(file_path).info

        if audio_info is not None:
            length = audio_info.length

    else:     # 動画への対応
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            length = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            

    if length == 100 * 3600:
        print(f"Warning: File length is unknown >> {file_path}")
    return length




def get_datetime(fpath, method="normal", timestamp_mode="start"):
    """ ファイル（の名前や保存時刻）から、日付情報を抽出する
    """
    #print(fpath)  # debug
    t = dt(2000,1,1)

    if method == "normal":
        r = re.compile(r"(?P<yymmdd>\d{6})_(?P<hhmm>\d{4})")
        bname = os.path.basename(fpath)
        m = r.search(bname)
        if m:
            yymmdd = m.group("yymmdd")
            hhmm = m.group("hhmm")
            date_str = f"{yymmdd}{hhmm}"
            t = dt.strptime(date_str, '%y%m%d%H%M')

    elif method == "SF":
        bname = os.path.basename(fpath)
        date_str = bname[:14]
        t = dt.strptime(date_str, '%Y%m%d%H%M%S')
        
    elif method == "timestamp":  # ファイルの更新日時（ICレコーダーの場合は大抵録音開始時刻）
        unix_time = os.path.getmtime(fpath)

        if timestamp_mode != "start":   # "end"ならTrue
            length = get_length(fpath)
            if length < 48 * 3600:
                unix_time -= length
            else:
                print(">>> Warning: can't get file length... @{fpath}")

        t = dt.fromtimestamp(unix_time)

    # デバッグ用に出力
    #print("get_datetime() sample: ", bname, t)

    return t



def count(df1, tags, dirs_list, delta, setting, skips=set()):

    for tag_ in tags:
        if tag_ in skips:
            continue
        print(f"**now {tag_}**")

        # 集計の期間を決める
        t_min = df1["time"].min()
        t_max = df1["time"].max()
        ts = dt(t_min.year, t_min.month, t_min.day)
        te = dt(t_max.year, t_max.month, t_max.day) + relativedelta(days=1)

        # 集計の時間分解能を決める
        step = "daily"
        if delta == relativedelta(months=1):  # 月ごとの処理なら、毎月1日を起点とする
            ts = dt(ts.year, ts.month, 1)
            step = "monthly"
        elif delta == relativedelta(weeks=1): # こちらは週ごとの処理の場合
            if ts.isoweekday() != 7:  # 日曜日起点
                ts = ts - td(days=ts.isoweekday())
            step = "weekly"

        # 検出数をカウント
        df4 = pd.DataFrame()
        count_all = 0
        
        for dir_ in dirs_list:
            df2 = df1.copy()
            if dir_ != "all":
                df2 = df2[df2["dir_name"] == dir_]
            t = ts
            counts = []
            dates = []

            while t < te:
                df3 = df2[(df2["time"] >= t) & (df2["time"] < t + delta)]
                if len(df3) != 0:   # 観測データの無い日を回避
                    df5 = df3[df3[tag_] == 1]
                    counts.append(len(df5))
                else:
                    counts.append(float("nan"))
                
                dates.append(t)
                t += delta
            df4[f"{dir_}"] = counts
            count_all += np.sum(np.nan_to_num(np.array(counts)))
        df4["date"] = dates

        
        # 少数しかないクラスを無視する
        if count_all < setting["min_limit"]:
            skips.add(tag_)
            continue

        # 結果の保存
        #df4 = df4.sort_index(axis=1)
        df4 = df4[natsorted(df4.columns)]   # 自然な感じで列をソート
        df4 = df4.set_index("date")
        df4.to_excel(os.path.join(setting["save_dir"] , f"stat_result_{step}_{tag_}.xlsx"), index=True)

        # グラフの保存
        if setting["graph_save"]:
            # 折れ線グラフ
            plt.figure()
            plt.style.use('tableau-colorblind10')
            for c in df4.columns:
                df_ = df4.dropna(subset=[c])
                plt.plot(df_.index, df_[c], label=c)
            plt.legend()
            plt.grid(True)
            plt.ylabel(f"{tag_} count")
            #df4.plot(grid=True, title=f"{tag_} counts")
            plt.savefig(os.path.join(setting["save_dir"] , f"stat_result_line_{step}_{tag_}.png"))
            plt.close('all')

            # 積み立てグラフ
            plt.figure()
            df4.plot.area()
            plt.grid(True)
            plt.ylabel(f"{tag_} count")
            plt.savefig(os.path.join(setting["save_dir"] , f"stat_result_area_{step}_{tag_}.png"))
            plt.close('all')

    return skips



# song_detection6.pyより移植
def extend(arr):
    """ 膨張
    """
    roll_l = np.roll(arr, -1)
    roll_r = np.roll(arr,  1)
    roll_or = np.logical_or(roll_l, roll_r)
    extended_arr = np.logical_or(roll_or, arr)
    return extended_arr


# song_detection6.pyより移植
def shrink(arr):
    """ 収縮
    """
    roll_l = np.roll(arr, -1)
    roll_r = np.roll(arr,  1)
    roll_and = np.logical_and(roll_l, roll_r)
    shrinked_arr = np.logical_and(roll_and, arr)
    return shrinked_arr


def save_timeslot(df, tags, setting, skips=set()):
    """ 種ごと・音源ファイルごと、時間ごとに検出されたフレーム数を保存する
    例えば、渡りをする鳥の個体数の把握に役立つと思う。
    """
    time_step = setting["stat2"]["time_step"]  # minute
    delta = relativedelta(minutes=time_step)
    extend_times = setting["stat2"]["extend_times"]
    shrink_times = setting["stat2"]["shrink_times"]

    for tag_ in tags:
        if tag_ in skips:
            continue
        print(f"**now {tag_}**")
        file_unique = sorted(df["fname"].unique())

        with open(os.path.join(setting["save_dir"] , f"stat_result_timeslot_{tag_}.csv"), "w", encoding="utf-8-sig") as fw:
            fw.write(f"fname,start_time,time_step[min],t1,t2,・・・\n")

            for i, file in enumerate(file_unique):
                
                # 結果や時間などを収集
                df_ = df.loc[df["fname"] == file_unique[i], ["fname", tag_, "time"]]
                result = df_[tag_].values
                t_min = df_["time"].min()
                t_max = df_["time"].max()
                ts = dt(t_min.year, t_min.month, t_min.day, t_min.hour, t_min.minute // time_step * time_step, 0)
                te = dt(t_max.year, t_max.month, t_max.day, t_max.hour, t_max.minute // time_step * time_step, 0) + relativedelta(minutes=time_step)

                # 個体数をカウントするために、連続区間をつなげる。これを膨張・収縮処理で実現する
                ## この処理は検出結果が時間的に連続であることを前提にするので注意
                ## 膨張回数は鳥によって変えた方がいいと思う。
                if np.sum(result) > 0:
                    b = result
                    for _ in range(extend_times):
                        b = extend(b)
                    for _ in range(shrink_times):
                        b = shrink(b)
                    result = np.array(b)

                # 鳴き始めだけ1にする
                result_ = [result[0]]
                for j in range(1, len(result)):
                    if result[j-1] == 0 and result[j] == 1:
                        result_.append(1)
                    else:
                        result_.append(0)

                # 集計
                t = ts
                counts = []
                while t < te:
                    df_a = df_[(df_["time"] >= t) & (df_["time"] < t + delta)]
                    counts.append(np.sum(df_a[tag_].values))
                    t += delta
                
                # 保存
                counts = [str(x) for x in counts]   # 文字列に変換
                fw.write(f"{file},{ts},{time_step},{','.join(counts)}\n")




def save_detected_hour(df, tags, dirs_list, setting, skips=set()):
    """ 鳴いた時刻と通算日の関係がわかる散布図を保存する
    """

    # 日の出・日の入り時刻の計算
    naze = ephem.city(setting["stat3"]["city"])
    naze.lat = str(setting["stat3"]["lat"])
    naze.lon = str(setting["stat3"]["lon"])
    origin = dt(2024, 1, 1, tzinfo=datetime.timezone(datetime.timedelta(hours=9)))
    x = []
    y_rising = []
    y_setting = []

    for i in range(365):
        day = origin + td(days=i)
        naze.date = day
        sun = ephem.Sun()

        t = ephem.localtime(naze.next_rising(sun))
        h = t.hour
        m = t.minute
        Hr = h + m / 60
        
        t = ephem.localtime(naze.next_setting(sun))
        h = t.hour
        m = t.minute
        Hs = h + m / 60
        
        x.append(i)
        y_rising.append(Hr)
        y_setting.append(Hs)


    # グラフ化
    marker = [".", ",", "o", "^", "v", ">", "<", "s", "+", "D", "d", "1", "2", "3", "4"]

    for tag_ in tags:
        if tag_ in skips:
            continue
        print(f"**now {tag_}**")
        plt.figure()

        df1 = df[df[tag_] == 1]
        for i, dir_ in enumerate(dirs_list):
            df2 = df1.copy()
            if dir_ != "all":
                df2 = df2[df2["dir_name"] == dir_]
        
            plt.scatter(df2["total_day"], df2["timeH"], marker=marker[i], label=f"{dir_}", alpha=0.6)
            
        plt.plot(x, y_setting, linestyle="--", label="Sun Setting")
        plt.plot(x, y_rising, linestyle="-", label="Sun Rising")
        plt.xlabel("通算日 [day]")
        plt.ylabel("時刻 [hour]")
        plt.ylim([0, 24])
        plt.xlim([0, 365])
        plt.grid(True)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()   # 凡例のはみだし防止
        plt.savefig(os.path.join(setting["save_dir"] , f"stat_result_scatter_{tag_}.png"))
        plt.close('all')




def create_detected_calenders(setting):
    tag = setting["tag"]
    dir_info_change = True
    os.makedirs(setting["save_dir"], exist_ok=True)


    # データを読み込み
    ## 画像から作製
    if setting["source"] == "image":   
        df, dirs_list = read_images(setting)    
        if len(df) == 0:
            return

        if not os.path.exists(df.at[0, "fname"]):
            dir_info_change = False   # 予測に使われた音源の情報と突き合わせてfnameを構築していれば、Trueで良い+
        
        #df.to_csv(os.path.join(setting["save_dir"] , "df_from_image.csv"))   # for debug

    ## 尤度が記載された識別結果のファイルから作製
    elif setting["source"] == "likelihood_file": 
        df = read_likelihoods(setting["likelihood_file"])   # 予測結果を読み込み
        if len(df) == 0:
            print("warning: DataFrame size from likelihood_file is zero.")
            return

    # 音源のディレクトリの情報を整理
    if dir_info_change:
        fnames = df["fname"].values
        dirs = [os.path.basename(os.path.dirname(fpath)) for fpath in fnames]
        dirs_list = list(set(dirs))
        df["dir_name"] = dirs

    # パスの編集
    for str_pair in setting["path_replace"]:
        a, b = str_pair
        if a == "":
            continue

        for i in range(len(df)):
            fname = df.at[i, "fname"]
            df.at[i, "fname"] = fname.replace(a, b)

    #df.to_csv(os.path.join(setting["save_dir"] , "df.csv"))   # for debug


    # 2値化
    if tag == "":
        tags = list(df.columns[3:-1])
    elif isinstance(tag, str):
        tags = [tag]
    else:
        tags = tag

    df = df.reset_index(drop=True)
    df1 = df.copy()
    df1 = to_binary(df, tags, setting["th"], ignore="dir_name")


    # 日時の列を作る
    print("--日時の列を作成中--")
    fnames = df1["fname"].values
    times = [get_datetime(fpath, method=setting["datetime_method"], timestamp_mode=setting["timestamp_mode"]) for fpath in fnames]
    df1 = df1.copy()
    df1["recording_start_time"] = times
    df2 = df1.loc[~df1["s"].isna()]
    df1.loc[~df1["s"].isna(), "time"] = df2["recording_start_time"] + df2["s"].astype('timedelta64[s]')
    if setting["debug"]:
        df1.to_excel(os.path.join(setting["save_dir"] , "df1.xlsx"), index=False)   # for debug

        # 音源ファイルの録音開始時刻（推定）のみを保存
        df_ = df1.copy()
        df_ = df_.drop_duplicates(keep="first", subset="fname")
        df_ = df_[["fname", "recording_start_time"]]
        df_.to_excel(os.path.join(setting["save_dir"] , "df_audio.xlsx"), index=False)   # for debug
    

    # サブディレクトリの扱い
    dirs_list = natsorted(dirs_list)
    if setting["basedir_separation"] == False:
        dirs_list = ["all"]


    print(df1.head())

    # 日毎の集計
    skips = set()
    if "daily" in setting["stat1"]["target"]:
        print("--日毎に集計中--")
        skips = count(df1, tags, dirs_list, relativedelta(days=1), setting, skips)

    if "weekly" in setting["stat1"]["target"]:
        print("--週毎に集計中--")
        skips = count(df1, tags, dirs_list, relativedelta(weeks=1), setting, skips)

    if "monthly" in setting["stat1"]["target"]:
        print("--月毎に集計中--")
        skips = count(df1, tags, dirs_list, relativedelta(months=1), setting, skips)

    # 年毎は必要か？

    

    # ファイル毎に分単位での時間スロットで処理（渡り調査用）
    if setting["stat2"]["available"]:
        print("--ファイル毎に集計中--")
        save_timeslot(df1, tags, setting, skips)



    # 検出された日時での散布図（季節変動・日修運動との関係を示す）
    if setting["stat3"]["available"]:
        df3 = df1.copy()

        # 年だけの列を作る
        df3["year"] = df3["time"].dt.year

        print("--時刻をhour単位で作製--")
        df3["timeH"] = df3["time"].dt.hour + df3["time"].dt.minute / 60 + df3["time"].dt.second / 3600

        # 通算日追加
        print("--通算日を追加--")
        df3["total_day"] = 0
        for i in range(len(df3)):
            date = df3.at[i, "time"]
            df3.at[i, "total_day"] = int(date.strftime("%j"))

        print("--検出された時刻で散布図を作成中--")
        save_detected_hour(df3, tags, dirs_list, setting, skips)








def set_default_setting():
    """ デフォルトの設定をセットして返す
    """
    params = {}
    params["tag"] = ""         # 評価対象のタグ
    params["source"] = "likelihood_file"
    params["likelihood_file"] = "prediction_likelihoods*.csv"
    params["image_source_dir"] = "."   # 画像が入っているフォルダのパス
    params["audio_source"] = ""        # 画像の作製に使われた音源の検索パターン。 /sample/*.mp2　など
    params["path_replace"] = [["", ""]]
    params["th"] = 0.15
    params["basedir_separation"] = False
    params["datetime_method"] = "SF"     # ファイルから日時を取り出す関数の選択。normal, SF, timestamp
    params["timestamp_mode"] = "start"   # タイムスタンプが録音開始時刻を刺しているなら"start"とすること。末尾なら"end"とする。
    params["graph_save"] = True
    params["save_dir"] = "."    # 保存先のフォルダ。　"."でカレントディレクトリを指す。
    params["min_limit"] = 200   # 検出数に対する閾値。これ以下だと保存されない。
    params["stat1"] = {"target": ["daily", "weekly", "monthly"]}
    params["stat2"] = {"available": True, "extend_times":3, "shrink_times":3, "time_step":1}
    params["stat3"] = {"available": True}
    params["debug"] = False
    return params



def read_setting(fname=""):
    """ 設定を読み込む。返り値は辞書で返す
    eval()を使っているので、不正なコードを実行しないように、気をつけてください。
    """
    param = set_default_setting()

    if fname != "":
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
        if "*" in param["likelihood_file"] and param["source"] == "likelihood_file":
            param["likelihood_file"] = glob.glob(param["likelihood_file"])[0]

    return param



def main():
    # 設定を読み込み
    setting = read_setting("statistics_setting.yaml")

    # 日付毎の検出数を保存する
    create_detected_calenders(setting)
    
    # 修了処理
    print("proccess is finished.")


if __name__ == "__main__":
    main()


