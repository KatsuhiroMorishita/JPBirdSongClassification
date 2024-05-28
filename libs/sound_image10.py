# purpose: 指定されたファイルを指定された区間でスペクトログラムを作成する。
# memo: デフォルトでは、5秒間隔でスペクトログラムを作成する。
#       区間の指定がなければ、全音源領域で画像を作る。
# author: Katsuhiro Morishita, Tamaki Shinmura
# history:
#             ver.3  前後にノイズだけでなく、元の音源を利用できるように改造
#                    get_location_ID()がMacでも動くように修正
#  2020-10-08 ver.4  研究室のNAS以外のファイルを処理する際に、既に作ったリストを流用するために、パスを書き換える機能を追加
#                    siの計算後にスライスで存在しない要素番号を指定してエラーが出ることが有ったので対応
#                    人の声の区間を画像化しないような機能を追加した。
#  2020-11-16 ver.5  静寂と車の通過時のスペクトルグラムの区別がつきやすい様に、音の強弱をスペクトログラムの中に埋め込む処理を追加 
#  2020-12-14        term==-1のとき、負の時間からの切り出しが指示されているとエラーになっていたので、負の時間が指定された場合に対応した。
#  2020-12-15        term==-1の時でもpad_mode=="noise"の時でもマスク処理が働くように改造
#  2021-03-02 ver.6  cut_bandを複数指定可能にして、ノイズ強度の基準を上端に取ったり下端に取ることを可能にした。
#  2021-03-09        cut_bandとemphasize_bandを併用するとちぐはぐな画像となってたので、スケーリングの順序を変更した。バージョンはそのまま。
#  2021-03-27        extendを選んだ際にどこまで画像として出力するか、指定している部分の計算ミスを修正した。バージョンはそのまま。
#  2021-07-28 ver.7  変数名の意味が合わないので、overlap_rateをshift_rateに変更した。
#  2021-11-20        保存フォルダを自動で作成するように変更した
#  2021-12-07 ver.8  locaiton判別にて、B1などに対応。アルファベットもABCDX以外にも対応した。
#  2022-09-26 ver.9  load_sound()にて、ファイルの読み込み失敗した場合への対応処理を追加
#  2022-11-06        librosa.feature.melspectrogram()でfeature warningが出ていたので、対応
#  2022-12-02        save_spectrogram_with_timelist()のバグ修正。以前もやった気がする…。
#  2022-12-05        音源のLRの読み込み分けに対応
#  2022-12-09        mp3音源のフォーマットによっては読み込みに失敗するらしく、その対応を入れた。
#  2022-12-20        load_sound()を外部から呼び出しやすいように修正
#  2023-08-30        音源データのスライスでまれにempty sliceデータになり、エラーが出ていたので対応。なぜかarray_data.sizeでは判定できなかった。
#  2023-11-02        location IDの一覧を保存する様にした。
#  2023-11-09        get_location_ID()が作るIDを4桁から6桁に拡張。これでフォルダ名称の文字数 + MD5 hash head 4 char となった。 
#  2023-11-22        引数で渡されたリスト内のファイルのパスから、IDの一覧を作成・保存するsave_ID_list()を作成（処理をmainから分離した）
#  2023-11-28        音源と同じフォルダにスペクトログラムを保存する機能を追加
#  2024-03-05 ver.10 背景にホワイトノイズを加える機能を追加した。cut_bandの機能も拡張して、特定帯域の音圧を0にする機能も追加。
# created: 2019-01-31      
import os, re, glob, time
import hashlib
import librosa
import librosa.display
import numpy as np
from PIL import Image, ImageOps
import copy
import unicodedata   # MacとWindowsのファイル名の扱いの違いを吸収するために使う




def get_melspectrogram_image(data, sr, n_mels=128, fmax=10000, n_fft=2048, hop_length=512, top_remove=0, 
                             emphasize_band=None, cut_band=None, raw=False, noise=0):
    """ 波形データをスペクトログラム画像に変換して返す。ただし、メルスケールになっているので注意
    data: list<float> or ndarray<float>, 音の波形データ（切り出したものでもOK）
    sr: float, 音源を読み込む際のリサンプリング周波数[Hz]
    n_mels: int, 周波数方向の分解能（画像の縦方向のピクセル数）
    fmax: int, スペクトログラムの最高周波数[Hz]。fmax < sr / 2でないと、警告がでる。
    n_fft: int, フーリエ変換に使うポイント数
    top_remove: int, 作成したスペクトログラムの上部（周波数の上端）から削除するピクセル数。フィルタの影響を小さくするため。
    emphasize_band: list<float or int>, 例：[1000, 3000, 0.5] 1 kHz〜3 kHzを強調する場合で、輝度を小さい順にソートした際の0レベルの閾値。0.5なら中央値を輝度0に調整する。
    cut_band: list<tuple<float or int>>, 例：[(0, 700, "upper")]。複数指定も可能。
    raw: bool, Trueだと、スペクトログラムに振幅強度を埋め込む
    noise: float, 0より大きいと、ノイズを加える。値は波形の標準偏差に対する相対値。
    """
    #print("debug info: ", data.size, len(data), np.sum(data), np.std(data), sr, n_mels, fmax, n_fft, hop_length)
    #data = np.nan_to_num(data, nan=np.nanmean(data))    # 非値を平均値に置換する

    data = data + np.std(data) * np.random.rand(len(data)) * noise  # add noise
    S = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=n_mels, fmax=fmax, n_fft=n_fft, hop_length=hop_length)  # スペクトログラムのデータを取得（2次元のndarray型）
    #log_S = librosa.power_to_db(S, ref=np.max)    # 対数を掛ける
    log_S = librosa.power_to_db(S, ref=np.max(S))  # 対数を掛ける
    if n_mels - top_remove < 0 or top_remove < 0:
        raise ValueError("n_mels - top_remove < 0 or top_remove < 0.")
    if top_remove > 0:
        log_S = log_S[:-top_remove]                # 上の方を削除
    min_ = np.min(log_S)
    max_ = np.max(log_S) - min_  # 0-255にスケーリングする高低差
    shape_ = log_S.shape         # 画像サイズを取得
    img_add = np.zeros(shape_)   # 画像と同じサイズで0行列を作成
    upper = 255

    # もし強調する帯域が指定されていれば、対応
    if emphasize_band is not None:
        # やや冗長だが、互換性のためにこういう構造にしている
        f1, f2, th = emphasize_band
        scale = librosa.core.mel_frequencies(n_mels=n_mels, fmax=fmax)   # メルスケールに相当する周波数のリストを取得
        if top_remove > 0:
            scale = scale[:-top_remove]         # 上の方を削除
        scale2 = (scale >= f1) & (scale <= f2)  # 指定範囲に該当する行がTrueになる1次元配列を作成
        masked_log_S = log_S[scale2]        # 指定帯域の画像を作成
        arr = masked_log_S.flatten()        # 画像を1次元化
        arr.sort()                          # 小さい順に並べる
        min_ = arr[int(len(arr) * th)]      # 最小値とする閾値を定める（この処理では、最小値扱いとする値が画像全体での最小値とは限らないことに注意）
        max_ = np.max(masked_log_S) - min_  # 0-250にスケーリングする高低差
        upper = 250                         # 輝度の最大値を250とする（あとでノイズを足すので小さくしている）
        img_add = np.random.normal(20, 7, shape_)   # 完全に0にならないように、ノイズを足す（0だと学習が進みにくい。のっぺりでも。なのでノイズを足す。）

    # スケーリング
    log_S = (log_S - min_) / max_ * upper     # 最小値が0、最大値がupperになるように調整（この時点では0以下やupper超えはありえる）

    # 帯域カット
    if cut_band is not None:
        for f1, f2, ref in cut_band:
            scale = librosa.core.mel_frequencies(n_mels=n_mels, fmax=fmax)   # メルスケールに相当する周波数のリストを取得
            if top_remove > 0:
                scale = scale[:-top_remove]         # 上の方を削除
            scale2 = (scale >= f1) & (scale <= f2)  # 指定範囲に該当する行がTrueになる1次元配列を作成
            
            # 該当帯域の音圧を0にする場合
            if ref == "zero":
                log_S[scale2] = 0.0
                continue

            # それ以外の場合
            masked_log_S = log_S[scale2]        # 指定帯域の画像を作成
            if ref == "all":            # 基準に全体を使う場合
                arr = masked_log_S
            elif ref == "upper":
                arr = masked_log_S[-1]  # 最高周波数対の1行分を取り出す（1次元配列）, 帯域の上を取る。
            else:  # lowerを想定
                arr = masked_log_S[0]   # 最高周波数対の1行分を取り出す（1次元配列）, 帯域の下を取る。
            log_S[scale2] = np.random.normal(np.median(arr), np.std(arr), masked_log_S.shape)  # 指定帯域をノイズで上書き




    # 上下反転
    img = np.flip(log_S, axis=0)
    
    # 画像に加工するために、最大値を255, 最小値を0にする    
    img = img + img_add          # 輝度値を足す（通常はゼロ行列なので意味はないが、場合によってはノイズ画像を足す）
    img[img < 0] = 0             # 輝度値の下限制限
    img[img > 255] = 255         # 輝度値の上限制限
    img = img.astype(np.int32)   # 画像として扱えるように、型を変える


    # 生波形の振幅を画像に埋め込む（これがないと、静寂と車の通過時のスペクトルグラムに差が生じない）
    if raw:
        h, w = shape_
        iw = int(len(data) / w)
        for i in range(w):
            wav = data[i * iw:(i + 1) * iw]
            amp = 10000
            m = np.log10(np.mean(np.abs(wav)) * amp + 1) / np.log10(amp) * 255  # 振幅の平均を輝度に変える. 
            if m > 255:
                m = 255
            img[0][i] = m

    return img


def save_spectrogram_with_window(data, params, rp0=0, terminal=None):
    """ 波形データを渡されたら、時間窓をずらしながらスペクトログラム画像を作成＆保存する
    引数のparamに格納されたtermが-1の場合は時間窓を使用しない。渡された音源で1つの画像を作成します。
    data: list<float> or ndarray<float>, 音の波形データ（切り出したものでもOK）
    params: dict, 設定を書き込んだ辞書
    rp0: int, 元々の音源データ先頭からの要素番号（スライスされていることを念頭に）
    """
    sr = params["sr"]
    hop = params["hop"]
    tag = params["tag"]
    fmax = params["fmax"]
    root = params["root"]
    term = params["term"]
    n_fft = params["n_fft"]
    n_mels = params["n_mels"]
    top_remove = params["top_remove"]
    shift_rate = params["shift_rate"]
    emphasize_band = params["emphasize_band"]
    cut_band = params["cut_band"]
    raw = params["raw"]
    noise_level = params["noise"]
    
    rp = 0   # read point
    exit_flag = False

    while True:
        # どこまで読み込むか決める
        if term == -1:
            ep = len(data)
        else:
            ep = int(rp + sr * term)   # end point

        # 最後の切り出しで、要素数が足りない場合を判断し、足りない分を付け加える
        if ep > len(data):    # 必然的にterm != -1の時のみTrueになり得る。基本的には、term!=-1で最後の要素の場合
            print("ep > len(data).")
            amp = np.median(np.abs(data))   # 振幅
            noise = list(amp * np.random.rand(ep - len(data)))  # 乱数のリスト
            data_ = list(data) + noise
            data = np.array(data_)   # ndarrayに変換
            exit_flag = True

        # 音源の切り出しと、サイズのチェック
        sliced_data = data[rp : ep]
        #print(f"**size of data is {len(sliced_data)}.")
        if len(sliced_data) < ep - rp or len(sliced_data) <= 0:
            print("len(data) is small...")
            break

        # データチェック
        if np.isnan(np.sum(sliced_data)):
            print("data is nan.")
            break

        # スペクトログラム作成
        hop_length = int(sr * hop)
        msg = "split"
        #try:
        img = get_melspectrogram_image(sliced_data, sr=sr, n_mels=n_mels, fmax=fmax, n_fft=n_fft, 
                                       hop_length=hop_length, top_remove=top_remove, 
                                       emphasize_band=emphasize_band, cut_band=cut_band, 
                                       raw=raw, noise=noise_level)  # スペクトログラムのデータを取得（2次元のndarray型）
        #except Exception as e:
        #    img = np.zeros((120,120), dtype=np.int8)
        #    msg = "error"

        # 加工
        #img[img < 120] = 0

        # 画像として保存
        im = Image.fromarray(np.uint8(img))
        t_start = (rp + rp0) / sr    # 切り出し開始時刻[s]
        t_width = (ep - rp) / sr     # 切り出し幅[s]
        fname = "{3}__{0}__{1:.1f}_{2:.1f}.png".format(tag, t_start, t_width, msg)
        path_list = [root, fname]
        path = os.path.join(*path_list)
        im.save(path)

        # rp更新
        rp += int(sr * term * shift_rate)   # shift_rateが0.5なら半分だけ重ねながらずらす。1で重なりなし。2なら1つ分開く。

        # 読み込み開始地点がデータサイズを超えたら終わる
        if rp > len(data):
            break

        # 終了を指示された地点をまたがったら終了
        if terminal is not None and ep >= terminal:
            print("reach the terminal.")
            break

        # 既に全て画像に変換済みなので、終了
        if term == -1:
            break

        # 足りない分を付け加えて処理したなら、終了
        if exit_flag:
            break



def change_path(file_path, replaced_str, replace_str):
    """ テキストに含まれるパス構造の中で、特定の文字列を特定の文字列に置換する
    ただし、置換後の文字列はreplace_strに一致するとは限らない（パスの指定として正しく機能すれば良い）
    """
    def normarize(path):
        field = re.split(r"\\|¥|/", path)    # パスをバックスラッシュ、円マーク、スラッシュで分解
        print("fuga", path, field)
        new_path = "/".join(field)           # パスを/で結合
        new_path2 = unicodedata.normalize("NFC", new_path)  # MacはNFD形式でファイル名を扱うので、Windowsと同じNFC形式に変換
        return new_path2

    n_path = normarize(file_path)  # パスを比較するために、パスの区切り文字を正規化
    str1 = normarize(replaced_str)
    str2 = normarize(replace_str)
    out = n_path.replace(str1, str2)   # パスを置換

    return out



def check_interference(mask_list: list, t: float, margin=0.1, side="right"):
    """ mask_listに格納された区間と干渉している場合は、干渉しない範囲（とりあえず始点）を返す
    人の声を避けるための処理です。
    mask_list: list<tuple<float, float>>, tが含まれてはならない区間リスト。時系列に並んでいることを前提に処理します。
    """
    if t < 0:
        t = 0

    ml = copy.deepcopy(mask_list)
    if side != "right":
        ml = ml[::-1]       # 逆順に並べる（時系列が逆になる）

    for s, w in ml:
        if s < t < (s + w):
            if side == "right":
                t = s + w + margin  # 区間の右側に変更（次以降のループで変更語が干渉しないかもチェック）
            else:
                t = s - margin      # 区間の左側に変更（次以降のループで変更語が干渉しないかもチェック）
            print("hoge", t, s, w, margin)
    return t


def mask(mask_list: list, size: int, t0: float, sr: float, margin: float):
    """ mask_listに格納された区間と干渉している場合は、干渉している部分が1の配列を返す
    人の声を避けるための処理です。
    mask_list: list<tuple<float, float>>, tが含まれてはならない区間リスト。時系列に並んでいることを前提に処理します。
    size: int, 配列のサイズ
    t0: 配列の最初の該当する時間[s]
    sr: int, サンプリング周波数[Hz]
    margin: float, マスク処理をかける際に、前後の方向に少し余裕を見て消すための時間幅[s]。0以上を設定のこと。
    """
    bins = np.zeros(size)     # 0の配列

    # mask_listに記載された区間を1にした配列を作成
    for s, w in mask_list:     # 開始時刻[s]とそこからの時間幅[s]
        s -= margin
        w += 2 * margin
        si = int((s - t0) * sr)
        ei = int(w * sr) + si
        if si < 0:
            si = 0
        if si < size and ei > 0:
            bins[si:ei] = 1

    return bins





def load_sound(file_path: str, params: dict, try_times=4):
    """ 指定された音源ファイルを読み込む
    file_path: str, 音源ファイルのパス
    params: dict, 設定を書き込んだ辞書
    """
    sr = params["sr"]
    load_mode = params["load_mode"]
    mono = True
    if "lr" in params:
        lr = params["lr"]
        if lr == "right" or lr == "left":
            mono = False

    # パスの編集が指示されていた場合、パスを置換する
    if "path_replace" in params and len(params["path_replace"]) == 2:   
        str1 = params["path_replace"][0]
        str2 = params["path_replace"][1]
        file_path = change_path(file_path, str1, str2)

    # 音源ファイルの読み込み。yに波形が配列で入り、srにはサンプリング周波数が入る
    if not os.path.exists(file_path):
        print("{} is not exists.".format(file_path))
        return
    else:
        print("{} is processing...".format(file_path))


    # 音源ファイル読み込み　失敗したときへの対応を含む
    success = False
    for _ in range(try_times):
        try:
            y, sr = librosa.load(file_path, sr=sr, res_type=load_mode, mono=mono)  # sampling frequency == sampling rate
            success = True
            break
        except Exception as e:
            print(str(e))
            print("--retry. wait 20 seconds.--")
            time.sleep(20)

    # 左右の音源の読み込み分け
    if mono == False:
        i = ["right", "left"].index(lr)
        y = y[i]

    if success:
        return y, sr, file_path
    else:
        print("file reading failed.")
        return None, sr, file_path




def save_spectrogram_with_timelist0(y, sr, file_path: str, loc: str, params: dict):
    """ 指定された音源ファイルから、スペクトログラムを作成・保存する
    y: ndarray, 音源の波形データ
    sr: float, サンプリングレート
    file_path: str, 音源ファイルのパス
    loc: str, 音源の録音された場所
    params: dict, 設定を書き込んだ辞書
    """
    # 設定の読み込み
    term = params["term"]
    positive = params["positive"]
    time_list = params["time_list"]
    mask_list = params["mask_list"]
    pad_mode = params["pad_mode"]


    # positive == Falseに備えて、対象時刻のリストを整理。Falseだとtime_list内の時刻範囲外を対象に処理を行う。
    if positive == True or len(time_list) == 0:
        _new_time_list = copy.deepcopy(time_list)   # len(time_list) == 0がTrueの時は、コピーしても意味はないのだけど、elseに行かないという意味。なお、len(time_list) == 0なのにelseに行っても処理は正常に行われる。
    else:
        _new_time_list = []
        start = 0
        for i in range(len(time_list) + 1):
            if i >= len(time_list):   # 最後までtime_listを読み切ったあと
                s = len(y) / sr
                w = 0
            else:
                s, w = time_list[i]    # 区間の始点[s]と時間幅[s]のペア
            if s - start > 0:
                _new_time_list.append((start, s - start))   # 始点までの時間が対象の区間となる
            start = s + w


    basename = os.path.basename(file_path)    # ファイル名のパスからディレクトリを除く
    name, ext = os.path.splitext(basename)    # 拡張子を取得
    tag = "{}__{}".format(loc, name)          # ファイルを保存する際のタグを生成
    params["tag"] = tag                       # パラメータとして格納

    # time_listに従い、スペクトログラムを作成する
    print("_new_time_list: ", _new_time_list)
    if len(_new_time_list) > 0:           # 区間リストが空じゃなかったら
        for start_time, width in _new_time_list:
            terminal = None
            si = int(start_time * sr)               # 切り出し開始インデックス番号
            ei = int((start_time + width) * sr)     # 切り出し終了インデックス番号
            si_ = si
            if si < 0:    # 時間start_timeが負だった場合に、スライス開始インデックスを制限する
                si = 0
            pad_width = term * 0.5 * np.random.rand()   # 先頭部分に加える時間の幅[s]。時間幅は最大でもtermの半分
            s0 = start_time

            # 一定の時間ごとに音源を切断
            ## 一定の時間幅で切り出す、かつ、前後にノイズの挿入が求められる場合
            if term != -1 and pad_mode == "noise":      # 先頭に乱数を加える。
                s0 = start_time - pad_width
                y_ = list(y[si : ei])                   # 時間のリストに基づき、音源データをスライスして、list型に変換
                amp = np.median(np.abs(y_))             # 振幅
                noise1 = list(amp * np.random.rand(int(pad_width * sr)))  # データの先頭に加えるノイズ
                noise2 = list(amp * np.random.rand(int(width * sr)))      # データの最後に加えるノイズ
                y_ = noise1 + list(y_) + noise2
                # 実質的なsiの再計算
                terminal = len(noise1) + (ei - si)
                si =  int(s0 * sr)
            
            ## 一定の時間幅で切り出す、かつ、指定時間より少し前からの切り出しが求められる場合
            elif term != -1 and pad_mode == "extend":    # 指定された区間よりも前から切り出す場合
                s0 = start_time - pad_width
                s1 = s0
                ss = 0
                if s1 < 0:    # 負だとスライスでエラーになるので、値の制限
                    ss = abs(s1)
                    s1 = 0
                e = start_time + width + 2 * term   # ちょっと先まで伸ばす
                si = int(s1 * sr)
                ei = int(e * sr)      # 切り出し終了インデックス番号の再計算
                # 音声の切り出しと、足りない部分にノイズを付け加える
                y_ = list(y[si : ei])  # 波形切り出し
                amp = np.median(np.abs(y_))             # 振幅
                noise1 = list(amp * np.random.rand(int(ss * sr)))     # データの先頭に加えるノイズ
                noise2 = list(amp * np.random.rand(int(term * sr)))   # データの最後に加えるノイズ
                y_ = np.array(noise1 + list(y_) + noise2)
                # 実質的なsiの再計算
                terminal = int((width + term) * sr)
                si =  int(s0 * sr) 
            
            ## termで切り出さない場合（start_timeからwidth秒間切り出す）
            else:
                y_ = list(y[si : ei])
                amp = np.median(np.abs(y_))   # ノイズを加えることになった場合のノイズの振幅
                # 負の時間からの切り出しが指示されていた場合、ノイズを加える
                if si_ < 0:
                    noise1 = list(amp * np.random.rand(abs(si_)))  # データの先頭に加えるノイズ
                    y_ = noise1 + list(y_)
                si = si_
                # 切り出した音源の長さが予定の幅に満たない場合、ノイズを加える
                if ei - si > len(y_):
                    noise2 = list(amp * np.random.rand(ei - si - len(y_)))  # データの最後に加えるノイズ
                    y_ = list(y_) + noise2
            
            # マスク処理が必要な場合
            if len(mask_list) > 0:
                margin = params["mask_margin"]
                amp = np.median(np.abs(y_))             # 振幅
                mask_bin = mask(mask_list, len(y_), s0, sr, margin)  # maskの部分が1のバイナリ列
                mask_inv = 1 - mask_bin   # 0と1を反転して、音の部分を1にする
                mask_noise = mask_bin * (amp * np.random.rand(len(mask_bin)))
                y_ = mask_inv * y_ + mask_noise

            y_ = np.array(y_)     # データ型をndarrayに変換
            save_spectrogram_with_window(y_, params, si, terminal)
    else:
        save_spectrogram_with_window(y, params)




def save_spectrogram_with_timelist(file_path: str, loc: str, params: dict):
    """ 指定された音源ファイルから、スペクトログラムを作成・保存する
    file_path: str, 音源ファイルのパス
    loc: str, 音源の録音された場所
    params: dict, 設定を書き込んだ辞書
    """

    # 音源ファイルの読み込み。yに波形が配列で入り、srにはサンプリング周波数が入る
    y, sr, file_path = load_sound(file_path, params)
    
    # 読み込みに失敗したら戻る
    if y is None:
        return 

    # スペクトログラムの作成と保存
    if len(y) > 0:
        save_spectrogram_with_timelist0(y, sr, file_path, loc, params)
    else:
        print("--size of sound data is zero.--")

    




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



def save_ID_list(fnames):
    """ 引数で渡されたリスト内のファイルのパスから、IDの一覧を作成・保存する 
    """
    dir_id_list = []
    
    for fname in fnames:
        # 親フォルダのID作製とその保存
        location = get_location_ID(fname)
        dir_id_list.append((fname, location))
        
    # 個々のファイルのリスト作成
    with open("location_files.csv", "w", encoding="utf-8-sig") as fw:
        fw.write("File Path,Directory ID\n")
        for fname, loc in dir_id_list:
            fw.write(f"{fname},ID_{loc}\n")

    # フォルダとIDのリストを作成
    with open("location_dirs.csv", "w", encoding="utf-8-sig") as fw:
        dir_dict = {}
        fw.write("Directory Name,Directory ID\n")
        for fname, loc in dir_id_list:
            dir_ = os.path.basename(os.path.dirname(fname))
            dir_dict[dir_] = loc

        for key, value in dir_dict.items():
            fw.write(f"{key},ID_{value}\n")

    return




def read_setting(fname, param={}):
    """ 設定を読み込む。返り値は辞書で返す
    eval()を使っているので、不正なコードを実行しないように、気をつけてください。
    param: dict, パラメータを格納した辞書
    """
    with open(fname, "r", encoding="utf-8-sig") as fr:
        lines = fr.readlines()

        for line in lines:
            line = line.rstrip()  # 改行コード削除
            if "#" in line:       # もしコメントが入っていたら、それ以降を削除
                line = line[:line.index("#")]
            line = line.rstrip()  # 右からスペースを消す
            if line == "" or "," not in line:
                continue
            param_name, value = line.split(",", 1)   # 最大1回の分割
            param[param_name] = eval(value)          # 文字列のパラメータは""等を付けること

    return param



def set_default_setting():
    """ デフォルトの設定をセットして返す
    """
    params = {}
    params["term"] = 5              #: int, スペクトログラムを作成する時間幅[s]。-1を渡すと、time_list以外では分割しない。
    params["fmax"] = 10000          #: int, スペクトログラムの最高周波数[Hz]。fmax < sr / 2でないと、警告がでる。
    params["shift_rate"] = 0.5  #: float, スペクトログラムを作成する際に、既に作成済みのものとどの程度離すかを決める割合。1.0で重なり量は0。0.5だと半分重なる。
    params["time_list"] = []    #: list<tuple<int, int>>, 区間指定。指定はタプルで(開始時間[s], 時間幅[s])と行う。空のリストの場合は音源の最初から最後までが画像作成の対象となる。
    params["positive"] = True   #: bool, Trueだと、time_listにある区間を処理対象とする。Falseだと、time_listの範囲外を処理対象とする。
    params["sr"] = 44100        #: float, 音源を読み込む際のリサンプリング周波数[Hz]
    params["load_mode"] = "kaiser_fast"    #: str, librosa.load()でres_typeに代入する文字列。読み込み速度が変わる。kaiser_fastとkaiser_bestではkaiser_fastの方が速い。
    params["top_remove"] = 8               #: int, 作成したスペクトログラムの上部（周波数の上端）から削除するピクセル数。フィルタの影響を小さくするため。
    params["hop"] = 0.025       #: int, 時間分解能[s]
    params["root"] = ""         #: str, ファイルを保存するフォルダ。""だと、カレントディレクトリに保存する。
    params["n_mels"] = 128      #: int, 周波数方向の分解能（画像の縦方向のピクセル数）
    params["n_fft"] = 2048      #: int, フーリエ変換に使うポイント数
    params["pad_mode"] = None   # 切り出した音源の前と後につけるデータのモード。noiseとextendがある。
    params["file_names"] = []
    params["emphasize_band"] = None   # 強調する帯域。例：[1000, 3000, 0.5] 1 kHz〜3 kHzを強調する場合で、輝度を小さい順にソートした際の0レベルの閾値。0.5なら中央値を輝度0に調整する。
    params["cut_band"] = None    # カットしたい帯域。例：[(0, 700, "upper")]
    params["path_replace"] = []  # ファイルのパスを置換する場合に利用（普段はNAS内のファイルを処理しているが、高速化のために外付けSSDにデータを移した場合を想定）
    params["mask_list"] = []     # スペクトログラムを作ってはならない区間のリスト
    params["raw"] = False        # スペクトログラムに生の音声の振幅情報を埋め込むかどうか。Trueで埋め込む。
    params["mask_margin"] = 0    # マスク処理をかける際に、前後の方向に少し余裕を見て消す場合は0以上を設定のこと。
    params["lr"] = ""            # 音源のLRの読み込み分けを行うかを決める。"right"でR。
    params["location_save_only"] = False  # Trueだと、location IDのみを保存する
    params["noise"] = 0.0        # 0より大きい値で、ノイズを加える
    return params



def main():
    # 設定を読み込み
    setting = set_default_setting()
    setting = read_setting("sound_image_setting10.txt", setting)
    fnames = sorted(setting["file_names"])
    print(fnames)

    
    # 保存先の作成
    root = setting["root"]
    if root != "" and root != "--same_dir":
        os.makedirs(root, exist_ok=True) 

    # 処理対象のファイルを1つずつ処理
    for fname in fnames:
        # 親フォルダのID作製
        location = get_location_ID(fname)
        

        # スペクトログラムの保存
        if not setting["location_save_only"]:

            # 保存先が音源と同じフォルダを指定されていた場合
            if root == "--same_dir":
                setting["root"] = os.path.dirname(fname)

            # スペクトログラムの作成と保存
            save_spectrogram_with_timelist(fname, location, setting)   # 区間リストは空なので、全範囲が対象


    save_ID_list(fnames)


    # 修了処理
    print("proccess is finished.")



if __name__ == "__main__":
    main()
