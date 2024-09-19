#!/usr/bin/python
# -*- coding:utf-8 -*-
# writen for python 3
# purpose: 尤度のリストからAUCを計算する
# history: 
#   2020-04-27 ver.1
#   2020-12-20 ver.2  sound_file_prediction2.pyに対応
#   2021-03-07 ver.3  AUC計算の検算機能を実装。F値の最大なども求めるようにした。
#   2021-03-09 ver.3v2 v3で実装した処理の高速化。numbaを使う。
#   2021-04-05 ver.4  複数のファイルを処理できるように改造（他のファイルから処理を呼び出しやすくもなった）
#   2021-10-28        設定の読み込みをより安全にした。また、結果の保存先を変更した。サンプル用だが、ファイル名だけでの比較にも対応した。
#   2022-01-06        yamlの保存をunicode形式に変更。
#   2022-12-14        既存のフォルダ探索に使う正規表現でsearchからmatchに変更した。これで文字列先頭の一致をチェックするようにした。
#   2023-01-12        感度も求めるように変更して、PR曲線とその曲線化の面積を求めるように変更
#   2023-01-13        numbaで高速化した部分で、parallel=Trueとするとエラーになるので、Falseに変更した（20倍くらい遅くなるかな）
#   2023-10-02        predict.pyの保存フォルダに名前を付ける機能に対応して、last_dirnumber(), last_dirpath(), next_dirpath()を修正。
#   2024-03-19        実行時の引数に対応し、パスの比較で、音源ファイルの上にあるフォルダの階層数を設定できるようにした。これで文字列置換がほとんど不要になる。
#                     鳴き声区間ごとの検出率（感度に近い）を計算するように機能追加
#   2024-03-21        鳴き声区間ごとの検出率のグラフを保存する様にして、偽陽性・偽陰性の区間リストを保存する様に機能を追加した。
#                     正解の区間リストに#でコメントを入れられるようにもした。
#   2024-03-27        鳴き声単位で閾値ごとの感度を求める処理が非常に時間がかかるので、省略できるようにした。また、予測結果を引数で指定する際に指定しやすくなるよう機能を追加した。
#   2024-04-01        引数処理のバグを修正
#   2024-05-15        区間リストを読み込む処理を更新して、対応書式を広げた。また、区間同士の比較ロジックを変更し、暫定でマージンを廃止した。
#                     閾値ごとのPrecision/Recallを保存するようにした。
#   2024-08-06        マージンの処理を廃止
#                     comp2で、区間の比較ロジックを修正した。また、和集合を求めるように変更した。
#                     正解区間と予測区間が一部でも被っていたら良しとみなす場合のFalse Negativeを求めるロジックのバグも取れたと思う。
#  2024-09-19         runsフォルダが無い場合に作るように変更
# author: Katsuhiro Morishita
# created: 2020-04-27
# license: MIT. If you use this program for your study, you should write Acknowledgement in your paper.
import os, re, glob, copy, time, pprint, shutil, ast, argparse
from datetime import datetime as dt
from numba import jit, njit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, recall_score, precision_score
import pandas as pd
import yaml





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




def read_likelihoods(likelihood_file, compare_depth=-1):
    """ 予測で出力された尤度のリストを読み込む
    likelihood_file: str, 尤度が格納されたテキストファイルのパス
    compare_depth: int, パスを比較する深度。0だとファイル名のみで比較する。
    """
    df = pd.read_csv(likelihood_file, encoding="utf-8-sig")
    #print("-----------------")
    #print2(df.head())

    # パスの加工の前に、バックアップとしてパスをコピー
    df["fname_back"] = df["fname"].values

    # パスの加工（ファイル名だけとか、1つ上の親フォルダまでとかで加工）
    if compare_depth >= 0:
        path_change = lambda x: os.path.join(*re.split(r"[\\/]", x)[-1 - compare_depth:])
        df.iloc[:, 0] = df.iloc[:, 0].map(path_change)

    print("++++++++ likelihood_file after path processing +++++++++")
    print2(df.head())

    return df



# save_spectimage_with_list2.pyより移植、改編
## コメントへの対応
## パスの円マークを/に置換するように変更
## パスの深さを指定できる様に変更
## パス名に"や'があっても削除
## 区間数が少ない場合に空の辞書を返す
def read_list(fname, compare_depth=-1, ignore_size=None):
    """ 音源と時間区間のリストを読み込む
    fname: str, 読み込むファイル名
    compare_depth: int, パスを比較する深度。0だとファイル名のみで比較する。
    """
    ans = {}
    count = 0
    path_change = lambda x: x.replace("\\", "/")   # 円マーク（windowsでは円マーク）を/に書き換えるラムダ関数

    with open(fname, "r", encoding="utf-8-sig") as fr:
        lines = fr.readlines()  # 全行読み込み
        for line in lines:
            line = line.rstrip()  # 改行コード削除
            if "#" in line:       # コメント文への対応
                line = line[:line.find("#")]
                line = line.rstrip()  # 空白削除
            if len(line) == 0:    # 空行を避ける
                continue

            field = line.split(",")
            path = field[0]       # ファイルのパスを取得

            # パスの区切り文字を統一
            path = path_change(path)

            # パスに余計な文字があれば排除
            if '"' in path:
                path = path.replace('"', "")
            if "'" in path:
                path = path.replace("'", "")


            # パスの加工（ファイル名だけとか、1つ上の親フォルダまでとかで加工）
            if compare_depth >= 0:
                path = os.path.join(*re.split(r"[\\|/]", path)[-1 - compare_depth:])

            times = field[1:]     # 時刻のリストを格納
            if len(times) < 2:    # 時刻が入っていない行は避ける
                continue

            # 時刻のリストを、(読み出し開始時刻, 時間幅)によるタプルのリストに加工
            time_list = []
            for i in range(0, len(times), 2):
                if times[i] == "":           # 文字が空だったら次へ（終端のはずだが）
                    continue
                start = float(times[i])      # 文字列を数値に変換
                width = float(times[i + 1])
                time_list.append((start, width))

            #print(path)
            #print(time_list)

            ans[path] = time_list  # 辞書として保存
            count += len(time_list)
    
    # 規定サイズ以下だったら空の辞書を返す
    if ignore_size is not None and ignore_size >= count:
        return {}
    else:
        return ans




class DetectChecker:
    """ 正解の区間リストと比較して、指定された時刻[秒]が区間内であるかどうかを返すクラス
    なぜクラスにしたのか、忘れた・・・。
    """
    def __init__(self, time_list_dict):
        self._time_list_dict = time_list_dict

    def check_time_pair(self, fpath, time_pair):
        """ time_pairで指定した範囲がコンストラクタでセットされた辞書内のリストと合致するか確認し、合致するとTrueを返す
        fpath: str, 検査したいファイルのフルパス
        time_pair: list or tuple <float, float>, 開始時間[s]と時間幅[s]のペア
        """
        #print(self._time_list_dict.keys())
        #exit()

        if fpath in self._time_list_dict:
            time_lists = self._time_list_dict[fpath]

            s2, w2 = time_pair
            e2 = s2 + w2
            for s1, w1 in time_lists:
                e1 = s1 + w1
                if (s1 <= s2 <= e1) or (s1 <= e2 <= e1) or ((s2 <= s1) and (e2 >= e1)) :
                    return True
        else:
            pass
            #print(f"warning::  {fpath} is not inclued in timelist.")
            #exit()

        return False




# if文とPythonのsum()を排除して1000倍速、さらにnumbaの力で20倍で、トータル1万倍速くなった。
@njit("f8[:,:](bool_[:], f8[:], f8[:])", parallel=False)   # parallel=Trueとすると、いつの間にかエラーになるようになった…
def check_amount(true, score, thresholds):
    """ AUC計算の検算
    true: ndarray<bool>, 真値の入った一次元配列
    score: ndarray<float>, 予測された尤度
    thresholds: list or ndarray <float>, TPRなどを計算する閾値
    """
    s, = thresholds.shape
    fpr, tpr, tp, p, pp, fp, n, P, F = np.zeros(s), np.zeros(s), np.zeros(s), np.zeros(s), np.zeros(s), np.zeros(s), np.zeros(s), np.zeros(s), np.zeros(s)  # bad positive rate（笑）
    #print(true[:5])
    #print(score[:5])

    # 何度も計算しなくてよいものはここで計算
    TP_FN = np.sum(true) + 0.0000000001   # 真に陽性の数
    FP_TN = len(true) - TP_FN  # 真に陰性の数

    i = 0
    for th in thresholds:
        y_bin = score >= th       # 閾値以上のものを1、それ以外を0にする
        tp_list = y_bin & true    # 真陽性（陽性と判定して、実際に陽性だったものが1）
        fp_list = (y_bin == 1) & (true == 0)   # 偽陽性（真は陰性で、予想で陽性だったもの。間違えて陽性にしたものが1。）
        #print("hoge: ", y_bin == 1)
        #print("fuga: ", true == 0)
        #fn_list = (y_bin == 0) & (true == 1)   # 偽陰性
        
        # 真陽性の数などを求める
        TP = np.sum(tp_list)   # 真陽性の数
        FP = np.sum(fp_list)   # 偽陽性の数
        TP_FP = np.sum(y_bin) + 0.0000000001  # 予想で陽性となった数

        # 特異度・真陰性率
        #tn_list = (~y_bin) & (~true)    # 真陰性
        #tnr = sum(tn_list) / FP_TN

        # 適合率precision（陽性と判定したもののうち、実際に陽性であった割合）
        P_ = TP / TP_FP
        
        # 真陽性率・感度recall（真に陽性のものを陽性と判定できた割合）
        tpr_ = TP / TP_FN
        
        # F値
        f = 2 * tpr_  * P_ / (tpr_ + P_ + 0.0000000001)

        # 偽陽性率（真に陰性のものを陽性と判定した割合）
        fpr_ = FP / FP_TN    # 分母は真に陰性の数

        # 配列に格納
        tpr[i] = tpr_     # 真陽性率
        fpr[i] = fpr_     # 偽陽性率
        tp[i] = TP        # 真陽性の数（陽性と判定したなかで当たっていた数）
        p[i] = TP_FN      # 真に陽性の数
        pp[i] = TP_FP     # 予想で陽性となった数
        fp[i] = FP        # 偽陽性の数
        n[i] = FP_TN      # 真に陰性の数
        P[i] = P_     # 適合率
        F[i] = f      # F値

        i += 1

    # 結果を返すために、すべての配列をまとめる（2次元配列になる）
    hoge = np.vstack((fpr, tpr, tp, p, pp, fp, n, P, F))

    return hoge



def check_amount0(true, score, thresholds):
    t = time.perf_counter()
    a = check_amount(true.astype(np.bool_), score, thresholds)

    # DataFrameへ格納（DataFrameへのappendは時間がかかるので、配列で列単位で入れる）
    df = pd.DataFrame({'fpr2': a[0], 
                       'tpr2': a[1], 
                       'TP': a[2], 
                       "真に陽性の数 TP+FN": a[3], 
                       "予想された陽性の数 TP+FP": a[4], 
                       "FP": a[5], 
                       "真に陰性の数 FP+TN": a[6], 
                       "適合率P": a[7], 
                       "F値": a[8],
                       }
                     )
    df["感度R"] =  df["TP"] / df["真に陽性の数 TP+FN"]

    print('Elapsed:', time.perf_counter() - t)

    return df




# timelist_from_likelihoods7.pyより移植・改変
def to_binary(df, target_kind, th_sets, positive=True, view=False):
    """ 尤度を基に特定のクラスに対する識別結果を返す
    df: pandas.DataFrame, ファイル名、時間区間、尤度を格納したpandasのDataFrame
    target_kind: str, 読み込むタグ
    th_sets: tuple<float>, 尤度に対する閾値
    positive: bool, Trueなら、target_kindの結果を読み込む。Falseならtarget_kind以外の結果を読み込む。
    """
    body = df[target_kind].values
    th1, th2 = th_sets
    if positive:
        results = (body >= th1) & (body < th2)
    else:
        results = ~((body >= th1) & (body < th2))
    df2 = df[results]
    df2 = df2.reset_index(drop=True)

    if view:
        print("df2, ", df2.shape)
        print(df2.head())

    ans = {}
    for i in range(len(df2)):
        path = df2.at[i, "fname"]
        start = df2.at[i, "s"]
        width = df2.at[i, "w"]
        if path not in ans:
            ans[path] = []
        ans[path].append((start, width))
    count = len(df2)

    return ans, count




# fusion_timelist.pyより引用・改変
def comp2(set_A_dict, set_B_dict, view=False):
    """ 和集合 A ∪ B, 積集合 A ∩ B, 差集合 B − A, 差集合 A − B を返す。第1引数がAで、第２引数がB。
    set_A_dict: dict<str: list>, 基準となるリスト。ファイル名をkeyとして、valueには区間のリストを格納すること。
    set_B_dict: dict<str: list>, 比較対象のリスト。構造はlist1と同じ。
    """
    union_ab, inter_ab, diff_ba, diff_ab = {}, {}, {}, {}    # intersection（積集合）, difference（差集合）

    # 和集合を求める（バグがあるかも）
    fpath_A = set(set_A_dict.keys())
    fpath_B = set(set_B_dict.keys())
    fpath_AB = fpath_A.union(fpath_B)

    for fpath in fpath_AB:
        if fpath in set_A_dict and fpath in set_B_dict:
            time_list1 = set_A_dict[fpath].copy()
            time_list2 = set_B_dict[fpath].copy()
            time_list = time_list1 + time_list2

            new_list = []
            while len(time_list) > 0:
                s0, w0 = time_list[0]
                e0 = s0 + w0

                remove = []
                for i, val in enumerate(time_list):
                    s, w = val
                    e = s + w    
                    if s0 <= s <= e0 or s0 <= e <= e0 or (s0 <= s and e <= e0):
                        set_ = [s, e, s0, e0]
                        s0 = min(set_)
                        e0 = max(set_)
                        remove.append(i)

                remove.reverse()
                for k in remove:
                    time_list.pop(k)

                new_list.append((s0, e0 - s0))

            union_ab[fpath] = new_list

        elif fpath in set_A_dict:
            union_ab[fpath] = set_A_dict[fpath].copy()
        else:
            union_ab[fpath] = set_B_dict[fpath].copy()

    # 積集合 A ∩ B、差集合 B − Aを求める
    for fpath in set_B_dict:
        time_list2 = set_B_dict[fpath]

        if fpath in set_A_dict and len(set_A_dict[fpath]) != 0:   # そもそも辞書にない、もしくは空のリストだったらループ内に入らない
            time_list1 = set_A_dict[fpath]    # 同じファイルを比較

            for s2, w2 in time_list2:
                e2 = s2 + w2
                flag = False

                for s1, w1 in time_list1:
                    e1 = s1 + w1
                    if (s1 <= s2 <= e1) or (s1 <= e2 <= e1) or ((s2 <= s1) and (e2 >= e1)) :  # かすっていたらOK
                        flag = True
                        break
                    #if (s2 >= (s1 - m) and (s2 + w2) <= (s1 + w1 + m))  or  \
                    #   (s1 >= (s2 - m) and (s1 + w1) <= (s2 + w2 + m)):   # 区間が包含関係にあった場合
                    #    flag = True
                    #    break

                if flag:
                    if fpath not in inter_ab:    # 格納したことがなければ、空のリストを用意
                        inter_ab[fpath] = []
                    inter_ab[fpath] += [(s2, w2)]
                else:
                    if fpath not in diff_ba:
                        diff_ba[fpath] = []
                    diff_ba[fpath] += [(s2, w2)]

        else:
            if fpath not in diff_ba:    # 格納したことがなければ、空のリストを用意
                diff_ba[fpath] = []
            diff_ba[fpath] += time_list2

    if view:
        print("inter_ab", inter_ab)
        print("diff_ba", diff_ba)

    # 差集合 A − B  == （A - (A ∩ B)）を求める
    for fpath in set_A_dict:
        time_list2 = set_A_dict[fpath]    # 同じファイルを比較

        if fpath in inter_ab and len(inter_ab[fpath]) != 0:
            time_list1 = inter_ab[fpath]
            for s2, w2 in time_list2:
                e2 = s2 + w2
                flag = False

                for s1, w1 in time_list1:
                    e1 = s1 + w1
                    if (s1 <= s2 <= e1) or (s1 <= e2 <= e1) or ((s2 <= s1) and (e2 >= e1)) :  # かすっていたらOK
                        flag = True
                        break
                    #if (s2 >= (s1 - m) and (s2 + w2) <= (s1 + w1 + m))  or  (s1 >= (s2 - m) and (s1 + w1) <= (s2 + w2 + m)):
                    #    flag = True
                    #    break

                if flag == False:
                    if fpath not in diff_ab:    # 格納したことがなければ、空のリストを用意
                        diff_ab[fpath] = []
                    diff_ab[fpath] += [(s2, w2)]
        else:
            if fpath not in diff_ab:    # 格納したことがなければ、空のリストを用意
                diff_ab[fpath] = []
            diff_ab[fpath] += time_list2

    if view:
        print("diff_ab", diff_ab)

    return union_ab, inter_ab, diff_ba, diff_ab




# create_human_voice_list4.pyからの移植。
def fusion(time_list, th=1.2):
    """ 時間的に接近している検出区間を統合したものを返す
    time_list:  list<tuple<float, float>>, 開始時刻と終了時刻をペアにしたタプルを多数格納したリスト
    th: float, 繋げる間隔[s]
    """
    # まずは、時系列になる様に並べ替え
    time_list = sorted(time_list, key=lambda x:x[0])

    # 近いものを結合した新しいリストを作成
    ans = []
    #print(time_list)
    s1, e1 = time_list[0]
    for i in range(1, len(time_list)):  # 要素数が2以上の時に実行される
        s2, e2 = time_list[i]
        if s2 - e1 < th:       # 時間的に接近していれば、くっつける
            e1 = e2
        else:
            ans.append((s1, e1))   # 十分に時間的に分離しているとみなして、追加
            s1, e1 = s2, e2

        if i == len(time_list) - 1:   # 最後の要素だったら追加
            ans.append((s1, e1))

    if len(time_list) == 1:   # 要素数が1の時は上のfor文は実行されないので、要素をここで追加
        ans = [(s1, e1)]

    return ans



# timelist_compare.pyより引用
def save_timelist(time_list, file_name):
    """ timelist（ファイルのパスをキーとする辞書）をファイルに保存する
    """
    with open(file_name, "w", encoding="utf-8-sig") as fw:
        for fpath in time_list:
            time_pairs = time_list[fpath]
            time_pairs = ["{:.2f},{:.2f}".format(s, w)  for s, w in time_pairs]
            txt = ",".join(time_pairs)
            fw.write("{},{}\n".format(fpath, txt))



# timelist_compare.pyより引用・改変
def save_diff_timelist(time_list_correct, time_list_test, save_dir="", tag=""):
    """ 区間リストを比較し、その積集合（両方のリストに該当するもの）と差集合（いずれかのリストにしか掲載の無いもの）を保存する
    積集合は正解に相当し、差集合はFalse PositiveとFalse Negativeに相当する。
    ただし、リストの掲載漏れについては関知しないので、注意すること。
    time_list_correct: dict[str]<lsit>, 正解とみなす区間リスト。ファイルパスをキーとし、時間の区間を格納した辞書
    time_list_test: dict[str]<lsit>, 正解と比較したい区間リスト。ファイルパスをキーとし、時間の区間を格納した辞書
    save_dir: str, 保存先のフォルダへのパス（相対パスでも絶対パスでもよい）
    tag: str, 保存するファイルに付ける任意の文字列
    """

    # 比較
    union_ab, inter_ab, diff_ba, diff_ab = comp2(time_list_correct, time_list_test)   # 積集合と差集合２つが得られる

    # 結果の保存
    save_timelist(union_ab, os.path.join(save_dir, f"timelist_union_{tag}.txt"))            # A ⋃ Bを保存
    save_timelist(inter_ab, os.path.join(save_dir, f"timelist_intersection_{tag}.txt"))     # A ∩ Bを保存
    save_timelist(diff_ba,  os.path.join(save_dir, f"timelist_FalsePositive_{tag}.txt"))    # B - Aを保存
    save_timelist(diff_ab,  os.path.join(save_dir, f"timelist_FalseNegative_{tag}.txt"))    # A - Bを保存

    
    # それぞれの件数をカウント
    counts = {}

    count = 0
    for key in inter_ab:
        count += len(inter_ab[key])
    counts["intersection"] = count

    count = 0
    for key in diff_ba:
        count += len(diff_ba[key])
    counts["FalsePositive"] = count

    count = 0
    for key in diff_ab:
        count += len(diff_ab[key])
    counts["FalseNegative"] = count

    counts["detected"] = counts["intersection"] + counts["FalsePositive"]   # 検出された全数


    return counts





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
    parser.add_argument("-lf", "--likelihood_files")
    parser.add_argument("-cf", "--correct_fpath")
    parser.add_argument("-t", "--tag")
    parser.add_argument("-ct", "--class_terget")
    parser.add_argument("-lpu", "--last_predict_use")
    parser.add_argument("-bnu", "--basename_use")
    parser.add_argument("-pr", "--path_replace_for_likelifood_files")
    parser.add_argument("-cd", "--compare_depth")
    parser.add_argument("-prrl", "--pseudo_recall_r_limit")
    parser.add_argument("-prtl", "--pseudo_recall_th_limit")

    # 引数を解析
    args = parser.parse_args()
    
    # 設定に反映
    if args.likelihood_files:    # 処理対象ファイルの指定への対応。スクリプトの引数内では、\は\\としてエスケープすること。
        value = args.likelihood_files
        print("***", value, "***")
        # globで指定された場合
        if "glob.glob" == value[:9]:
            index = value.find(")")
            if index > 0:
                order = value[:index + 1]
                v = eval(order)
                params["likelihood_files"] = sorted(v)

        # listで指定された場合
        elif value[0] == "[" and value[-1] == "]":
            index = value.find("]")
            if index > 0:
                order = value[:index + 1]
                v = ast.literal_eval(order)      # -f "[r'./a/fuga.mp3', r'E:\fuga.wav', 'c']"
                params["likelihood_files"] = sorted(v)
        
        # ただ１つだけパスが書かれていた場合
        elif os.path.exists(value):
            params["likelihood_files"] = [value]

    if args.correct_fpath: params["correct_fpath"] = str(args.correct_fpath)
    if args.tag: params["tag"] = str(args.tag)
    if args.class_terget: params["class_terget"] = str(args.class_terget)
    if args.last_predict_use: 
        params["last_predict_use"] = strtobool(args.last_predict_use)

        # 最後の予測フォルダの結果の利用を指示されていた場合、最後のフォルダ内のファイルを探す。
        if params["last_predict_use"]:
            last_dir = last_dirpath("predict")
            params["likelihood_files"] = glob.glob(last_dir + "/prediction_likelihoods*.csv") + \
                                         glob.glob(last_dir + "/**/prediction_likelihoods*.csv")
    if args.basename_use: params["basename_use"] = strtobool(args.basename_use)

    if args.path_replace_for_likelifood_files:    # パスの変換ルール
        value = args.path_replace_for_likelifood_files
        if value[0] == "[" and value[-1] == "]":
            index = value.find("]")
            if index > 0:
                order = value[:index + 1]
                v = ast.literal_eval(order)      # -f "['hoge', 'fuga']"
                params["path_replace_for_likelifood_files"] = v

    if args.compare_depth: params["compare_depth"] = int(args.compare_depth)

    # パラメータ間の調整
    if params["basename_use"]:
        params["compare_depth"] = 0
    if params["compare_depth"] < -1:
        params["compare_depth"] = -1

    if args.pseudo_recall_r_limit: params["pseudo_recall_r_limit"] = float(args.pseudo_recall_r_limit)
    if args.pseudo_recall_th_limit: params["pseudo_recall_th_limit"] = float(args.pseudo_recall_th_limit)


    return params






def set_default_setting():
    """ デフォルトの設定をセットして返す
    """
    params = {}
    params["tag"] = ""               # フォルダに付ける名前
    params["class_terget"] = ""      # 評価対象のクラス名
    params["likelihood_files"] = []           # 尤度で示された識別結果のファイルのリスト
    params["correct_fpath"] = "dummy_list.txt"    # 正解の区間リスト
    params["F_th"] = 0.25                     # F値を計算する際に使う閾値
    params["y_score_test"] = False   # 動作確認用に、ダミーデータと比較する場合はTrueとする
    params["graph_show"] = False     # グラフを描画するかどうか
    params["basename_use"] = False   # ファイル名だけで、正解と予測結果を突き合わせるならTrue。フルパス推奨なのでFalseがデフォルト。
    params["last_predict_use"] = False  # 最後の予測フォルダ内の結果を使うならTrue
    params["path_replace_for_likelifood_files"] = None    # 例: ["2.29", "3.46"]
    params["compare_depth"] = -1     # -1だとファイルに記載されているパスをそのまま使う

    if params["basename_use"]:    # コード的な働きはないが、意味的にはこうなので書いておく
        params["compare_depth"] = 0

    params["FP_FN_list"] = {                  # 見逃しなどの区間リストを作成パラメータ
                            "fusion":0,       # 検出区間の結合距離[s]
                            "th":[0.6, 0.7, 0.8]     # リストを作成する際の尤度に対する閾値
                            }
    params["pseudo_recall_r_limit"] = 0.95    # 鳴き声ごとの感度を計算する際に、計算時間が非常にかかる場合があるので感度がこの値を超えたらそれ以上の計算を省略する
    params["pseudo_recall_th_limit"] = 0.20    # 鳴き声ごとの感度を計算する際に、計算時間が非常にかかる場合があるので閾値がこの値を下回ったらそれ以上の計算を省略する
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
    ## 最後の予測フォルダの結果の利用を指示されていた場合、最後のフォルダ内のファイルを再帰的に探す。
    if param["last_predict_use"]:
        last_dir = last_dirpath("predict")
        param["likelihood_files"] = glob.glob(last_dir + "/prediction_likelihoods*.csv") + \
                                    glob.glob(last_dir + "/**/prediction_likelihoods*.csv")

    return param





def main():
    # 設定を読み込み
    setting = read_setting("evaluate_setting.yaml")

    # 引数のチェック
    setting = arg_parse(setting)
    print2("\n\n< setting >", setting)

    # 保存先の親フォルダを作成
    os.makedirs("runs", exist_ok=True)  # 保存先のフォルダを作成

    # 保存先のフォルダを作る
    save_root_dir = next_dirpath("evaluate")
    if setting["tag"] != "":
        save_root_dir += f"_{setting['tag']}"
    os.makedirs(save_root_dir, exist_ok=True)

    # 設定の保存（後でパラメータを追えるように）
    now_ = dt.now().strftime('%Y%m%d%H%M')
    fname = os.path.join(save_root_dir, "evaluate_setting_{}.yaml".format(now_))
    with open(fname, 'w', encoding="utf-8-sig") as fw:
        yaml.dump(setting, fw, encoding='utf8', allow_unicode=True)


    for fpath in setting["likelihood_files"]:
        print("\n\n--proccess start--: ", fpath)
        basename = os.path.basename(fpath)
        name, ext = os.path.splitext(basename)
        save_dir = os.path.join(save_root_dir, name)    # 処理結果を保存するフォルダのパスを作る

        # 処理結果を保存するフォルダを作成
        os.makedirs(save_dir, exist_ok=True)

        # 処理対象ファイルをコピーしておく（後で何を処理したのかわからなくなるので）
        shutil.copy2(fpath, os.path.join(save_dir, name + ".bak"))

        # 予測結果の尤度データを読み込み
        df = read_likelihoods(fpath, compare_depth=setting["compare_depth"])   # 予測結果を読み込み
        
        # 必要ならパス名を置換する
        if setting["path_replace_for_likelifood_files"] is not None:
            a, b = setting["path_replace_for_likelifood_files"]
            path_change2 = lambda x: x.replace(a, b)
            df["fname"] = df["fname"].map(path_change2)  # パスの中の文字列を置換

        # ファイルのパスの表記を統一
        path_change1 = lambda x: x.replace("\\", "/")
        df["fname"] = df["fname"].map(path_change1)  

        print("**************  fname heads in likelifood_files after path replacing.  ************")
        print(df["fname"].head())

        # 評価対象の列データを取り出す
        y_score = df[setting["class_terget"]].values       # 予想された尤度。tagの列をndarray型で取り出す
        
        # 正解の区間リスト読み込み
        time_dict = read_list(setting["correct_fpath"], compare_depth=setting["compare_depth"])
        file_paths = list(time_dict.keys())          # ファイルのパスの表記を統一するために、キー（ファイルのパス）を取り出す
        for path in file_paths:
            time_dict[path_change1(path)] = time_dict.pop(path)
            #print("path:         ", path)
            #print("changed path: ", path_change1(path))
        print("++++++++++++  time_dict  ++++++++++++")
        print2(time_dict)

        # 予測結果に合わせて、正解を作成
        checker = DetectChecker(time_dict)
        fnames = df.iloc[:, 0].values
        s = df.iloc[:, 1].values
        w = df.iloc[:, 2].values
        y_true = []
        for path_, s_, w_ in zip(fnames, s, w):
            #print(path_, s_, w_)
            y_true.append(checker.check_time_pair(path_, (s_, w_)))
        y_true = np.array(y_true)

        # 処理のサンプルが欲しい場合はこちらを実行（動作テスト用）
        if setting["y_score_test"]:
            y_bin = y_score > setting["F_th"]
            rand = np.random.rand(len(y_bin)) > 0.9995
            y_true = y_bin | rand
            print(sum(y_bin))

        # 正解を保存
        df_true = pd.DataFrame()
        df_true["path"] = fnames
        df_true["s"] = s
        df_true["w"] = w
        df_true["y_true"] = y_true
        df_true.to_excel(os.path.join(save_dir, "y_true.xlsx"), index=None)   # 保存
        print(sum(y_true))

        
        # ROC-AUCの計算
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)   # ftr: false_positive,  tpr: true_positive
        roc_auc = auc(fpr, tpr)
        print("roc_auc", roc_auc)

        # ROC-AUCの計算の検算
        df_checked = check_amount0(y_true, y_score, thresholds)
        print("df_checked is generated.")

  
        # 鳴き声区間ごとの検出率を算出
        print("**********  Detection Rate of Each Call/Song **********")
        fuga = []
        df_ = df[["fname", "s", "w", setting["class_terget"]]]
        for x in thresholds:
            print(f"*** now threshold is {x}.")
            time_dict_, size = to_binary(df_, setting["class_terget"], (x, 1.01))
            checker_ = DetectChecker(time_dict_)
            true_num, count = 0, 0

            for fpath2 in time_dict:
                for time_pair in time_dict[fpath2]:
                    s_, w_ = time_pair
                    true_num += checker_.check_time_pair(fpath2, (s_, w_))
                    count += 1
            if count == 0:
                pseudo_recall = float("nan")
            else:
                pseudo_recall = true_num / count
            fuga.append(pseudo_recall)

            if pseudo_recall >= setting["pseudo_recall_r_limit"] or x < setting["pseudo_recall_th_limit"]:    # これ以上計算しても無駄だし、時間がかかりすぎるので脱出
                break
        fuga += [pseudo_recall] * (len(thresholds) - len(fuga))
        df_checked["鳴き声区間ごとの検出率"] = fuga

        
        # ROC-AUCの保存 
        with open(os.path.join(save_dir, "ROC-AUC_result.txt"), "w", encoding="utf-8-sig") as fw:
            fw.write("auc,{}".format(roc_auc))
        
        # fpr, tprをpandasで保存
        print("~~~~~~  Save FPR/TPR to DataFrame ~~~~~~")
        df_roc = pd.DataFrame({'th': thresholds, 'fpr': fpr, 'tpr': tpr})  # 格納順はExcelでのグラフを作りやすいように決めた。
        df_roc = pd.concat([df_roc, df_checked], axis=1)    # DataFrameの結合
        df_roc.to_excel(os.path.join(save_dir, "data.xlsx"), index=False)
        print(df_roc)

        # fpr, tprの散布図（ROC曲線）を保存
        plt.scatter(fpr, tpr, marker='o', label='ROC curve (area = {0:.2f})'.format(roc_auc))
        plt.xlabel('FPR: False positive rate')
        plt.ylabel('TPR: True positive rate')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, 'ROC_curve.png'))
        if setting["graph_show"]:
            plt.show()
        plt.clf()
        
        # fpr, tprのグラフを保存
        plt.scatter(thresholds, tpr, marker='o', label='TPR')
        plt.scatter(thresholds, fpr, marker='o', label='FPR')
        plt.xlabel('thresholds')
        plt.ylabel('rate')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, 'FPR_TPR_thresholds.png'))
        if setting["graph_show"]:
            plt.show()
        plt.clf()

        # 適合率P・感度（再現率）Recallの散布図（PR曲線）を保存
        P = df_roc["適合率P"].values
        R = df_roc["感度R"].values      # 感度は再現率ともいう
        pr_auc = auc(R, P)
        plt.scatter(R, P, marker='o', label='PR curve (area = {0:.2f})'.format(pr_auc))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, 'PR_curve.png'))
        if setting["graph_show"]:
            plt.show()
        plt.clf()

        # 適合率P・鳴き声区間ごとの検出率の散布図を保存
        P = df_roc["適合率P"].values
        R = df_roc["鳴き声区間ごとの検出率"].values
        pr_auc = auc(R, P)
        plt.scatter(R, P, marker='o', label='PR curve (area = {0:.2f})'.format(pr_auc))
        plt.xlabel('Detection Rate of Each Call (nearly Recall)')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, 'P-DRE_curve.png'))
        if setting["graph_show"]:
            plt.show()
        plt.clf()

        # 横軸を閾値、縦軸を適合率Pと感度Recallとするグラフを保存
        plt.scatter(thresholds, P, marker='o', label='Precision')
        plt.scatter(thresholds, R, marker='o', label='Recall')
        plt.xlabel('thresholds')
        plt.ylabel('rate')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, 'Precision_Recall_thresholds.png'))
        if setting["graph_show"]:
            plt.show()
        plt.clf()

        # PR-AUCの保存 
        with open(os.path.join(save_dir, "PR-AUC_result.txt"), "w", encoding="utf-8-sig") as fw:
            fw.write("auc,{}".format(pr_auc))

        
        # F値を求める（ただし、閾値がテキトーなときは参考に留めること）
        print("=====  Calculation of F1 Value =====")
        R = recall_score(y_true, y_score >= setting["F_th"])
        P = precision_score(y_true, y_score >= setting["F_th"])
        if P + R == 0:
            F = float("nan")
        else:
            F = 2 * (P*R) / (P + R)
        print("R:{:.2f}, P:{:.2f}, F:{:.2f}".format(R, P, F))


        # 見逃しなどの区間リストを作成
        ## そのままだと相対パスだったりするので、正解の区間リストを識別結果に入っているパスに置換する
        time_dict_c = time_dict.copy() 
        keys = time_dict_c.keys()
        for key in list(keys):
            full_path = df.loc[df["fname"] == key, "fname_back"].values[0]
            time_dict_c[full_path] = time_dict_c.pop(key)
        
        ## 閾値毎に作成
        print("######  Save FP/FN time-list ######")
        df_ = df[["fname", "s", "w", setting["class_terget"]]]
        with open(os.path.join(save_dir, f"each_amount.txt"), "w", encoding="utf-8-sig") as fw:
            fw.write("th,intersection,FalsePositive,FalseNegative,detected,Precision,Recall,F1\n")

            for x in setting["FP_FN_list"]["th"]:
                print(f"+++ now threshold is {x}.")
                time_dict_, size = to_binary(df_, setting["class_terget"], (x, 1.01))

                # 区間の結合処理
                if setting["FP_FN_list"]["fusion"] > 0:
                    keys = time_dict_.keys()
                    for key in list(keys):
                        time_dict_[key] = fusion(time_dict_[key], th=setting["FP_FN_list"]["fusion"])

                # そのままだと相対パスだったりするので、識別結果に入っているパスに置換する
                keys = time_dict_.keys()
                for key in list(keys):
                    full_path = df.loc[df["fname"] == key, "fname_back"].values[0]
                    time_dict_[full_path] = time_dict_.pop(key)

                # 全予測結果の保存
                save_timelist(time_dict_, os.path.join(save_dir, f"timelist_predicted_th{x}.txt"))  # A ∩ Bを保存
                
                # 正解リストとの比較と保存
                counts = save_diff_timelist(time_dict_c, time_dict_, save_dir=save_dir, tag=f'{setting["class_terget"]}_th{x}')

                if counts["detected"] == 0:
                    p_ = float("nan")
                else:
                    p_ = counts["intersection"] / counts["detected"]

                if counts["intersection"] + counts["FalseNegative"] == 0:
                    r_ = float("nan")
                else:
                    r_ = counts["intersection"] / (counts["intersection"] + counts["FalseNegative"])
                if p_ + r_ == 0:
                    f1_ = float("nan")
                else:
                    f1_ = 2 * (p_* r_) / (p_ + r_)
                fw.write(f'{x},{counts["intersection"]},{counts["FalsePositive"]},{counts["FalseNegative"]},{counts["detected"]},{p_},{r_},{f1_}\n')
    
    
    # 修了処理
    print("proccess is finished.")


if __name__ == "__main__":
    main()

