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
# author: Katsuhiro Morishita
# created: 2020-04-27
# license: MIT. If you use this program for your study, you should write Acknowledgement in your paper.
import os, re, glob, copy, time, pprint, shutil
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
            

def last_dirnumber(pattern="train"):
    """ 既に保存されている学習・予測フォルダ等の最大の番号を取得
    """
    path = r'.\runs'
    dirs = os.listdir(path=path)
    numbers = [0]
    for d in dirs:
        if os.path.isdir(os.path.join(path, d)):
            m = re.search(r"{}(?P<num>\d+)".format(pattern), d)
            if m:
                numbers.append(int(m.group("num")))
    max_ = max(numbers)

    return max_, path


def last_dirpath(pattern="train"):
    """ 最大の番号を持つ既に保存されている学習・予測フォルダ等のパスを返す
    """
    max_, path = last_dirnumber(pattern)
    return os.path.join(path, pattern + str(max_)) 


def next_dirpath(pattern="train"):
    """ 学習・予測フォルダ等を新規で作成する場合のパスを返す
    """
    max_, path = last_dirnumber(pattern)
    return os.path.join(path, pattern + str(max_ + 1)) 




def read_likelihoods(likelihood_file, basename=False):
    """ 予測で出力された尤度のリストを読み込む
    likelihood_file: str, 尤度が格納されたテキストファイルのパス
    basename: str, ファイル名のパスがフルパスでは都合が悪い場合（ファイル名だけで比較したい場合）はTrueを指定してください。
    """
    df = pd.read_csv(likelihood_file, encoding="utf-8-sig")

    if basename:   # ファイルのパスからフォルダを排除して、ファイル名のみとする
        path_change = lambda x: os.path.basename(x)
        df.iloc[:, 0] = df.iloc[:, 0].map(path_change)  # ファイルのパスをファイル名だけに変更
    print(df.head())

    return df



# save_spectimage_with_list2.pyより移植、改編
def read_list(fname, basename=False):
    """ 音源と時間区間のリストを読み込む
    fname: str, 読み込むファイル名
    basename: str, ファイル名のパスがフルパスでは都合が悪い場合（ファイル名だけで比較したい場合）はTrueを指定してください。
    """
    ans = {}
    with open(fname, "r", encoding="utf-8-sig") as fr:
        lines = fr.readlines()  # 全行読み込み
        for line in lines:
            line = line.rstrip()  # 改行コード削除
            if len(line) == 0:    # 空行を避ける
                continue
            field = line.split(",")
            path = field[0]       # ファイルのパスを取得
            if basename:                       # add.
                path = os.path.basename(path)  # add. パスからファイル名を取り出す。
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

            print(path)
            print(time_list)

            ans[path] = time_list  # 辞書として保存
    return ans



class DetectChecker:
    """ 正解の区間リストと比較して、指定された時刻[秒]が区間内であるかどうかを返すクラス
    なぜクラスにしたのか、忘れた・・・。
    """
    def __init__(self, time_list_dict):
        self._time_list_dict = time_list_dict

    def check_time_pair(self, fpath, time_pair, margin=0):
        """ time_pairで指定した範囲がコンストラクタでセットされた辞書内のリストと合致するか確認し、合致するとTrueを返す
        fpath: str, 検査したいファイルのフルパス
        time_pair: list or tuple <float, float>, 開始時間[s]と時間幅[s]のペア
        margin: float or int, 許容誤差[s]
        """
        m = margin

        if fpath in self._time_list_dict:
            time_lists = self._time_list_dict[fpath]

            s2, w2 = time_pair
            for s1, w1 in time_lists:
                if (s2 >= (s1 - m) and (s2 + w2) <= (s1 + w1 + m))  or  (s1 >= (s2 - m) and (s1 + w1) <= (s2 + w2 + m)):
                    return True
        else:
            pass
            #print("{} is not inclued in timelist.".format(fpath))
            #exit()

        return False








# if文とPythonのsum()を排除して1000倍速、さらにnumbaの力で20倍で、トータル1万倍速くなった。
@njit("f8[:,:](bool_[:], f8[:], f8[:])", parallel=True)
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
    a = check_amount(true.astype(np.bool), score, thresholds)

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

    print('Elapsed:', time.perf_counter() - t)

    return df





def set_default_setting():
    """ デフォルトの設定をセットして返す
    """
    params = {}
    params["tag"] = ""         # 評価対象のタグ
    params["likelihood_files"] = []           # 尤度で示された識別結果のファイルのリスト
    params["list_name"] = "dummy_list.txt"    # 正解の区間リスト
    params["F_th"] = 0.25                     # F値を計算する際に使う閾値
    params["y_score_test"] = False   # 動作確認用に、ダミーデータと比較する場合はTrueとする
    params["graph_show"] = False     # グラフを描画するかどうか
    params["margin"] = 10            # 正解扱いする範囲を広げるマージン
    params["basename_use"] = False   # ファイル名だけで、正解と予測結果を突き合わせるならTrue。フルパス推奨なのでFalseがデフォルト。
    params["last_predict_use"] = False  # 最後の予測フォルダ内の結果を使うならTrue
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
        param["likelihood_files"] += glob.glob(last_dir + "/prediction_likelihoods*.csv") + \
                                     glob.glob(last_dir + "/**/prediction_likelihoods*.csv")

    return param


def print_setting(params):
    """いい感じに設定内容の辞書を表示する
    """
    params_ = copy.deepcopy(params)
    pprint.pprint(params_)




def main():
    # 設定を読み込み
    setting = read_setting("evaluate_setting.yaml")
    print2("\n\n< setting >", setting)

    # 保存先のフォルダを作る
    save_root_dir = next_dirpath("evaluate")
    os.makedirs(save_root_dir, exist_ok=True)

    # 設定の保存（後でパラメータを追えるように）
    now_ = dt.now().strftime('%Y%m%d%H%M')
    fname = os.path.join(save_root_dir, "evaluate_setting_{}.yaml".format(now_))
    with open(fname, 'w', encoding="utf-8-sig") as fw:
        yaml.dump(setting, fw, encoding='utf8', allow_unicode=True)


    for fpath in setting["likelihood_files"]:
        print("--proccess start--: ", fpath)
        basename = os.path.basename(fpath)
        name, ext = os.path.splitext(basename)
        save_dir = os.path.join(save_root_dir, name)    # 処理結果を保存するフォルダのパスを作る

        # 処理結果を保存するフォルダを作成
        os.makedirs(save_dir, exist_ok=True)

        # 処理対象ファイルをコピーしておく（後で何を処理したのかわからなくなるので）
        shutil.copy2(fpath, os.path.join(save_dir, name + ".bak"))

        # 予測結果の尤度データを読み込み
        df = read_likelihoods(fpath, basename=setting["basename_use"])   # 予測結果を読み込み
        path_change = lambda x: x.replace("\\", "/")
        df["fname"] = df["fname"].map(path_change)  # ファイルのパスの表記を統一
        y_score = df[setting["tag"]].values       # 予想された尤度。tagの列をndarray型で取り出す
        
        # 正解の区間リスト読み込み
        time_dict = read_list(setting["list_name"], basename=setting["basename_use"])
        file_paths = list(time_dict.keys())          # ファイルのパスの表記を統一するために、キー（ファイルのパス）を取り出す
        for path in file_paths:
            time_dict[path_change(path)] = time_dict.pop(path)
            #print("path:         ", path)
            #print("changed path: ", path_change(path))

        # 予測結果に合わせて、正解を作成
        checker = DetectChecker(time_dict)
        fnames = df.iloc[:, 0].values
        s = df.iloc[:, 1].values
        w = df.iloc[:, 2].values
        y_true = []
        for path_, s_, w_ in zip(fnames, s, w):
            #print(path_, s_, w_)
            y_true.append(checker.check_time_pair(path_, (s_, w_), setting["margin"]))
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

        
        # AUCの計算
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)   # ftr: false_positive,  tpr: true_positive
        auc_value = auc(fpr, tpr)

        # AUCの計算の検算
        df_checked = check_amount0(y_true, y_score, thresholds)
        
        # AUCの保存 
        with open(os.path.join(save_dir, "auc_result.txt"), "w", encoding="utf-8-sig") as fw:
            fw.write("auc,{}".format(auc_value))
        
        # fpr, tprをpandasで保存
        df_roc = pd.DataFrame({'th': thresholds, 'fpr': fpr, 'tpr': tpr})  # 格納順はExcelでのグラフを作りやすいように決めた。
        df_roc = pd.concat([df_roc, df_checked], axis=1)    # DataFrameの結合
        df_roc.to_excel(os.path.join(save_dir, "ROC_curve.xlsx"), index=False)
        print(df_roc)

        # fpr, tprの散布図（ROC曲線）を保存
        plt.scatter(fpr, tpr, marker='o', label='ROC curve (area = {0:.2f})'.format(auc_value))
        plt.xlabel('FPR: False positive rate')
        plt.ylabel('TPR: True positive rate')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, 'sklearn_roc_curve.png'))
        if setting["graph_show"]:
            plt.show()
        plt.clf()
        
        # fpr, tprのグラフを保存
        plt.scatter(thresholds, tpr, marker='o', label='tpr')
        plt.scatter(thresholds, fpr, marker='o', label='fpr')
        plt.xlabel('thresholds')
        plt.ylabel('rate')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_dir, 'sklearn_tpr_fpr_curve.png'))
        if setting["graph_show"]:
            plt.show()
        plt.clf()

        
        # F値を求める（ただし、閾値がテキトーなときは参考に留めること）
        R = recall_score(y_true, y_score >= setting["F_th"])
        P = precision_score(y_true, y_score >= setting["F_th"])
        F = 2 * (P*R) / (P + R)
        print("R:{:.2f}, P:{:.2f}, F:{:.2f}".format(R, P, F))
    
    
    # 修了処理
    print("proccess is finished.")


if __name__ == "__main__":
    main()

