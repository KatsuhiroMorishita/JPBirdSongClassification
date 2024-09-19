#!/usr/bin/python
# -*- coding:utf-8 -*-
# writen for python 3
# purpose: 尤度のリストから区間リストを作成する
# history: 
#   2020-04-27 ver.1
#   2020-12-20 ver.2  sound_file_prediction2.pyに対応
#   2023-03-01 ver.4  1件以上の物を出力するように変更
#                     抽出数が多すぎる場合にも対応
#   2023-08-28 ver.5  冒頭部分を保存する設定を追加して、さらに複数の閾値での処理を実行できるように変更
#   2024-02-12 ver.6  閾値の設定を範囲指定に変更した。これで低尤度での発火状況を確認しやすくなる。
#   2024-02-13 ver.7  複数の設定ファイルに対応
#   2024-04-18 ver.8  尤度のファイル指定にglobを使ったパターンを指定できるように変更
#   2024-07-10        関数の位置を変更
#   2024-07-14        予測結果に含まれないクラスを設定ファイルにて指定していた場合でもエラーでとまらない様にした。
#   2024-09-06        指定された尤度の範囲の件数や、実際の保存した件数もログとして残すようにした。また、抽出部分のif文の評価式を修正した。
# author: Katsuhiro Morishita
# created: 2020-04-27
# license: MIT. If you use this program for your study, you should write Acknowledgement in your paper.
import os, glob
import pandas as pd
import numpy as np




def read_likelihoods(likelihood_file):
    """ 予測で出力された尤度のリストを読み込む
    likelihood_file: str, 尤度が格納されたテキストファイルのパス
    """
    df = pd.read_csv(likelihood_file, encoding="utf-8-sig")
    print(df.head())

    return df




def to_binary(df, target_kind, th_sets, positive=True):
    """ 尤度を基に特定のクラスに対する識別結果を返す
    df: pandas.DataFrame, 
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
    print("df2, ", df2.shape)
    print(df2.head())

    ans = {}
    for i in range(len(df2)):
        path = df2.loc[i, "fname"]
        start = df2.loc[i, "s"]
        width = df2.loc[i, "w"]
        if path not in ans:
            ans[path] = []
        ans[path].append((start, width))
    count = len(df2)

    return ans, count




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










def create_timelist(setting_file):
    # 設定を読み込み
    setting = set_default_setting()
    setting = read_setting(setting_file, setting)

    # データを読み込み
    likelifood_path = sorted(glob.glob(setting["likelihood_file"]))[0]
    df = read_likelihoods(likelifood_path)   # 予測結果を読み込み
    
    if "all" in setting["tag"]:
        setting["tag"] = list(df.columns)[3:]

    setting_name, setting_ext = os.path.splitext(os.path.basename(setting_file))
    with open(f"ratio_{setting_name}.csv", "w", encoding="utf-8-sig") as fw_ratio:
        fw_ratio.write(f'tag,th1,th2,count,setting["max"],r,under,real_saved_amount\n')
    
        # 検出結果を取得する
        for tag in setting["tag"]:

            if tag not in df.columns:
                print("#" * 80)
                print("#" * 80)
                print(f"Tag: {tag} is not exists in DataFrame. So, check your setting file.")
                continue

            for th_sets in setting["th"]:
                th1, th2 = th_sets
                if not (th1 >= 0 and th2 >= 0 and th1 < th2):
                    print(f"th_sets is wrong. {th_sets}")
                    continue
                pre_result, count = to_binary(df, tag, th_sets, setting["positive"])
                
                # 結果の保存
                r = float("nan")
                under = ""
                count_ = 0

                if count > 0:
                    # 保存割合の計算
                    r = setting["max"] / count
                    
                    if r < 1:
                        under = "_limited"
                    print(f"r={r}, count={count}.")
 
                    # 冒頭のみ保存する枚数
                    h = setting["head_save_file"]


                    with open(f"timelist_likelihoods_result_{tag}_th{th1}to{th2}{under}.txt", "w", encoding="utf-8-sig") as fw:
                        files = sorted(pre_result.keys())
                        
                        for fpath in files:
                            time_list = pre_result[fpath]

                            
                            m = setting["min_per_file"]
                            
                            time_list_ = []

                            # 冒頭で必要な分を取得
                            if h > 0:
                                time_list_ = time_list[:h]  # 実際にはh個も取れないかも
                                time_list = time_list[h:]
                                h -= len(time_list_)

                            # 多すぎる場合は減らす
                            if m >= 0 and len(time_list) > m and r < 1:
                                r_ = ((len(time_list) - m) * r + m) / len(time_list)   # 実質的に残す割合
                                x = np.random.rand(len(time_list))      # 乱数を作って
                                time_list = np.array(time_list)[x < r_]  # 必要数を選択

                            time_list_ += list(time_list)
                            print(f"r2={len(time_list_) / count}.")
                            
                            count_ += len(time_list_)
                            if len(time_list_) == 0:  # 中身が0個なら次のループ
                                continue

                            th = setting["fusion_th"]
                            if th is not None:      # 結合
                                time_list_se = [(s, s + w) for s, w in time_list_]   # 区間の開始時刻と幅から開始時刻と終了時刻のペアを作る
                                time_list_se = fusion(time_list_se, th)
                                time_list_ = [(s, e-s) for s, e in time_list_se]
                            time_list = ["{},{}".format(s, w) for s, w in time_list_]
                            time_list = ",".join(time_list)


                            fw.write("{},{}\n".format(fpath, time_list))


                # 状況を保存
                fw_ratio.write(f'{tag},{th1},{th2},{count},{setting["max"]},{r},{under},{count_}\n')






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
    params["tag"] = [""]         # 評価対象のタグ
    params["positive"] = True  # 評価対象のタグか、それ以外か。Falseだとtag以外をまとめて読む。
    params["fusion_th"] = None  # 結合する時間間隔
    params["likelihood_file"] = "prediction_likelihoods.csv"
    params["th"] = [(0.15, 1.01)]
    params["max"] = 2000            # 生成画像数の制限枚数（必ずしもピッタリにはならない）
    params["min_per_file"] = 10     # 1音源あたりの画像数の最小値（生成画像数が大量な場合に利用される）
    params["head_save_file"] = 10   # 冒頭で保存する枚数
    return params






def main():
    # 設定読み込み
    setting_files = glob.glob("timelist_likelihoods_setting*.txt")

    # 処理の実行
    for file in setting_files:
        print("--" * 20)
        print(f"process start. {file}")
        create_timelist(file)

    # 修了処理
    print("proccess is finished.")


if __name__ == "__main__":
    main()

