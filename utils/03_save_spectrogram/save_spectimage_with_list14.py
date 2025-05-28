# purpose: ファイルで指定されたファイルを指定された区間でスペクトログラムを作成する。
# author: Katsuhiro Morishita (morimori.ynct@gmail.com)
# history: 
#  2020-10-07 ver.3  fusion()を追加して、3にバージョンを上げた。
#  2020-10-08 ver.4  スペクトログラムに加工しない区間を読み込めるようにした。
#                    mask_fnameというキーワードでCDの人の声区間を格納したファイルを指定すれば良い。
#  2020-11-20        sound_image4から5に変更した。
#  2020-12-11        fusionの設定値が0でも結合処理が動いていたので、0では動作しないようにした。
#  2021-03-02 ver.5  sound_image6を使うように変更したのでバージョンアップ
#  2021-07-29 ver.6  sound_image7を使うように変更したのでバージョンアップ
#  2021-10-31 ver.7  区間リストを複数指定できるように変更した。保存先のフォルダは区間リストのファイル名から自動生成する。
#  2021-12-07 ver.8  sound_image8を使うように変更したのでバージョンアップ
#  2022-09-26        音源ファイルの読み込み失敗した場合への対応処理を追加
#  2023-06-27 ver.9  ファイルのパスが""で囲われている場合に対応した。バージョンは以前から9だったが、いつ9にしたっけ？ 
#  2023-08-30        ファイルのサイズが0だった場合に対応（ほぼないと思うが）し、ログファイルの文字コードを変更した。
#                    ファイルの読み込みに失敗した場合に対応
#                    ログファイルに時刻を残すように変更
#  2023-11-22 ver.10 一定数に満たないファイルを無視する設定を追加 
#                    音源ファイルの親フォルダ毎にフォルダを分ける機能を追加（音源名が同じでフォルダで区別されている場合にわかりやすい）
#                    ファイルのID一覧も保存するようにした。
#  2024-03-05 ver.11 sound_imageの依存バージョンを9から10に変更した。
#  2024-07-29        コメントに対応したread_listと、最近の仕様に合致したfusionをfusion_timelists.txtより移植した。普段の挙動は変わらないと思う。
#  2025-02-03 ver.12 作成した画像の保存先のルートフォルダを指定できるように変更した。
#  2025-03-21 ver.13 sound_imageの依存バージョンを10から11に変更した。
#  2025-04-08 ver.14 sound_imageの依存バージョンを11から12に変更した。
#  2025-04-11        sound_imageで対応する保存形式を増やしたので、そのあおりで設定ファイルの形式がyamlになった。
# created: 2019-08-30
# license: 
#  Please contact us for commercial use.
#  It is free to use for non-commercial purposes, hobbies, and study.
#  If you use this program for your study, you should write reference in your paper.

import re, os
from datetime import datetime as dt

import sound_image12 as si




# fusion_timelists.txtより移植
def read_list(fname, compare_depth=-1, ignore_size=None):
    """ 音源と時間区間のリストを読み込む
    fname: str, 読み込むファイル名
    compare_depth: int, パスを比較する深度。パスを比較する際に使用する。0だとファイル名のみで比較し、-1だとパス全体で比較する。
                        音源を読み込んで加工するなど、比較が不要の場合は無視してよい。
    ignore_size: int, 読み込まれた区間数がこの数より小さい場合、空の辞書を返す。
    """
    ans = {}
    count = 0
    path_change = lambda x: x.replace("\\", "/")   # 円マーク（windowsでは円マーク）を/に書き換えるラムダ関数

    with open(fname, "r", encoding="utf-8-sig") as fr:
        lines = fr.readlines()  # 全行読み込み
        #print(lines)   # debug

        for line in lines:
            line = line.rstrip()  # 改行コード削除
            if "#" in line:       # コメント文への対応
                line = line[:line.find("#")]
                line = line.rstrip()  # 空白削除
            if len(line) == 0:    # 空行を避ける
                #print("no text")   # debug
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
        print(f"The size of time field is less than ignroe_size. So skip this time list file. : {fname}")
        return {}
    else:
        return ans


            

# fusion_timelists.pyより移植
def fusion(time_list, th=1.2):
    """ 時間的に接近している検出区間を統合したものを返す
    time_list:  list<tuple<float, float>>, 開始時刻と終了時刻をペアにしたタプルを多数格納したリスト
    th: float, 繋げる間隔[s]
    """
    # まずは、時系列になる様に並べ替え
    time_list = sorted(time_list, key=lambda x:x[0])

    # 近いものを結合した新しいリストを作成
    ans = []
    s1, w1 = time_list[0]
    e1 = s1 + w1
    for i in range(1, len(time_list)):  # 要素数が2以上の時に実行される
        s2, w2 = time_list[i]
        e2 = s2 + w2
        if s2 - e1 < th:       # 時間的に接近していれば、くっつける
            e1 = e2
        else:
            ans.append((s1, e1 - s1))   # 十分に時間的に分離しているとみなして、追加
            s1, e1 = s2, e2

        if i == len(time_list) - 1:   # 最後の要素だったら追加
            ans.append((s1, e1 - s1))

    if len(time_list) == 1:   # 要素数が1の時は上のfor文は実行されないので、要素をここで追加
        ans = [(s1, e1 - s1)]

    return ans




def save_spectrogram0(y, sr, file_path, file_name, time_list, mask_list, params):
    """ 辞書で指定された音源ファイルを指定された区間で切り出しながらスペクトログラムを作成する
    time_list: list<tuple<float, float>>, 切り出す時間のタプル（開始時刻、時間幅）のリスト
    mask_dict: dict[str]<tuple<float, float>>, time_dictと構造は同じだが、スペクトログラム化しないリストを入れた辞書
    params: dict, 設定を書き込んだ辞書
    """

    # 区間の結合
    th = params["fusion"]
    if th > 0:
        tl = [(s, s + w)  for s, w in time_list]
        tl2 = fusion(tl, th)
        time_list = [(s, e - s)  for s, e in tl2]

    # 空のリストチェック
    if len(time_list) == 0:
        print("len of {} is zero.".format(file_name))
        return

    params["time_list"] = time_list
    params["mask_list"] = mask_list

    # ファイルのパスから、録音場所を識別して、略称を作る
    location = si.get_location_ID(file_path)

    # スペクトログラム画像の作成と保存
    si.save_spectrogram_with_timelist0(y, sr, file_path, location, params)




def save_spectrogram(time_dict, mask_dict, params):
    """ 辞書で指定された音源ファイルを指定された区間で切り出しながらスペクトログラムを作成する
    time_dict: dict[str]<tuple<float, float>>, 音源ファイルのパスと切り出す時間のタプル（開始時刻、時間幅）
    mask_dict: dict[str]<tuple<float, float>>, time_dictと構造は同じだが、スペクトログラム化しないリストを入れた辞書
    params: dict, 設定を書き込んだ辞書
    """
    for path in time_dict:
        time_list = time_dict[path]
        mask_list = []
        if path in mask_dict:
            mask_list = mask_dict[path]

        th = params["fusion"]
        if th > 0:
            time_list = fusion(time_list, th)

        # 空のリストチェック
        if len(time_list) == 0:
            print("len of {} is zero.".format(path))
            continue
        params["time_list"] = time_list
        params["mask_list"] = mask_list

        # ファイルのパスから、録音場所を識別して、略称を作る
        location = si.get_location_ID(path)

        # スペクトログラム画像の作成と保存
        si.save_spectrogram_with_timelist(path, location, params)



def set_default_setting(params: dict):
    """ デフォルトの設定をセットして返す
    """
    params["list_names"] = []
    params["fusion"] = 0
    params["mask_fname"] = ""
    params["mask_margin"] = 0    # マスク処理をかける際に、前後の方向に少し余裕を見て消す場合は0以上を設定のこと。
    params["ignore_size"] = 0
    params["save_each_dir"] = False   # Trueだと、音源の親フォルダと同じ名前のフォルダを作成する
    params["save_dir_root"] = ""  # 保存先のフォルダ ""だとカレントディレクトリとなる。
    return params



def main():
    # 設定の読み込み
    setting = si.set_default_setting()
    setting = set_default_setting(setting)
    setting = si.read_setting("save_spectimage_with_list_setting14.yaml", setting)
    print(setting)

    # マスク情報（例えばCDにおける人の声）の読み込み
    mask_dict = {}
    mask_fname = setting["mask_fname"]
    if mask_fname != "":
        mask_dict = read_list(mask_fname)

    # 保存先の作成　その1
    root = setting["save_dir_root"]
    if root != "":
        os.makedirs(root, exist_ok=True) 

    # 保存先の作成　その2
    time_dicts = {}
    for list_name in setting["list_names"]:
        # 設定ファイルを使った処理
        print("** now list file name is", list_name)
        list_ = read_list(list_name, ignore_size=setting["ignore_size"])
        if len(list_) == 0:
            print("The size of list is zero. so goto next list or finish creating dir.")
            continue
            
        time_dicts[list_name] = list_
        
        # 保存先の設定とフォルダの作成
        name, ext = os.path.splitext(os.path.basename(list_name))
        save_dir_ = name.replace("timelist_", "")
        save_dir = os.path.join(root, save_dir_)
        os.makedirs(save_dir, exist_ok=True)


    # 処理対象のファイルの一覧を作成
    files = set()
    for list_name in time_dicts:
        time_dict = time_dicts[list_name]
        files = files | set(time_dict.keys())

    files = sorted(list(files))


    # ファイルのID一覧を保存
    si.save_ID_list(files)


    # スペクトログラムを作成
    for file in files:
        # ファイルを読み込む
        sound_data = si.load_sound(file, setting)

        if sound_data is not None:
            y, sr, file_path = sound_data
        else:
            with open("error_log.txt", "a", encoding="utf-8-sig") as fw:
                fw.write(f"{dt.now().strftime('%Y/%m/%d %H:%M:%S.%f')},{file} reading is failed.\n")
            continue

        if y is None:
            with open("error_log.txt", "a", encoding="utf-8-sig") as fw:
                fw.write(f"{dt.now().strftime('%Y/%m/%d %H:%M:%S.%f')},{file} is not readable. check it.\n")
            continue

        print(f"---data size is {len(y)}.---")
        if len(y) == 0:
            with open("error_log.txt", "a", encoding="utf-8-sig") as fw:
                fw.write(f"{dt.now().strftime('%Y/%m/%d %H:%M:%S.%f')},size of {file} is zero.\n")
            continue

        # 各辞書を確認しつつ、処理
        for list_name in time_dicts:
            time_dict = time_dicts[list_name]

            if file in time_dict:
                # 保存先の設定とフォルダ名作成
                name, ext = os.path.splitext(os.path.basename(list_name))
                save_dir_ = name.replace("timelist_", "")
                save_dir = os.path.join(root, save_dir_)

                # 音源の親フォルダ毎にフォルダを分ける場合
                if setting["save_each_dir"]:
                    audio_dir = os.path.basename(os.path.dirname(file))
                    #audio_dir = audio_dir + "_ID-" + si.get_location_ID(file)   # 後でわかるように、ID名もくっつける
                    save_dir = os.path.join(save_dir, audio_dir)
                    os.makedirs(save_dir, exist_ok=True)

                # 保存先のフォルダ名を格納
                setting["root"] = save_dir     

                # 対象区間の読み出し
                time_list = time_dict[file]
                mask_list = []
                if file in mask_dict:
                    mask_list = mask_dict[file]

                # スペクトログラムの保存
                save_spectrogram0(y, sr, file_path, file, time_list, mask_list, setting)

    

if __name__ == "__main__":
    main()
