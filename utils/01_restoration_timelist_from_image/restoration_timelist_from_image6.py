# purpose: 画像ファイル名から、区間リストを生成する
# history: 
#              ver.3  fusionに渡す区間リストが時系列でなくても良いように差し替えた。
#                     設定ファイルの読み込みに対応した。
#                     沖縄で録音したもの以外の音源にも対応した。拡張子の大文字小文字の差にも対応した。
#   2021-10-31 ver.4  画像が複数のフォルダにあっても一気に処理できるように変更。
#   2021-11-01        フォルダ名を変更していた場合へ対応。かもしれない処理を実装。やややっつけなので、近日中にファイル整理が必要。
#   2023-12-11 ver.5  get_location_ID()を最新版に差し替えた。
#   2023-12-13        最近のIDと古いIDの両方に対応。また、画像作成時にミスって__を___にしているものがあったので、それにも対応した。
#   2024-04-26        画像の中に間違ってコピーしたファイルがあっても無視して進めるように変更
#   2024-07-10 ver.6  read_setting()を差し替えた。設定ファイルはyaml形式にした方が良いかも。
#                     全体的にコメントを調整し、画像と音源のマッチングの機能をクラスにまとめた（他で利用できるように）
#                     また、マッチングでは類似性を考慮できるように変更したので、音源ファイルのファイル名変更に強くなったと思う。
# author: Katsuhiro Morishita
# created: 2020-03-12
# license: MIT. If you use this program for your study, you should write Acknowledgement in your paper.
import os, re, glob, copy, hashlib, unicodedata

import numpy as np
import Levenshtein




def print2(*args):
    """ いい感じにstrやdictやlistを表示する
    """
    for arg in args:
        if isinstance(arg, dict) or isinstance(arg, list):
            pprint.pprint(arg)
        else:
            print(arg)



# sound_image9.pyよりコピー＆改編
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



def split_fname(path_image, ext):
    """ スペクトログラム画像のファイル名のパスから、画像ファイル名単体と元音源ファイルのファイル名とIDと抜出時間のタプルを返す
    path_image: str, 画像ファイルのパス名
    ext: str, 音源の拡張子。例：".mp3"
    """
    fname_image = os.path.basename(path_image)         # ファイル名取り出し
    dummy, loc, name_and_times = re.split(r'_{2,4}', fname_image, 2)  # __で分割

    # ファイル名と時間区間の取得（ファイル名の中に__があっても対応）
    name_and_times_mirror = name_and_times[::-1]
    time_mirror, name_mirror = re.split(r'_{2,4}', name_and_times_mirror, 1)
    name = name_mirror[::-1]
    times = time_mirror[::-1]                    # この時点では拡張子が含まれる
    time_pair, ext2 = os.path.splitext(times)    # 拡張子を分離
    #print(">>>> ", time_pair)
    t_start, t_width = time_pair.split("_")[:2]      # _で分離（たまにファイルのコピーしたやつがあるので、2つだけ取り出す

    # 音源ファイルのフルパスを作成
    fname_music = name + ext.lower()

    return (fname_image, fname_music, loc, float(t_start), float(t_width))





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
        time_mirror, name_mirror = re.split(r'_{2,4}', image_file_path_temp_mirror, 1)
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
    params["target_images"] = []  # 処理対象となる画像ファイルの存在するフォルダ
    params["audio_files"] = []    # マッチングさせたい音源ファイルのパスのリスト
    params["fusion_th"] = None    # 結合する際の閾値
    params["exchange_dict"] = dict()   # 例：　{"mp3": "wav"}　など。ファイル名の一部が規則的に変更されたときに利用する。　
    return params






def main():
    # 設定を読み込み
    setting = set_default_setting()
    setting = read_setting("restoration_setting6.txt", setting)

    # 値の取り出し
    target_images = setting["target_images"]
    audio_files = setting["audio_files"]
    fusion_th = setting["fusion_th"]
    ext = ".mp3"   # wavかもしれないが、とりあえず入れておく


    print(audio_files)
    am = audio_matcher(audio_files)

    # チェック
    if am.ready == False:
        print("Please check parameter audio_files in the setting file.")
        return

    
    
    # フォルダ名のセットを作成
    dirs = set([os.path.basename(os.path.dirname(path)) for path in target_images])
    print2("\n\n\n---image dir--", dirs)

    # 画像のあるフォルダ毎に処理
    for dir_ in dirs:
        # 同一のディレクトリの画像だけのリストを作成
        target_images_sub = [path for path in target_images if os.path.basename(os.path.dirname(path)) == dir_]

        with open("timelist_{}.txt".format(dir_), "w", encoding="utf-8-sig") as fw:
            
            # ファイル名を処理しながら、ファイル名と時間のデータを整理
            time_dict = {}
            for path_img in target_images_sub:
                #print(path_img)

                audio_path, ts, te, tw = am.match(path_img)

                if audio_path != "":
                    # 画像ファイルごとに、切り出し時刻を格納していく
                    if audio_path not in time_dict:
                        time_dict[audio_path] = []
                    time_dict[audio_path].append((ts, te))


            # 時系列になる様に並べ替え（fusion()は実行されないこともあるので、当てにしない）
            for fpath in time_dict:
                time_list = time_dict[fpath]
                time_dict[fpath] = sorted(time_list, key=lambda x:x[0])

            # 近い区間や連続区間の結合
            if fusion_th is not None:
                for fpath in time_dict:
                    time_list = time_dict[fpath]
                    time_dict[fpath] = fusion(time_list, fusion_th)

            
            # ファイル名と音の区間の時間を整理してファイルに保存
            fpaths = sorted(time_dict.keys())

            for fpath in fpaths:
                dir_ = os.path.basename(os.path.dirname(fpath))
                time_list = sorted(time_dict[fpath])
                time_list = ["{:.2f},{:.2f}".format(s, e-s) for s,e in time_list]
                txt = ",".join(time_list)                # デリミタで結合

                # パスの書き換え
                for key in setting["exchange_dict"]:
                    if key in fpath:
                        fpath = fpath.replace(key, setting["exchange_dict"][key])

                # 書き込み
                fw.write("{},{}\n".format(fpath, txt))   # ファイルに書き込み


if __name__ == "__main__":
    main()