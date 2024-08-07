# 概要
　ここでは、主に予測結果を基に教師データの作り直しを行ったり、検出結果の統計を取るためのスクリプトファイルを格納しています。各ディレクトリとその中に格納されたプログラムの説明を以下に示します。　


## 01_restoration_timelist_from_image

　このディレクトリでは、教師画像の作り直しを目的として、03_save_spectrogramに格納されているrestoration_timelist_from_image*.pyやsound_image*.pyを使って作成されたスペクトログラム画像から区間リスト（timelist）を作成するスクリプトrestoration_timelist_from_image*.pyを格納しています。

例えば、多数のフォルダに分散して保存されている教師画像を設定を変えて再作成する場合や、03_save_spectrogramで作成した画像を整理して正解データを作成する際に利用します。正解データを整備すれば、evaluate.pyを使って適合率や再現率を求めることができます。


## 02_create_timelist_from_likelifoods

　このディレクトリに格納されたtimelist_from_likelihoods*.pyは、predict.pyが作成した予測結果ファイル（likelifood.csv）からスペクトログラム画像を作成することを目的として、尤度が閾値以上の区間リスト（timelist）をクラス毎に作成します。このスクリプトを使って区間リストを作成した後に03_save_spectrogramを実行すれば、区間リストで指定された音源と時間のスペクトログラム画像が作成されます。

　作成する区間リストは尤度の範囲を複数指定したり、クラス別に設定を変えることができます。これにより、特定のクラスのみ適合率を高くしたり（尤度の閾値を上げる）、教師画像作成に向けて再現率を高めたり（尤度の閾値を下げる）、尤度の範囲毎の誤識別傾向を把握することができます。


## 03_save_spectrogram

　このディレクトリは、区間リストに基づいてスペクトログラム画像を作成するrestoration_timelist_from_image*.pyと、設定ファイルに基づいてスペクトログラム画像を作成するsound_image*.pyを格納しています。予測結果の正しさを確認するために、02_create_timelist_from_likelifoodsと組み合わせて使用します。

なお、設定ファイルや区間リストを編集すれば、任意の音源の任意の時間帯から教師画像を作成するためにも利用できます。


## 04_statistics

　このディレクトリに格納されたstatistics.pyは、predict.pyが作成した予測結果ファイル（likelifood.csv）や、03_save_spectrogramで作成して整理したスペクトログラム画像からクラス毎に検出結果を集計します、時系列での検出数の折れ線グラフや積み立てグラフ、年間の通算日と検出時刻の散布図、音源毎の検出時間を出力します。これを利用すれば、野鳥の囀りと日周運動との関係や、月毎の囀り頻度の変化、録音場所毎の囀り傾向を可視化できます。

予測結果が十分に信用できれば、クラス毎の尤度が記載された予測結果ファイル（likelifood.csv）から直接集計できます。一方で信用できないのであれば、03_save_spectrogramを使ってスペクトログラムを作成した後に誤識別した画像を削除してからstatistics.pyを実行してください。どちらを利用するかは設定ファイルに書き込みます。


# 使い方

　基本的には、予測結果が格納されているディレクトリと同じ場所（predict*直下）に上記4つのディレクトリをコピーして使用します。スクリプトを実行する際は、それぞれのスクリプトのあるディレクトリにカレントディレクトリを移動してから実行してください。なお、それぞれの設定ファイル内に記載されているパスの設定は自由に変更できますので、任意のフォルダにスクリプトファイルと設定ファイルをコピーして実行することも可能です。


