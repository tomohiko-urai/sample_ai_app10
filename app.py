#-*- coding:utf-8 -*-
import model as tomato_chk 
import sys, os
from PIL import Image
import numpy as np
#import cv2 
import streamlit as st
#import matplotlib.pyplot as plt
#import datetime

image_size = 50
## 2022 categories = [
## 2022    "ng","pass" ]
## 2022 camerapos = ["NG","GOOD"]

categories = ["Blue","white","red"]
##### 20230313 categories = ["Blue-aa","white-aa","red-aa"]

camerapos = ["0","1","2"]

st.set_option("deprecation.showfileUploaderEncoding", False)

st.sidebar.title("トマト画像収穫認識アプリ")
st.sidebar.write("オリジナルの画像認識モデルを使ってトマト収穫判定します。")

st.sidebar.write("")

img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    with st.spinner("推定中..."):
        img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")

        img = img.convert("RGB")
        img = img.resize((image_size,image_size))
        in_data = np.asarray(img)
        X = []
        X.append(in_data)
        X = np.array(X)
        # CNNのモデルを構築 --- (※3)
        model = tomato_chk.build_model(X.shape[1:])

        model.load_weights("tomato-color5a-model3b-10-100.hdf5")
        ####20230313  model.load_weights("tomato-color2-model2.hdf5")
# データを予測 --- (※4)
      
        pre = model.predict(X)
        y=0
        y = pre.argmax()
        #st.image(img, caption="対象の画像", width=480)
        #st.write("")

        # 予測
        #results = predict(img)

        # 結果の表示
        st.subheader("判定結果")

        st.write(categories[y] + "です。")
