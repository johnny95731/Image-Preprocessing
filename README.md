# Image preprocessing 影像預處理
主要為[Gonzalez & Woods & 繆紹綱譯 : 數位影像處理](https://www.books.com.tw/products/0010855652?sloc=main)內容之實作，`interfact/main.py`為GUI進入點。<br>
使用Python, numpy, numba以及 OpenCV撰寫影像處理部分，並使用PyQt6和Pyqtgraph作為使用者介面及繪圖工具。<br>
空間域影像多以uint8資料型態處理，頻率域影像以complex64資料型態處理，以縮短執行時間。


## 運行環境需求(requirement)
<pre>
  Python         3.9.12
  Cython         0.29.32
  matplotlib     3.5.3
  numba          0.56.2
  numexpr        2.8.1
  numpy          1.22.4
  opencv-python  4.7.0
  pyFFTW         0.13.0
  PyQt6          6.4.0
  QDarkStyle     3.0.2
</pre>
