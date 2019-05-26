# Binocular-Stereo-Vision
基于SIFT特征匹配的双目立体视觉测距

1.DisparityMap 是第一种测距方法，景深测距可按照BM或SGBM计算

2.Similar-Triangles是主要的相似三角形测距法，主要公式 d=Bf/abs(x2-x1),B为相机的平移向量模 ，f为相机焦距
头文件的形式以便用于程序引用

3.VedioCaptrue主要是用于左右相机同时截图以用于标定

4.data 为标定得到的相机内外参数形式

5.calibration为使用opencv立体标定的方法，实验中比较常采用MATLAB标定工具箱
