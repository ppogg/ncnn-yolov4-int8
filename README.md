# ncnn-yolov4-int8
NCNN+Int8+YOLOv4 quantitative modeling and real-time inference

**Int8** quantization and inference for **yolov4-tiny** using **ncnn** is performed as follows.


 - **inference**

Equipment | Computing backend | System | Framework | input_size| Run time
 :-----:|:-----:|:-----:|:----------:|:----:|:----:|
Intel | Core i5-4210 | window10（x64） | ncnn@fp16  | 320| 36ms
Intel | Core i5-4210 | window10（x64） | ncnn@int8 | 320| 57ms
Raspberrypi 3B| 4xCortex-A53 | Linux(arm64) | ncnn@fp16 | 320| 313ms
Raspberrypi 3B| 4xCortex-A53 | Linux(arm64) | ncnn@int8 | 320| 217ms


 - **Installing NCNN**
 
    git clone  [https://github.com/Tencent/ncnn.git](https://github.com/Tencent/ncnn.git)
    cd ncnn
    mkdir build
    cd build
    cmake ..
    make
    make install
    
中文教程：[https://zhuanlan.zhihu.com/p/368653551](https://zhuanlan.zhihu.com/p/368653551)


 - **Run Demo**
 
    git clone  [https://github.com/pengtougu/ncnn-yolov4-int8.git](https://github.com/pengtougu/ncnn-yolov4-int8.git)
      Open VS-2019 and create project
      copy yolov4.cpp
      run
 
 中文教程：[https://zhuanlan.zhihu.com/p/372278785](https://zhuanlan.zhihu.com/p/372278785)

- **Note**

 Please watch out the following code changes
 ![](https://pic3.zhimg.com/80/v2-b989f8bf5e0a01a30723f18d8b84ff5e_720w.jpg)
 
