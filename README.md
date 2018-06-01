# objdet_darknet

使用pytorch实现YOLO（v3）。

---
## 参考资料

[How to implement a YOLO (v3) object detector from scratch in PyTorch: Part 1](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)

[How to implement a YOLO (v3) object detector from scratch in PyTorch: Part 2](https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/)

[darknet](https://github.com/pjreddie/darknet)

[YOLO_v3_tutorial_from_scratch](https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch)

[yolo-face](https://github.com/imistyrain/yolo-face)

[PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) 

[pytorch-caffe-darknet-convert](https://github.com/marvis/pytorch-caffe-darknet-convert)


---
## TODO

使用人脸数据集实现人脸检测（face detection）。

---
## 网络结构


```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 416, 416]             864
       BatchNorm2d-2         [-1, 32, 416, 416]              64
         LeakyReLU-3         [-1, 32, 416, 416]               0
            Conv2d-4         [-1, 64, 208, 208]          18,432
       BatchNorm2d-5         [-1, 64, 208, 208]             128
         LeakyReLU-6         [-1, 64, 208, 208]               0
            Conv2d-7         [-1, 32, 208, 208]           2,048
       BatchNorm2d-8         [-1, 32, 208, 208]              64
         LeakyReLU-9         [-1, 32, 208, 208]               0
           Conv2d-10         [-1, 64, 208, 208]          18,432
      BatchNorm2d-11         [-1, 64, 208, 208]             128
        LeakyReLU-12         [-1, 64, 208, 208]               0
           Conv2d-13        [-1, 128, 104, 104]          73,728
      BatchNorm2d-14        [-1, 128, 104, 104]             256
        LeakyReLU-15        [-1, 128, 104, 104]               0
           Conv2d-16         [-1, 64, 104, 104]           8,192
      BatchNorm2d-17         [-1, 64, 104, 104]             128
        LeakyReLU-18         [-1, 64, 104, 104]               0
           Conv2d-19        [-1, 128, 104, 104]          73,728
      BatchNorm2d-20        [-1, 128, 104, 104]             256
        LeakyReLU-21        [-1, 128, 104, 104]               0
           Conv2d-22         [-1, 64, 104, 104]           8,192
      BatchNorm2d-23         [-1, 64, 104, 104]             128
        LeakyReLU-24         [-1, 64, 104, 104]               0
           Conv2d-25        [-1, 128, 104, 104]          73,728
      BatchNorm2d-26        [-1, 128, 104, 104]             256
        LeakyReLU-27        [-1, 128, 104, 104]               0
           Conv2d-28          [-1, 256, 52, 52]         294,912
      BatchNorm2d-29          [-1, 256, 52, 52]             512
        LeakyReLU-30          [-1, 256, 52, 52]               0
           Conv2d-31          [-1, 128, 52, 52]          32,768
      BatchNorm2d-32          [-1, 128, 52, 52]             256
        LeakyReLU-33          [-1, 128, 52, 52]               0
           Conv2d-34          [-1, 256, 52, 52]         294,912
      BatchNorm2d-35          [-1, 256, 52, 52]             512
        LeakyReLU-36          [-1, 256, 52, 52]               0
           Conv2d-37          [-1, 128, 52, 52]          32,768
      BatchNorm2d-38          [-1, 128, 52, 52]             256
        LeakyReLU-39          [-1, 128, 52, 52]               0
           Conv2d-40          [-1, 256, 52, 52]         294,912
      BatchNorm2d-41          [-1, 256, 52, 52]             512
        LeakyReLU-42          [-1, 256, 52, 52]               0
           Conv2d-43          [-1, 128, 52, 52]          32,768
      BatchNorm2d-44          [-1, 128, 52, 52]             256
        LeakyReLU-45          [-1, 128, 52, 52]               0
           Conv2d-46          [-1, 256, 52, 52]         294,912
      BatchNorm2d-47          [-1, 256, 52, 52]             512
        LeakyReLU-48          [-1, 256, 52, 52]               0
           Conv2d-49          [-1, 128, 52, 52]          32,768
      BatchNorm2d-50          [-1, 128, 52, 52]             256
        LeakyReLU-51          [-1, 128, 52, 52]               0
           Conv2d-52          [-1, 256, 52, 52]         294,912
      BatchNorm2d-53          [-1, 256, 52, 52]             512
        LeakyReLU-54          [-1, 256, 52, 52]               0
           Conv2d-55          [-1, 128, 52, 52]          32,768
      BatchNorm2d-56          [-1, 128, 52, 52]             256
        LeakyReLU-57          [-1, 128, 52, 52]               0
           Conv2d-58          [-1, 256, 52, 52]         294,912
      BatchNorm2d-59          [-1, 256, 52, 52]             512
        LeakyReLU-60          [-1, 256, 52, 52]               0
           Conv2d-61          [-1, 128, 52, 52]          32,768
      BatchNorm2d-62          [-1, 128, 52, 52]             256
        LeakyReLU-63          [-1, 128, 52, 52]               0
           Conv2d-64          [-1, 256, 52, 52]         294,912
      BatchNorm2d-65          [-1, 256, 52, 52]             512
        LeakyReLU-66          [-1, 256, 52, 52]               0
           Conv2d-67          [-1, 128, 52, 52]          32,768
      BatchNorm2d-68          [-1, 128, 52, 52]             256
        LeakyReLU-69          [-1, 128, 52, 52]               0
           Conv2d-70          [-1, 256, 52, 52]         294,912
      BatchNorm2d-71          [-1, 256, 52, 52]             512
        LeakyReLU-72          [-1, 256, 52, 52]               0
           Conv2d-73          [-1, 128, 52, 52]          32,768
      BatchNorm2d-74          [-1, 128, 52, 52]             256
        LeakyReLU-75          [-1, 128, 52, 52]               0
           Conv2d-76          [-1, 256, 52, 52]         294,912
      BatchNorm2d-77          [-1, 256, 52, 52]             512
        LeakyReLU-78          [-1, 256, 52, 52]               0
           Conv2d-79          [-1, 512, 26, 26]       1,179,648
      BatchNorm2d-80          [-1, 512, 26, 26]           1,024
        LeakyReLU-81          [-1, 512, 26, 26]               0
           Conv2d-82          [-1, 256, 26, 26]         131,072
      BatchNorm2d-83          [-1, 256, 26, 26]             512
        LeakyReLU-84          [-1, 256, 26, 26]               0
           Conv2d-85          [-1, 512, 26, 26]       1,179,648
      BatchNorm2d-86          [-1, 512, 26, 26]           1,024
        LeakyReLU-87          [-1, 512, 26, 26]               0
           Conv2d-88          [-1, 256, 26, 26]         131,072
      BatchNorm2d-89          [-1, 256, 26, 26]             512
        LeakyReLU-90          [-1, 256, 26, 26]               0
           Conv2d-91          [-1, 512, 26, 26]       1,179,648
      BatchNorm2d-92          [-1, 512, 26, 26]           1,024
        LeakyReLU-93          [-1, 512, 26, 26]               0
           Conv2d-94          [-1, 256, 26, 26]         131,072
      BatchNorm2d-95          [-1, 256, 26, 26]             512
        LeakyReLU-96          [-1, 256, 26, 26]               0
           Conv2d-97          [-1, 512, 26, 26]       1,179,648
      BatchNorm2d-98          [-1, 512, 26, 26]           1,024
        LeakyReLU-99          [-1, 512, 26, 26]               0
          Conv2d-100          [-1, 256, 26, 26]         131,072
     BatchNorm2d-101          [-1, 256, 26, 26]             512
       LeakyReLU-102          [-1, 256, 26, 26]               0
          Conv2d-103          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-104          [-1, 512, 26, 26]           1,024
       LeakyReLU-105          [-1, 512, 26, 26]               0
          Conv2d-106          [-1, 256, 26, 26]         131,072
     BatchNorm2d-107          [-1, 256, 26, 26]             512
       LeakyReLU-108          [-1, 256, 26, 26]               0
          Conv2d-109          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-110          [-1, 512, 26, 26]           1,024
       LeakyReLU-111          [-1, 512, 26, 26]               0
          Conv2d-112          [-1, 256, 26, 26]         131,072
     BatchNorm2d-113          [-1, 256, 26, 26]             512
       LeakyReLU-114          [-1, 256, 26, 26]               0
          Conv2d-115          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-116          [-1, 512, 26, 26]           1,024
       LeakyReLU-117          [-1, 512, 26, 26]               0
          Conv2d-118          [-1, 256, 26, 26]         131,072
     BatchNorm2d-119          [-1, 256, 26, 26]             512
       LeakyReLU-120          [-1, 256, 26, 26]               0
          Conv2d-121          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-122          [-1, 512, 26, 26]           1,024
       LeakyReLU-123          [-1, 512, 26, 26]               0
          Conv2d-124          [-1, 256, 26, 26]         131,072
     BatchNorm2d-125          [-1, 256, 26, 26]             512
       LeakyReLU-126          [-1, 256, 26, 26]               0
          Conv2d-127          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-128          [-1, 512, 26, 26]           1,024
       LeakyReLU-129          [-1, 512, 26, 26]               0
          Conv2d-130         [-1, 1024, 13, 13]       4,718,592
     BatchNorm2d-131         [-1, 1024, 13, 13]           2,048
       LeakyReLU-132         [-1, 1024, 13, 13]               0
          Conv2d-133          [-1, 512, 13, 13]         524,288
     BatchNorm2d-134          [-1, 512, 13, 13]           1,024
       LeakyReLU-135          [-1, 512, 13, 13]               0
          Conv2d-136         [-1, 1024, 13, 13]       4,718,592
     BatchNorm2d-137         [-1, 1024, 13, 13]           2,048
       LeakyReLU-138         [-1, 1024, 13, 13]               0
          Conv2d-139          [-1, 512, 13, 13]         524,288
     BatchNorm2d-140          [-1, 512, 13, 13]           1,024
       LeakyReLU-141          [-1, 512, 13, 13]               0
          Conv2d-142         [-1, 1024, 13, 13]       4,718,592
     BatchNorm2d-143         [-1, 1024, 13, 13]           2,048
       LeakyReLU-144         [-1, 1024, 13, 13]               0
          Conv2d-145          [-1, 512, 13, 13]         524,288
     BatchNorm2d-146          [-1, 512, 13, 13]           1,024
       LeakyReLU-147          [-1, 512, 13, 13]               0
          Conv2d-148         [-1, 1024, 13, 13]       4,718,592
     BatchNorm2d-149         [-1, 1024, 13, 13]           2,048
       LeakyReLU-150         [-1, 1024, 13, 13]               0
          Conv2d-151          [-1, 512, 13, 13]         524,288
     BatchNorm2d-152          [-1, 512, 13, 13]           1,024
       LeakyReLU-153          [-1, 512, 13, 13]               0
          Conv2d-154         [-1, 1024, 13, 13]       4,718,592
     BatchNorm2d-155         [-1, 1024, 13, 13]           2,048
       LeakyReLU-156         [-1, 1024, 13, 13]               0
          Conv2d-157          [-1, 512, 13, 13]         524,288
     BatchNorm2d-158          [-1, 512, 13, 13]           1,024
       LeakyReLU-159          [-1, 512, 13, 13]               0
          Conv2d-160         [-1, 1024, 13, 13]       4,718,592
     BatchNorm2d-161         [-1, 1024, 13, 13]           2,048
       LeakyReLU-162         [-1, 1024, 13, 13]               0
          Conv2d-163          [-1, 512, 13, 13]         524,288
     BatchNorm2d-164          [-1, 512, 13, 13]           1,024
       LeakyReLU-165          [-1, 512, 13, 13]               0
          Conv2d-166         [-1, 1024, 13, 13]       4,718,592
     BatchNorm2d-167         [-1, 1024, 13, 13]           2,048
       LeakyReLU-168         [-1, 1024, 13, 13]               0
          Conv2d-169          [-1, 512, 13, 13]         524,288
     BatchNorm2d-170          [-1, 512, 13, 13]           1,024
       LeakyReLU-171          [-1, 512, 13, 13]               0
          Conv2d-172         [-1, 1024, 13, 13]       4,718,592
     BatchNorm2d-173         [-1, 1024, 13, 13]           2,048
       LeakyReLU-174         [-1, 1024, 13, 13]               0
          Conv2d-175          [-1, 255, 13, 13]         261,375
          Conv2d-176          [-1, 256, 13, 13]         131,072
     BatchNorm2d-177          [-1, 256, 13, 13]             512
       LeakyReLU-178          [-1, 256, 13, 13]               0
        Upsample-179          [-1, 256, 26, 26]               0
          Conv2d-180          [-1, 256, 26, 26]         196,608
     BatchNorm2d-181          [-1, 256, 26, 26]             512
       LeakyReLU-182          [-1, 256, 26, 26]               0
          Conv2d-183          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-184          [-1, 512, 26, 26]           1,024
       LeakyReLU-185          [-1, 512, 26, 26]               0
          Conv2d-186          [-1, 256, 26, 26]         131,072
     BatchNorm2d-187          [-1, 256, 26, 26]             512
       LeakyReLU-188          [-1, 256, 26, 26]               0
          Conv2d-189          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-190          [-1, 512, 26, 26]           1,024
       LeakyReLU-191          [-1, 512, 26, 26]               0
          Conv2d-192          [-1, 256, 26, 26]         131,072
     BatchNorm2d-193          [-1, 256, 26, 26]             512
       LeakyReLU-194          [-1, 256, 26, 26]               0
          Conv2d-195          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-196          [-1, 512, 26, 26]           1,024
       LeakyReLU-197          [-1, 512, 26, 26]               0
          Conv2d-198          [-1, 255, 26, 26]         130,815
          Conv2d-199          [-1, 128, 26, 26]          32,768
     BatchNorm2d-200          [-1, 128, 26, 26]             256
       LeakyReLU-201          [-1, 128, 26, 26]               0
        Upsample-202          [-1, 128, 52, 52]               0
          Conv2d-203          [-1, 128, 52, 52]          49,152
     BatchNorm2d-204          [-1, 128, 52, 52]             256
       LeakyReLU-205          [-1, 128, 52, 52]               0
          Conv2d-206          [-1, 256, 52, 52]         294,912
     BatchNorm2d-207          [-1, 256, 52, 52]             512
       LeakyReLU-208          [-1, 256, 52, 52]               0
          Conv2d-209          [-1, 128, 52, 52]          32,768
     BatchNorm2d-210          [-1, 128, 52, 52]             256
       LeakyReLU-211          [-1, 128, 52, 52]               0
          Conv2d-212          [-1, 256, 52, 52]         294,912
     BatchNorm2d-213          [-1, 256, 52, 52]             512
       LeakyReLU-214          [-1, 256, 52, 52]               0
          Conv2d-215          [-1, 128, 52, 52]          32,768
     BatchNorm2d-216          [-1, 128, 52, 52]             256
       LeakyReLU-217          [-1, 128, 52, 52]               0
          Conv2d-218          [-1, 256, 52, 52]         294,912
     BatchNorm2d-219          [-1, 256, 52, 52]             512
       LeakyReLU-220          [-1, 256, 52, 52]               0
          Conv2d-221          [-1, 255, 52, 52]          65,535
================================================================
Total params: 61,949,149
Trainable params: 61,949,149
Non-trainable params: 0
----------------------------------------------------------------
```
