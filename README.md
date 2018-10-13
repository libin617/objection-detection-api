# objection-detection-api
tensorflow目标检测的api
从github下载代码
同样在cmd进入到models文件夹，编译Object Detection API的代码：protoc object_detection/protos/*.proto --python_out=.
运行notebook demo
输入 jupyter-notebook
浏览器自动开启
进入到进入object_detection文件夹中的object_detection_tutorial.ipynb：
点击Cell内的Run All，等待三分钟左右即可识别图片以及视频
修改文件路径即可识别检测自己的图片
注意：要将图片名称设置的和代码描述相符合，如image1.jpg 
可直接将MODEL_NAME修改为如下值调用其他模型：MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'

MODEL_NAME = 'rfcn_resnet101_coco_11_06_2017'

MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'

MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'
其中video1.mp4已经从电脑中上传至object_detection文件夹
感谢csdn中withzheng博主 提供的步骤，让我完美运行，所以上传daogithub中，供大家喜欢tensorflow的用户参考交流
本人微信bin617，欢迎一起交流

