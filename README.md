# DAMO-YOLO-detect-onnxrun-cpp-py
使用ONNXRuntime部署DAMO-YOLO目标检测，包含C++和Python两个版本的程序
起初，我想使用opencv做部署的，但是opencv的dnn模块读取onnx文件出错，
无赖只能使用onnxruntime做部署了。本套程序一共提供了27个onnx模型，
onnx文件需要从百度云盘下载，
链接：https://pan.baidu.com/s/10-5ke_fs2omqUMSgKTJV0Q 
提取码：w9kp

其中在百度云盘里一共有30个onnx模型文件，但是逐个读取onnx文件做推理时，
发现有3个onnx文件在onnxruntime读取时出错了，在程序里的choices参数里声明了
27个模型文件的名称。
