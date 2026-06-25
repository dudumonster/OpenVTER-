# road config

使用[labelme](https://github.com/wkentaro/labelme)对视频第一帧进行标记。因此，需要安装labelme，具体安装可参考[labelme](https://github.com/wkentaro/labelme)

步骤如下：
1. 用于提取需要计算的道路的多边形用Polygons标注（可不标注，但为减少计算量推荐标注）：label为road
2. 用于稳定视频的跟踪点用Polygons或Rectangle标注（4个及以上）：label为fp，代表fixed point，选取固定建筑物等具有特征角点信息的区域
3. 标注基准长度，采用Line，可标注多个：
   - 路面主标尺使用`length_长度`（单位米），例如`length_3.5`。代码使用稳健中位数估计比例并剔除明显异常标线；至少需要1条，建议在画面不同区域标注3条以上。
   - 可选的车辆辅助标尺使用`vehicle_length_长度`，例如`vehicle_length_4.3`。至少5条有效车辆标尺才参与计算；仅标注已确认长度接近标签值、无遮挡且不含阴影的车辆。
   - 仅有路面标尺时使用路面稳健值；仅有车辆标尺时，至少5条有效标线即可使用车辆稳健值。两组同时存在且比例差异不超过8%时，按路面75%、车辆25%融合；超过8%时忽略车辆标尺、回退路面结果并在标定报告中报警。
4. 坐标轴绘制，采用Line，绘制line时第一个点为原点，第二个点为x轴上的点，以此来确定原点和x轴方向：label为x。同时在线的某一侧绘制Point，lable为y，代表y轴方向（由于无法准确绘制90度角因此采用绘制点的方式）
5. （可选）车道绘制，采用Polygons标注：labe为lane_***，例如lane_1 ，***需为数字
6. （可选）标记车辆行驶线，计算车辆沿道路行驶的距离
  a. 标记投影区域，采用Polygons标注，label为drivingline_{line name}_{position_id}_region
  b. 标注投影线，一般为道路中心线，采用LineStrip标注，label为drivingline_{line name}_{position_id}_line
  c. 标注投影线的坐标零点，采用Point标注，label为drivingline_{line name}_{position_id}_point_{base distance}
  d. 标注xxxx点，采用Point标注，label为drivingline_{line name}_{position_id}_connect_{position_id1}_{position_id2}

保存.json文件

注意：以上标注中label名不可出错，不然不能识别。
