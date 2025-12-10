#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2021-12-20 21:24
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : video_process.py 
@Software: PyCharm
@desc: 
'''
import os

import cv2
import json
import pickle
import numpy as np
import torch
from mmcv.ops import nms_rotated


from detection.VehicleDetModule import VehicleDetModule
from tracking.VehicleTrackingModule import VehicleTrackingModule

from stabilization.VideoStabilization import VideoStabilization

import time
import datetime
from utils import Config,RoadConfig,isPointinPolygon
from utils.VideoTool import get_all_video_info,get_srt,splitbase

class DroneVideoProcess:
    def __init__(self,config_json):
        config_dict = Config.fromfile(config_json)
        self.pipeline = config_dict.get('pipeline',['det'])
        self.video_file = self._get_video_file(config_dict)
        self.road_config = self._load_road_config(config_dict.get('road_config',None))
        self.mask = self.road_config['det_mask']
        self.axis_image = self.road_config['axis_image']
        self.length_per_pixel = self.road_config.get('length_per_pixel', None)
        # self.split = splitbase('','',subsize_height=812,
        #          subsize_width=940)
        self.split_gap = config_dict.get('split_gap',100)
        subsize_height = config_dict.get('subsize_height',640)
        subsize_width = config_dict.get('subsize_width',640)
        self.split = splitbase('', '',gap=self.split_gap,subsize_height=subsize_height,subsize_width=subsize_width)

        # self.num_classes = config_dict.get('num_classes') # 分类类别
        # self.checkpoints = config_dict.get('checkpoints') # 模型路径
        # self.phi = config_dict.get('phi')

        self.output_video = config_dict.get('output_video',1) # 是否输出视频,0:不输出，1:输出
        self.output_img = config_dict.get('output_img',0) # 是否输出图片,0:不输出，1:输出

        self.sub_positions = None
        self.out_fps = config_dict.get('out_fps') #输出数据的fps，考虑抽帧的方式
        self.conf_thresh = config_dict.get('conf_thresh',0.25) # 目标检测置信度

        self.save_folder = config_dict.get('save_folder') # 输出路径
        self.inference_batch_size = config_dict.get('inference_batch_size',1) # 推理时候的batch大小，batch中图片是小图片

        self.video_start_frame = config_dict.get('video_start_frame',0) # 视频开始帧
        self.video_end_frame = config_dict.get('video_end_frame',0) # 视频结尾截取掉的帧

        self.stabilize_scale = config_dict.get('stabilize_scale', 1)
        self.stabilize_smooth = config_dict.get('stabilize_smooth', 1)
        self.stabilize_translate = config_dict.get('stabilize_translate', 1)

        #可视化相关
        self.output_background = config_dict.get('output_background', 1) # 输出背景图片
        self.debug_lane_log = config_dict.get('debug_lane_log', False)
        self.debug_lane_draw = config_dict.get('debug_lane_draw', False)
        self.background_image_ls = []

        # 掩膜调试：只打印一次掩膜统计，避免刷屏
        self._mask_logged = False

        self.bbox_label = config_dict.get('bbox_label',['id','score','xy'])
        _,video_name_ext = os.path.split(self.video_file[0])
        self.video_name, extension = os.path.splitext(video_name_ext)
        if len(self.video_file)==1:
            self.save_folder = os.path.join(self.save_folder,self.video_name)
        else:
            self.save_folder = os.path.join(self.save_folder, self.video_name+"_Num_%d"%len(self.video_file))
        os.makedirs(self.save_folder,exist_ok=True)
        # 目标检测
        self.det_model = self._get_det_model(config_dict.get('detection'))

        # 跟踪模型
        self.mot_tracker = self._get_tracking_model(config_dict.get('tracking'))

        # 稳定
        if 'stab' in self.pipeline:
            self.stabilize_transformers_path = os.path.join(self.save_folder,config_dict.get('stabilize_file',None))
            self._init_stabilizer()

        # —— 写视频延迟初始化
        self.video_writer = None
        self.writer_size = None
        self.writer_path = None

        # —— 掩膜诊断：只打印一次
        self._mask_warned = False

        self.det_bbox_result = {'video_info': [], 'output_info': {'output_fps': self.out_fps}, 'traj_info': [],
                                'process_time': datetime.datetime.now(),'raw_det':[]}
        # 存储检测结果
        #video_info: 视频信息
        #output_info: 输出信息
        #traj_info: 轨迹信息
        #process_time: 处理时间
        #raw_det: 原始检测结果

    def _get_video_file(self,config_dict):
        if 'video_file' in config_dict:
            return [config_dict.get('video_file')]
        elif 'first_video_name' in config_dict and 'video_num' in config_dict:
            video_folder = config_dict.get('video_folder')
            first_video_name = config_dict.get('first_video_name')
            video_num = int(config_dict.get('video_num'))
            video_file_ls = []
            for i in range(video_num):
                video_name = first_video_name.format(i+1)
                video_file_ls.append(os.path.join(video_folder, video_name))
            return video_file_ls
        else:
            video_folder = config_dict.get('video_folder')
            video_name_ls = config_dict.get('video_name')
            video_file_ls = []
            for video_name in video_name_ls:
                video_file_ls.append(os.path.join(video_folder, video_name))
            return video_file_ls

    def _get_det_model(self,config):
        det_model = VehicleDetModule(**config)
        det_model.load_model()
        return det_model

    def _get_tracking_model(self,config):
        tracking_model = VehicleTrackingModule(**config)
        return tracking_model

    def _get_mask(self,mask_json_file):
        if mask_json_file is None or mask_json_file == '':
            return None
        with open(mask_json_file, "r",encoding='utf-8') as f:
            tmp = f.read()
        annotation = json.loads(tmp)
        imageHeight = annotation['imageHeight']
        imageWidth = annotation['imageWidth']
        mask = np.zeros((imageHeight,imageWidth,3))
        if annotation['shapes']:
            for s in annotation["shapes"]:
                if s['label'] == 'road':
                    points = s["points"]
                    points = np.array(points, np.int32)
                    cv2.fillPoly(mask, [points], (1, 1, 1))
        return mask

    def process_video(self):
        gap_length = self.road_config['length_per_pixel']*self.split_gap
        print('gap_length:%f m'%gap_length)
        num_frame_ls,all_num_frame,width,height,fps = get_all_video_info(self.video_file)
        srt_info_ls = get_srt(self.video_file)
        if len(srt_info_ls)==0:
            print('SRT file is not used')
        video_info = {'video_name': self.video_file, 'width': width, 'height': height, 'fps': fps,
                      'total_frames': all_num_frame}
        self.det_bbox_result['video_info'].append(video_info)
        print("Output Folder:%s" % self.save_folder)
        print('Process Pipeline:','->'.join(self.pipeline))
        gap = round(fps/self.out_fps)
        print('Frame gap:%d'%gap)
        frame_index = self.video_start_frame
        output_frame = 0

        self.video_writer = None
        self.writer_size = None
        self.writer_path = None

        try:
            s_time = time.time()
            for video_index,video_file in enumerate(self.video_file):
                cap = cv2.VideoCapture(video_file)
                valid_frames = num_frame_ls[video_index]
                if video_index == 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,self.video_start_frame)
                    valid_frames -= self.video_start_frame
                if video_index == len(self.video_file) - 1:
                    valid_frames -= self.video_end_frame
                print('Input Video:%s'%video_file)
                print("Input Video:width:{} height:{} fps:{} num_frames:{} valid_frames:{}".format(width, height, fps, num_frame_ls[video_index], valid_frames))
                current_video_frame = 0
                video_frame_index = 0
                while video_frame_index<valid_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_index == self.video_start_frame:
                        first_frame_name = os.path.join(self.save_folder,'first_frame_'+self.video_name+'.jpg')
                        if self.axis_image is not None:
                            new_frame = cv2.add(frame, self.axis_image)
                            cv2.imwrite(first_frame_name, new_frame)
                        else:
                            cv2.imwrite(first_frame_name, frame)
                    if frame_index%gap==0:

                        self._process_img(frame,output_frame,frame_index,srt_info_ls,self.save_folder)
                        output_frame += 1
                        e_time = time.time()
                        remain_time = (e_time - s_time) * (all_num_frame//gap - output_frame - 1)
                        process_fps = 1/(e_time-s_time)
                        print('\rvideo index:%d,process frame:%d/%d,current video:%d/%d, FPS:%.1f, remain time:%.2f min'%(video_index,frame_index,all_num_frame-self.video_end_frame,current_video_frame,valid_frames,process_fps,remain_time/60),end="",flush=True)
                        s_time = e_time
                    video_frame_index += 1
                    frame_index += 1
                    current_video_frame += 1
                cap.release()
        finally:
            if self.output_video and self.video_writer is not None:
                self.video_writer.release()
            self._save_det_bbox(self.save_folder)
            print('save video')


    # —— 延迟初始化写视频 + 安全写帧
    def _init_video_writer_safe(self, first_frame):
        h, w = first_frame.shape[:2]
        size = (int(w), int(h))
        out_path = os.path.join(self.save_folder, 'tracking_output_' +"_".join(self.pipeline)+"_"+ self.video_name + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(out_path, fourcc, float(self.out_fps), size)
        if not vw.isOpened():
            out_path = out_path[:-4] + '.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            vw = cv2.VideoWriter(out_path, fourcc, float(self.out_fps), size)
        assert vw.isOpened(), "VideoWriter open failed"
        self.writer_size = size
        self.writer_path = out_path
        self.video_writer = vw
        print("Output Tracking Video:%s" % out_path)

    def _safe_write(self, frame):
        if self.video_writer is None:
            self._init_video_writer_safe(frame)
        w, h = self.writer_size[0], self.writer_size[1]
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
        if frame.dtype != np.uint8:
            mx = float(frame.max()) if frame.size else 1.0
            if mx <= 1.0:
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        self.video_writer.write(frame)

    def _process_img(self,frame,output_frame,frame_index,srt_info_ls,save_file_folder=None):
        t1 = time.time()
        # 稳像
        if 'stab' in self.pipeline:
            frame = self.video_stabilizer.stabilize_frame(frame,frame_index,scale=self.stabilize_scale,
                                                          smooth_xya=self.stabilize_smooth,
                                                          translate=self.stabilize_translate)
        # 背景
        if self.output_background and (not self.background_image_ls is None):
            if len(self.background_image_ls)<50:
                self.background_image_ls.append(frame)
            else:
                background_img = np.zeros(frame.shape, dtype=np.float32)
                for image_b in self.background_image_ls:
                    background_img += image_b.astype(np.float32)
                background_img = background_img / len(self.background_image_ls)
                background_img = np.clip(background_img, 0, 255).astype(np.uint8)
                background_path = os.path.join(save_file_folder, 'background_%s.jpg'%self.video_name)
                cv2.imwrite(background_path, background_img)
                print('output background image to:%s'%background_path)
                self.background_image_ls = None

        # —— 关键修改：掩膜兜底，避免把整帧抹黑
        if self.mask is not None:
            m = self.mask
            if isinstance(m, np.ndarray):
                if m.ndim == 3:
                    m = m[:, :, 0]
                if m.shape[:2] != frame.shape[:2]:
                    m = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                # 归一为 0/255 的 uint8
                if m.dtype != np.uint8:
                    # 只要非 0 就当 255，用于 bitwise_and
                    m = (m != 0).astype(np.uint8) * 255
                # 仅首次打印掩膜统计，确认掩膜已生效
                if not self._mask_logged:
                    print(f"[mask_debug] sum={int(m.sum())}, shape={m.shape}, dtype={m.dtype}")
                    self._mask_logged = True
                # 空掩膜就跳过，避免黑屏
                if int(m.sum()) == 0:
                    if not self._mask_warned:
                        print("Warning: det_mask is empty -> skip masking (frame kept as-is).")
                        self._mask_warned = True
                else:
                    frame = cv2.bitwise_and(frame, frame, mask=m)
                    # 缓存处理后的 mask，后续帧直接用
                    self.mask = m
            else:
                if not self._mask_warned:
                    print("Warning: det_mask is not a numpy array -> skip masking.")
                    self._mask_warned = True

        # 切分
        if frame_index == self.video_start_frame:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sub_imgs, sub_positions = self.split.split_image(frame_rgb)
            self.sub_positions = sub_positions
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sub_imgs = self.split.split_image_with_position(frame_rgb,self.sub_positions)

        # 检测
        t2 = time.time()
        new_nms_ls = []
        for i in range(0,len(sub_imgs),self.inference_batch_size):
            s = i
            e = min(i+self.inference_batch_size,len(sub_imgs))
            select_imgs = sub_imgs[s:e]
            select_positions = self.sub_positions[s:e]
            nms_results_ls = self.det_model.inference_img_batch(select_imgs)
            for nms_results,position in zip(nms_results_ls,select_positions):
                if nms_results is None:
                    continue
                x, y = position

                position_arr = np.array([x, y, 0, 0, 0, x, y, x, y, x, y, x, y,0, 0],dtype=np.float32)#字段偏移量
                position_arr_t = torch.from_numpy(position_arr).to(self.det_model.device)# 将位置信息转换为tensor
                new_nms = nms_results + position_arr_t# 将检测结果与位置信息相加，得到新的检测结果
                new_nms_ls.append(new_nms)
        t3 = time.time()
        # 坐标轴
        if self.axis_image is not None:
            frame = cv2.add(frame, self.axis_image)# 将坐标轴图像添加到帧中
        # 跟踪与可视化
        det_raw = np.empty((0, 15))
        lane_centers = np.empty((0, 2))
        lane_ids = []
        if len(new_nms_ls)==0:
            self.mot_tracker.update()
            nms_img = frame
            if self.road_config['pixel2xy_matrix'] is not None:
                if len(self.road_config['lane'])==0:
                    o_bboxs_res = np.empty((0, 19))
                else:
                    o_bboxs_res = np.empty((0, 20))
            else:
                o_bboxs_res = np.empty((0, 11))
            lane_ids = []
        else:
            all_bbox = torch.vstack(new_nms_ls)
            dets, keep_inds = nms_rotated(all_bbox[:, :5], all_bbox[:, 5], 0.3)
            nms_all_bbox = all_bbox[keep_inds]
            # —— 简单截断，避免候选过多导致跟踪/IoU 计算爆内存
            max_det = getattr(self, "max_object_num", None)
            if max_det is None:
                # 从检测配置里读取（与 centernet_bbavectors 配置保持一致）
                max_det = self.det_model.__dict__.get("max_object_num", None)
            if max_det is None:
                max_det = 500  # 默认限制 500 个框，可按需调小
            if nms_all_bbox.shape[0] > max_det:
                scores = nms_all_bbox[:, 5]
                _, topk_idx = torch.topk(scores, k=max_det)
                nms_all_bbox = nms_all_bbox[topk_idx]
            if nms_all_bbox.device.type == 'cpu':
                det_raw = nms_all_bbox.numpy()
            else:
                det_raw = nms_all_bbox.data.cpu().numpy()
            o_bboxs_res = self.mot_tracker.update(nms_all_bbox,frame)
            o_bboxs_res = self._pixel_to_xy(o_bboxs_res)
            o_bboxs_res = self._assign_vehicle_cat_by_length(o_bboxs_res)
            o_bboxs_res, lane_centers, lane_ids = self._get_lane_id(o_bboxs_res, frame_index)
            nms_img = self.det_model.draw_oriented_bboxs(frame, o_bboxs_res,self.bbox_label)
            if self.debug_lane_draw and len(lane_centers) > 0:
                self._draw_lane_debug(nms_img, lane_centers, lane_ids)

        t4 = time.time()
        if self.output_img:
            file_folder = os.path.join(save_file_folder,'det_img_'+self.video_name)
            if not os.path.exists(file_folder):
                os.makedirs(file_folder)
            img_name = os.path.join(file_folder,'%06d.jpg'%output_frame)
            cv2.imwrite(img_name, nms_img)

        if self.output_video:
            self._safe_write(nms_img)

        # 结果存储
        if len(srt_info_ls) == 0:
            self.det_bbox_result['traj_info'].append((frame_index,output_frame,o_bboxs_res))#
            self.det_bbox_result['raw_det'].append((frame_index,output_frame,det_raw))
        else:
            if frame_index >= len(srt_info_ls):
                frame_time = srt_info_ls[-1][0]
            else:
                frame_time = srt_info_ls[frame_index][0]
            self.det_bbox_result['traj_info'].append((frame_index,output_frame, o_bboxs_res,frame_time))#轨迹信息
            self.det_bbox_result['raw_det'].append((frame_index, output_frame, det_raw,frame_time))#原始检测结果
        t5 = time.time()

    def _save_det_bbox(self,save_file_folder):
        if save_file_folder is None:
            return
        if not os.path.exists(save_file_folder):
            os.makedirs(save_file_folder)
        file_path = os.path.join(save_file_folder,'det_bbox_result_'+self.video_name+'.pkl')
        print('\nstart writing detection result:%s'%file_path)
        with open(file_path,'wb') as f:
            pickle.dump(self.det_bbox_result,f)

    def _rotated_bbox_to_bbox(self,rotated_bbox):
        object_num = rotated_bbox.shape[0]
        rotated_bbox_np = rotated_bbox.data.cpu().numpy()
        res = []
        for i in range(object_num):
            x = rotated_bbox_np[i]
            cen_x, cen_y = x[0], x[1]
            bbox_w, bbox_h = x[2], x[3]
            theta = x[4] * 180 / np.pi
            pts_4 = cv2.boxPoints(((cen_x, cen_y), (bbox_w, bbox_h), theta))  # 4 x 2
            x1_y1 = pts_4.min(0)
            x2_y2 = pts_4.max(0)
            x1,y1 = x1_y1[0],x1_y1[1]
            x2, y2 = x2_y2[0], x2_y2[1]
            score = x[5]
            h_bbox = [x1,y1, x2,y2, score]
            res.append(h_bbox)
        res = np.array(res,dtype=np.float32)
        return res

    def _init_stabilizer(self):
        self.video_stabilizer = VideoStabilization()
        self.video_stabilizer.load_transforms(self.stabilize_transformers_path)

    def _save_result_video(self):
        pass

    def _load_road_config(self,path):
        if not path:
            return
        return RoadConfig.fromfile(path)

    def _pixel_to_xy(self,nms_result):
        pixel2xy_matrix = self.road_config['pixel2xy_matrix']
        if pixel2xy_matrix is not None:
            pixel_data = nms_result[:,:8].copy()
            pixel_data = pixel_data.reshape(-1, 2)
            b = np.ones(pixel_data.shape[0])
            pixel_data = np.column_stack((pixel_data, b))
            xy_data = np.matmul(pixel2xy_matrix, pixel_data.T).T.reshape(-1,8)
            return np.hstack((nms_result,xy_data))
        else:
            return nms_result

    def _get_lane_id(self,nms_result, frame_index=None):
        lanes = self.road_config['lane']
        if len(lanes) == 0 or nms_result.shape[0] == 0:
            return nms_result, np.empty((0, 2)), []
        pixel_cts = nms_result[:,:8].reshape(-1,4,2).mean(axis=1)
        lane_name_ls = []
        for pixel_ct in pixel_cts:
            lane_name = -1
            for key, lane_polygon in lanes.items():
                if isPointinPolygon(pixel_ct,lane_polygon):
                    lane_name = key
                    break
            lane_name_ls.append(lane_name)
        if self.debug_lane_log and frame_index is not None:
            track_ids = nms_result[:,10] if nms_result.shape[1] > 10 else np.arange(nms_result.shape[0])
            for tid, center, lane_name in zip(track_ids, pixel_cts, lane_name_ls):
                print(f"[lane_debug] frame={frame_index} track={int(tid)} center=({center[0]:.1f},{center[1]:.1f}) lane={lane_name}")
        lane_array = np.array(lane_name_ls).reshape(-1,1)
        return np.hstack((nms_result,lane_array)), pixel_cts, lane_name_ls

    def _draw_lane_debug(self, image, centers, lane_ids):
        for center, lane in zip(centers, lane_ids):
            pt = (int(center[0]), int(center[1]))
            color = (0, 255, 255) if lane != -1 else (0, 0, 255)
            cv2.circle(image, pt, 6, color, 2, cv2.LINE_AA)
            cv2.putText(image, f"L{lane}", (pt[0]+4, pt[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # —— 按车长规则重写类别：避免重训，输出更细分类型
    def _assign_vehicle_cat_by_length(self, nms_result):
        """
        根据旋转框的物理长度（优先使用 pixel2xy 后的世界坐标）重新分配类别。
        返回修改后的 nms_result（原地副本被修改）。
        """
        if nms_result is None or nms_result.shape[0] == 0:
            return nms_result

        # 确定类别列索引，兼容 11/19/20 列格式
        cat_idx = 9 if nms_result.shape[1] > 9 else None
        if cat_idx is None:
            return nms_result

        # 提取像素四点
        pts_px = nms_result[:, :8].reshape(-1, 4, 2)

        # 优先使用世界坐标（存在于末尾 8 列）
        use_world = nms_result.shape[1] >= 19
        if use_world:
            pts_world = nms_result[:, -8:].reshape(-1, 4, 2)
            pts_for_len = pts_world
        else:
            pts_for_len = pts_px

        def quad_max_len(q):
            # 四点两两距离的最大值
            dif = q[:, None, :] - q[None, :, :]
            dist = np.sqrt((dif ** 2).sum(-1))
            return dist.max()

        lengths = np.array([quad_max_len(p) for p in pts_for_len], dtype=np.float32)
        # 若使用像素长度且有 length_per_pixel，则转为米
        if (not use_world) and self.length_per_pixel:
            lengths = lengths * float(self.length_per_pixel)

        new_cats = []
        for L in lengths:
            # 车长粗分 5 类：car < van < truck < bus < freight_car（长挂/重挂）
            if L < 4.8:
                new_cats.append(0)  # car 小轿车/小客车
            elif L < 6.8:
                new_cats.append(4)  # van 面包/轻客/小型厢式
            elif L < 9.5:
                new_cats.append(1)  # truck 中/重型货车（非挂车）
            elif L < 12.0:
                new_cats.append(2)  # bus 公交/大客车
            else:
                new_cats.append(3)  # freight_car 长挂/重挂/超长厢式

        nms_result[:, cat_idx] = np.array(new_cats, dtype=nms_result.dtype)
        return nms_result
    #

if __name__ == '__main__':

    config_json = '../config/mixed_roads_deepsort/A/20220617_A3_F2.json'
    v = DroneVideoProcess(config_json)
    # v.process_img()
    v.process_video()
