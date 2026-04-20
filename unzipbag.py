import cv2
import numpy as np
import os
from pathlib import Path
from rosbags.highlevel import AnyReader # 【关键修改】使用高级提取器

# ==========================================
# 配置区
# ==========================================
bag_path = 'd:/Projects/Underwater/Datasets/stereo-vi-underwater/bus_outside.bag'
output_folder = 'd:/Projects/Underwater/extracted_images/cam1' 

# 对应的压缩图像话题
image_topic = '/slave1/image_raw/compressed'  
# ==========================================

def main():
    os.makedirs(output_folder, exist_ok=True)
    print(f"正在打开 bag 文件: {bag_path}\n")
    
    try:
        # AnyReader 接受的是一个包含 Path 对象的列表
        with AnyReader([Path(bag_path)]) as reader:
            print(f"✅ 成功打开，开始提取 '{image_topic}' 的压缩图像...")
            frame_count = 0
            
            # 遍历消息
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == image_topic:
                    
                    # 1. 直接使用 AnyReader 的解码方法
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    
                    # 2. 从 ROS 的 CompressedImage 中读取字节流
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    
                    # 3. 让 OpenCV 自动解码 JPEG/PNG 字节流
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        print(f"⚠️ 警告：第 {frame_count} 帧解码失败，已跳过。")
                        continue

                    # 4. 保存为本地图片
                    filename = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
                    cv2.imwrite(filename, img)
                    
                    if frame_count % 50 == 0 and frame_count > 0:
                        print(f"已提取 {frame_count} 帧...")
                        
                    frame_count += 1

            print(f"\n🎉 提取完成！共保存了 {frame_count} 张图片到 {output_folder}")
            
    except Exception as e:
        print(f"❌ 读取出错: {e}")

if __name__ == '__main__':
    main()
    