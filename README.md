
深度学习YOLOV8模型如何训练智慧课堂行为检测数据集 识别低头 睡觉 听讲 阅读 侧视等行为智能课堂行为分析系统，教室摄像头、在线教学平台或教育管理平台，实现学生学习状态的自动化评估与教学反馈优化
---

### 📊 **学生课堂行为检测数据集信息表**

| 项目 | 内容 |
|------|------|
| **数据集名称** | 学生课堂行为检测数据集（智慧课堂） |
| **总图像数量** | 1,810 张 |
| **数据划分** | - 训练集（Train）：1,268 张<br>- 验证集（Val）：361 张<br>- 测试集（Test）：181 张 |
| **类别数量** | 6 类 |
| **标注格式** | YOLO 格式（`.txt`） |
| **应用场景** | 智慧教育、课堂专注度分析、教学行为评估、AI助教系统 |

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/684586c0f0d343cfa7ebeb547094e8fa.png)

### 📈 **各类别分布详情**

| 行为类别（英文） | 中文含义 | 图像数量 | 标注框数量 |
|------------------|----------|----------|------------|
| `focused`        | 专注听讲 | 641      | 8,416      |
| `reading`        | 阅读     | 512      | 3,553      |
| `hand movement`  | 手部动作 | 908      | 11,064     |
| `head down`      | 低头     | 769      | 8,052      |
| `looking aside`  | 侧视     | 503      | 6,065      |
| `sleeping`       | 睡觉     | 572      | 8,528      |

> 💡 注：因单张图像可包含多个行为标签（如一名学生同时“低头”+“手部动作”），各类图像数量之和 > 总图像数（1,810）。

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fd9468f7d9144261b550ff8d0e797208.png)

### 📉 **模型在验证集上的性能对比（mAP@0.5）**

| 模型版本 | mAP@0.5 |
|----------|---------|
| **YOLOv5m** | 0.86045 |
| **YOLOv8m** | 0.85897 |
| **YOLOv11m** | 0.85820 |

> ⚠️ 说明：截至 2025 年，**YOLOv11 并非 Ultralytics 官方发布模型**，可能为第三方命名或误标。建议以 **YOLOv5 / YOLOv8** 作为标准训练与部署基准。

---

该数据集适用于构建**智能课堂行为分析系统**，可集成至教室摄像头、在线教学平台或教育管理平台，实现学生学习状态的自动化评估与教学反馈优化。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0c15451145ec473796d828692b4eff08.png)
 **【学生课堂行为检测数据集】**（6类，1810张图像）的完整 **YOLOv8 训练代码**，包含数据配置、训练脚本、评估与部署支持，适用于智慧教育场景。

---
以下文字及代码仅供参考学习使用。
## 📁 一、推荐项目目录结构

```
Classroom_Behavior_Detection/
├── classroom.yaml                # 数据集配置文件
├── train_classroom.py            # YOLOv8 训练脚本
├── val_classroom.py              # 验证/测试评估脚本
├── export_model.py               # 模型导出（ONNX/TensorRT）
├── dataset/                      # 数据集根目录
│   ├── images/
│   │   ├── train/      # 1268 张
│   │   ├── val/        # 361 张
│   │   └── test/       # 181 张
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── requirements.txt              # Python 依赖
```

---

## 📦 二、依赖安装（`requirements.txt`）

```txt
ultralytics==8.2.0
torch>=1.10.0
torchvision>=0.11.0
opencv-python>=4.6.0
numpy>=1.21.0
Pillow>=9.0.0
```

安装命令：
```bash
pip install -r requirements.txt
```

---

## 📄 三、数据配置文件：`classroom.yaml`

```yaml
# classroom.yaml
path: ./dataset
train: images/train
val: images/val
test: images/test

nc: 6
names: [
  'focused',
  'reading',
  'hand movement',
  'head down',
  'looking aside',
  'sleeping'
]
```

> ✅ 注意：类别名称需与标注文件中的 `class_id` 严格对应（从 0 开始）。

---

## 🚀 四、YOLOv8 完整训练脚本：`train_classroom.py`

```python
# train_classroom.py
from ultralytics import YOLO
import torch

def main():
    # 自动选择设备（GPU/CPU）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 使用设备: {device}")

    # 加载预训练模型（推荐 yolov8m 平衡精度与速度；也可选 yolov8s/n）
    model = YOLO('yolov8m.pt')

    # 开始训练
    results = model.train(
        data='classroom.yaml',                  # 数据配置文件
        epochs=100,                             # 训练轮数（建议 80~150）
        imgsz=640,                              # 输入图像尺寸
        batch=32,                               # 批大小（根据显存调整：16/32/64）
        name='classroom_behavior_yolov8m',      # 实验名称
        device=device,
        patience=20,                            # 早停：20 轮验证指标无提升则停止
        save=True,                              # 保存 best.pt 和 last.pt
        plots=True,                             # 生成训练曲线（loss, PR, confusion matrix）
        exist_ok=False,                         # 若实验名已存在则报错
        workers=8,                              # 数据加载线程数
        optimizer='AdamW',                      # 优化器（也可用 'SGD'）
        lr0=0.01,                               # 初始学习率
        lrf=0.01,                               # 最终学习率 = lr0 * lrf
        momentum=0.937,                         # SGD 动量（若使用 SGD）
        weight_decay=0.0005                     # 权重衰减
    )

    print("\n✅ 训练完成！模型保存路径:")
    print("  最佳模型: runs/detect/classroom_behavior_yolov8m/weights/best.pt")
    print("  最后模型: runs/detect/classroom_behavior_yolov8m/weights/last.pt")

    # 在测试集上评估最终性能
    print("\n🔍 正在测试集上评估模型...")
    metrics = model.val(data='classroom.yaml', split='test')
    print(f"\n🎯 测试集结果:")
    print(f"  mAP@0.5:         {metrics.box.map50:.5f}")
    print(f"  mAP@0.5:0.95:    {metrics.box.map:.5f}")
    print(f"  Precision:       {metrics.box.mp:.5f}")
    print(f"  Recall:          {metrics.box.mr:.5f}")

if __name__ == '__main__':
    main()
```

### ▶️ 启动训练
```bash
python train_classroom.py
```

---

## 🔍 五、单独验证/测试脚本（可选）：`val_classroom.py`

```python
# val_classroom.py
from ultralytics import YOLO

# 加载最佳模型
model = YOLO('runs/detect/classroom_behavior_yolov8m/weights/best.pt')

# 在验证集评估
val_metrics = model.val(data='classroom.yaml', split='val')
print(f"✅ 验证集 mAP@0.5: {val_metrics.box.map50:.5f}")

# 在测试集评估
test_metrics = model.val(data='classroom.yaml', split='test')
print(f"✅ 测试集 mAP@0.5: {test_metrics.box.map50:.5f}")
```

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d0c7c8ba53ac40659a96477c0a5d06bf.png)

## 📤 六、模型导出（用于边缘设备部署）

```python
# export_model.py
from ultralytics import YOLO

model = YOLO('runs/detect/classroom_behavior_yolov8m/weights/best.pt')

# 导出为 ONNX（通用格式，支持 OpenVINO / RKNN / Axera）
model.export(format='onnx', imgsz=640, opset=12, simplify=True)

# 导出为 TensorRT（Jetson 推荐）
model.export(format='engine', imgsz=640, half=True)  # FP16加速

# 导出为 TorchScript（备用）
model.export(format='torchscript', imgsz=640)
```

> 💡 **部署建议**：
> - 教室摄像头终端：使用 ONNX + OpenCV DNN  
> - 边缘盒子（如华为 Atlas）：编译 ONNX 至 NPU 模型  
> - Web 端演示：使用 `model.predict()` + Flask

---

## 🖼️ 七、推理示例（单图检测）

```python
from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/classroom_behavior_yolov8m/weights/best.pt')
results = model('test_image.jpg')  # 支持图片/视频/摄像头

# 可视化结果
annotated = results[0].plot()
cv2.imshow('Classroom Behavior Detection', annotated)
cv2.waitKey(0)
```

---

## ✅ 八、训练技巧提示

| 问题 | 建议 |
|------|------|
| **小目标漏检**（如手部动作） | 尝试 `imgsz=1280` 或启用 `mosaic=1.0` |
| **类别不平衡** | YOLOv8 自动处理，但可检查 `labels` 分布 |
| **过拟合** | 启用 `augment=True`（默认开启），或增加 `dropout`（需自定义模型） |

---


