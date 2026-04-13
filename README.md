# PCBAIQualityInspect

基于 **语义分割 + 焊盘 ROI 分类** 的 PCB 焊点/缺陷质检桌面应用：从视频抽帧、数据准备，到加载 **BiSeNetV2** 与 **EfficientNet** 系列模型进行组合推理，并在 **PyQt5** 界面中可视化结果（标签：`OK` / `NG` / `Pre`）。

## 功能概览

| 模块 | 说明 |
|------|------|
| **数据标注 / 抽帧** | 浏览视频目录、按帧间隔导出 PNG，可选 SSIM 去重；支持批量增强入口（见 `GUI/utils.py`） |
| **模型训练** | `Classifier/` 下基于 PyTorch 训练分类器；支持 Notebook 交互实验（`train_efficient.ipynb` 等） |
| **在线质检** | 加载分割与分类权重，对视频逐帧推理，原图与叠加结果分栏显示 |

## 技术栈

- **界面**：PyQt5（含 `QtMultimedia` 视频播放）
- **视觉**：OpenCV、NumPy
- **深度学习**：PyTorch、torchvision
- **训练评估**：scikit-learn、matplotlib、tqdm 等

## 目录结构

```
.
├── app.py                 # 程序入口
├── GUI/                   # 界面、线程、组合推理 tools
├── Classifier/            # 分类数据集、EfficientNet 训练脚本与 Notebook
├── Segmentation/          # BiSeNet 等分割网络实现
├── Settings/
│   └── model_setting.json # 可选模型名称（供界面下拉等使用）
└── README.md
```

## 环境要求

- Python 3.8+（建议 3.10）
- Windows 下建议安装 **带 CUDA 的 PyTorch** 以使用 GPU 推理（质检线程中对分割推理有 CUDA 相关假设，详见 `GUI/tools.py` 注释）

### 依赖安装（示例）

项目未附带 `requirements.txt` 时，可按实际环境安装，例如：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install PyQt5 opencv-python numpy scikit-learn matplotlib seaborn tqdm
```

> 训练脚本中若报缺少其它包，按报错补装即可（如 `fonttools` 等）。

## 快速开始

### 启动桌面程序

在仓库根目录执行：

```bash
python app.py
```

### 质检流程（简要）

1. 在 **质检** 相关页签中选择视频目录与文件  
2. 加载 **分割**、**分类** 模型权重（`.pth`）  
3. 初始化模型后开始播放/推理，查看分割叠加与分类结果  

具体按钮与配置以界面为准；模型架构需与 `GUI/utils.py` 中 `create_model` 所支持名称一致（如 `BiSeNetV2`、`EfficientV2`）。

### 训练分类器

```bash
cd Classifier
python train.py --epoch 80
```

或使用 `train_efficient.ipynb` 进行交互训练与评估。数据与标注格式见 `Classifier/datasets.py`（LabelMe 风格 JSON + 焊盘裁剪逻辑）。

## 配置说明

- **`Settings/model_setting.json`**：声明可选的分类/分割模型名称，便于与界面选项对齐。  
- **分割推理均值方差**（可选）：若存在 `GUI/temp/runtime_cfg.py` 且导出 `cfg` 字典，可被 `GUI/tools.py` 读取用于归一化（详见该文件内说明）。

## 数据与权重

- 训练用图像与 JSON 默认放在 **`Classifier/data/`** 下（具体子目录以 `datasets.py` 为准）。  
- 本仓库的 `.gitignore` 已忽略 **`Classifier/data/`**、**`*.pth` / `*.pt`** 及训练输出目录，避免大文件进入 Git。本地使用请自行放置数据与权重。

## 许可证

未特别声明时，以仓库所有者约定为准；使用第三方模型代码请遵守各自开源协议。
