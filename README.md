```*


project_name/
├── config/                  # 配置管理：超参数、环境设置（如YAML文件）
│   ├── environment.yml      # Conda环境依赖
│   ├── requirements.txt     # Python包依赖
│   └── model_config.yml     # 模型和训练配置
├── data/                    # 数据管理：版本化所有数据阶段
│   ├── raw/                 # 原始不可变数据（如下载的MNIST数据集）
│   ├── interim/             # 中间处理数据（临时，用于原型，如归一化后）
│   ├── processed/           # 最终处理数据（训练/验证/测试集，如TFRecords或CSV）
│   └── external/            # 第三方数据源，外部基准数据
├── docs/                    # 文档：API参考、设计文档、教程
│   ├── api/                 # API文档
│   ├── design_docs/         # 模型架构设计
│   ├── tutorials/           # 使用指南
│   └── diagrams/            # 网络架构图、流程图
├── notebooks/               # 探索性分析：Jupyter notebooks
│   ├── eda/eda.ipynb                 # 数据探索和可视化
│   ├── model_experiments/   # 模型原型和实验
│   └── results/results_analysis.ipynb             # 结果可视化与比较
├── src/                     # 源代码：核心逻辑，模块化设计
│   ├── __init__.py          # 使src成为Python模块
│   ├── data_pipeline/       # 数据管道：采集、预处理、增强
│   │   ├── sourcing/        # 数据采集（下载、爬取、验证）
│   │   ├── preprocessing/   # 数据清洗、转换、编码、分裂
│   │   ├── augmentations/   # 数据增强（图像/文本/音频）
│   │   ├── loaders/         # 数据加载（Dataset类、生成器）
│   │   ├── validation/      # 数据验证和统计
│   │   └── utils/utils.py          # 辅助工具（可视化、日志）,如Informer模型需要的时间序列工具（e.g., sliding window）
│   ├── models/              # 模型定义：架构、层、损失、指标
│   │   ├── architectures/   # 不同网络架构
│   │   ├── layers/          # 自定义层
│   │   ├── losses/          # 自定义损失函数
│   │   └── metrics/         # 自定义评估指标
│   ├── training/            # 训练逻辑：实验、超参调优、回调
│   │   ├── experiments/     # 每个实验的子文件夹（config、logs、results）
│   │   ├── scripts/         # 训练/验证/恢复脚本
│   │   ├── hyperparameters/ # 搜索算法和结果
│   │   ├── callbacks/       # LR调度、早停、检查点
│   │   ├── strategies/      # 单/多GPU、TPU策略
│   │   ├── metrics/         # 训练指标
│   │   └── utils/           # 梯度裁剪、权重初始化, AMP混合精度
│   ├── inference/           # 推理与部署：模型转换、基准测试
│   │   ├── deployment/deployment.py      # Docker、云函数、API端点、边缘设备  ONNX导出、API导出
│   │   ├── tools/           # 模型转换、基准、可视化
│   │   └── utils/           # 预/后处理、日志
│   ├── testing/             # 测试：单元/集成测试
│   │   ├── unit_tests/      # 数据管道/模型/训练测试
│   │   ├── integration_tests/ # 端到端管道测试
│   │   └── utils/           # fixture、mock、可视化
│   └── utils/               # 通用工具：文件处理、可视化
├── models/                  # 训练模型：权重、检查点，按版本组织
├── results/                 # 输出：模型结果、图表、表格
│   ├── models/              # 序列化模型
│   ├── plots/               # 可视化图表
│   └── tables/              # CSV/Excel结果
├── reports/                 # 报告：分析输出、图形
│   └── figures/             # 生成的图像和图表，如注意力热度图
├── scripts/                 # 自动化脚本
│   ├── ETTh1.sh             # 运行ETTh1实验
│   ├── train.sh             # 通用训练
│   └── deploy.sh            # 部署到Kubernetes
├── tests/                   # 整体测试（可选，与src/testing重叠）
├── references/              # 参考材料：数据字典、手册
├── logs/                    # 日志文件：训练/推理日志
|-- ci_cd/		     # CI/CD: github actions
|   |__workflow.yml	     # 自动化测试/部署
├── LICENSE                  # 许可协议
├── Makefile                 # 自动化命令（如make train）
├── README.md                # 项目概述、安装指南
└── setup.cfg                # 项目配置（可选，使项目pip安装）


# 环境设置
- 创建conda环境：`conda env create -f environment.yml`
- 依赖：PyTorch, NumPy, Matplotlib（从requirements.txt）
- setup.py  pip install -e .

# 数据准备：
- 下载ETTh1到data/raw/。
- 示例预处理（src/data_pipeline/preprocessing.py）：

# 多卡训练：--distributed 自动用 DDP 启动多进程
# 混合精度：--amp 自动用 AMP
# 超参搜索：--optuna 自动用 Optuna 搜索学习率/批量大小
# 自动实验管理：每次实验自动新建目录，归档所有日志、配置、结果、可视化
# 断点续训：自动保存/加载 last_checkpoint.pth
# 训练/验证/测试三分，准确率/损失全记录
# 可视化：loss/acc/confusion matrix
# notebooks/analysis.ipynb 可用于更复杂的分析

```
