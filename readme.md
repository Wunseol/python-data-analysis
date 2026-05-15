# Python 数据分析案例学习项目

[![Python](https://img.shields.io/badge/Python-%E2%89%A53.9-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Cases](https://img.shields.io/badge/Cases-211%2B-orange.svg)](docs/模块详解.md)

按数据分析工作流组织的 Python 案例学习项目 — **5 大方向 · 26 个子方向 · 211+ 代码案例**，从数据获取到综合应用，系统掌握数据分析全流程。

## ✨ 项目亮点

- **工作流驱动** — 按真实数据分析流程组织：获取 → 处理 → 可视化 → 建模 → 应用
- **案例即学即用** — 每个案例独立可运行，无需额外数据文件
- **统一学习路线** — 每个阶段含📚概念理解层 + 💻代码实战层，开发者和非开发者同路线不同侧重
- **代码可跳转** — 文档中的代码链接支持 VSCode / GitHub 点击跳转
- **技术时效标注** — 标注所有 API 弃用与变更，避免踩坑

## 🚀 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/<your-username>/python-data-analysis.git
cd python-data-analysis

# 2. 安装依赖（Python ≥ 3.9）
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt

# 3. 运行案例
python "02_数据处理/01_NumPy/1_创建一个单位矩阵.py"
```

## 📁 项目结构

```
python-data-analysis/
├── 01_数据获取/          网页基础 · 网络爬虫 · 数据库查询 · API数据获取
├── 02_数据处理/          NumPy · Pandas · 数据清洗 · 特征工程
├── 03_数据可视化/        Matplotlib · Seaborn · 交互式可视化 · 地理空间分析
├── 04_分析建模/          EDA · 统计分析 · 监督学习 · 无监督学习 · 模型评估 · 时序分析 · 推荐系统 · 异常检测 · AB测试 · 深度学习
├── 05_综合应用/          NLP基础 · 数据报告与导出 · 商业分析方法 · 数据工程与ETL
└── docs/                 项目文档
```

## 📖 项目文档

| 文档 | 说明 |
|------|------|
| [项目概览](docs/项目概览.md) | 项目定位、学习路线、技术栈一览 |
| [架构设计](docs/架构设计.md) | 11 张 Mermaid 图（架构图、依赖图、数据流图、用例图、ER图…） |
| [模块详解](docs/模块详解.md) | 26 个子方向逐一详解（含代码跳转链接） |
| [学习路线](docs/学习路线.md) | 统一学习路线：每个阶段含概念理解层 + 代码实战层 |
| [开发指南](docs/开发指南.md) | 环境搭建、编码规范、运行方式、贡献指南 |
| [依赖管理](docs/依赖管理.md) | 23 个依赖包说明、弃用变更、安装指引 |

## ⚠️ 技术时效性

部分库存在重要 API 变更，详见 [依赖管理 → 弃用与变更](docs/依赖管理.md#2-重要弃用与变更)：

| 库 | 变更 |
|----|------|
| Seaborn ≥0.13 | `distplot` 已移除 → `displot`/`histplot` |
| Scikit-learn ≥1.3 | `load_boston` 已移除 → `fetch_california_housing` |
| Pandas ≥2.0 | `append` → `pd.concat`；`ix` → `loc`/`iloc` |
| SQLAlchemy ≥2.0 | 2.0 语法与 1.x 差异较大 |
| Prophet ≥1.1 | 包名从 `fbprophet` 更改为 `prophet` |
| PyTorch ≥2.0 | `torch.cuda.amp.autocast` → `torch.amp.autocast` |

## 📜 开源协议与免责声明

- 本项目基于 [MIT License](LICENSE) 开源
- 使用前请阅读 [免责声明](DISCLAIMER.md)
