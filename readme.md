<div align="center">

# Python 数据分析案例学习

**按数据分析工作流组织的案例驱动学习项目**

[![Python](https://img.shields.io/badge/Python-%3E%3D3.9-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Stages](https://img.shields.io/badge/学习阶段-9-blue.svg)
![Topics](https://img.shields.io/badge/子方向-26-9cf.svg)
![Cases](https://img.shields.io/badge/代码案例-211+-orange.svg)

9 个学习阶段 · 26 个子方向 · 211+ 代码案例 · 每个案例独立可运行

[快速开始](#快速开始) · [学习路线](docs/01_学习路线.md) · [模块详解](docs/02_模块详解.md) · [贡献指南](#贡献)

</div>

---

## 特性

| | |
|:--|:--|
| 路线驱动 | 9 阶段循序渐进：理解概念 → 运行代码 → 进阶练习 |
| 即学即用 | 每个案例独立可运行，无需额外数据文件 |
| 代码可跳转 | 文档中的代码链接支持 VSCode / GitHub 点击跳转 |
| 时效标注 | 标注所有 API 弃用与变更，避免踩坑 |

---

## 学习路线

```
阶段一          阶段二          阶段三            阶段四          阶段五
数据基础   →   数据处理   →   数据可视化   →   数据获取   →   分析建模基础
NumPy          数据清洗        Matplotlib        网页基础        EDA方法
Pandas         特征工程        Seaborn           网络爬虫        统计分析
                               交互式可视化      数据库查询
                               地理空间分析      API数据获取

阶段六          阶段七          阶段八          阶段九
机器学习   →   专项建模   →   深度学习   →   综合应用
监督学习        推荐系统        深度学习基础      NLP基础
无监督学习      异常检测                         商业分析方法
模型评估        AB测试                           数据工程与ETL
                时序分析                         数据报告与导出
```

> 阶段四（数据获取）可后置，先用现成数据练习后续阶段。详见 [学习路线](docs/01_学习路线.md)。

---

## 技术栈

<table>
<tr><td width="140"><strong>数据基础</strong></td><td><code>numpy</code> <code>pandas</code></td></tr>
<tr><td><strong>数据处理</strong></td><td><code>scipy</code> <code>scikit-learn</code></td></tr>
<tr><td><strong>数据可视化</strong></td><td><code>matplotlib</code> <code>seaborn</code> <code>plotly</code> <code>geopandas</code> <code>folium</code></td></tr>
<tr><td><strong>数据获取</strong></td><td><code>requests</code> <code>beautifulsoup4</code> <code>sqlalchemy</code> <code>httpx</code></td></tr>
<tr><td><strong>分析建模</strong></td><td><code>scikit-learn</code> <code>statsmodels</code> <code>scipy</code> <code>ydata-profiling</code> <code>prophet</code></td></tr>
<tr><td><strong>深度学习</strong></td><td><code>torch</code></td></tr>
<tr><td><strong>综合应用</strong></td><td><code>jieba</code> <code>wordcloud</code> <code>snownlp</code> <code>fpdf2</code> <code>openpyxl</code> <code>nbformat</code></td></tr>
</table>

---

## 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/<your-username>/python-data-analysis.git
cd python-data-analysis

# 2. 创建虚拟环境（Python ≥ 3.9）
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行第一个案例
python "02_数据处理/01_NumPy/1_创建一个单位矩阵.py"
```

安装完成后，进入 [学习路线](docs/01_学习路线.md) 开始系统学习。

如只需某一方向的依赖，可按需安装：

```bash
pip install numpy pandas scipy scikit-learn   # 数据处理
pip install matplotlib seaborn plotly          # 数据可视化
pip install requests beautifulsoup4 sqlalchemy # 数据获取
```

完整按方向安装指引见 [依赖管理](docs/06_依赖管理.md)。

---

## 项目结构

```
python-data-analysis/
├── 02_数据处理/              阶段一·数据基础 + 阶段二·数据处理
│   ├── 01_NumPy/            10 个案例
│   ├── 02_Pandas/           13 个案例
│   ├── 03_数据清洗/          6 个案例
│   └── 04_特征工程/          8 个案例
├── 03_数据可视化/            阶段三·数据可视化
│   ├── 01_Matplotlib/       14 个案例
│   ├── 02_Seaborn/          10 个案例
│   ├── 03_交互式可视化/      9 个案例
│   └── 04_地理空间分析/      7 个案例
├── 01_数据获取/              阶段四·数据获取（可后置）
│   ├── 01_网页基础/
│   ├── 02_网络爬虫/          10 个案例
│   ├── 03_数据库查询/        9 个案例
│   └── 04_API数据获取/       7 个案例
├── 04_分析建模/              阶段五~八·分析建模
│   ├── 01_EDA方法/           10 个案例
│   ├── 02_统计分析/
│   ├── 03_监督学习/          6 个案例
│   ├── 04_无监督学习/        6 个案例
│   ├── 05_模型评估与优化/    6 个案例
│   ├── 06_时序分析与预测/    9 个案例
│   ├── 07_推荐系统/          7 个案例
│   ├── 08_异常检测/          7 个案例
│   ├── 09_AB测试与实验设计/  7 个案例
│   └── 10_深度学习基础/      7 个案例
├── 05_综合应用/              阶段九·综合应用
│   ├── 01_NLP基础/           9 个案例
│   ├── 02_数据报告与导出/    8 个案例
│   ├── 03_商业分析方法/      8 个案例
│   └── 04_数据工程与ETL/     7 个案例
└── docs/                     项目文档
    ├── 01_学习路线.md
    ├── 02_模块详解.md
    ├── 03_项目概览.md
    ├── 04_架构设计.md
    ├── 05_开发指南.md
    └── 06_依赖管理.md
```

---

## 文档

**学习文档**

| 文档 | 说明 |
|:-----|:-----|
| [学习路线](docs/01_学习路线.md) | 9 阶段学习路线：理解概念 → 运行代码 → 进阶练习 |
| [模块详解](docs/02_模块详解.md) | 26 个子方向逐一详解（含代码跳转链接） |

**项目文档**

| 文档 | 说明 |
|:-----|:-----|
| [项目概览](docs/03_项目概览.md) | 项目定位、技术栈一览 |
| [架构设计](docs/04_架构设计.md) | 11 张 Mermaid 图（架构图、依赖图、数据流图、用例图、ER图…） |
| [开发指南](docs/05_开发指南.md) | 环境搭建、编码规范、贡献指南 |
| [依赖管理](docs/06_依赖管理.md) | 23 个依赖包说明、弃用变更、安装指引 |

---

## API 弃用与变更

部分库存在重要 API 变更，详见 [依赖管理 → 弃用与变更](docs/06_依赖管理.md#2-重要弃用与变更)：

| 库 | 变更 |
|:---|:-----|
| Seaborn ≥0.13 | `distplot` 已移除 → `displot` / `histplot` |
| Scikit-learn ≥1.3 | `load_boston` 已移除 → `fetch_california_housing` |
| Pandas ≥2.0 | `append` → `pd.concat`；`ix` → `loc` / `iloc` |
| SQLAlchemy ≥2.0 | 2.0 语法与 1.x 差异较大 |
| Prophet ≥1.1 | 包名从 `fbprophet` 更改为 `prophet` |
| PyTorch ≥2.0 | `torch.cuda.amp.autocast` → `torch.amp.autocast` |

---

## 贡献

欢迎贡献代码！请阅读 [开发指南](docs/05_开发指南.md) 了解编码规范和 PR 流程。

1. Fork 本仓库
2. 从 `main` 创建功能分支：`feature/xxx` 或 `fix/xxx`
3. 提交代码，commit 信息清晰描述变更内容
4. 向 `main` 分支提交 Pull Request

---

## 致谢

本项目的构建参考了以下优秀的开源项目和资源，感谢它们的作者和社区：

| 项目 | 说明 |
|:-----|:-----|
| [Python-100-Days](https://github.com/jackfrued/Python-100-Days) | 骆昊的 Python 百天学习路线，项目结构设计的参考 |
| [PythonDataScienceHandbook](https://github.com/jakevdp/PythonDataScienceHandbook) | Jake VanderPlas 的 Python 数据科学手册，数据分析知识体系的参考 |
| [100-Days-Of-ML-Code](https://github.com/MLEveryday/100-Days-Of-ML-Code) | 机器学习 100 天中文版，案例驱动学习方式的参考 |
| [awesome-python](https://awesome-python.com/) | Python 生态资源精选列表，技术栈覆盖面的参考 |
| [scikit-learn](https://scikit-learn.org/) / [pandas](https://pandas.pydata.org/) / [matplotlib](https://matplotlib.org/) | 官方文档和示例，案例代码的重要参考来源 |

同时感谢所有为本项目提供建议和反馈的贡献者。

---

## 开源协议与免责声明

- 本项目基于 [MIT License](LICENSE) 开源
- 使用前请阅读 [免责声明](DISCLAIMER.md)
