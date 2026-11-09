# TabPFN for Electrolytes

[English](README.md) · [简体中文](README.zh-CN.md)

论文《利用表格基础模型：从小样本电导率预测到电解液配方的贝叶斯优化》的分析代码和部分派生结果。

## 下载后先做什么

从仓库根目录运行以下命令。核对只使用随仓库发布的 CSV。

```sh
python -m pip install pandas
python study/check_saved_results.py
```

脚本从逐条绝对百分比误差重新计算 AMPERE 固定划分的 MAPE 和推理种子标准差，从分温度汇总计算搜索成本及相对基线的降幅，并读取 Clio 新起点的冻结策略成本。结果打印到终端，原 CSV 保持原样。`requirements.txt` 列出分析和测试的依赖，[MANIFEST.sha256](MANIFEST.sha256) 列出发布文件及校验值。

## 下载原始数据后放在哪里

脚本根据自身所在位置定位输入。请按[数据来源索引](study/datasets/README.md)中的精确文件名和相对路径，将原始数据放到 `study/datasets/<数据集名>/`。例如 Flores 的 CSV 路径是 `study/datasets/flores/HTE_dataset_PC_EMC_EC_LiPF6.csv`。建议指定单独的输出目录，避免将新产出和随仓库提供的结果混在一起：

```sh
python -m pip install -r requirements.txt
python study/prediction_benchmarks/run_benchmark.py flores audit --output-dir recomputed/flores
```

这条命令需要 Flores 原始 CSV，并将划分和审计产出写入 `recomputed/flores/`。上面的保存结果核对可以在下载仓库后直接运行。需要历史对照文件或托管推理的完整实验模式不在此次公开版的已验证运行范围内。

## 脚本与论文对应关系

“随仓库提供”指已保存的结果。“运行产出”指研究脚本在数据和缓存就绪时生成的文件。

| 论文位置 | 脚本 | 功能和输出 |
| --- | --- | --- |
| §2.1.1 / §3.1.1 | [run_benchmark.py](study/prediction_benchmarks/run_benchmark.py) | 按数据集选择基准，参数用 `python study/prediction_benchmarks/run_benchmark.py flores --help` 查看。运行目录默认为 `study/prediction_benchmarks/<数据集>/results/`。本次提供的指标汇总统一放在 [study/results/benchmarks/](study/results/benchmarks/)。 |
| §2.1.1 | [benchmark_utils.py](study/prediction_benchmarks/benchmark_utils.py) | 离子液体、Electrolytomics 和水系共用的划分核对、缓存处理和指标汇总逻辑。运行时输出划分、预测、指标 CSV 和审计记录。 |
| §2.1.2 / §3.1.2 | [run_ampere_conductivity.py](study/run_ampere_conductivity.py)、[ampere_conductivity.py](study/analysis/ampere_conductivity.py) | 固定 40/60 划分和嵌套标签预算对照的入口与实现。[结果目录](study/results/ampere_conductivity/)包含划分索引、逐样本汇总、固定划分指标及学习曲线统计。 |
| §2.1.3 / §3.1.3 | [run_domain_adaptation.py](study/run_domain_adaptation.py)、[domain_adaptation.py](study/analysis/domain_adaptation.py) | 构造目标域划分，对照随机与多样性选样。[结果目录](study/results/domain_adaptation/)包含划分、预测、配对差异和按标签预算汇总的指标。 |
| §2.2.1 / §3.2.1 | [run_conductivity_guided_search.py](study/run_conductivity_guided_search.py)、[conductivity_guided_search.py](study/analysis/conductivity_guided_search.py) | 构建电导率与盐浓度坐标，进行实测池 GP 批量回放。[结果目录](study/results/conductivity_guided_search/)包含参考浓度预测、坐标、采集规则排名、命中率和 UCB 推荐批次。 |
| §2.2.2 / §3.2.2 | [run_representation_ablation.py](study/run_representation_ablation.py)、[representation_ablation.py](study/analysis/representation_ablation.py) | 比较组成、单坐标、双坐标及随机路线，并分析推理种子敏感性。[结果目录](study/results/representation_ablation/)包含成本、AUC、命中率、配对差异和种子汇总。 |
| §2.1–2.2 | [data_sources.py](study/analysis/data_sources.py)、[`__init__.py`](study/analysis/__init__.py) | 定位数据和历史来源记录、初始化分析包，不生成结果表。 |
| 数值核对 | [check_saved_results.py](study/check_saved_results.py) | 从随仓库提供的 CSV 核对关键数字，输出到终端。 |

### 独立电导率数据集

| 脚本 | 数据准备与评价 | 随仓库提供 |
| --- | --- | --- |
| [calisol23.py](study/prediction_benchmarks/calisol23.py) | 清理、聚合、按阈值筛选，分别评价 mol/L 和 mol/kg 并诊断误差尾部。 | [CALiSol-23 指标](study/results/benchmarks/calisol23.csv) |
| [flores.py](study/prediction_benchmarks/flores.py) | 聚合重复实验条件，构造预测划分并建立符号公式基线。 | [Flores 指标](study/results/benchmarks/flores.csv) |
| [clio.py](study/prediction_benchmarks/clio.py) | 按配方聚合实测重复数据，审计 AEM 表，建立静态预测基准和同划分 GP。 | [Clio 预测指标](study/results/benchmarks/clio.csv) |
| [ionic_liquids.py](study/prediction_benchmarks/ionic_liquids.py) | 将 nh 和 ILThermo 展开为物种–温度条件，分别进行对数电导率评价。 | [离子液体指标](study/results/benchmarks/ionic_liquids.csv) |
| [chemarr.py](study/prediction_benchmarks/chemarr.py) | 计算聚合物与盐描述符，按规范化结构分组，评价各折的对数误差和秩相关。 | [ChemArr 指标](study/results/benchmarks/chemarr.csv) |
| [electrolytomics.py](study/prediction_benchmarks/electrolytomics.py) | 匹配实验条件和 QSAR 特征，构造对数电导率划分。 | [Electrolytomics 指标](study/results/benchmarks/electrolytomics.csv) |
| [aqueous.py](study/prediction_benchmarks/aqueous.py) | 从原液体积、密度和质量摩尔浓度重构配方特征，按配方分组。 | [水系指标](study/results/benchmarks/aqueous.csv) |

### 直接目标代理案例

| 脚本 | 功能及输出 |
| --- | --- |
| [prepare_polymer_search.py](study/objective_search/ionic_polymer/prepare_polymer_search.py) | 核对 Song 等发布的实验表和 GP 参考表、构建候选网格、以已观测标签定义综合目标。审计结果写到案例的 `results/`。 |
| [evaluate_polymer_rankings.py](study/objective_search/ionic_polymer/evaluate_polymer_rankings.py) | 从原始实测表和保存的预测分布评价受限池与完整网格排序，并计算分数归一化敏感性。 |
| [compare_polymer_acquisitions.py](study/objective_search/ionic_polymer/compare_polymer_acquisitions.py) | 比较采集规则、融合排名，计算覆盖率和已知分数下界。[随仓库提供](study/objective_search/ionic_polymer/results/)排名轨迹、目标排名和汇总表。 |
| [replay_clio_search.py](study/objective_search/clio/replay_clio_search.py) | 定义实测池逐轮反馈、GP 采集、条件随机基线和轨迹审计。运行时输出起点、轨迹及指标。 |
| [compare_clio_acquisition_rules.py](study/objective_search/clio/compare_clio_acquisition_rules.py) | 用开发起点比较采集规则，在新起点评价冻结策略。[随仓库提供](study/objective_search/clio/acquisition_results/)发现成本、预算指标、配对差异和初始预测诊断。 |
| [summarize_clio_validation_progress.py](study/objective_search/clio/summarize_clio_validation_progress.py) | 从保存的验证轨迹汇总各预算下的最好电导率及命中率。对应 CSV 位于 `acquisition_results/figure_source_data/`。 |

结果目录下的 `tables/` 存放分析表，`figure_source_data/` 存放部分补充图使用的数值。四个测试脚本的链接、检查内容和运行命令见[英文版测试说明](README.md#verification-and-inference-scope)。

## 数据来源与推理范围

原始输入由来源论文的作者随论文公开发布，下载入口和脚本所需的相对路径见[上游数据索引](study/datasets/README.md)。发布版保留本研究的派生结果和划分索引。完整实验脚本还依赖相应原始输入，部分步骤也需要历史审计文件或预测缓存。下载本仓库后可直接运行保存结果核对和合成测试。

论文中的 AMPERE 电导率基准使用托管检查点 `tabpfn-v3-regressor-v3_default.ckpt`，客户端为 `tabpfn-client` 0.5.3，使用八个估计器和六个推理种子。发布代码已移除托管推理和账户访问实现，这些名称仅用于记录原实验配置。
