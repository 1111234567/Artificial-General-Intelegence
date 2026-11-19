# 框架入口与维护工具

本目录的共享框架工具依赖 Python 与 NumPy，文档/历史文本检查只用标准库。当前框架配置与实现位于 [src](../src/README.md)，独立研究线和历史实现的入口见 [experiments](../experiments/README.md)，报告、证据与被忽略的本机数据对应见 [证据索引](../docs/evidence/README.md)。`run.py list` 只列共享框架配置，不是全部研究脚本的清单。

| 工具 | 用途 |
|---|---|
| [run.py](run.py) | 框架配置查看、前置检查、运行与结果汇总 |
| [align_layers.py](align_layers.py) | 当前框架与 `experiments/legacy/` 中旧实现的逐层比较 |
| [check_docs.py](check_docs.py) | 现行 Markdown 的本地链接、必要路径和 UTF-8 编码检查；只用标准库 |
| [package_seed_evolution_cloud.py](package_seed_evolution_cloud.py) | 检查或打包 11 月初始化进化的 Debian CPU 续跑存档，启动脚本和解包说明在 [cloud_cpu_bundle](cloud_cpu_bundle/README.md) |
| [historical](historical/README.md) | 归档格式检查和文本差异报告 |

11 月 19 日发布的 NumPy 游戏运行包可直接下载与解压。[参考个体和使用说明](../data/pulse-game/morning-20261119/README.md)

## 轻量入口检查

```powershell
python scripts/run.py list
python scripts/run.py show base
python scripts/run.py check base
python scripts/check_docs.py
```

`test/run.py` 保留相同命令的兼容入口。脚本支持从其它工作目录以绝对路径启动。

这些命令不会训练。`list` / `show` 查看声明，`check` 只检查已实现的前置条件，文档检查只核对链接和编码。运行、扫描、基准测量与逐层行为比较另按下表选择，不能把它们当作同成本的资料检查。

## 框架命令

| 命令 | 行为与边界 |
|---|---|
| `list` / `show <name>` | 列出声明与配置；列入声明不代表配置得到验证 |
| `check [name]` | 检查配置可行性；省略名称会包含历史上有意不通过的配置。当前失败写入输出，退出码仍为 0，不能据此作为 CI 判据 |
| `run <name> [nseed] [nproc]` | 训练并追加到仓库 `data/results.jsonl`；文件环境还会写 `data/env/` |
| `results` / `gaps` | 查看本机结果与指纹差集；指纹未覆盖 seed、episode 预算等全部运行条件 |
| `align` | 按历史指纹汇总成功率跨度，提示人工核验；跨种子和预算的跨度不能证明逐步对齐或逻辑错误 |
| `scan [homeo\|calib] [nseed] [nproc]` / `bench` | 参数扫描与成本测量；可能耗时，不能视为入口冒烟 |

逐层比较可从 `python scripts/align_layers.py construct 1` 开始。通过只说明所选层、字段、配置与种子上的一致性，不能证明机制正确或持续学习能力；见 [复核记录](../docs/review-2026-10-26.md)。

`run` 会追加结果，文件任务还会创建/重置 `data/env/<seed>/`；已有文件环境不宜被同种子的并行进程共用。独立研究线通常保存完整状态与冻结来源绑定，不能用共享框架的 `results` / `gaps` / `align` 汇总代替各线报告。整理已有材料时先读结果文件，重跑研究应按该线说明使用新目录，保留正式协议、检查点和已绑定的分析产物。

## 文档检查

`python scripts/check_docs.py` 在本机核对 UTF-8、Markdown 链接和标题锚点。默认范围包含根文档、现行设计与分主题报告、证据和归档索引、两份本机数据索引，以及框架、工具和实验的 README、实验计划和游戏脚本导航。历史原件保留当时的路径和表述。

单查一批文件时，可以从仓库根目录运行下面的命令。

```powershell
python scripts/check_docs.py experiments/README.md scripts/README.md docs/evidence/README.md
```

本地链接通过，只说明目标在当前工作树存在。换机器阅读或复现时，要另外确认哪些材料随仓库携带、哪些位于本机的数据目录。[证据索引](../docs/evidence/README.md)
