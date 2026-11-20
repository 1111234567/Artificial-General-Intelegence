# 11 月 20 日研究的公开复算包

报告、图、源码、协议和汇总 JSON 都可直接在 GitHub 阅读。需要加载个体、从真实动作重新计分或核对训练账本时，再下载对应的 Git LFS 压缩包。以下大小是 ZIP 大小，按 MiB 四舍五入；“原件”按包内清单计数，不含清单自身。各包内路径均以仓库根目录为起点。[逐项实验索引](../experiments.md)

| 研究 | 审计包 | 原件 | 运行依赖与覆盖 |
|---|---|---:|---|
| [交错生活与旧能力](../interleaved-learning-2026-11-20.md) | [125 MiB](interleaved-learning-2026-11-20/audit-data.zip) | 1,649 | 含正式审计的行为、优化器、当时运行的源码副本及三角形示例。 |
| [主动复习与写入](../active-review-2026-11-20.md) | [132 MiB](active-review-20261120/audit-data.zip) | 721 | 含独立世界、实际选路账本和冻结头部。下列几项研究从这里取原先的亲本。 |
| [该写哪些连接](../edge-site-2026-11-20.md) | [256 MiB](edge-site-20261120/audit-data.zip) | 153 | 另取主动复习包。含单边/组合干预、共享规则的四个检查点阶段及等剂量、配平回放。 |
| [一个草稿与显式重演](draft-rehearsal-20261120/README.md) | [239 MiB](draft-rehearsal-20261120/audit-data.zip) | 94 | 另取主动复习包。含单路拟合账本、三种重演、接受门、草稿邻域和正式留出。 |
| [无输入时的整图活动](../native-rehearsal-2026-11-20.md) | [232 MiB](native-rehearsal-20261120/audit-data.zip) | 121 | 另取主动复习包。含五组开发终点、内部回流与前瞻测试、生命内读取本体现行参数的对照。 |
| [本体向克隆区发问](../fork-lesson-2026-11-20.md) | [204 MiB](fork-lesson-20261120/audit-data.zip) | 198 | 另取主动复习包。含发送轴突、首次问答、严格 40 节点和侧别替换检验。原封存加载器的协议路径按 Windows 格式核对。 |
| [房内改道](../reroute-2026-11-20.md) | [137 MiB](reroute-20261120/audit-data.zip) | 138 | 含 30 个训练终点、六个所需上游亲本、动作留出与状态移植。 |
| [双复查](../double-visit-2026-11-20.md) | [145 MiB](double-visit-20261120/audit-data.zip) | 85 | 先解压改道包以取得六个继承终点。含本轮 12 个模型与训练账本、状态探针、配对行动和读出干预。 |

## 例：从冻结账本复算双复查

在一份新检出的仓库根目录，用 Git LFS 取下列两包。使用已安装项目依赖的 Python 环境执行最后一步；原实验运行版本见改道、双复查包内的 `audit-manifest.json`。复算命令会在这份检出中写入汇总，已有本机实验可另取一份工作副本。

```powershell
git lfs pull --include="docs/evidence/reroute-20261120/audit-data.zip,docs/evidence/double-visit-20261120/audit-data.zip"
python -m zipfile -e docs/evidence/reroute-20261120/audit-data.zip .
python -m zipfile -e docs/evidence/double-visit-20261120/audit-data.zip .
python -m experiments.pulse_game.double_visit_eval
```

每份 ZIP 解开后，会在自己的证据目录中留下原件清单和 SHA-256。LFS 指针记录整个包的 SHA-256，可用 `git show HEAD:docs/evidence/double-visit-20261120/audit-data.zip` 与本机 `Get-FileHash docs/evidence/double-visit-20261120/audit-data.zip -Algorithm SHA256` 对照。公开包覆盖各项正式审计和指定的诊断；历次试跑、进程日志、运行环境及更早的完整本机档案按研究各自保存。[双复查协议和计数](double-visit-20261120/eval-plan.md) · [本机数据索引](../../data/README.md)
