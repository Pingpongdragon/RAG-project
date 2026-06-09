# archive/ — 历史备份与过时材料

这里只放**不再属于当前 tex 主线**的内容：旧代码备份、tex 草稿、运行日志、过时文档、早期快照。
保留以备追溯，正常工作不需要看这里。实验数据（`motivation_*/data/`）与 embedding 缓存
（`motivation_*/cache/`）按约定**未移入此处、未删除**，仍在各自原目录。

| 子目录 | 内容 |
|---|---|
| `tex_backups/` | `motivation.tex` 的历史草稿（preLeanBaselines / preNaturalDrift / preV2） |
| `code_backups/` | `strategies.py` / `plot_main.py` 的旧版本备份 |
| `doc_backups/` | 过时或被取代的文档：DATASET_AUDIT、DOC_2_LOGIC_QA、早期"实验想法"、旧 STORYLINE |
| `logs/` | `motivation.log` 编译日志 + `runlogs/`（各实验 stdout/stderr，含 mo1/mo2/v3 各跑次） |
| `assets/` | 散落的截图素材（`mo_screenshots/`）与 `fig1_view.png`，未被 tex/plotting 引用 |
| `snapshots_50w/` | `snapshot_20260422/` — 50-window 时期的旧实验快照（data + figures） |

> 这些内容大多被根 `.gitignore` 的 `*.bak` / `*.log` 规则忽略，不在版本控制中。
> 若确定不再需要，可整体删除以释放空间。
