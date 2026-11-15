# 实际执行状态

## 用户调整后的交付范围

仅MetaDiffuser主方法代码与运行验证；不要求本机正式训练和其他baseline。原Point预训练、Prompt-DT长训练已通过进程句柄停止，数据/checkpoint保留；Cheetah-Dir专家批处理已自行完成。以下旧进度条目为历史记录，不表示仍在训练。

主方法23项测试重新通过（`runs/verification/metadiffuser-delivery.xml`）。服务器指南为`SERVER_RUN.md`；新增`fetch_references.py`恢复Git忽略的固定依赖，`smoke_metadiffuser.py`执行独立端到端验证。Git默认不携带datasets/runs，传输或重新采集方式已说明。

独立GPU端到端验证已完成，退出码0，结果在`runs/delivery_smoke`；本次交付验证汇总见`VALIDATION.md`。按最新范围交付主方法代码，不把历史实验未完成清单当作仍需在本机执行的任务。

本记录只列实际执行，不把脚本存在当作实验完成。

已完成：

- 论文正式PDF存档及公式/附录原页检查，4个参考仓库固定commit。
- Windows项目venv：Python3.11.5，PyTorch2.6.0+cu126；RTX4060 Laptop GPU可用。
- CPU隔离备用运行时`.runtime/cpu`，PyTorch2.6.0+cpu。
- 21项核心测试通过；CUDA发行版安装后再次运行21项通过；新增原始CORRO采集入口及相关数据测试6项通过（含已有5项）。记录在`runs/verification/*.xml`。
- GPU默认Point网络10次训练更新通过，参数有限，峰值张量显存约32.7MB。该短测含首次CUDA开销，不能直接外推完整训练耗时。
- `datasets/smoke`：3训练任务/2测试任务、每任务8episode，明确controller_smoke_only。
- 100步上下文预训练：验证奖励MSE约.000429，动力学MSE约.01723；100步扩散噪声MSE约.5650。
- 两种符号各2测试任务的20步闭环：paper_plus平均回报-14.668，consistency_minus -16.608。这只是流程验证；均远未收敛，动作裁剪比例为1，模型引导误差很高，不能作为论文性能或符号优劣证据。
- 独立SAC首任务10k步试跑结束，回报约-2.181；checkpoint在`runs/sac_pilot`，不混入CORRO专家来源。
- CORRO原始SAC的40个Point任务均完成1000iterations（每任务40,200环境步、10,000次更新），保留中途策略及best checkpoint。目录`runs/point_experts_corro_long`；原始100iterations试跑不足以达到专家质量，明确延长采集预算。
- 正式Point数据`datasets/point_sac`完成：2700训练/300验证expert episodes；10个测试任务每种expert/medium/random各1000条，总计6000episode/120000transition。manifest哈希`c029b479b996ae27b60dbd31bd6984cc6e45b7b9f995a22b5daa6445ab70b93e`。
- 全部正式数据通过文件哈希、连续性、任务/切分验证；每组完整轨迹重复率0；10个测试任务均满足平均expert>medium>random。等权测试任务平均回报分别-3.27244、-9.45441、-17.06534。此为采集策略回报，**不是MetaDiffuser回报**。详见`runs/verification/point_sac_data_audit.json`。
- CVAE-Planner完成独立实现及短训练/闭环测试；其未披露架构明确列为实施选择。
- Prompt-DT直接导入固定原始模型，完成隔离数据适配、训练、精确续训与相同warm-start索引的闭环入口；新增2项测试通过，见`runs/verification/pytest-prompt.xml`。它们验证适配，不是正式基线成绩。
- 正式Point专家数据已导出基线交换格式，`datasets/point_baseline_exchange`包含训练/验证/测试expert共4000条episode；导出manifest与正式数据SHA256一致。medium/random交换格式尚未另行导出，现有Prompt-DT适配直接读取原始NPZ。
- 原始MuJoCo1.31/2.0官方二进制与公开免费激活文件已下载，WSL项目内Python3.10、mujoco-py2.0.2.13及旧依赖已安装。
- 六环境真实核验全部通过，见`runs/verification/all_environments.json`：非零动作奖励独立重算、维度、种子reset/step重现、终止/超时、Hopper/Walker四类参数精确写入。Walker零动作第58步真实跌倒，其他环境达到设定上限；这是环境检查而非训练策略成绩。

进行中/未完成：

- Point种子0正式预训练已启动：`runs/point/seed_0/pretrain`，1000epoch=718000次更新，GPU运行；日志`runs/point/pretrain_seed_0.log`。
- Point种子0 Prompt-DT基线正式50000次更新已启动：`runs/point/seed_0/prompt_dt`，CPU运行，日志`runs/point/prompt_dt_seed_0.log`。
- Cheetah-Dir单任务原版SAC短训练通过：10800环境步/400更新，平均回报-11.574，只作运行验证；`runs/cheetah_dir_corro_pilot`。其4个任务的500iteration正式专家训练已启动，输出`runs/cheetah_dir_experts_corro`，尚需检查收敛与策略质量。
- 全部MuJoCo任务manifest已固定在`datasets/*_tasks.json`，包括Walker/Hopper四类参数；尚需其余环境专家训练和正式数据。
- Windows NumPy2检查点在Linux NumPy1.23中发现`numpy._core`兼容问题；已加入可信本地检查点的模块路径兼容，并将新RNG存储改为纯Python列表。8项数据/续训/跨格式相关测试通过，Linux实机成功加载Windows第53000步预训练检查点，复验通过。
- 正式1000epoch/1e6step、多种子及消融训练尚未完成。
- FOCAL/CORRO统一数据与评估适配、各基线正式多种子训练及全部对照实验尚未完成。
- 论文未披露细节及内部矛盾仍见`PAPER_AUDIT.md`；不宣称无差异。

用户已明确同意：没有现成数据，继续自行采集。
