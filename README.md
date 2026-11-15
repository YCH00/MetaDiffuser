# MetaDiffuser 复现工程

**当前交付范围：MetaDiffuser主方法代码和运行验证，正式训练由用户在服务器执行，不包含其他baseline复现。** 服务器安装、Git同步后的依赖恢复、数据采集、训练及评估请直接参照[服务器运行说明](docs/SERVER_RUN.md)。已有本地长训练已停止，数据和检查点保留；历史基线代码与记录不属于必要运行依赖。

本目录按照 ICML 2023 正式版和用户提供的实施文档构建。当前为可审计的独立实现；论文存在符号和维度矛盾，原始数据、全部结构细节及实验预算并未公开。**代码可运行/测试通过不代表六环境论文成绩已复现。** 详见 [逐项核对记录](docs/PAPER_AUDIT.md)。

## 安装

Python 3.10+，建议单独venv。Windows将下列python替换为`.venv\Scripts\python`；Linux可使用`.venv/bin/python`。

```bash
python -m venv .venv
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[test]"
python -m metadiffuser.cli doctor
python -m pytest -q
```

先激活venv再运行上述pip，避免装到系统Python。旧MuJoCo单独配置，见 [环境说明](docs/ENVIRONMENTS.md)。

## 快速验证

此配置缩小网络和预算，使用标明`controller_smoke_only`的手工控制器数据，不构成SAC专家或论文成绩。

```bash
python -m metadiffuser.cli collect --config configs/smoke.json --output datasets/smoke --controller-smoke
python -m metadiffuser.cli pretrain --config configs/smoke.json --dataset datasets/smoke --output runs/smoke/pretrain --max-steps 100
python -m metadiffuser.cli train --config configs/smoke.json --dataset datasets/smoke --pretrained runs/smoke/pretrain/latest.pt --output runs/smoke/diffusion --max-steps 100
python -m metadiffuser.cli evaluate --checkpoint runs/smoke/diffusion/latest.pt --dataset datasets/smoke --output runs/smoke/eval --episodes 1
```

默认`paper_plus`保留论文Eq.8加号，可能导致动力学不稳定。减号是独立解释，必须显式运行并分开报告：

```bash
python -m metadiffuser.cli evaluate --checkpoint runs/smoke/diffusion/latest.pt --dataset datasets/smoke --output runs/smoke/eval_minus --episodes 1 --overrides '{"dynamics_sign":"consistency_minus"}'
```

PowerShell对JSON参数的转义取决于版本，可用Python API `evaluate(..., overrides={"dynamics_sign": "consistency_minus"})`，或实验矩阵脚本避免shell转义问题。

## 正式数据与训练入口

```bash
python -m metadiffuser.cli tasks --config configs/point_robot.json --output datasets/point_tasks.json
python -m metadiffuser.cli expert --collector corro --config configs/point_robot.json --tasks datasets/point_tasks.json --output runs/point_experts
python -m metadiffuser.cli collect --config configs/point_robot.json --tasks datasets/point_tasks.json --policy-root runs/point_experts --output datasets/point_sac
python -m metadiffuser.cli pretrain --config configs/point_robot.json --dataset datasets/point_sac --output runs/point/pretrain
python -m metadiffuser.cli train --config configs/point_robot.json --dataset datasets/point_sac --pretrained runs/point/pretrain/latest.pt --output runs/point/diffusion
python -m metadiffuser.cli evaluate --checkpoint runs/point/diffusion/latest.pt --dataset datasets/point_sac --output runs/point/eval
```

默认直接调用固定CORRO的SAC网络与更新代码，预算来自其公开配置，须检查`evaluation.jsonl`及独立评估确认质量；`--iterations`可调整预算。`expert --task-id ...`可指定单任务。`--collector independent --steps 100000`保留独立SAC诊断实现，来源明确区分。
每环境一份完整JSON；种子修改后另存配置。复现论文训练预算不要使用`--max-steps`。每个run保存解析配置、版本、优化器、RNG、训练指标和checkpoint。续训追加`--resume runs/.../latest.pt`并保持配置及数据不变。`--max-steps`表示目标总步数，不是新增步数。

正式Point数据已自行采集，当前实际执行状态见[执行记录](docs/EXECUTION_STATUS.md)。训练入口保存每次启动/续训的源码快照与哈希，便于追溯长期运行所用版本。数据质量审查入口为`python scripts/audit_dataset.py --dataset datasets/point_sac --output runs/verification/point_sac_data_audit.json`。该报告中的策略回报属于采集策略，不是MetaDiffuser成绩。

CVAE-Planner、Prompt-DT及其他基线的入口和已知适配差异见[基线说明](docs/BASELINES.md)。Prompt-DT数值模型运行依赖可用`python scripts/setup_prompt_runtime.py`安装到已激活的项目venv。

## 评估与消融

`--quality expert|medium|random`，`--overrides`允许`dynamics_sign`、`dynamics_weight`、`reward_weight`、`cfg_scale`、`guide_scale`、`temperature`、`context_mode`。K=20/50/100、dropout、history、horizon属于训练配置，须重训，不能偷换已有checkpoint的schedule。

每条`episodes.jsonl`包含真实回报、warm-start窗口、动作裁剪比例、延迟和模型诊断。`aggregate`依次对episode、task、model seed平均，并报告种子间样本标准差。单种子不伪造误差条。文献值明确命名为`paper_table_1_literature_value`，不作为本地运行结果。

## 工程结构

- `metadiffuser/`：任务、环境、采集、数据、SAC、网络、扩散、训练、评估及CLI。
- `configs/`：六环境完整参数和单独的smoke参数。
- `tests/`：闭式公式、梯度方向、inpainting、数据隔离、短训练、精确续训验证。
- `references/`：论文及来源锁定记录。
- `docs/`：冲突说明、工程选择及未完成实验清单。
- `third_party/`：固定commit的CORRO与Diffuser参考代码。

研究来源：[MetaDiffuser正式论文](https://proceedings.mlr.press/v202/ni23a.html)、[CORRO](https://github.com/PKU-RL/CORRO)、[Diffuser](https://github.com/jannerm/diffuser)。
