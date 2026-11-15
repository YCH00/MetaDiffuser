# MetaDiffuser 主方法服务器运行说明

按用户最新范围：只交付MetaDiffuser主方法代码及运行验证，不要求本机完成正式训练或复现其他baseline。之前产生的基线实验仅为历史产物，不是当前验收条件。

## Git同步与依赖

当前目录尚未初始化Git仓库。可自行`git init`并设置服务器使用的远端；不需要提交本机虚拟环境或运行产物。`.gitignore`排除了`.venv/`、`.venv-legacy/`、`.runtime/`、`third_party/`、`datasets/`和`runs/`。所以Git同步**不会携带数据或checkpoint**：已有正式数据`datasets/point_sac`及相应任务清单需要另行传输；也可按下文重新采集。

Linux GPU训练环境（从项目根目录执行）：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[test]'
python scripts/fetch_references.py
python -m metadiffuser.cli doctor
python -m pytest tests/test_math.py tests/test_overfit.py tests/test_data_pipeline.py tests/test_corro_collector.py tests/test_checkpoint_portability.py -q
python scripts/smoke_metadiffuser.py --device cuda --output runs/server_smoke
```

CUDA发行版需匹配服务器驱动；`doctor`应确认GPU可用。可先使用`--device cpu`完成smoke。重复执行smoke须换一个空输出目录。smoke包含采集、100步预训练、100步扩散训练和真实闭环评估，仅验证代码链路，不表示论文性能。

模型训练只依赖PyTorch/NumPy；不需要安装FOCAL、Prompt-DT或CVAE基线依赖。CORRO仓库用于原始环境和专家采集；Diffuser只作固定参考源码，不需安装其旧训练环境。

## 生成正式数据

Point无MuJoCo依赖：

```bash
python -m metadiffuser.cli tasks --config configs/point_robot.json --output datasets/point_tasks.json
python scripts/train_expert_cohort.py --config configs/point_robot.json --tasks datasets/point_tasks.json --output runs/point_experts --iterations 1000 --workers 2
python -m metadiffuser.cli --device cpu collect --config configs/point_robot.json --tasks datasets/point_tasks.json --policy-root runs/point_experts --output datasets/point_sac
python scripts/audit_dataset.py --dataset datasets/point_sac --output runs/point_data_audit.json
```

1000次SAC迭代是本地采集预算选择，并非MetaDiffuser论文公开预算。训练/验证expert数据与测试warm-start分开保存，模型训练不读取测试数据。

其余五个环境使用独立Linux旧环境，详见`ENVIRONMENTS.md`：

```bash
python scripts/fetch_legacy_runtime.py
bash scripts/fetch_legacy_system_libs.sh
bash scripts/setup_legacy_wsl.sh
bash scripts/verify_legacy.sh
bash scripts/run_legacy.sh scripts/generate_tasks.py --env ant_dir cheetah_dir cheetah_vel walker_param hopper_param
```

脚本名称含`wsl`，但内容也是普通Linux bash；目前实测系统为Ubuntu24.04 x86_64。`fetch_legacy_system_libs.sh`锁定了Ubuntu24.04包版本，其他发行版须按实际发行版配置GL/OSMesa编译依赖，不能直接套用该apt脚本。原引擎固定MuJoCo2.0（Ant/Cheetah）及1.31（Hopper/Walker），不能直接替换成Gymnasium同名环境。旧环境的Python/NumPy单独固定，不混装进GPU训练venv。采集命令同Point，替换config、tasks和输出路径，并通过`bash scripts/run_legacy.sh -m metadiffuser.cli ...`调用。

## 正式训练、续训与评估

```bash
python -m metadiffuser.cli --device cuda pretrain --config configs/point_robot.json --dataset datasets/point_sac --output runs/point/pretrain
python -m metadiffuser.cli --device cuda train --config configs/point_robot.json --dataset datasets/point_sac --pretrained runs/point/pretrain/latest.pt --output runs/point/diffusion
python -m metadiffuser.cli --device cuda evaluate --checkpoint runs/point/diffusion/latest.pt --dataset datasets/point_sac --output runs/point/evaluation
```

预训练1000epoch，扩散100万更新。正式运行不传`--max-steps`；中断后保持相同配置和数据，原命令追加`--resume 对应目录/latest.pt`。每次启动/续训保存源码快照。MuJoCo闭环评估可在旧环境用`bash scripts/run_legacy.sh -m metadiffuser.cli --device cpu evaluate ...`运行；模型仍可在GPU环境中训练。Windows NumPy2检查点到Linux NumPy1.23的加载已实测通过。

## 论文一致性边界

完整审计见`PAPER_AUDIT.md`。默认严格保留正文Eq.8的`paper_plus`，其动力学正误差符号与论文文字意图冲突；`consistency_minus`仅是独立诊断解释。正文噪声损失、CFG和丢弃概率优先于有矛盾的附录伪代码。未公开的网络层级、概率分布、归一化等采用明确记录的实施选择。可验证本机代码运行与所列公式约束，不能保证未知服务器驱动兼容、正式训练成绩，或作者未公开实现的逐行一致性。
