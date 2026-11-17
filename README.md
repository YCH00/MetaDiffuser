# MetaDiffuser 复现工程

**当前交付范围：MetaDiffuser主方法代码和运行验证，正式训练由用户在服务器执行，不包含其他baseline复现。** 服务器安装、Git同步后的依赖恢复、数据采集、训练及评估请直接参照[服务器运行说明](docs/SERVER_RUN.md)。已有本地长训练已停止，数据和检查点保留；历史基线代码与记录不属于必要运行依赖。

本目录按照 ICML 2023 正式版和用户提供的实施文档构建。当前为可审计的独立实现；论文存在符号和维度矛盾，原始数据、全部结构细节及实验预算并未公开。**代码可运行/测试通过不代表六环境论文成绩已复现。** 详见 [逐项核对记录](docs/PAPER_AUDIT.md)。

## 环境配置

```bash
conda create -n metadiffuser python=3.10 pip -y
conda activate metadiffuser

python -m pip install \
  'pip<24.1' \
  'setuptools>=68,<70' \
  'wheel<0.40' \
  numpy==1.23.5 \
  Cython==0.29.37

python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

python -m pip install \
  gym==0.17.3 \
  scipy==1.10.1 \
  matplotlib==3.7.5 \
  six==1.17.0 \
  glfw==2.6.5 \
  imageio==2.31.6 \
  fasteners==0.19 \
  cffi==1.16.0 \
  pycparser==2.22 \
  cloudpickle==1.6.0 \
  requests==2.32.3 \
  PyOpenGL==3.1.7

sudo apt-get update
sudo apt-get install -y build-essential libosmesa6-dev libgl1-mesa-dev libglu1-mesa-dev patchelf

use_mujoco 131 200
python -m pip install --no-build-isolation  mujoco-py==2.0.2.13

cd MetaDiffuser
python -m pip install -e '.[test]'
python scripts/fetch_references.py


```

cp -r ./pkg/.mujoco ~/
vim ~/.bashrc
source ~/.bashrc

use_mujoco() {
    local version
    local mjpro_path=""
    local mujoco_path=""
    local clean_path=""
    local path_item
    local -a path_items

    # 检查参数数量
    if [ "$#" -eq 0 ] || [ "$#" -gt 2 ]; then
        echo "用法: use_mujoco 131|150|200 [200]"
        return 1
    fi

    # 解析版本参数
    for version in "$@"; do
        case "$version" in
            131)
                if [ -n "$mjpro_path" ]; then
                    echo "错误: 131 和 150 不能同时启用"
                    return 1
                fi
                mjpro_path="$HOME/.mujoco/mjpro131"
                ;;
            150)
                if [ -n "$mjpro_path" ]; then
                    echo "错误: 131 和 150 不能同时启用"
                    return 1
                fi
                mjpro_path="$HOME/.mujoco/mjpro150"
                ;;
            200)
                if [ -n "$mujoco_path" ]; then
                    echo "错误: 200 不能重复指定"
                    return 1
                fi
                mujoco_path="$HOME/.mujoco/mujoco200"
                ;;
            *)
                echo "错误: 不支持 MuJoCo $version"
                echo "用法: use_mujoco 131|150|200 [200]"
                return 1
                ;;
        esac
    done

    # 检查安装目录
    if [ -n "$mjpro_path" ] && [ ! -d "$mjpro_path/bin" ]; then
        echo "错误: 目录不存在 $mjpro_path/bin"
        return 1
    fi

    if [ -n "$mujoco_path" ] && [ ! -d "$mujoco_path/bin" ]; then
        echo "错误: 目录不存在 $mujoco_path/bin"
        return 1
    fi

    # 清理旧 MuJoCo 动态库路径
    IFS=':' read -r -a path_items <<< "${LD_LIBRARY_PATH:-}"

    for path_item in "${path_items[@]}"; do
        case "$path_item" in
            "$HOME/.mujoco/mjpro131/bin"|\
            "$HOME/.mujoco/mjpro150/bin"|\
            "$HOME/.mujoco/mujoco200/bin"|\
            "$HOME/.mujoco/mjpro/bin"|"")
                ;;
            *)
                clean_path="${clean_path:+$clean_path:}$path_item"
                ;;
        esac
    done

    # 清理旧环境变量
    unset MUJOCO_PY_MJPRO_PATH
    unset MUJOCO_PY_MUJOCO_PATH

    # 设置新环境变量
    local new_path=""

    if [ -n "$mujoco_path" ]; then
        export MUJOCO_PY_MUJOCO_PATH="$mujoco_path"
        new_path="$mujoco_path/bin"
    fi

    if [ -n "$mjpro_path" ]; then
        export MUJOCO_PY_MJPRO_PATH="$mjpro_path"
        new_path="${new_path:+$new_path:}$mjpro_path/bin"
    fi

    export MUJOCO_PY_MJKEY_PATH="$HOME/.mujoco/mjkey.txt"
    export MUJOCO_PY_FORCE_CPU=1

    export LD_LIBRARY_PATH="$new_path${clean_path:+:$clean_path}"

    echo "MuJoCo 环境切换成功"
    echo "MUJOCO_PY_MJPRO_PATH=${MUJOCO_PY_MJPRO_PATH:-<未设置>}"
    echo "MUJOCO_PY_MUJOCO_PATH=${MUJOCO_PY_MUJOCO_PATH:-<未设置>}"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
}

先激活venv再运行上述pip，避免装到系统Python。旧MuJoCo单独配置，见 [环境说明](docs/ENVIRONMENTS.md)。

## 数据收集
训练专家策略
```bash
(
  set -e

  for spec in \
    point_robot:1000 \
    ant_dir:200 \
    cheetah_dir:500 \
    cheetah_vel:500 \
    walker_param:3000 \
    hopper_param:700
  do
    env_name="${spec%%:*}"
    iterations="${spec##*:}"

    python scripts/train_expert_cohort.py \
      --config "configs/${env_name}.json" \
      --tasks "datasets/${env_name}_tasks.json" \
      --output "runs/${env_name}_experts" \
      --iterations "$iterations" \
      --workers 2
  done
)
```
检查各任务 evaluation.jsonl 中的回报是否已达到合理水平（主要看能不能达到论文中报告的oracle）
```bash
for ENV in ant_dir point_robot cheetah_dir cheetah_vel hopper_param walker_param; do
    echo "$ENV start:"
    for f in ./runs/"$ENV"_experts/"$ENV"_*/evaluation.jsonl; do
        echo -n "$(basename "$(dirname "$f")"): "
        jq -s -r 'max_by(.mean) | "mean=\(.mean), iteration=\(.iteration)"' "$f"
    done
    echo "$ENV done."
done
```
确认策略质量后，采集数据
```bash
(
  set -e

  for env_name in \
    point_robot ant_dir cheetah_dir cheetah_vel walker_param hopper_param
  do
    python -m metadiffuser.cli --device cpu collect \
      --config "configs/${env_name}.json" \
      --tasks "datasets/${env_name}_tasks.json" \
      --policy-root "runs/${env_name}_experts" \
      --output "datasets/${env_name}_sac"

    python scripts/audit_dataset.py \
      --dataset "datasets/${env_name}_sac" \
      --output "runs/${env_name}_data_audit.json"
  done
)
```

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

