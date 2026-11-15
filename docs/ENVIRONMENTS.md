# 环境与数据来源

本地两个参考仓库已固定：

| 来源 | commit |
|---|---|
| https://github.com/PKU-RL/CORRO | `210ad407a0a7807b7789ab426d8b5be6c72268a7` |
| https://github.com/jannerm/diffuser | `7ea422860cc0106e5ca5949d980f04b799d5462c` |

保留第三方原始文件与许可证；本项目不修改其环境奖励公式。

## Point-Robot

本地实现不依赖Gym。方形目标按论文E，状态二维，动作每维[-.1,.1]，状态更新为s+a，奖励为更新后状态与目标的负欧氏距离；20步超时，无提前成功终止。起点原点为实施选择（配置可选square）。不使用半圆目标来对齐论文方形实验。

## 旧版 MuJoCo

`backend=corro_legacy` 动态导入固定CORRO代码，不把Gymnasium同名环境当作等价环境。

- Ant-Dir：仅方向投影速度奖励，无控制/接触/存活项；观测qpos[2:]+qvel（27维）；frame_skip=5。
- Cheetah：观测qpos[1:]+qvel+torso位置（20维），不是标准Gym的17维；速度任务奖励 `-|v-target|-0.05*sum(a²)`，方向任务 `direction*v-0.05*sum(a²)`；目标速度上限3源于CORRO代码而非其过时注释中的2。
- Hopper：全部qpos+qvel（12维），无存活奖励和健康终止；速度减0.001动作代价。
- Walker：全部qpos+qvel（18维），存活奖励1，速度减0.001动作代价；height<=.5终止。
- 参数基数质量/惯量/摩擦1.5，阻尼1.3，指数均匀[-3,3]，范围来自CORRO，论文未给出。
- 200步超时与真实终止分开保存。独立SAC只对真实终止停止bootstrap；默认CORRO采集器保留其原实现的全部非goal转移持续bootstrap规则（包括Walker跌倒），详见PAPER_AUDIT。

建议隔离Linux/WSL采集环境。上游requirements是旧完整开发环境快照，包含不相关且互相冲突包，不应在训练venv中混装。
Ant/Cheetah至少需旧Gym及mujoco-py 2.0.2.13与相应MuJoCo；Hopper/Walker源码额外要求MuJoCo1.31二进制与许可证（见其`mujoco_py/config.py`、`mjlib.py`）。官方已提供免费旧版激活文件：https://roboti.us/license.html 。已从官方站下载1.31/2.0引擎及公开激活文件到`.runtime/legacy`，下载URL/哈希在该目录`downloads.json`。运行依赖仍在配置验证中。

`scripts/fetch_legacy_runtime.py`下载官方运行时；在WSL先运行`bash scripts/fetch_legacy_system_libs.sh`将构建依赖解包到项目内sysroot，再运行`bash scripts/setup_legacy_wsl.sh`建立项目内Python3.10环境。运行前`source scripts/legacy_env.sh`同时配置两个引擎路径。旧版MuJoCo2.0绑定已实际编译成功；全部Python依赖锁定在`requirements-legacy.lock.txt`。不会自动改成新版物理引擎。

运行`bash scripts/verify_legacy.sh`进行真实环境核对，报告保存到`runs/verification/environments.json`。包括非零动作下从物理位移独立重算奖励、同种子reset/step逐元素一致、episode终止、观测/动作维度和四类随机参数写入。零动作回报仅作环境诊断，不作为策略性能。
获取合法旧运行时后，先逐环境reset/step及随机策略200步验收，再启动专家训练。切勿把版本迁移结果混作旧物理引擎的精确复现。

## 数据约定

`manifest.json`持久化task_id、任务参数、任务采样种子、train/test split及每条episode哈希。episode使用无pickle的NPZ，metadata为JSON。
train/validation仅来自训练任务；warm_start仅来自测试任务。Cheetah-Dir的参数重合为论文明确例外，但数据文件及采集种子仍不同。
读入时校验哈希、任务参数、split、episode连续性与终止位置。归一化只拟合train。模型checkpoint绑定manifest SHA256，避免误用其他数据。
测试任务的SAC只作为论文Oracle/生成warm-start数据；其数据不会用于MetaDiffuser梯度更新。

