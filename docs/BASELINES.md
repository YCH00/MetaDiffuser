# 对照方法的状态与公平比较要求

按论文附录G下载原始实现，并锁定提交（`references/sources.lock.json`）：

- FOCAL：`third_party/FOCAL`，论文引用的ICLR仓库。README指出后来版本修复过问题；这里不擅自切换版本。
- CORRO：`third_party/CORRO`，包含generative/contrastive encoder及offline policy训练入口。
- Prompt-DT：`third_party/prompt-dt`，原始Transformer与prompt训练入口。

**下载仓库不代表已复现基线。** 还须适配任务manifest、环境奖励/观测、相同离线数据与统一评估。CVAE-Planner提供了可训练的独立实现，但其具体结构与调参未由原论文公开，不能据此声称精确复现作者的CVAE网络。

本地CVAE使用两隐藏层MLP、64维latent、标准高斯KL和单位方差重构损失；显式上下文与初始状态条件，预测相同H的状态/动作，复用同一数据/冻结world encoder/闭环评估。以上架构与KL权重均为配置化实施选择，无扩散梯度引导。入口：

```bash
python -m metadiffuser.cli train-cvae --config configs/point_robot.json --dataset datasets/point_sac --pretrained runs/point/pretrain/latest.pt --output runs/point/cvae
python -m metadiffuser.cli evaluate --checkpoint runs/point/cvae/latest.pt --dataset datasets/point_sac --output runs/point/cvae_eval
```

数据交换工具：

```bash
python scripts/export_baseline_data.py --dataset datasets/point_sac --output datasets/baseline_exchange
```

工具输出CORRO数组、Prompt-DT轨迹pickle、FOCAL四列episode数组、精确任务参数和源数据哈希。它保持train/validation/warm_start分开，未替基线训练器做隐式归一化或任务重采样。pickle文件仅用于本地产生的可信基线输入。

需要核对的实际差异：

1. FOCAL原始loader把每条episode末尾统一设terminal，没有区分timeout；其环境也不保证和本项目一致。
2. Prompt-DT的公共环境初始化及target return配置只覆盖部分实验；论文额外Point/Walker/Hopper需要独立调参，参数未发布。
3. CORRO示例PointRobot-v1是半圆任务，必须更换为论文方形目标，保留精确参数而非解析目录名中的round(3)结果。
4. 原始baseline评估需要改为本项目相同warm-start索引/任务/回报口径；直接引用论文表1只能标为文献值。

Prompt-DT目前已提供训练、精确断点续训及闭环评估适配，直接导入固定提交的原始模型，不下载预训练语言模型：

```bash
python scripts/prompt_dt_baseline.py train --config configs/point_robot.json --dataset datasets/point_sac --output runs/point/prompt_dt --device cuda
python scripts/prompt_dt_baseline.py evaluate --checkpoint runs/point/prompt_dt/latest.pt --dataset datasets/point_sac --output runs/point/prompt_dt_eval --device cuda
```

该适配继承公共入口的K=20、hidden128、3层1头、dropout=.1、batch16、AdamW1e-4、weight_decay1e-4、warmup10000、50000更新、梯度裁剪.25。与MetaDiffuser作者未知设置的差异：prompt使用同一h历史窗口；均值/方差只来自训练集，避免原入口合并train/test；动作映射到真实bounds对应的[-1,1]；prompt RTG只累加窗口内奖励；回报scale默认1，target默认训练episode回报95%分位数，均可通过settings JSON明确更改。每个测试任务和episode的warm-start索引算法与主方法一致，当前只支持单窗口上下文。

原模型按prompt长度是否等于K判断是否把prompt展平共享，且按RTG长度模10猜测是否截断。适配在prompt左侧增加两个attention_mask=0的padding token，使论文h=4/10/20避开这两个分支，保留原模型文件不变。测试覆盖批间任务隔离、当前动作标签不可见、精确断点续训和实际闭环。Transformers固定4.5.1；本项目使用数值轨迹嵌入，不调用tokenizer，未安装与Python3.11不兼容的旧tokenizers二进制。

基线的正式训练结果、FOCAL/CORRO统一评估适配，以及CVAE与作者实现的结构确认仍属于尚未完成的工作，不能用主方法的单元测试替代验收。
