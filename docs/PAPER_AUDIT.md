# 论文逐项核对与差异记录

用户最新要求已调整为：只交付MetaDiffuser主方法代码并验证可运行，正式训练在用户服务器进行，不复现其他baseline。下文“论文级验收”的多种子成绩、基线和作者确认清单描述未达到的完整数值复现边界，不再是本次代码交付的必需任务。公式冲突与实施选择说明仍有效。

依据：ICML 2023 正式出版版，19 页，PMLR 202:26087–26105。
原文：https://proceedings.mlr.press/v202/ni23a/ni23a.pdf 。本地存档 `references/metadiffuser.pdf`。
用户文档 `MetaDiffuser复现步骤.md` 保持原样。用户要求论文优先；文档中实施建议并非原论文事实。

## 状态与可验证边界

本工程是独立复现，不是作者原始实现。作者项目页的 Code 链接回到项目页，尚未获得原始 MetaDiffuser 代码、数据或 checkpoint。
“完整正确且与原论文无任何差异”目前无法作出这一保证：论文有自相矛盾及未披露内容；数值复现还须完成实验。
代码测试、短程 smoke、正式训练预算、论文数值复现是不同验收项目，不能互相替代。

## 核心算法核对表

| 论文依据 | 本地实现 | 处理及状态 |
|---|---|---|
| §4.1 / Eq.1，上下文和两个预测器联合训练 | `models.py:WorldModel`, `train.py:world_loss` | 输入连续 `(s,a,r,s_next)` 窗口；三隐藏层编码器；64 维默认输出；同时优化三个网络 |
| Eq.1 条件概率预测 | `TransitionPredictor` | 固定单位方差高斯 NLL，事件维求和；概率族与方差是未披露实施选择 |
| Eq.2 联合生成状态和动作 | `ConditionalUNet`, `WindowSampler` | `[B,H,Ds+Da]`，状态在前，动作在后 |
| Eq.5 / App.D 噪声重构 | `GaussianDiffusion.loss` | 目标为真实注入噪声；冻结编码器；Bernoulli 逐样本丢弃上下文 |
| Eq.6 CFG | `predict_noise` | `uncond + omega*(cond-uncond)`；omega=1 为普通条件预测 |
| Eq.7/8 双重引导 | `DualGuide`, `sample` | 当前噪声轨迹上求梯度；在噪声预测中减去 `sqrt(1-alpha_bar)*gradient` |
| Eq.7 轨迹回报 | `DualGuide.components` | H 个状态动作上的奖励求和，不引入额外折扣 |
| Eq.7 动力学误差 | 同上 | 只对已有相邻状态的 H-1 个转移求平方误差和；不伪造终点 |
| §4.2 / App.D receding horizon | `evaluate.py` | 固定 warm-start 上下文；每步重规划；仅执行第一动作 |
| App.H temporal U-Net | `models.py` | 6 个残差块，每块两层 Conv1d/GroupNorm/Mish；时间与上下文各64维，两层嵌入 MLP 隐藏256 |
| App.H 优化器和预算 | 六份 `configs/*.json` | Adam；预训练2e-4/batch64/1000 epochs；扩散2e-4/batch32/1e6 steps |
| App.E / F.3 / H 任务数、h、H、episode长度 | `config.py:ENV_SPECS` | 30/10、Cheetah-Dir 2/2；h=4/10/20，H=4/16/32，episode=20/200 |
| Fig.4 五个训练随机种子 | 实验矩阵脚本 | 五种子是论文明确内容；用户文档将其列为实施选择不够准确 |

## 冲突及处理优先级

1. **动力学项正负号（原论文内部冲突）**：Eq.7 的 zeta 是正平方误差，而 Eq.8 使用 `J+lambda*zeta`，从标准 DDPM 反推均值会沿误差增大的方向移动。文字要求减少不一致性。默认 `paper_plus` 严格保留公式；`consistency_minus` 实现文字意图及用户文档的建议。后者不是作者确认的勘误。两者的结果必须分开报告，不能挑选结果后声称完全复现。
2. **附录 I 的 mask**：Bernoulli(drop_prob) 直接乘上下文实际上是保留概率。采用正文和附录 D 的 `1-drop_mask`。在嵌入 MLP 后 mask，消除 bias 残留。
3. **附录 I 的 loss**：伪代码把 noisy trajectory 放在 loss 目标位置；正文 Eq.5 是噪声预测。以 Eq.5 为准。
4. **附录 I 采样**：缺 CFG、dual-guide、首状态约束且有语法遗漏；采用正文 Eq.6/8、附录 D 和 DDPM 完整反向链，不把该伪代码当成可直接运行源码。
5. **附录 H 预测器输出**：文字称奖励和动力学均输出标量；Eq.1/7 要求下一状态向量。本实现奖励输出1维，动力学输出Ds维。这是为实现正文所需的维度解释，仍待作者确认。
6. **六残差块 vs Diffuser 默认网络**：固定的 Diffuser 仓库默认 U-Net 实际不止六个残差块；不能原样拷贝还声称符合 H。这里采用6个 residual blocks（2 contracting、2 middle、2 expanding）。层级/通道数/核大小未由论文确定，当前拓扑是明确实施选择。并不能据“六块”证明与作者网络逐层相同。
7. **Point-Robot**：CORRO README 的 `point_robot_v1` 是半圆目标和10步上限；论文 E 是方形目标和20步。用方形和20步；方形边长、起始分布未披露，默认 `[-1,1]^2`、原点；可改为 CORRO 方形初始分布，结果须注明。
8. **Walker 随机参数**：CORRO 构造函数只随机质量和摩擦；论文 E 指四类参数。调用完整参数采样，包含惯量与阻尼。CORRO `set_goal` 展平接口会丢掉两类参数，故直接调用 `set_task(dict)`。
9. **环境依赖**：CORRO Ant/Cheetah 用 mujoco-py 2.x；Hopper/Walker 的内置 ctypes binding 指向 MuJoCo 1.31 和旧 license。不能以安装一个新包或换成 Gymnasium v5 代替而声称物理一致。
10. **默认超参**：文档推荐统一 dropout=.2/CFG=1.4/K=100 是起跑建议。六环境配置采用表7的 dropout、表4的 lambda，以及与表1对应的表11 CFG / 表8 K。后两者是数值匹配推断，未发布联合最终配置。Hopper 在不同表中的356.4/356.8等差异原样登记，不将其视为同一实验。

## 所有已知实施选择

- 编码器隐藏宽度256及 Mish；上下文窗口展平；聚合默认单窗口。
- U-Net基础通道64，卷积核5，GroupNorm最多8组，两个尺度。预测器是 contracting blocks，单转移作为长度1输入，避免访问未来状态/动作。预测器无扩散时间输入，预训练使用干净离线转移；未擅自增加噪声训练。
- 状态用训练集均值/标准差（含训练 next_state）；动作用真实 action bounds 映射；奖励乘显式 scale。guidance在同一归一化坐标系计算，lambda尺度因此依赖归一化；论文未交代此点。
- cosine噪声调度，DDPM posterior variance；temperature 是协方差倍率，默认1；默认无EMA、无梯度裁剪、无额外guide缩放、无x0裁剪。
- 初始状态inpainting同时用于训练和采样；损失排除首状态维度；这是文档建议，论文未给细节。
- `epoch` 定义为 `ceil(valid_window_count/batch)` 个任务均衡有放回采样步骤。因此是等效数据遍历量，不是精确逐窗遍历；保存真实steps。
- 每任务100条episode起跑，10%独立验证episode；先均匀任务，再episode，再连续窗口，窗口不跨episode，短窗舍弃。
- 正式采集入口直接使用固定CORRO的SAC网络和更新代码，Point使用32x32 ReLU、固定alpha=.01、gamma=.9，其他环境遵循其公开配置。Cheetah-Dir没有公开采集配置，暂借Cheetah-Vel配置并标注。原论文未发布自身最终专家预算，不能把CORRO默认值视为已确认原始预算。独立实现SAC保留为诊断选项，其数据来源单独标记。
- CORRO原始collector把非goal环境的全部转移按非terminal bootstrap（含Walker跌倒），且Point初始随机动作在底层action bounds采样后再由wrapper映射。本地CORRO适配保留这两个实际行为并披露，独立SAC则区分真实终止。最终离线episode格式始终正确记录terminated/truncated。
- medium策略按独立评估回报最接近random/expert中点选择；若所有保存策略都已收敛，则并不构成有效medium，必须查看实际回报。不能仅用训练epoch定义质量。
- 每测试任务10个episode、各层等权均值、种子间样本标准差（ddof=1）。论文未明确10个episode及误差条定义。

## 尚未完成的论文级验收

- 原始任务/数据/专家或同分布专家质量确认。
- 六环境正式预算、多种子训练与回报比较。
- FOCAL、CORRO、Prompt-DT、CVAE-Planner在同一数据上的实际对照。
- 核心消融全量实验；附录MetaWorld额外实验。
- 对不明确结构、guide符号、概率模型等取得可核验的原实现或作者澄清。

以上项目完成前，不应宣称“完整论文数值复现完成”。

# 数据采集补充核对

正式采集默认 `collection_deterministic=false`，从选定SAC checkpoint的动作分布采样；每条episode设置独立且可重现的PyTorch种子。Point固定起点配确定性策略会产生完全重复的episode，不能增加状态覆盖。该选择未由MetaDiffuser论文披露，记录为实施选择。checkpoint质量选择仍依据独立确定性评估；最终数据质量依据采集后的真实回报复核，不能用checkpoint评估替代。

