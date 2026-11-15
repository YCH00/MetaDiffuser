# 主方法交付验证

验证日期：2026-11-15。范围按用户最新要求：MetaDiffuser代码可运行，正式训练由用户在服务器完成，其他baseline不属于交付验收。

- Windows Python3.11.5 / PyTorch2.6.0+cu126 / RTX4060 Laptop：主方法23项测试通过，4.74秒。命令见SERVER_RUN；原始结果`runs/verification/metadiffuser-delivery.xml`。
- 独立GPU端到端命令：`python scripts/smoke_metadiffuser.py --device cuda --output runs/delivery_smoke`，退出码0。生成5任务数据，预训练100更新、扩散100更新，2测试任务各完成20步真实闭环；噪声MSE末步0.56267。该短运行不用于论文性能宣称。
- 六环境实际核验通过：Point、Ant、Cheetah-Dir/Vel、Walker、Hopper，包含非零动作奖励公式、观测/动作维度、种子重现性、终止/超时及四类随机动力学参数。报告`runs/verification/all_environments.json`；原版MuJoCo在WSL Ubuntu24.04/Python3.10/NumPy1.23.5运行。
- Windows NumPy2检查点到Linux NumPy1.23加载通过；数据、续训及序列化相关8项测试通过。
- `python scripts/fetch_references.py`核验两个依赖提交与来源锁定文件一致。新机器需能访问其GitHub源以重新下载。

服务器硬件、驱动及发行版尚未实际验证；先执行SERVER_RUN的doctor、测试和smoke再启动长训练。本文证明已测试环境及列举路径可运行，不证明论文回报已复现，也不消除PAPER_AUDIT中的原文冲突与未公开细节。
