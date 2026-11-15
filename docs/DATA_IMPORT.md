# 导入已有 CORRO 数据

命令：`python -m metadiffuser.cli import-corro --index source_index.json --output datasets/imported`。

源目录需有 `obs.npy`、`actions.npy`、`rewards.npy`、`next_obs.npy`。原始 `terminals.npy` 不足以恢复所有超时和环形覆盖信息，因此入口要求下面的显式索引。任务参数必须完整，不能从只保留三位小数的文件夹名推测。

```json
{
  "env": "point_robot",
  "action_low": [-0.1, -0.1],
  "action_high": [0.1, 0.1],
  "tasks": [
    {"task_id": "point_robot_000", "parameters": {"goal": [0.25, 0.75]}, "task_seed": 2026, "split": "train"}
  ],
  "sources": [
    {
      "directory": "original_task_buffer",
      "task_id": "point_robot_000",
      "episodes": [
        {
          "indices": [0, 1, 2, 3],
          "end_reason": "terminated",
          "split": "train",
          "quality": "expert",
          "collection_seed": 1,
          "policy_checkpoint": "original_policy.pt"
        }
      ]
    }
  ]
}
```

上面仅演示schema，并不是合法完整的论文数据集。实际索引须包含所有训练/测试任务及独立train/validation/warm_start episode；Point-Robot完整episode应为20步且超时，不能采用示例中的4步terminal占位。

`indices`按时间排序，可表达环形buffer回绕。导入器检查状态连续性、重叠样本、合法结束类型、任务划分、完整任务覆盖和校验值。不能恢复的信息必须补齐，不能用padding或任意切片掩盖。

质量标签要依据真实回报和专家来源，不会仅因为文件来自CORRO就自动视作MetaDiffuser论文的原始专家数据。
