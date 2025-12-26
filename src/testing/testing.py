def test(config):
    """测试核心逻辑（替换原有ad-hoc测试脚本）"""
    print("\n========== 开始测试 ==========")
    # 从config读取参数
    experiment_id = config["experiment_id"]
    model_path = config["model_path"]
    test_dataset = config["test_dataset"]
    data_path = config["data_path"]
    
    # 模拟测试逻辑（替换为你的真实测试代码）
    print(f"实验ID：{experiment_id}")
    print(f"模型路径：{model_path}")
    print(f"测试集：{test_dataset}")
    print(f"数据路径：{data_path}")
    print("测试中...（加载模型→加载测试集→推理→计算指标）")
    print("========== 测试完成 ==========\n")
    
    # 返回测试结果（可选）
    return {"status": "success", "test_acc": 0.915}