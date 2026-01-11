# run_enhance_rl.py
import argparse
import os
from pathlib import Path
import json

def setup_directories():
    """设置目录结构"""
    directories = ['structures', 'results', 'checkpoints', 'graphs', 'logs']
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"确保目录存在: {dir_name}")

def main():
    parser = argparse.ArgumentParser(description='强化学习实验')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'analyze', 'benchmark'],
                       help='运行模式')
    parser.add_argument('--episodes', type=int, default=500,
                       help='训练episode数量')
    parser.add_argument('--steps', type=int, default=3,
                       help='每个episode的步数')
    parser.add_argument('--start', type=str, default='v1939',
                       help='起始结构')
    parser.add_argument('--load_agent', type=str, default=None,
                       help='加载已训练的代理')
    parser.add_argument('--test_sequence', type=str, default=None,
                       help='测试特定序列，格式: "op1,op2,op3"')
    parser.add_argument('--max_vertices', type=int, default=4000,
                       help='最大顶点数限制')
    parser.add_argument('--pruning', action='store_true', default=True,
                       help='启用删点操作')
    
    args = parser.parse_args()
    
    # 设置目录
    setup_directories()
    
    # 导入必要的模块
    from enhance_rl_env import EnhanceRLEnv
    from rl_agent import EnhanceRLAgent
    
    # 初始化环境
    env = EnhanceRLEnv(
        base_structures_path="structures/",
        max_steps_per_episode=args.steps,
        max_vertices=args.max_vertices,
        pruning_enabled=args.pruning
    )
    
    print(f"环境初始化完成:")
    print(f"  最大顶点数: {args.max_vertices}")
    print(f"  操作数量: {len(env.operations)}")
    print(f"  删点操作: {args.pruning}")
    
    if args.mode == 'train':
        from train_enhance_rl import train_agent, analyze_successful_sequences
        
        # 初始化代理
        agent = EnhanceRLAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n
        )
        
        # 加载已有代理（如果指定）
        if args.load_agent and os.path.exists(args.load_agent):
            agent.load(args.load_agent)
            print(f"从 {args.load_agent} 加载代理")
        
        # 训练
        print("\n开始训练...")
        env, agent, stats = train_agent(
            episodes=args.episodes,
            max_steps_per_episode=args.steps,
            start_structure=args.start,
            load_agent_path=args.load_agent
        )
        
        # 分析
        analyze_successful_sequences(agent, env)
        
    elif args.mode == 'test':
        from train_enhance_rl import test_optimized_sequence
        
        if args.test_sequence:
            # 解析操作序列
            op_names = [name.strip() for name in args.test_sequence.split(',')]
            
            # 转换为动作索引
            sequence_actions = []
            for op_name in op_names:
                found = False
                for idx, op in enumerate(env.operations):
                    if op['name'] == op_name:
                        sequence_actions.append(idx)
                        found = True
                        break
                
                if not found:
                    print(f"错误: 未找到操作 '{op_name}'")
                    return
            
            print(f"测试自定义序列: {op_names}")
            
            # 执行测试
            reward, chromatic = test_optimized_sequence(env, sequence_actions, "自定义序列")
            
            print(f"\n测试结果:")
            print(f"  总奖励: {reward:.2f}")
            print(f"  色数: {chromatic}")
            print(f"  顶点数: {len(env.current_points)}") # type: ignore
            
        else:
            # 测试预定义序列
            test_sequences = [
                ("自定义构造1", ["Minkowski_v31", "Filter_radius_2"]),
                ("自定义构造2", ["Minkowski_v31", "Rotate_60_merge", "Prune_redundant"]),
                ("自定义构造3", ["Minkowski_v31", "Rotate_theta4_merge", "Filter_radius_2"]),
            ]
            
            for seq_name, op_names in test_sequences:
                # 转换为动作索引
                sequence_actions = []
                for op_name in op_names:
                    found = False
                    for idx, op in enumerate(env.operations):
                        if op['name'] == op_name:
                            sequence_actions.append(idx)
                            found = True
                            break
                    
                    if not found:
                        print(f"警告: 未找到操作 '{op_name}'")
                        continue
                
                if sequence_actions:
                    test_optimized_sequence(env, sequence_actions, seq_name)
    
    elif args.mode == 'analyze':
        # 分析模式
        if args.load_agent and os.path.exists(args.load_agent):
            from train_enhance_rl import analyze_successful_sequences
            
            agent = EnhanceRLAgent(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n
            )
            agent.load(args.load_agent)
            
            analyze_successful_sequences(agent, env)
            
        else:
            print("请指定要分析的代理文件 (--load_agent)")
    
    elif args.mode == 'benchmark':
        # 基准测试模式
        print("运行基准测试...")
        
        # 测试所有操作的执行时间
        import time
        
        env.reset(args.start)
        benchmark_results = []
        
        for i, op in enumerate(env.operations):
            print(f"测试操作 {i+1}/{len(env.operations)}: {op['name']}")
            
            start_time = time.time()
            success, message, _ = env._apply_operation(i)
            elapsed = time.time() - start_time
            
            benchmark_results.append({
                'operation': op['name'],
                'success': success,
                'time': elapsed,
                'vertices': len(env.current_points) if env.current_points else 0
            })
            
            # 重置环境
            env.reset(args.start)
        
        # 保存基准测试结果
        with open('logs/benchmark_results.json', 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print("\n基准测试完成！结果已保存到 logs/benchmark_results.json")
        
        # 显示最快和最慢的操作
        sorted_results = sorted(benchmark_results, key=lambda x: x['time'])
        print("\n最快操作:")
        for result in sorted_results[:5]:
            print(f"  {result['operation']}: {result['time']:.3f}秒")
        
        print("\n最慢操作:")
        for result in sorted_results[-5:]:
            print(f"  {result['operation']}: {result['time']:.3f}秒")
    
    print("\n实验完成！")

if __name__ == "__main__":
    main()