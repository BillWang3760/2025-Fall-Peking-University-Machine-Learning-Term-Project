# train_enhance_rl.py
import numpy as np
import random
from enhance_rl_env import EnhanceRLEnv
from rl_agent import EnhanceRLAgent
import time
from typing import List, Optional
import json

def train_agent(episodes: int = 1000,
               max_steps_per_episode: int = 3,
               start_structure: str = 'v31',
               load_agent_path: Optional[str] = None):
    
    # 初始化环境
    env = EnhanceRLEnv(
        base_structures_path="structures/",
        max_steps_per_episode=max_steps_per_episode,
        max_vertices=4000,
        pruning_enabled=True
    )
    
    # 初始化代理
    agent = EnhanceRLAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    
    # 加载已有代理
    if load_agent_path:
        agent.load(load_agent_path)
    
    # 训练统计
    training_stats = {
        'episode_rewards': [],
        'episode_chromatics': [],
        'episode_vertices': [],
        'success_count': 0,
    }
    
    for episode in range(episodes):
        # 重置环境
        state = env.reset(start_structure)
        
        episode_reward = 0.0
        episode_sequence = []
        done = False
        step_count = 0
        
        # 运行一个episode
        while not done and step_count < max_steps_per_episode:
            # 选择动作
            action = agent.choose_action(state)
            episode_sequence.append(action)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 定期经验回放
            if step_count % 3 == 0:  # 每3步回放一次
                agent.replay(batch_size=32)
            
            # 更新状态
            state = next_state
            episode_reward += reward
            step_count += 1
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 记录统计
        training_stats['episode_rewards'].append(episode_reward)
        training_stats['episode_chromatics'].append(env.episode_chromatic)
        training_stats['episode_vertices'].append(len(env.current_points)) # type: ignore
        
        # 检查是否成功
        if env.episode_chromatic == 5:
            training_stats['success_count'] += 1
            agent.record_success(episode_sequence, episode_reward, env.episode_chromatic)
        
        # 更新代理统计
        agent.stats['episodes'] += 1
        
        # 定期打印进度
        if (episode + 1) % 20 == 0:
            # ... 进度打印逻辑 ...
            pass
    
    return env, agent, training_stats


def analyze_successful_sequences(agent, env):
    """分析成功序列的模式"""
    if not agent.success_sequences:
        print("没有成功序列可分析")
        return
    
    print("\n=== 成功序列分析 ===")
    
    # 统计操作频率
    op_counts = {}
    total_sequences = len(agent.success_sequences)
    
    for seq_data in agent.success_sequences:
        for action in seq_data['sequence']:
            op_name = env.operations[action]['name']
            op_counts[op_name] = op_counts.get(op_name, 0) + 1
    
    print("操作频率统计:")
    for op_name, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True):
        frequency = count / total_sequences
        print(f"  {op_name}: {count}次 ({frequency:.2f})")
    
    # 寻找常见模式（长度为2-3的子序列）
    pattern_counts_2 = {}
    pattern_counts_3 = {}
    
    for seq_data in agent.success_sequences:
        seq = seq_data['sequence']
        
        # 长度2的模式
        for i in range(len(seq) - 1):
            pattern = (env.operations[seq[i]]['name'], env.operations[seq[i+1]]['name'])
            pattern_counts_2[pattern] = pattern_counts_2.get(pattern, 0) + 1
        
        # 长度3的模式
        for i in range(len(seq) - 2):
            pattern = (env.operations[seq[i]]['name'], 
                      env.operations[seq[i+1]]['name'],
                      env.operations[seq[i+2]]['name'])
            pattern_counts_3[pattern] = pattern_counts_3.get(pattern, 0) + 1
    
    print("\n常见操作对(长度2):")
    for pattern, count in sorted(pattern_counts_2.items(), key=lambda x: x[1], reverse=True)[:10]:
        frequency = count / total_sequences
        print(f"  {pattern[0]} → {pattern[1]}: {count}次 ({frequency:.2f})")
    
    print("\n常见操作序列(长度3):")
    for pattern, count in sorted(pattern_counts_3.items(), key=lambda x: x[1], reverse=True)[:5]:
        frequency = count / total_sequences
        print(f"  {pattern[0]} → {pattern[1]} → {pattern[2]}: {count}次 ({frequency:.2f})")


def test_optimized_sequence(env, sequence_actions: List[int], name: str = "优化序列"):
    """测试优化后的序列"""
    print(f"\n=== 测试 {name} ===")
    
    # 执行序列
    env.reset('v31')
    total_reward = 0.0
    steps_info = []
    
    for i, action in enumerate(sequence_actions):
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps_info.append(info)
        
        print(f"步骤 {i+1}: {env.operations[action]['name']}")
        print(f"  结果: {info['message']}")
        print(f"  当前点数: {info['vertices']}")
        print(f"  步骤奖励: {reward:.2f}")
        
        if done:
            break
    
    # 最终评估
    if not done: # type: ignore
        final_reward = env._calculate_final_reward()
        total_reward += final_reward
    
    print(f"\n测试完成:")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  顶点数: {len(env.current_points)}") # type: ignore
    print(f"  色数: {env.episode_chromatic}")
    
    # 保存结果
    filename = f"results/test_{name.replace(' ', '_')}.txt"
    env.save_current_graph(filename)
    
    return total_reward, env.episode_chromatic


if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    random.seed(42)
    
    # 训练参数
    EPISODES = 500
    MAX_STEPS = 3
    START_STRUCTURE = 'v31'
    
    try:
        # 训练代理
        print("="*50)
        print("开始强化学习训练")
        print("="*50)
        
        env, agent, stats = train_agent(
            episodes=EPISODES,
            max_steps_per_episode=MAX_STEPS,
            start_structure=START_STRUCTURE,
            load_agent_path=None  # 可以指定已训练的代理路径
        )
        
        # 分析成功序列
        analyze_successful_sequences(agent, env)
        
        # 测试已知的优化序列
        known_sequences = {
            "Heule风格": [0, 3, 9],  # Minkowski_v31 → Rotate_theta4_merge → Filter_radius_2
            "精简构造": [0, 9, 10],  # Minkowski_v31 → Filter_radius_2 → Prune_low_degree_2
            "对称优化": [0, 5, 11],   # Minkowski_v31 → Rotate_60_merge → Prune_redundant
        }
        
        for seq_name, seq_actions in known_sequences.items():
            test_optimized_sequence(env, seq_actions, seq_name)
        
        print("\n✅ 训练完成！")
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
