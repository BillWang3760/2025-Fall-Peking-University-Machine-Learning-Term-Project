import numpy as np
from typing import Dict, List, Tuple
import random
import pickle
from collections import deque

class EnhanceRLAgent:
    """
    增强RL代理
    使用表格型Q-learning学习图构造操作序列
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 memory_size: int = 10000):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-learning参数
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q表（状态-动作价值）
        self.q_table = {}
        
        # 经验回放（使用deque提高效率）
        self.memory = deque(maxlen=memory_size)
        
        # 训练统计
        self.stats = {
            'episodes': 0,
            'steps': 0,
            'best_reward': -float('inf'),
            'best_chromatic': 0,
            'success_rate': 0.0
        }
        
        # 成功序列记录
        self.success_sequences = []
    
    def get_state_key(self, state: np.ndarray) -> str:
        """将状态转换为Q表的键"""
        discretized = []
        
        # 特征离散化策略
        discretization_levels = [5, 5, 5, 5, 5, 10, 10, 5, 5]  # 不同特征不同粒度
        
        for i, val in enumerate(state):
            if i >= len(discretization_levels):
                levels = 5  # 默认粒度
            else:
                levels = discretization_levels[i]
            
            # 确保值在合理范围内
            val_clipped = np.clip(float(val), 0.0, 1.0)
            
            # 离散化
            disc = int(val_clipped * levels)
            disc = min(disc, levels - 1)  # 确保不超过最大索引
            disc = max(disc, 0)  # 确保不小于0
            
            discretized.append(str(disc))
        
        return ','.join(discretized)
    
    def choose_action(self, state: np.ndarray) -> int:
        """选择动作"""
        state_key = self.get_state_key(state)
        
        # 确保状态在Q表中
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in range(self.action_dim)}
        
        # ε-greedy策略
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            action = random.randint(0, self.action_dim - 1)
        else:
            # 利用：选择Q值最大的动作
            q_values = self.q_table[state_key]
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = random.choice(best_actions)
        
        return action
    
    def update_q_table(self, 
                      state: np.ndarray, 
                      action: int, 
                      reward: float, 
                      next_state: np.ndarray, 
                      done: bool):
        """更新Q表"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # 初始化Q值如果状态不存在
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in range(self.action_dim)}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in range(self.action_dim)}
        
        # Q-learning更新规则
        current_q = self.q_table[state_key][action]
        
        if done:
            # 终止状态
            target = reward
        else:
            # 非终止状态
            next_max_q = max(self.q_table[next_state_key].values())
            target = reward + self.gamma * next_max_q
        
        # 更新Q值
        self.q_table[state_key][action] = current_q + self.learning_rate * (target - current_q)
        
        # 更新统计
        self.stats['steps'] += 1
    
    def remember(self, 
                state: np.ndarray, 
                action: int, 
                reward: float, 
                next_state: np.ndarray, 
                done: bool):
        """存储经验（使用deque）"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size: int = 32):
        """经验回放"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            self.update_q_table(state, action, reward, next_state, done)
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def record_success(self, sequence: List[int], reward: float, chromatic: int):
        """记录成功序列"""
        self.success_sequences.append({
            'sequence': sequence.copy(),
            'reward': reward,
            'chromatic': chromatic,
            'episode': self.stats['episodes']
        })
        
        if reward > self.stats['best_reward']:
            self.stats['best_reward'] = reward
            self.stats['best_chromatic'] = chromatic
    
    def save(self, filename: str):
        """保存代理"""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'stats': self.stats,
            'success_sequences': self.success_sequences
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"代理已保存到: {filename}")
    
    def load(self, filename: str):
        """加载代理"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = data['q_table']
        self.epsilon = data['epsilon']
        self.stats = data['stats']
        self.success_sequences = data.get('success_sequences', [])
        
        print(f"代理已从 {filename} 加载")
    
    def get_best_sequence(self) -> List[int]:
        """获取最佳序列"""
        if not self.success_sequences:
            return []
        
        # 按奖励排序
        sorted_sequences = sorted(self.success_sequences, key=lambda x: x['reward'], reverse=True)
        return sorted_sequences[0]['sequence']