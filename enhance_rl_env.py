# enhance_rl_env.py
import sympy as sp
import pickle
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Set
import random
from collections import deque, defaultdict
import time
import hashlib
import json

from main import GraphColoringSAT, UnitDistanceGraph
from minkovski import MinkowskiSumGenerator

class EnhanceRLEnv:
    """
    基于全局操作的增强强化学习环境，用于构造高色数单位距离图。
    核心特性：
    1. 自适应规模控制：操作前预估+操作后过滤，防止点数爆炸增长
    2. 度数导向奖励：显式鼓励构造稠密图结构
    3. 数学操作集：图集基于de Grey和Heule工作、以及RL迭代训练得到的中间结果

    状态空间：包含规模、度数、对称性等特征的9维向量
    动作空间：包含闵可夫斯基和、旋转合并、添加高对称结构等9种操作
    奖励函数：色数（主要）+ 平均度数 + 对称性 + 规模控制

    数学基础：所有点坐标保持在ℚ[√3, √5, √11]域中，确保精确算术。
    验证方法：SAT求解器严格验证色数。
    设计理念：通过数学上有意义的变换进行探索，而非随机修改。
    """
    
    def __init__(self, 
                 base_structures_path: str = "structures/",
                 max_steps_per_episode: int = 3,
                 max_vertices: int = 4000,
                 pruning_enabled: bool = True):
        
        # 环境参数
        self.max_steps_per_episode = max_steps_per_episode
        self.max_vertices = max_vertices
        self.pruning_enabled = pruning_enabled
        
        # 加载候选图
        self.base_structures = self._load_base_structures(base_structures_path)
        # Minkowski和生成器
        self.minkowski_gen = MinkowskiSumGenerator()
        
        # 当前状态
        self.current_points = None
        self.current_edges = None
        self.graph = None
        self.steps_taken = 0
        self.episode_reward = 0.0
        self.episode_chromatic = None
        self.episode_sequence = []  # 记录当前episode的操作序列
        
        # 历史记录
        self.history = deque(maxlen=100)
        self.best_chromatic = 0
        self.best_sequence = []
        
        # 操作统计
        self.op_stats = defaultdict(int)
        
        # 定义操作
        self.operations = self._define_operations()
        
        # 观察和动作空间
        self.observation_space = self._define_observation_space()
        self.action_space = self._define_action_space()
        
        print(f"环境初始化完成，包含 {len(self.operations)} 种操作")
        print(f"最大顶点数: {max_vertices}")
    
    def _load_base_structures(self, path: str) -> Dict[str, List[Tuple[sp.Expr, sp.Expr]]]:
        """加载候选图结构"""
        structures = {}
        
        # 候选图结构列表
        key_structures = ['v31', 'v151', 'v1939', 'S199', 'S397','w_graph']
        
        for name in key_structures:
            try:
                filename = f"{path}/{name}.txt"
                with open(filename, 'r') as f:
                    points = []
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            line = line.strip().replace('{', '').replace('}', '')
                            x_str, y_str = [s.strip() for s in line.split(',')]
                            
                            x_str = x_str.replace('Sqrt[', 'sqrt(').replace(']', ')')
                            y_str = y_str.replace('Sqrt[', 'sqrt(').replace(']', ')')
                            
                            try:
                                x = sp.sympify(x_str)
                                y = sp.sympify(y_str)
                                points.append((x, y))
                            except:
                                pass
                    
                    if points:
                        structures[name] = points
                        print(f"加载 {name}: {len(points)} 个点")
            except Exception as e:
                print(f"加载 {name} 失败: {e}")
        
        return structures
    
    def _define_operations(self) -> List[Dict[str, Any]]:
        """定义操作集合"""
        operations = [
            # Minkowski求和操作
            {
                'name': 'Minkovski_sum',
                'description': '当前图与自身求Minkowski和',
                'function': self._op_minkovski_sum,
                'weight': 1.0,
            },
            {
                'name': 'Minkowski_v31',
                'description': '当前图与v31求Minkowski和',
                'function': self._op_minkowski_v31,
                'weight': 1.0,
            },

            # 旋转合并操作
            {
                'name': 'Rotate_theta3_merge',
                'description': '当前图旋转θ₃后与自身合并',
                'function': self._op_rotate_theta3,
                'weight': 1.5,
            },
            {
                'name': 'Rotate_theta4_merge',
                'description': '当前图旋转θ₄后与自身合并',
                'function': self._op_rotate_theta4,
                'weight': 1.5,
            },
            {
                'name': 'Rotate_60_merge',
                'description': '当前图旋转60度后与自身合并',
                'function': self._op_rotate_60,
                'weight': 1.2,
            },
            {
                'name': 'Rotate_120_merge',
                'description': '当前图旋转120度后与自身合并',
                'function': self._op_rotate_120,
                'weight': 1.0,
            },
            
            # 添加结构操作
            {
                'name': 'Add_S199',
                'description': '添加S199结构',
                'function': self._op_add_s199,
                'weight': 1.4,
            },
            
            # 过滤和删点操作
            {
                'name': 'Filter_radius',
                'description': '过滤距离原点>sqrt(3)的点',
                'function': self._op_filter_radius,
                'weight': 1.0,
            },
            {
                'name': 'Prune_low_degree',
                'description': '删除度数<=4的顶点',
                'function': self._op_prune_low_degree,
                'weight': 1.0,
            },
        ]
        
        return operations
    
    def _apply_operation(self, operation_idx: int) -> Tuple[bool, str, Optional[List[Tuple[sp.Expr, sp.Expr]]]]:
        """执行操作"""
        if not self.current_points:
            return False, "当前图为空", None
        
        operation = self.operations[operation_idx]
        
        # 执行操作
        success, message = operation['function']()
        new_points = self.current_points if success else None

        return success, message, new_points
    
    # ==================== 操作实现 ====================
    def _op_minkovski_sum(self) -> Tuple[bool, str]:
        """操作: 与自身求Minkowski和"""
        if not self.current_points:
            return False, "当前图为空"
        
        # 生成Minkowski和
        new_points = self.minkowski_gen.minkowski_sum(self.current_points, self.current_points)
        
        # 过滤距离>2的点
        filtered = self.minkowski_gen.filter_points_by_radius(new_points, max_radius=2.0)
        
        self.current_points = filtered
        self.current_edges = None
        self.graph = None
        
        return True, f"Minkowski和: {len(self.current_points)}点"

    def _op_minkowski_v31(self) -> Tuple[bool, str]:
        """操作: 当前图与v31的Minkowski和"""
        if 'v31' not in self.base_structures:
            return False, "v31未加载"
        
        if not self.current_points:
            return False, "当前图为空"
        
        v31 = self.base_structures['v31']
        new_points = self.minkowski_gen.minkowski_sum(self.current_points, v31)
        
        # 过滤距离>2的点
        filtered = self.minkowski_gen.filter_points_by_radius(new_points, max_radius=2.0)
        
        self.current_points = filtered
        self.current_edges = None
        self.graph = None
        
        return True, f"Minkowski(v31): {len(self.current_points)}点"
    
    def _op_rotate_theta3(self) -> Tuple[bool, str]:
        """操作: 旋转θ₃后与自身合并"""
        if not self.current_points:
            return False, "当前图为空"
        
        cos_theta = sp.Rational(5, 6)
        sin_theta = sp.sqrt(1 - cos_theta**2) # type: ignore
        
        rotated_points = []
        for x, y in self.current_points:
            x_rot = sp.simplify(x * cos_theta - y * sin_theta) # type: ignore
            y_rot = sp.simplify(x * sin_theta + y * cos_theta) # type: ignore
            rotated_points.append((x_rot, y_rot))
        
        merged = self._merge_points(self.current_points, rotated_points)
        filtered = self.minkowski_gen.filter_points_by_radius(merged, max_radius=2.0)
        
        self.current_points = filtered
        self.current_edges = None
        self.graph = None
        
        return True, f"旋转θ₃合并: {len(self.current_points)}点"
    
    def _op_rotate_theta4(self) -> Tuple[bool, str]:
        """操作: 旋转θ₄后与自身合并"""
        if not self.current_points:
            return False, "当前图为空"
        
        cos_theta = sp.Rational(7, 8)
        sin_theta = sp.sqrt(1 - cos_theta**2) # type: ignore
        
        rotated_points = []
        for x, y in self.current_points:
            x_rot = sp.simplify(x * cos_theta - y * sin_theta) # type: ignore
            y_rot = sp.simplify(x * sin_theta + y * cos_theta) # type: ignore
            rotated_points.append((x_rot, y_rot))
        
        merged = self._merge_points(self.current_points, rotated_points)
        filtered = self.minkowski_gen.filter_points_by_radius(merged, max_radius=2.0)
        
        self.current_points = filtered
        self.current_edges = None
        self.graph = None
        
        return True, f"旋转θ₄合并: {len(self.current_points)}点"
    
    def _op_add_s199(self) -> Tuple[bool, str]:
        """操作: 添加S199结构"""
        if 'S199' not in self.base_structures:
            return False, "S199未加载"
        
        S199 = self.base_structures['S199']
        merged = self._merge_points(self.current_points or [], S199)
        filtered = self.minkowski_gen.filter_points_by_radius(merged, max_radius=2.0)
        
        self.current_points = filtered
        self.current_edges = None
        self.graph = None
        
        return True, f"添加S199: {len(self.current_points)}点"
    
    def _op_rotate_60(self) -> Tuple[bool, str]:
        """操作: 旋转60度后与自身合并"""
        if not self.current_points:
            return False, "当前图为空"
        
        cos_60 = sp.Rational(1, 2)
        sin_60 = sp.sqrt(3) / 2 # type: ignore
        
        rotated_points = []
        for x, y in self.current_points:
            x_rot = sp.simplify(x * cos_60 - y * sin_60) # type: ignore
            y_rot = sp.simplify(x * sin_60 + y * cos_60) # type: ignore
            rotated_points.append((x_rot, y_rot))
        
        merged = self._merge_points(self.current_points, rotated_points)
        filtered = self.minkowski_gen.filter_points_by_radius(merged, max_radius=2.0)
        
        self.current_points = filtered
        self.current_edges = None
        self.graph = None
        
        return True, f"旋转60度合并: {len(self.current_points)}点"
    
    def _op_rotate_120(self) -> Tuple[bool, str]:
        """操作: 旋转120度后与自身合并"""
        if not self.current_points:
            return False, "当前图为空"
        
        cos_120 = sp.Rational(-1, 2)
        sin_120 = sp.sqrt(3) / 2 # type: ignore
        
        rotated_points = []
        for x, y in self.current_points:
            x_rot = sp.simplify(x * cos_120 - y * sin_120) # type: ignore
            y_rot = sp.simplify(x * sin_120 + y * cos_120) # type: ignore
            rotated_points.append((x_rot, y_rot))
        
        merged = self._merge_points(self.current_points, rotated_points)
        filtered = self.minkowski_gen.filter_points_by_radius(merged, max_radius=2.0)
        
        self.current_points = filtered
        self.current_edges = None
        self.graph = None
        
        return True, f"旋转120度合并: {len(self.current_points)}点"
    
    def _op_filter_radius(self) -> Tuple[bool, str]:
        """操作: 过滤距离原点>sqrt(3)的点"""
        if not self.current_points:
            return False, "当前图为空"
        
        old_len = len(self.current_points)
        filtered = self.minkowski_gen.filter_points_by_radius(self.current_points, max_radius=sp.sqrt(3)) # type: ignore
        
        if len(filtered) == old_len:
            return False, f"过滤未删除任何点"
        
        self.current_points = filtered
        self.current_edges = None
        self.graph = None
        
        return True, f"过滤半径>sqrt(3): {old_len} → {len(filtered)}点"
    
    def _compute_vertex_degrees(self) -> List[int]:
        """计算每个顶点的度数"""
        if not self.current_points:
            return []
        
        if self.current_edges is None:
            self.current_edges = self.minkowski_gen.build_edge_table(self.current_points) # type: ignore
        
        degrees = [0] * len(self.current_points) # type: ignore
        for i, j in self.current_edges:
            degrees[i] += 1
            degrees[j] += 1
        return degrees
    
    def _op_prune_low_degree(self) -> Tuple[bool, str]:
        """删除度数≤4的顶点"""
        if not self.current_points or len(self.current_points) < 10:
            return False, "图太小不适合删点"
        
        degrees = self._compute_vertex_degrees()
        old_len = len(self.current_points)
        
        # 找出度数≤4的顶点索引
        low_degree_indices = [i for i, deg in enumerate(degrees) if deg <= 4]
        
        if not low_degree_indices:
            return False, "没有度数≤4的顶点"
        
        # 删除这些顶点
        new_points = []
        keep_indices = []
        
        for i, point in enumerate(self.current_points):
            if i not in low_degree_indices:
                new_points.append(point)
                keep_indices.append(i)
        
        # 重新构建边表（只保留原有边中两个端点都保留的边）
        if self.current_edges:
            new_edges = []
            for i, j in self.current_edges:
                if i not in low_degree_indices and j not in low_degree_indices:
                    # 重新映射索引
                    new_i = keep_indices.index(i)
                    new_j = keep_indices.index(j)
                    new_edges.append((new_i, new_j))
            self.current_edges = new_edges
        else:
            self.current_edges = None
        
        self.current_points = new_points
        self.graph = None
        
        return True, f"删除度数≤4的顶点: {old_len} → {len(new_points)}点"
    
    def _merge_points(self, points1: List[Tuple[sp.Expr, sp.Expr]], 
                     points2: List[Tuple[sp.Expr, sp.Expr]]) -> List[Tuple[sp.Expr, sp.Expr]]:
        """合并两个点集并去重"""
        if not points1:
            return points2
        if not points2:
            return points1
        
        merged = []
        seen = set()
        
        for points in [points1, points2]:
            for x, y in points:
                x_norm = sp.nsimplify(x, [sp.sqrt(3), sp.sqrt(11), sp.sqrt(33)])
                y_norm = sp.nsimplify(y, [sp.sqrt(3), sp.sqrt(11), sp.sqrt(33)])
                
                key = f"{x_norm},{y_norm}"
                if key not in seen:
                    seen.add(key)
                    merged.append((x_norm, y_norm))
        
        return merged
    
    def _build_unit_distance_graph(self):
        """构建UnitDistanceGraph对象"""
        if self.current_points is None or len(self.current_points) == 0:
            return None
        
        if self.current_edges is None:
            self.current_edges = self.minkowski_gen.build_edge_table(self.current_points)
        
        complex_points = []
        for x, y in self.current_points:
            point_expr = x + sp.I * y
            complex_points.append(point_expr)
        
        self.graph = UnitDistanceGraph(complex_points)
        
        if hasattr(self.graph, 'edge_list'):
            self.graph.edge_list = self.current_edges # type: ignore
        elif hasattr(self.graph, 'edges'):
            self.graph.edges = list(self.current_edges) # type: ignore
        
        return self.graph
    
    def _evaluate_chromatic_number_sat(self) -> int:
        """使用SAT求解器评估色数"""
        print(f"开始SAT评估色数，图有 {len(self.current_points)} 个顶点...") # type: ignore
        start_time = time.time()
        
        self._build_unit_distance_graph()
        if self.graph is None:
            print("图对象构建失败")
            return 3

        try:
            # 检查3-可着色性
            sat_3 = GraphColoringSAT(num_colors=3)
            is_3_colorable = sat_3.check_colorability(self.graph)
            print(f"3-可着色: {is_3_colorable}")
            
            if is_3_colorable:
                chromatic = 3
            else:
                # 检查4-可着色性
                sat_4 = GraphColoringSAT(num_colors=4)
                is_4_colorable = sat_4.check_colorability(self.graph)
                print(f"4-可着色: {is_4_colorable}")
                
                if is_4_colorable:
                    chromatic = 4
                else:
                    # 检查5-可着色性
                    sat_5 = GraphColoringSAT(num_colors=5)
                    is_5_colorable = sat_5.check_colorability(self.graph)
                    print(f"5-可着色: {is_5_colorable}")
                    
                    if is_5_colorable:
                        chromatic = 5
                    else:
                        chromatic = 6

            elapsed = time.time() - start_time
            print(f"SAT评估完成: 色数 = {chromatic}, 耗时: {elapsed:.2f}秒")
            
            return chromatic
            
        except Exception as e:
            print(f"SAT评估出错: {e}")
            return 0
    
    def _calculate_symmetry_score(self) -> float:
        """计算对称性分数"""
        if not self.current_points or len(self.current_points) < 3:
            return 0.0
        
        numerical_points = []
        for x, y in self.current_points:
            try:
                x_val = float(sp.N(x, 15))
                y_val = float(sp.N(y, 15))
                numerical_points.append((x_val, y_val))
            except:
                numerical_points.append((0.0, 0.0))
        
        center_x = sum(p[0] for p in numerical_points) / len(numerical_points)
        center_y = sum(p[1] for p in numerical_points) / len(numerical_points)
        
        symmetry_scores = []
        test_angles = [60, 120, 180, 240, 300]
        
        for angle_deg in test_angles:
            angle_rad = np.deg2rad(angle_deg)
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)
            
            rotated_points = []
            for x, y in numerical_points:
                x_centered = x - center_x
                y_centered = y - center_y
                x_rot = x_centered * cos_angle - y_centered * sin_angle + center_x
                y_rot = x_centered * sin_angle + y_centered * cos_angle + center_y
                rotated_points.append((x_rot, y_rot))
            
            match_count = 0
            threshold = 1e-4
            
            for i, (x1, y1) in enumerate(numerical_points):
                min_dist = float('inf')
                for j, (x2, y2) in enumerate(rotated_points):
                    if i == j:
                        continue
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < min_dist:
                        min_dist = dist
                
                if min_dist < threshold:
                    match_count += 1
            
            symmetry_score = match_count / len(numerical_points)
            symmetry_scores.append(symmetry_score)
        
        avg_symmetry = np.mean(symmetry_scores) if symmetry_scores else 0.0
        
        # 检查特定对称性
        if len(symmetry_scores) > 0:
            symmetry_60 = symmetry_scores[0]
            if symmetry_60 > 0.7:
                avg_symmetry = max(avg_symmetry, symmetry_60 * 1.2)
        
        return float(avg_symmetry)
    
    def _define_action_space(self):
        """定义动作空间"""
        try:
            from gym import spaces
            return spaces.Discrete(len(self.operations))
        except ImportError:
            class SimpleDiscrete:
                def __init__(self, n):
                    self.n = n
                def sample(self):
                    return random.randint(0, self.n-1)
            return SimpleDiscrete(len(self.operations))
    
    def _define_observation_space(self):
        """定义观察空间"""
        try:
            from gym import spaces
            obs_length = 9  # 观察维度
            return spaces.Box(low=0.0, high=1.0, shape=(obs_length,), dtype=np.float32)
        except ImportError:
            class SimpleBox:
                def __init__(self, shape):
                    self.shape = shape
                def sample(self):
                    return np.random.uniform(0, 1, self.shape)
            return SimpleBox((9,))
    
    def _estimate_vertices(self, operation_idx: int) -> int:
        """预估执行操作后可能产生的顶点数"""
        if not self.current_points:
            return 0
        
        operation = self.operations[operation_idx]
        op_name = operation['name']
        current_size = len(self.current_points)
        
        # 基于操作类型的预估规则
        estimation_rules = {
            'Minkovski_sum': current_size ** 2 * 0.1,
            'Minkowski_v31': current_size * len(self.base_structures.get('v31', [])) * 0.1,
            'Rotate_theta3_merge': current_size * 2,
            'Rotate_theta4_merge': current_size * 2,
            'Rotate_60_merge': current_size * 2,
            'Rotate_120_merge': current_size * 2,
            'Add_S199': current_size + len(self.base_structures.get('S199', [])),
            'Filter_radius': current_size * 0.7,
            'Prune_low_degree': current_size * 0.8,
        }
        
        estimated = estimation_rules.get(op_name, current_size * 1.5)
        return int(estimated)

    def _pre_operation_pruning(self, estimated_size: int) -> bool:
        """如果预估点数过大，先进行删点"""
        threshold = self.max_vertices * 1.25  # 超过最大顶点数的1/4时开始删点
        
        if estimated_size > threshold:
            print(f"预估计点{estimated_size} > 阈值{threshold}，执行预删点...")
            
            # 执行多次删点操作直到规模可控
            old_size = len(self.current_points) # type: ignore
            pruning_attempts = 0
            
            while len(self.current_points) > threshold * 0.7 and pruning_attempts < 3: # type: ignore
                success, msg = self._op_prune_low_degree()
                if not success:
                    success, msg = self._op_filter_radius()
                pruning_attempts += 1
            
            new_size = len(self.current_points) # type: ignore
            print(f"预删点完成: {old_size} → {new_size}")
            
            return True
        return False
    
    def reset(self, start_structure: str = 'v31') -> np.ndarray:
        """重置环境"""
        if start_structure in self.base_structures:
            self.current_points = self.base_structures[start_structure].copy()
        else:
            self.current_points = self.base_structures.get('v31', []).copy()
        
        self.current_edges = None
        self.graph = None
        self.steps_taken = 0
        self.episode_reward = 0.0
        self.last_symmetry_score = 0.0
        self.episode_chromatic = None
        self.episode_sequence = []
        self.last_vertex_count = len(self.current_points)

        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步操作"""
        if action < 0 or action >= len(self.operations):
            return self._get_observation(), -10.0, True, {"error": f"无效动作: {action}"}
        # 预估操作后规模
        estimated_size = self._estimate_vertices(action)
    
        # 如果预估点数过大，先进行删点
        self._pre_operation_pruning(estimated_size)

        # 记录操作
        self.episode_sequence.append(action)
        self.op_stats[self.operations[action]['name']] += 1
        
        # 执行操作
        success, message, new_points = self._apply_operation(action)
        
        if not success:
            # 操作失败，小惩罚
            reward = -2.0
            done = False
            info = {
                "status": "操作失败",
                "message": message,
                "operation": self.operations[action]['name']
            }
            return self._get_observation(), reward, done, info
        
        self.steps_taken += 1
        
        # 操作后执行过滤删点
        self._op_filter_radius()
        self._op_prune_low_degree()

        # 检查顶点数限制
        if len(self.current_points) > self.max_vertices: # type: ignore
            done = True
            reward = -10.0
            info = {
                "status": "超出最大顶点数",
                "vertices": len(self.current_points), # type: ignore
                "operation": self.operations[action]['name'],
                "message": message
            }
            return self._get_observation(), reward, done, info
        
        # 检查是否达到最大步数
        done = (self.steps_taken >= self.max_steps_per_episode)
        
        # 计算奖励
        if not done:
            reward = self._calculate_intermediate_reward()
        else:
            reward = self._calculate_final_reward()
            self.episode_reward = reward
        
        info = {
            "step": self.steps_taken,
            "operation": self.operations[action]['name'],
            "vertices": len(self.current_points), # type: ignore
            "message": message,
            "done": done,
            "reward": reward,
        }
        
        return self._get_observation(), reward, done, info
    
    def _calculate_intermediate_reward(self) -> float:
        """计算中间步骤的奖励"""
        reward = 0.0
        
        if self.current_points:
            n_points = len(self.current_points)
            n_edges = len(self.current_edges) # type: ignore
            self.last_symmetry_score = self._calculate_symmetry_score()

            # 平均度数奖励
            avg_degree = n_edges * 2 / n_points if n_points > 0 else 0
            reward += avg_degree * 0.5

            # 规模奖励
            if n_points > 100:
                reward += 0.5
            if n_points > 500:
                reward += 1.0
            
            # 惩罚过大
            if n_points > 2000:
                reward -= 2.0
            
            # 顶点数变化奖励（鼓励删点）
            if hasattr(self, 'last_vertex_count') and self.last_vertex_count: # type: ignore
                delta = self.last_vertex_count - n_points
                if delta > 0:
                    reward += delta * 0.1  # 删除顶点奖励
                elif delta < 0:
                    reward += delta * 0.05  # 添加顶点惩罚（较小）
            
            self.last_vertex_count = n_points
        
        # 操作多样性奖励
        reward += 0.1
        
        return reward
    
    def _calculate_final_reward(self) -> float:
        """计算episode结束时的最终奖励"""
        if not self.current_points:
            return -20.0
        
        # 构建图
        if self.current_edges is None:
            self.current_edges = self.minkowski_gen.build_edge_table(self.current_points)
        
        # SAT验证色数
        self.episode_chromatic = self._evaluate_chromatic_number_sat()
        chromatic = self.episode_chromatic

        # 基础奖励
        reward = 0.0
        
        # 色数奖励
        if chromatic == 5:
            reward = 1000.0
            reward -= len(self.current_points) * 0.1
            
            if chromatic > self.best_chromatic:
                self.best_chromatic = chromatic
                self.best_sequence = self.episode_sequence.copy()
                print(f"🎉 找到新的最佳色数: {chromatic}")
        elif chromatic == 4:
            reward = 100.0
            reward -= len(self.current_points) * 0.05
        elif chromatic == 3:
            reward = 10.0
            reward -= len(self.current_points) * 0.02
        else:
            reward = -len(self.current_points) * 0.1
        
        # 平均度数奖励
        n_points = len(self.current_points)
        n_edges = len(self.current_edges)
        avg_degree = n_edges * 2 / n_points
        reward += avg_degree * 10.0

        # 对称性奖励
        symmetry_score = self._calculate_symmetry_score()
        reward += symmetry_score * 50.0
        
        # 惩罚过大的图
        if len(self.current_points) > 3000:
            reward -= 100.0
        elif len(self.current_points) > 2000:
            reward -= 50.0
        
        # 归一化奖励
        reward = min(max(reward, -100.0), 2000.0)
        
        print(f"最终奖励: {reward:.2f} (色数: {chromatic}, 顶点数: {len(self.current_points)})") # type: ignore
        print(f"对称性分数: {symmetry_score:.3f}")
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """获取观察值"""
        features = np.zeros(9, dtype=np.float32)
        
        if self.current_points:
            n_points = len(self.current_points)
            
            # 1. 规模特征
            features[0] = min(n_points / self.max_vertices, 1.0)
            
            # 2. 步数进度
            features[1] = self.steps_taken / self.max_steps_per_episode
            
            # 3. 密度特征
            if n_points > 1 and self.current_edges:
                max_edges = n_points * (n_points - 1) / 2
                density = len(self.current_edges) / max_edges if max_edges > 0 else 0
                features[2] = density
            
            # 4. 平均度数
            if n_points > 0 and self.current_edges:
                avg_degree = len(self.current_edges) * 2 / n_points
                features[3] = min(avg_degree / 20.0, 1.0)
            
            # 5. 对称性特征
            if hasattr(self, 'last_symmetry_score'):
                features[4] = self.last_symmetry_score
            else:
                features[4] = 0.0
            
            # 6. 操作多样性
            if self.episode_sequence:
                unique_ops = len(set(self.episode_sequence))
                features[5] = unique_ops / len(self.operations)
            
            # 7. 历史奖励趋势
            if len(self.history) > 0:
                last_rewards = [h['reward'] for h in list(self.history)[-5:]]
                avg_last_reward = np.mean(last_rewards) if last_rewards else 0
                features[6] = (avg_last_reward + 100) / 200
            
            # 8. 顶点数变化趋势
            if hasattr(self, 'last_vertex_count') and self.last_vertex_count and n_points > 0:
                features[7] = (self.last_vertex_count - n_points) / self.last_vertex_count
            
            # 9. 操作权重特征
            if self.episode_sequence:
                last_op = self.episode_sequence[-1] if self.episode_sequence else 0
                features[8] = self.operations[last_op]['weight']
        
        return features
    
    def get_operation_stats(self) -> Dict:
        """获取操作统计"""
        total_ops = sum(self.op_stats.values())
        
        stats = {}
        for op_name, count in self.op_stats.items():
            freq = count / total_ops if total_ops > 0 else 0
            stats[op_name] = {'count': count, 'frequency': freq}
        
        return stats
    
    def save_current_graph(self, filename: str):
        """保存当前图到文件"""
        if not self.current_points:
            print("当前图为空")
            return
        
        with open(filename, 'w') as f:
            f.write(f"# 生成图，共 {len(self.current_points)} 个点\n")
            f.write(f"# 操作序列: {[self.operations[a]['name'] for a in self.episode_sequence]}\n")
            f.write(f"# 色数: {self.episode_chromatic}\n")
            
            for x, y in self.current_points:
                x_str = str(x).replace('sqrt', 'Sqrt')
                y_str = str(y).replace('sqrt', 'Sqrt')
                f.write(f"{{ {x_str} , {y_str} }}\n")
        
        if self.current_edges:
            edge_file = filename.replace('.txt', '_edges.txt')
            with open(edge_file, 'w') as f:
                f.write(f"# 边表，共 {len(self.current_edges)} 条边\n")
                for i, j in self.current_edges:
                    f.write(f"{i} {j}\n")
        
        # 保存统计信息
        info_file = filename.replace('.txt', '_info.txt')
        with open(info_file, 'w') as f:
            f.write(f"图信息:\n")
            f.write(f"顶点数: {len(self.current_points)}\n")
            f.write(f"边数: {len(self.current_edges) if self.current_edges else 0}\n")
            f.write(f"色数: {self.episode_chromatic}\n")
            f.write(f"操作序列: {[self.operations[a]['name'] for a in self.episode_sequence]}\n")
            f.write(f"对称性分数: {self._calculate_symmetry_score()}\n")
        
        print(f"图已保存到: {filename}")