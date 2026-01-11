import sympy as sp
import pickle
from typing import List, Tuple, Set, Optional
import numpy as np

def load_points_from_file(filename: str) -> List[Tuple[sp.Expr, sp.Expr]]:
    """
    从文本文件加载点集（格式：{x , y}），返回 sympy 表达式元组列表
    处理 sqrt() 转 sympy.sqrt
    """
    points = []
    sp.sqrt3 = sp.sqrt(3) # type: ignore
    sp.sqrt11 = sp.sqrt(11) # type: ignore
    sp.sqrt33 = sp.sqrt(33) # type: ignore
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            clean_line = line.strip().replace('{', '').replace('}', '').strip()
            if not clean_line:
                continue
            
            try:
                # 分割 x 和 y（兼容中英文逗号）
                if ',' in clean_line:
                    x_str, y_str = [part.strip() for part in clean_line.split(',')]
                else:
                    x_str, y_str = [part.strip() for part in clean_line.split('，')]
                
                # 将 sqrt() 替换为 sympy 格式
                x_str = x_str.replace('sqrt(', 'sp.sqrt(')
                y_str = y_str.replace('sqrt(', 'sp.sqrt(')
                
                # 解析为 sympy 表达式
                x_expr = sp.sympify(x_str, locals={'sp': sp})
                y_expr = sp.sympify(y_str, locals={'sp': sp})
                
                # 简化表达式
                x_expr = sp.nsimplify(x_expr, [sp.sqrt(3), sp.sqrt(11), sp.sqrt(33)])
                y_expr = sp.nsimplify(y_expr, [sp.sqrt(3), sp.sqrt(11), sp.sqrt(33)])
                
                points.append((x_expr, y_expr))
                
            except Exception as e:
                raise ValueError(f"第 {line_num} 行解析失败：{line}，错误：{str(e)}")
    
    if not points:
        raise ValueError("文件中未加载到任何有效点")
    
    print(f"成功加载 {len(points)} 个点")
    return points

class MinkowskiSumGenerator:
    """Minkowski和生成器"""
    
    def __init__(self, distance_cache: Optional[dict] = None):
        self.distance_cache = distance_cache or {}
    
    def minkowski_sum(self, 
                     points_a: List[Tuple[sp.Expr, sp.Expr]], 
                     points_b: List[Tuple[sp.Expr, sp.Expr]]) -> List[Tuple[sp.Expr, sp.Expr]]:
        """计算两个点集的Minkowski和，返回去重后的点集"""
        # 输入验证
        for idx, p in enumerate(points_a):
            if not isinstance(p, tuple) or len(p) != 2:
                raise TypeError(f"points_a 第 {idx} 个元素不是 (x, y) 元组：{p}")
        for idx, p in enumerate(points_b):
            if not isinstance(p, tuple) or len(p) != 2:
                raise TypeError(f"points_b 第 {idx} 个元素不是 (x, y) 元组：{p}")
        
        result_set = set()
        print(f"计算Minkowski和: |A|={len(points_a)}, |B|={len(points_b)}")
        
        for (x1, y1) in points_a:
            for (x2, y2) in points_b:
                x_sum = sp.simplify(x1 + x2) # type: ignore
                y_sum = sp.simplify(y1 + y2) # type: ignore
                
                x_norm = sp.nsimplify(x_sum, [sp.sqrt(3), sp.sqrt(11), sp.sqrt(33)])
                y_norm = sp.nsimplify(y_sum, [sp.sqrt(3), sp.sqrt(11), sp.sqrt(33)])
                
                result_set.add((x_norm, y_norm))
        
        result = list(result_set)
        print(f"Minkowski和生成 {len(result)} 个不同的点")
        return result
    
    def distance_to_origin(self, point: Tuple[sp.Expr, sp.Expr]) -> sp.Expr:
        """
        计算点到原点的距离平方（避免开方，提升精度）
        返回：x² + y² 的简化表达式
        """
        x, y = point
        dist_sq = sp.simplify(x**2 + y**2) # type: ignore
        return dist_sq
    
    def filter_points_by_radius(self, 
                               points: List[Tuple[sp.Expr, sp.Expr]], 
                               max_radius: float = 1.0) -> List[Tuple[sp.Expr, sp.Expr]]:
        """
        过滤掉距离原点 > max_radius 的点（默认过滤 > 1的点）
        :param points: 待过滤点集
        :param max_radius: 最大允许半径（默认1.0）
        :return: 过滤后的点集
        """
        filtered = []
        max_radius_sq = max_radius ** 2
        
        for idx, point in enumerate(points):
            dist_sq = self.distance_to_origin(point)
            
            # 方式1：精确符号比较
            try:
                # 检查 dist_sq <= max_radius_sq
                is_less = sp.simplify(dist_sq - max_radius_sq) <= 0 # type: ignore
                if is_less:
                    filtered.append(point)
                
            except:
                # 方式2：浮点兜底（符号比较失败时）
                dist_float = float(sp.N(dist_sq, 20)) ** 0.5
                if dist_float <= max_radius:
                    filtered.append(point)

        return filtered
    
    def distance_check(self, 
                       p1: Tuple[sp.Expr, sp.Expr], 
                       p2: Tuple[sp.Expr, sp.Expr]) -> bool:
        """距离检查"""
        x1, y1 = p1
        x2, y2 = p2
        
        try:
            x1_f = float(sp.N(x1, 20))
            y1_f = float(sp.N(y1, 20))
            x2_f = float(sp.N(x2, 20))
            y2_f = float(sp.N(y2, 20))
            
            dx, dy = x1_f - x2_f, y1_f - y2_f
            dist_sq_approx = dx*dx + dy*dy
            
            if abs(dist_sq_approx - 1.0) > 1e-8:
                return False
            
            dx_exact = x1 - x2 # type: ignore
            dy_exact = y1 - y2 # type: ignore
            dist_sq_exact = dx_exact**2 + dy_exact**2
            dist_sq_simplified = sp.simplify(dist_sq_exact)
            
            return sp.simplify(dist_sq_simplified - 1) == 0
        
        except Exception as e:
            print(f"距离检查失败（{p1} vs {p2}）：{str(e)}")
            return False

    def build_edge_table(self, points: List[Tuple[sp.Expr, sp.Expr]]) -> List[Tuple[int, int]]:
        """构建边表：连接所有距离为1的点对"""
        edge_table = []
        n = len(points)
        print(f"构建边表，检查 {n} 个点...")
        
        for i in range(n):
            if i % 50 == 0 and i > 0:
                print(f"  已处理 {i}/{n} 个点")
            for j in range(i+1, n):
                cache_key = tuple(sorted((i, j)))
                if cache_key in self.distance_cache:
                    if self.distance_cache[cache_key]:
                        edge_table.append((i, j))
                    continue
                
                is_unit = self.distance_check(points[i], points[j])
                self.distance_cache[cache_key] = is_unit
                if is_unit:
                    edge_table.append((i, j))
        
        print(f"找到 {len(edge_table)} 条边")
        return edge_table
    
    def save_points(self, points: List[Tuple[sp.Expr, sp.Expr]], filename: str):
        """保存点集到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            for idx, (x, y) in enumerate(points):
                x_simple = sp.nsimplify(x, [sp.sqrt(3), sp.sqrt(11), sp.sqrt(33)])
                y_simple = sp.nsimplify(y, [sp.sqrt(3), sp.sqrt(11), sp.sqrt(33)])
                
                x_str = str(x_simple).replace('sqrt', 'sqrt').replace('sp.', '')
                y_str = str(y_simple).replace('sqrt', 'sqrt').replace('sp.', '')
                
                f.write(f"{{ {x_str} , {y_str} }}\n")
        
        print(f"点集已保存到 {filename}")
    
    def save_edges(self, edge_table: List[Tuple[int, int]], filename: str):
        """保存边表到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            for i, j in edge_table:
                f.write(f"{i} {j}\n")
        print(f"边表已保存到 {filename}")
    
    def save_serialized(self, 
                       points: List[Tuple[sp.Expr, sp.Expr]], 
                       edge_table: List[Tuple[int, int]], 
                       base_filename: str):
        """保存序列化数据"""
        points_serializable = [(str(x), str(y)) for x, y in points]
        data = {
            'points': points_serializable,
            'edges': edge_table,
            'counts': {'points': len(points), 'edges': len(edge_table)}
        }
        
        if not base_filename.endswith('.txt'):
            base_filename += '.txt'
        pkl_filename = base_filename.replace('.txt', '_edges.pkl')
        
        with open(pkl_filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"序列化数据已保存到 {pkl_filename}")
    
    def process(self, input_file_a: str, input_file_b: str, output_file: str, max_radius: float = 1.0):
        """
        主处理流程
        """
        # 1. 加载点集
        base_points1 = load_points_from_file(input_file_a)
        base_points2 = load_points_from_file(input_file_b)
        
        # 2. 计算Minkowski和
        current_points = self.minkowski_sum(base_points1, base_points2)
        
        # 3. 过滤距离原点>max_radius的点
        print(f"\n开始过滤距离原点>{max_radius}的点...")
        current_points = self.filter_points_by_radius(current_points, max_radius)
        
        # 4. 构建边表（基于过滤后的点集）
        edge_table = self.build_edge_table(current_points)
        
        # 5. 保存结果
        self.save_points(current_points, output_file)
        base_name = output_file.replace('.txt', '') if output_file.endswith('.txt') else output_file
        self.save_edges(edge_table, f"{base_name}_edges.txt")
        self.save_serialized(current_points, edge_table, output_file)
        
        # 6. 统计信息
        print(f"\n===== 结果统计 =====")
        print(f"  过滤后点数: {len(current_points)}")
        print(f"  总边数: {len(edge_table)}")
        if current_points:
            avg_degree = 2 * len(edge_table) / len(current_points)
            print(f"  平均度数: {avg_degree:.2f}")
        
        return current_points, edge_table


if __name__ == "__main__":
    # 初始化生成器
    generator = MinkowskiSumGenerator()
    
    try:
        # 执行计算
        points, edges = generator.process(
            # 配置文件路径
            input_file_a= "structures/v31.txt",
            input_file_b= "structures/w_graph.txt",
            output_file= "v1999.txt",
            max_radius=2.0
        )
        print("\n✅ 计算完成！")
    except Exception as e:
        print(f"\n❌ 执行失败：{str(e)}")
        raise