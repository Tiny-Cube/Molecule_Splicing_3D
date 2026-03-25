"""
获取分子骨架合并位点 (getBackboneMergedPoint.py)

本模块提供分子结合位点识别功能，用于自动找出分子上适合作为连接位点的原子。
潜在结合位点定义：与H原子相连的非H原子（X-H键）。

依赖：RDKit (>=2025.9.3)
"""

from rdkit import Chem
from rdkit.Geometry import Point3D
import numpy as np


def load_mol(path: str, removeHs: bool = False, sanitize: bool = True):
    """
    加载分子文件（仅支持.mol格式）

    参数：
        path (str): 分子文件路径（.mol）
        removeHs (bool): 是否移除H原子，默认False（保留H）
        sanitize (bool): 是否sanitize分子，默认True

    返回：
        mol: RDKit分子对象，失败返回None
    """
    if not path.endswith(".mol"):
        raise ValueError(f"仅支持.mol格式，当前文件：{path}")
    mol = Chem.MolFromMolFile(path, removeHs=removeHs, sanitize=sanitize)
    return mol


def find_potential_merging_sites(mol: Chem.Mol) -> dict:
    """
    查找潜在的可合并位点（适合形成新化学键的原子）

    位点定义：与H原子相连的非H原子（X-H键），即可以断裂形成新键的位置

    参数：
        mol: RDKit分子对象

    返回：
        dict: {
            "c_sites": [(idx, [h_idx1, h_idx2, ...]), ...],  # 碳原子及邻居H索引
            "n_sites": [(idx, [h_idx1, h_idx2, ...]), ...]    # 氮原子及邻居H索引
        }
    """
    if mol is None:
        return {"c_sites": [], "n_sites": []}

    c_sites = []
    n_sites = []

    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)

        # 跳过H原子
        if atom.GetAtomicNum() == 1:
            continue

        # 获取邻居H原子索引
        h_neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]

        # 必须与至少一个H原子相连
        if not h_neighbors:
            continue

        # 计算不含H的配位数（GetDegree包含显式H，需减去）
        explicit_h = len(h_neighbors)
        degree = atom.GetDegree() - explicit_h

        atomic_num = atom.GetAtomicNum()

        if atomic_num == 6:  # 碳原子
            if degree <= 3 and degree > 1:
                c_sites.append((idx, h_neighbors))

        elif atomic_num == 7:  # 氮原子
            if degree == 2:
                n_sites.append((idx, h_neighbors))

    return {"c_sites": c_sites, "n_sites": n_sites}


def get_symm_sites(mol: Chem.Mol) -> dict:
    """
    按对称性分组查找潜在结合位点（碳和氮原子）

    使用RDKit的CanonicalRankAtoms获取原子的对称序，
    对称序相同的原子在化学环境上完全等价。

    参数：
        mol: RDKit分子对象

    返回：
        dict: {
            "c_symm_dict": [[idx1, idx2, ...], ...],  # 按对称性分组的碳原子列表
            "n_symm_dict": [[idx1, idx2, ...], ...]    # 按对称性分组的氮原子列表
        }
        每个内部列表包含互为对称的原子索引
    """
    if mol is None:
        return {"c_symm_dict": [], "n_symm_dict": []}

    # 获取所有原子的canonical rank（对称序）
    canon_ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))

    # 初始化对称性字典
    c_symm_dict = {}  # {对称序: [原子索引列表]}
    n_symm_dict = {}

    # 按对称序分组
    for idx, order in enumerate(canon_ranks):
        atom = mol.GetAtomWithIdx(idx)
        atomic_num = atom.GetAtomicNum()

        # 检查是否连接H原子（显式H或隐式H）
        has_h = any(n.GetAtomicNum() == 1 for n in atom.GetNeighbors())

        # 计算不含H的配位数（GetDegree包含显式H，需减去）
        explicit_h = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
        degree_no_h = atom.GetDegree() - explicit_h

        if atomic_num == 6:  # 碳原子
            if degree_no_h <= 3 and degree_no_h > 1 and has_h:
                if order not in c_symm_dict:
                    c_symm_dict[order] = []
                c_symm_dict[order].append(idx)

        elif atomic_num == 7:  # 氮原子
            if degree_no_h == 2 and has_h:
                if order not in n_symm_dict:
                    n_symm_dict[order] = []
                n_symm_dict[order].append(idx)

    # 转换为列表（去除空列表），按对称序排序
    return {
        "c_symm_dict": [v for v in sorted(c_symm_dict.values()) if v],
        "n_symm_dict": [v for v in sorted(n_symm_dict.values()) if v]
    }




# ========================== 命令行接口 ==========================

if __name__ == "__main__":
    
    ph = "ph.mol"
    c6 = "c6.mol"

    ph_mol = load_mol(ph, removeHs=False, sanitize=False)
    if ph_mol is None:
        print(f"错误: 无法加载 {ph}")
        exit(1)


    c6_mol = load_mol(c6, removeHs=False, sanitize=False)
    if c6_mol is None:
        print(f"错误: 无法加载 {c6}")
        exit(1)

    print("ph")
    seits=find_potential_merging_sites(ph_mol)
    print(seits)
    print(seits['c_sites'][0][1][0])
        
    print("c6")
    seits=find_potential_merging_sites(c6_mol)
    print(seits)
    print(seits['c_sites'][0][1][0])
    
    # 按对称性分组显示潜在结合位点
    result = get_symm_sites(ph_mol)
    print(f"\n{ph} - 按对称性分组的潜在结合位点:")
    if result["c_symm_dict"]:
        for i, group in enumerate(result["c_symm_dict"]):
            print(f"    组{i+1}: {group}")
    else:
        print(f"\n  碳原子: 无")
    if result["n_symm_dict"]:
        for i, group in enumerate(result["n_symm_dict"]):
            print(f"    组{i+1}: {group}")
    else:
        print(f"\n  氮原子: 无")


