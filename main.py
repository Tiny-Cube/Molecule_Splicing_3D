"""
分子向量对齐拼接工具 (main.py)

功能说明：
    本程序实现两个分子之间的3D空间拼接，核心思想是将分子B旋转一定角度，
    使其键向量与分子A的键向量呈反向平行（夹角180°），然后在指定位点
    形成新化学键，最终输出标准.xyz格式文件。

核心算法：
    1. 向量识别：自动识别X-H键对，确定拼接方向向量
    2. Rodrigues旋转：使用Rodrigues旋转公式计算3D旋转矩阵
    3. 刚性变换：先绕原点旋转，再平移，使B分子到位
    4. 分子合并：RDKit CombineMols合并，删除冗余H原子，成键

依赖：RDKit (>=2025.9.3)

作者/用途：计算化学、分子设计、药物研发中的分子拼接操作
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import numpy as np
import os
from getBackboneMergedPoint import find_potential_merging_sites




# ========================== 加载分子文件 ==========================

def load_mol(path):
    """
    加载分子文件（.mol/.sdf/.pdb），严格保留3D构象和所有H原子
    参数：
        path (str): 分子文件路径
    返回：
        mol: RDKit分子对象，包含3D构象信息
    注意事项：
        - removeHs=False：保留所有H原子，因为H是拼接位点
        - sanitize=False：暂不 sanitize，待成键操作完成后再 sanitize
        - 仅支持MDL Molfile(.mol)、SDfile(.sdf)、PDB(.pdb)三种格式
    """
    mol = None
    if path.endswith(".mol"):
        # .mol格式：MDL Molfile格式，最常用的分子文件格式
        mol = Chem.MolFromMolFile(path, removeHs=False, sanitize=False)
    elif path.endswith(".sdf"):
        # .sdf格式：多个分子的存储格式，此处取第一个分子
        suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
        mol = suppl[0]
    elif path.endswith(".pdb"):
        # .pdb格式：蛋白质数据银行格式，也可存储小分子
        mol = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False)
    else:
        raise ValueError(
            f"仅支持.mol/.sdf/.pdb，当前输入：{os.path.splitext(path)[1]}"
        )

    # 检查分子是否加载成功且包含3D构象
    if mol is None or mol.GetNumConformers() == 0:
        raise Exception(f"分子{path}加载失败/无3D构象，请确认是带3D坐标的文件")

    # 成键操作前进行分子结构检查和标准化
    Chem.SanitizeMol(mol)
    return mol


# ========================== 位点识别函数 ==========================
def get_H_nonH(mol, idx_pair):
    """
    从1-based索引对识别X-H键，返回0-based索引和3D坐标

    参数：
        mol: RDKit分子对象
        idx_pair (tuple): 1-based原子索引对，例如(54, 5)表示第54和第5号原子

    返回：
        nonH_idx (int): 非H原子（X）的0-based索引
        H_idx (int): H原子的0-based索引
        nonH_coords (np.array): 非H原子的三维坐标
        H_coords (np.array): H原子的三维坐标

    算法说明：
        - 根据原子序数判断：氢原子序数为1，其他原子序数>1
        - 自动将用户输入的1-based索引转换为程序使用的0-based索引
        - 严格校验索引范围，防止越界访问
    """
    idx1, idx2 = idx_pair  # 解包索引对
    num_atoms = mol.GetNumAtoms()  # 获取分子总原子数

    # ------------------------- 索引边界校验 -------------------------
    # 防止用户输入超出范围的索引导致程序崩溃
    if idx1< 0 or idx1 >= num_atoms:
        raise ValueError(
            f"分子共{num_atoms}原子（0~{num_atoms - 1}），点位{idx1}越界！"
        )
    if idx2 < 0 or idx2 >= num_atoms:
        raise ValueError(
            f"分子共{num_atoms}原子（0~{num_atoms - 1}），点位{idx2}越界！"
        )

    # -------------------- 识别H原子和非H原子 --------------------
    atom1 = mol.GetAtomWithIdx(idx1)
    atom2 = mol.GetAtomWithIdx(idx2)

    # 判断哪个是H哪个是非H原子
    # 原子序数1为H，>1为非H（C/O/N/S等）
    if atom1.GetAtomicNum() == 1 and atom2.GetAtomicNum() != 1:
        H_idx, nonH_idx = idx1, idx2
    elif atom2.GetAtomicNum() == 1 and atom1.GetAtomicNum() != 1:
        H_idx, nonH_idx = idx2, idx1
    else:
        raise Exception(
            f"点位对{idx_pair}，无唯一H/非H组合！请确保是X-H键对"
        )

    # -------------------- 提取3D坐标 --------------------
    # RDKit的Conformer对象存储原子三维坐标
    conf = mol.GetConformer()
    # Lambda表达式：获取指定原子的坐标并转换为numpy数组
    # dtype=np.float64确保精度，避免后续向量计算的精度问题
    get_coords = lambda idx: np.array(conf.GetAtomPosition(idx), dtype=np.float64)
    nonH_coords = get_coords(nonH_idx)  # 非H原子坐标
    H_coords = get_coords(H_idx)  # H原子坐标

    return nonH_idx, H_idx, nonH_coords, H_coords


# ========================== 向量运算和旋转矩阵函数 ==========================
def normalize_vector(v):
    """
    向量归一化（单位化）
    """
    norm = np.linalg.norm(v)  # 计算向量长度（欧几里得范数）
    return v / norm if norm > 1e-8 else v  # 防止除以零

def rotate_matrix_from_vectors(v1, v2):
    """
    使用Rodrigues旋转公式计算从v1旋转到v2的3x3旋转矩阵
    参数：
        v1 (np.array): 原始向量（分子B的键向量）
        v2 (np.array): 目标向量（分子A键向量的反向，即-vecA）
    返回：
        np.array: 3x3旋转矩阵
    Rodrigues旋转公式说明：
        任意3D旋转都可以通过 Rodrigues公式 实现：
        R = I + sin(θ)×K + (1-cos(θ))×K²
        其中：
        - I 是3x3单位矩阵
        - θ 是旋转角度
        - K 是由旋转轴构成的反对称矩阵
        - 旋转轴 = v1 × v2（叉乘）

    特殊情况处理：
        - v1与v2同向：返回单位矩阵（无需旋转）
        - v1与v2反向：返回-1倍单位矩阵（180°旋转）
        - 叉乘为零向量时：使用简化公式
    """
    # 归一化向量，消除长度影响
    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)

    # -------------------- 特殊情况处理 --------------------
    if np.allclose(v1, v2):
        return np.eye(3)  # 两向量同向，无需旋转
    if np.allclose(v1, -v2):
        return -np.eye(3)  # 两向量反向，180°旋转

    # -------------------- Rodrigues公式核心计算 --------------------
    # 1. 计算旋转轴（叉乘）
    cross = np.cross(v1, v2)
    cross_norm = np.linalg.norm(cross)
    k = cross / cross_norm  # 旋转轴单位向量

    # 2. 计算旋转角度（点乘求cosθ，再反余弦）
    theta = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    # 3. 构建反对称矩阵K（用于Rodrigues公式）
    # 反对称矩阵性质：Kᵀ = -K，用于表示旋转轴
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])

    # 4. 计算最终旋转矩阵
    # Rodrigues公式：R = I + sin(θ)×K + (1-cos(θ))×K²
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R

def transform_mol_coords(mol, R, T):
    """
    对分子进行刚性坐标变换（旋转+平移）

    参数：
        mol: RDKit分子对象（将被原地修改）
        R (np.array): 3x3旋转矩阵
        T (np.array): 3元素平移向量

    返回：
        mol: 变换后的分子对象

    变换公式：
        new_coord = R × old_coord + T

        即：先对每个原子坐标乘以旋转矩阵R，再加上平移向量T

    技术实现：
        - 使用mol.GetConformer()获取构象对象
        - 使用conf.SetAtomPosition()直接修改原子坐标
        - 无需创建新分子对象，原地修改节省内存
        - 兼容低版本RDKit（无SetConformer时代码）
    """
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()

    # -------------------- 逐原子进行坐标变换 --------------------
    for idx in range(num_atoms):
        # 获取当前原子坐标
        coords = np.array(conf.GetAtomPosition(idx), dtype=np.float64)
        # 刚性变换核心公式：旋转 + 平移
        new_coords = np.dot(R, coords) + T
        # 将numpy数组转换回RDKit的Point3D并设置
        # Point3D是RDKit的三维点类，内部存储为float
        conf.SetAtomPosition(idx, Point3D(*new_coords))

    return mol

def merge_mols_with_vector_align(
    molA_path: str,
    molB_path: str,
    pairA: tuple,  # 分子A点位对 (54,5) - 1开始索引，无需修改
    pairB: tuple,  # 分子B点位对 (35,29) - 1开始索引，无需修改
    out_format: str = "mol",  # 默认输出.mol格式
    out_path: str = "merged_aligned_mol",
    bond_type: Chem.BondType = Chem.BondType.SINGLE,
    target_bond_length: float = 1.54
):
    """
    分子拼接主函数：向量180°对正 + 刚性坐标变换 + RDKit成键

    参数说明：
        molA_path (str): 分子A的.mol文件路径
        molB_path (str): 分子B的.mol文件路径
        pairA (tuple): 分子A的X-H键原子索引对，0-based
        pairB (tuple): 分子B的X-H键原子索引对，0-based
        out_format (str): 输出文件格式，默认"mol"
        out_path (str): 输出文件路径前缀（不含扩展名）
        bond_type (Chem.BondType): 成键类型，SINGLE/DOUBLE/TRIPLE，默认单键

    返回值：
        merged_mol: RDKit分子对象，拼接后的分子

    工作流程：
        1. 加载两个.mol分子文件（保留3D坐标和所有H原子）
        2. 识别X-H键对，确定拼接位点和方向向量
        3. 计算Rodrigues旋转矩阵，使分子B的向量旋转至与分子A呈180°
        4. 对分子B进行刚性变换（旋转+平移）
        5. 合并两个分子，删除冗余H原子，建立新化学键
        6. 输出标准.xyz格式文件

    技术要点：
        - 使用Rodrigues旋转公式实现3D空间旋转，无万向锁问题
        - 刚性变换保持分子内部构象不变
        - 删除冗余H原子避免价态超标
        - 输出.xyz格式兼容Avogadro/PyMOL/Gaussian等软件
    """


    
    # ========================== 第4步：主流程开始 ==========================

    # -------------------- 步骤4.1：加载分子文件 --------------------
    # 严格保留3D构象（不删除H原子，不改变坐标）
    molA = load_mol(molA_path)
    molB = load_mol(molB_path)
    numA = molA.GetNumAtoms()
    numB = molB.GetNumAtoms()
    print(f"分子A（.mol）加载完成：{numA}个原子，3D构象保留（0~{numA - 1}）")
    print(f"分子B（.mol）加载完成：{numB}个原子，3D构象保留（0~{numB - 1}）")

    # -------------------- 步骤4.2：识别拼接位点 --------------------
    # 自动完成1→0索引转换，识别X-H键对
    Xa_idx, Ha_idx, Xa, Ha = get_H_nonH(molA, pairA)
    Xb_idx, Hb_idx, Xb, Hb = get_H_nonH(molB, pairB)
    print(f"分子A：非H位点{Xa_idx}，H位点{Ha_idx}")
    print(f"分子B：非H位点{Xb_idx}，H位点{Hb_idx}")

    # -------------------- 步骤4.3：计算方向向量 --------------------
    # 方向向量定义：从非H原子指向H原子的向量
    # 这个向量代表化学键的方向
    vecA = Ha - Xa  # 分子A的键向量
    vecB = Hb - Xb  # 分子B的键向量
    vecA_norm = normalize_vector(vecA)  # 归一化
    vecB_norm = normalize_vector(vecB)  # 归一化
    print(f"分子A方向向量（归一化）：{vecA_norm.round(3)}")
    print(f"分子B方向向量（归一化）：{vecB_norm.round(3)}")

    # -------------------- 步骤4.4：计算旋转矩阵 --------------------
    # 目标：将分子B旋转，使其键向量与分子A呈180°（反向平行）
    # 180°反向对齐后的向量 = -vecA_norm
    target_vecB = -vecA_norm
    # 使用Rodrigues公式计算旋转矩阵
    R = rotate_matrix_from_vectors(vecB_norm, target_vecB)
    print(f"旋转矩阵：\n{R.round(3)}")

    # -------------------- 步骤4.5：分子B的刚性变换 --------------------
    # 第一阶段：绕Xb点（分子B的非H原子）旋转180°
    # 注意：Rodrigues旋转默认绕原点旋转，因此实际上先平移到原点，旋转，再平移回去
    # 为简化实现，这里采用：绕原点旋转（旋转矩阵已考虑方向），然后再平移到位
    target_bond_length = target_bond_length  # 典型C-C单键长度（Å），可根据实际情况调整

    molB_rotated = transform_mol_coords(molB, R, T=np.zeros(3))

    # 获取旋转后的Xb坐标
    confB_rot = molB_rotated.GetConformer()
    Xb_rot = np.array(confB_rot.GetAtomPosition(Xb_idx), dtype=np.float64)

    # -------------------- 步骤4.6：平移分子B到位 --------------------
    # 计算目标位置：Xa沿着vecA方向延伸1.54 Å的位置
    target_Xb = Xa + vecA_norm * target_bond_length
    # 计算平移向量：从旋转后Xb位置到目标位置的距离
    T = target_Xb - Xb_rot
    # 执行平移（旋转矩阵此时为单位矩阵）
    molB_transformed = transform_mol_coords(molB_rotated, R=np.eye(3), T=T)

    # -------------------- 步骤4.7：验证变换结果 --------------------
    # 计算最终Xb-Xa距离
    confB_final = molB_transformed.GetConformer()
    Xb_final = np.array(confB_final.GetAtomPosition(Xb_idx), dtype=np.float64)
    final_distance = np.linalg.norm(Xb_final - Xa)
    print(f"平移向量：{T.round(3)}")
    print(f"Xb-Xa距离：{final_distance:.3f} Å（目标：{target_bond_length} Å）")

    # -------------------- 步骤4.8：验证向量对齐 --------------------
    # 计算拼接后分子B的键向量
    Hb_final = np.array(confB_final.GetAtomPosition(Hb_idx), dtype=np.float64)
    vecB_final = Hb_final - Xb_final
    vecB_final_norm = normalize_vector(vecB_final)
    # 计算A和B的键向量夹角（应为180°或接近180°）
    angle = (
        np.arccos(np.clip(np.dot(vecA_norm, vecB_final_norm), -1.0, 1.0)) * 180 / np.pi
    )
    print(f"向量对齐结果：A/B夹角 = {angle:.1f}°（目标180°）")

    # ========================== 第5步：分子合并与成键 ==========================

    # -------------------- 步骤5.1：合并两个分子 --------------------
    # Chem.CombineMols：将两个分子的原子和坐标合并到同一个分子对象中
    # 此时两个分子只是空间上放在一起，还没有形成化学键
    merged_mol = Chem.CombineMols(molA, molB_transformed)

    # -------------------- 步骤5.2：准备成键 --------------------
    # EditableMol：RDKit中用于修改分子拓扑结构（增删键、原子）的类
    ed = Chem.EditableMol(merged_mol)

    # 计算分子B中原子在合并后分子中的索引
    # 合并后，分子A的原子索引不变（0~numA-1）
    # 分子B的原子索引需要偏移numA
    Xb_merged_idx = Xb_idx + numA  # B的非H原子在合并分子中的索引
    Hb_merged_idx = Hb_idx + numA  # B的H原子在合并分子中的索引

    # -------------------- 步骤5.3：建立化学键 --------------------
    # 在Xa和Xb之间建立化学键
    print(f"建立化学键：分子A非H位点 {Xa_idx} + 分子B非H位点 {Xb_merged_idx}")
    ed.AddBond(int(Xa_idx), int(Xb_merged_idx), bond_type)

    # -------------------- 步骤5.4：删除冗余H原子 --------------------
    # 成键后，Xa和Xb各自还连着原来的H原子
    # 如果不删除，Xa和Xb都会变成五价（价态超标），导致分子不稳定
    # 删除策略：先删索引大的，后删索引小的（避免索引变化问题）
    if Ha_idx < Hb_merged_idx:
        # B的H原子索引更大，先删B的H
        ed.RemoveAtom(int(Hb_merged_idx))
        # 再删A的H
        ed.RemoveAtom(int(Ha_idx))
    else:
        # 反之亦然
        ed.RemoveAtom(int(Ha_idx))
        # B的H原子在A的H删除后索引会减1
        ed.RemoveAtom(int(Hb_merged_idx) - 1)

    # -------------------- 步骤5.5：获取最终分子并校验 --------------------
    merged_mol = ed.GetMol()
    # SanitizeMol：检查分子价态、 Kekulize 等，确保分子化学合理
    Chem.SanitizeMol(merged_mol)
    print("已删除连接位点冗余H，分子价态正常，无成键错误")

    # ========================== 第6步：输出结果 ==========================

    # -------------------- 步骤6.1：准备输出文件 --------------------
    out_file = f"{out_path}.{out_format.lower()}"
    num_total_atoms = merged_mol.GetNumAtoms()  # 合并后总原子数

    # -------------------- 步骤6.2：写入.mol文件 --------------------
    # 使用RDKit的MolToMolFile保存为MDL Molfile格式
    Chem.MolToMolFile(merged_mol, out_file)

    # -------------------- 步骤6.3：打印完成信息 --------------------
    print(f"\n所有步骤完成！.mol格式结果已导出至：{out_file}")
    print(
        f" 最终信息：总原子数{num_total_atoms} | 向量夹角{angle:.1f}°"
    )
    return merged_mol


# ========================== 示例调用 ==========================

if __name__ == "__main__":
    """
    程序入口示例

    使用说明：
        1. 将molA_path和molB_path替换为实际要拼接的分子文件路径
        2. pairB是对应的X-H键原子索引对（1-based）
           - pairA: 自动从分子A的第一个潜在结合位点获取
        3. out_path是输出文件的路径前缀
        4. 运行后会在当前目录生成 A_B_merged_180_xyz.xyz 文件
    """

    # 自动获取分子A和分子B的潜在结合位点
    molA_for_sites = Chem.MolFromMolFile("mol.mol", removeHs=False, sanitize=False)
    sitesA = find_potential_merging_sites(molA_for_sites)

    # 使用第一个碳原子位点作为pairA和pairB
    auto_pairA = None

    if sitesA['c_sites']:
        first_c_site = sitesA['c_sites'][0]
        auto_pairA = (first_c_site[0], first_c_site[1][0])

    merge_mols_with_vector_align(
        molA_path="mol.mol",
        molB_path="chain.mol",
        pairA=auto_pairA,
        pairB=(4,5),
        out_format="mol",
        out_path="A_B_merged_180",
        bond_type=Chem.BondType.SINGLE,
        target_bond_length = 1.54,
    )
