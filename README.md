# Molecule_Splicing_3D

分子3D空间拼接工具 - 将两个分子在3D空间中按指定位点拼接，形成新的化学键。

## 功能特点

- **向量对齐拼接**：将两个分子的键向量旋转至180°反向平行后拼接
- **保留3D构象**：拼接过程中保持原有的3D坐标
- **自动位点识别**：自动识别分子骨架上的潜在结合位点（碳原子和氮原子）
- **Rodrigues旋转**：使用Rodrigues公式实现精确的3D空间旋转
- **可调键长**：支持自定义目标键长（默认1.54 Å）

## 安装

### 依赖

- Python >= 3.8
- RDKit >= 2025.9.3

### 安装RDKit

```bash
# 使用uv（推荐）
uv sync

# 或使用pip
pip install rdkit
```

## 使用方法

### 1. 向量对齐拼接 (main.py)

自动识别分子结合位点并进行向量对齐拼接：

```bash
uv run python main.py
```

参数说明：
- `pairA`, `pairB`: X-H键原子索引对，0-based格式
- `target_bond_length`: 目标键长，默认1.54 Å
- 输出格式：`.mol`

### 2. 查找潜在结合位点 (getBackboneMergedPoint.py)

查找分子骨架上适合作为连接位点的原子（与H相连的非H原子）：

```python
from getBackboneMergedPoint import find_potential_merging_sites, get_symm_sites

# 加载分子
mol = Chem.MolFromMolFile("mol.mol", removeHs=False, sanitize=False)

# 查找潜在结合位点
sites = find_potential_merging_sites(mol)
# 返回: {"c_sites": [(idx, [h_idx1, h_idx2]), ...], "n_sites": [...]}

# 按对称性分组
symm = get_symm_sites(mol)
# 返回: {"c_symm_dict": [[idx1, idx2], ...], "n_symm_dict": [...]}
```

输出示例：
```
mol.mol - 按对称性分组的潜在结合位点:
  碳原子 (sp3, degree=3, ≥1H):
    组1: [0]
    组2: [1]
  氮原子 (degree=3, 1H):
    组1: [28]
    组2: [29]
```

### 3. 简单拼接 (merge_molecules.py)

在指定位点拼接两个分子：

```bash
# 查看分子原子信息
uv run python merge_molecules.py --info mol.mol

# 合并分子
uv run python merge_molecules.py mol.mol chain.mol 0 0 output.mol [bond_type]
```

参数说明：
- `mol.mol`: 第一个分子文件
- `chain.mol`: 第二个分子文件
- `0`: 第一个分子中要连接的原子索引（0开始）
- `0`: 第二个分子中要连接的原子索引（0开始）
- `output.mol`: 输出文件名
- `bond_type`: 键类型（1=单键, 2=双键, 3=三键，默认为1）

## 算法原理

### 向量对齐拼接流程

1. **加载分子**：读取`.mol`文件，保留3D坐标和H原子
2. **位点识别**：识别X-H键对，确定拼接方向向量
3. **计算旋转**：使用Rodrigues旋转公式计算旋转矩阵
4. **刚性变换**：对分子B进行旋转+平移，使其键向量与分子A呈180°
5. **分子合并**：使用RDKit的`CombineMols`合并两个分子
6. **成键删除**：删除冗余H原子，在指定位点形成新化学键
7. **输出结果**：保存为`.mol`格式

### Rodrigues旋转公式

任意3D旋转可通过Rodrigues公式实现：

```
R = I + sin(θ)×K + (1-cos(θ))×K²
```

其中：
- `I` 是3×3单位矩阵
- `θ` 是旋转角度
- `K` 是由旋转轴构成的反对称矩阵
- 旋转轴 = v1 × v2（叉乘）

## 文件结构

```
Molecule_Splicing_3D/
├── main.py                      # 向量对齐拼接主程序
├── merge_molecules.py           # 简单拼接工具
├── getBackboneMergedPoint.py   # 潜在结合位点识别
├── pyproject.toml               # 项目配置
└── README.md                    # 本文件
```

## 潜在结合位点定义

潜在结合位点是指满足以下条件的原子：

1. 是非H原子（C、N、O、S等）
2. 与至少一个H原子相连（形成X-H键）
3. 不含H的配位数在指定范围内
   - 碳原子：degree ∈ (1, 3]（即2或3）
   - 氮原子：degree = 2
4. 按对称性分组：相同化学环境的原子归为一组

返回格式：
- `find_potential_merging_sites()`: `{"c_sites": [(idx, [h_idx1, ...]), ...], "n_sites": [...]}`
- `get_symm_sites()`: `{"c_symm_dict": [[idx1, idx2], ...], "n_symm_dict": [...]}`

## 注意事项

- 输入的`.mol`文件必须包含3D坐标
- 原子索引从0开始
- pair参数为0-based索引对：(非H原子索引, H原子索引)
- 拼接后会自动删除连接位点的冗余H原子
- 默认键长为1.54 Å（可调整）

## License

MIT
