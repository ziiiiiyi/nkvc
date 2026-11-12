#!/usr/bin/env python3
"""NK aging atlas with UNAGI-inspired within-stage graph encoding."""
import argparse###用来读取数据
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc ###用来读取.h5ad数据格式的
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components##检查细胞之间的连接
from sklearn.decomposition import TruncatedSVD ###用来降维
from sklearn.neighbors import NearestNeighbors###用来给每个细胞找特征最相似的其他细胞
from sklearn.preprocessing import StandardScaler###标准化数据

GROUPS = ("cord", "young", "old")##脐血组、青年组、老年组
MARKERS = {###把markers想象成一个字典，给它一个名字，他就能找到这个名字对应的内容
    "bright": ("IL7R XCL1 XCL2 GZMK KLRC1 SELL", ""),##bright对应的CD56bright NK。##第一段列出给 bright 分数加分的基因，基因名之间用空格隔开。"" 是空字符串，表示这一项没有设置减分基因
    "dim": ("FCGR3A FGFBP2 PRF1 GNLY GZMB", ""),##对应CD56dim NK
    "terminal_maturation": ("B3GAT1 LILRB1 ZEB2 CX3CR1", ""),
    "adaptive": ("KLRC2 LILRB1 CD52", "FCER1G ZBTB16 SYK"),
    "memory_like": ("IL2RA IFNG GZMB", ""),
    "precursor_like": ("TCF7 SELL IL7R KIT", ""),
    "cytotoxicity_proxy": ("PRF1 GNLY GZMB NKG7 CTSW", ""),
    "exhaustion_proxy": ("TIGIT HAVCR2 LAG3 PDCD1 TOX", ""),
    "senescence_proxy": ("CDKN1A CDKN2A GLB1 SERPINE1", ""),
}###这个是AI给的打分项目；如果所给数据包含这些指标，可以当作一套待验证的打分名单。先跑一遍，看是否能区分出细胞群，效果不好再更改打分的基因。


def arguments():###定义了一个函数，给一组操作起了一个名字，以后需要做这件事情，直接用这个名字调用它。
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("rna", help="NK-enriched raw count h5ad")
    p.add_argument("out", type=Path)####在终端换成自己的数据文件和想要的输出文件名
    for name in ("adt", "atac", "regulons", "reference-module", "lock-module",
                 "senolytic-genes", "annotations", "response"):
        p.add_argument("--" + name)###for循环，把括号里的名字一个一个拿出来，每拿到一个，就执行下面缩进的那行。有蛋白数据时，在终端命令中加入--adt proteins.tsv
    p.add_argument("--response-margin", type=float)###登记参数名
    p.add_argument("--velocity", choices=["off", "stochastic", "dynamical"], default="off")###登记了--velocity，让你选择要不要做 RNA velocity（RNA 速度）分析。它利用已剪接和未剪接的 RNA 计数，估计细胞的表达状态接下来可能往哪个方向变化。
    p.add_argument("--embedding", choices=["pca", "graph-vae"], default="graph-vae")###登记了embeeding参数，用来选择脚本要不要额外运行图VAE，VAE会参考细胞邻居信息
    for name, default in dict(hvg=2000, neighbors=15, latent=32, epochs=40,
                              min_genes=200, min_counts=500, min_cells=20,
                              min_donors=5, permutations=999, seed=8848).items():###准备了一批参数，都设有默认值，若终端里没有指定，就用这些默认值
        p.add_argument("--" + name.replace("_", "-"), type=int, default=default)###准备了一批参数，type=float表示可以接受小数
    for name, default in dict(max_mt=15, self_weight=0.5, resolution=0.8, beta=0.1,
                              module_weight=0, lr=0.001).items():
        p.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    p.add_argument("--max-counts", type=int)###用来设置单个细胞的RNA总计数上限。
    p.add_argument("--device", default="cpu")###用来选择图 VAE 在什么设备上训练
    a = p.parse_args()###读取我在终端输入的参数，读完以后存到变量a，理解为读参数的工具。
    if min(a.neighbors, a.epochs, a.latent, a.min_cells, a.permutations) < 1 or a.min_donors < 3 or a.hvg < 3:
        p.error("Require positive sizes, hvg >= 3, min-donors >= 3")###这两行开始检查这些值能不能用
    if not 0 <= a.self_weight <= 1 or min(a.beta, a.module_weight) < 0 or a.lr <= 0:
        p.error("Invalid graph or training weights")###继续检查参数
    if a.response and (a.response_margin is None or a.response_margin <= 0):
        p.error("--response requires a positive --response-margin")###填已经做过的功能实验的数据文件
    if a.module_weight and (not a.regulons or a.embedding != "graph-vae"):
        p.error("--module-weight requires --regulons and graph-vae")##只检查你给的设置是否齐全
    return a###“把结果交回给调用这个函数的地方”


def save(table, a, name, index=False):
    table.to_csv(a.out / (name + ".tsv"), sep="\t", index=index)###定义函数save用来保存文件

def read_indexed(path):##读取文件路径
    index = pd.read_csv(path, sep="\t", nrows=0).columns[0]###`nrows=0` 表示只读列名，`.columns[0]` 取第一个列名
    return pd.read_csv(path, sep="\t", dtype={index: str}).set_index(index)###读整张表



def counts(x, name):
    x = sparse.csr_matrix(x, dtype=np.float32)###把X转成了CSR的稀疏矩阵
    if not np.isfinite(x.data).all() or (x.data < 0).any() or not np.allclose(x.data, np.rint(x.data), rtol=0):###`x.data` 取出稀疏矩阵里实际存储的数值，省略的位置按零处理。
        raise ValueError(f"{name} requires nonnegative raw integer counts")###在任意一项检查失败时抛出错误
    return x###在检查全部通过时执行，把上一行已经转好的稀疏矩阵交回给调用者


def prepare(a):###定义读取和准备NK数据的函数
    ad = sc.read_h5ad(a.rna)###adata包含表达矩阵ad.X和细胞信息ad.obs
    ad.var_names = ad.var_names.str.upper()###把所有基因名称改成大写`ad.var_names`是基因名单。
    if not ad.obs_names.is_unique or not ad.var_names.is_unique:
        raise ValueError("Require unique cell IDs and gene symbols")###检查细胞ID和基因名是否自个唯一
    cols = ["donor", "age_group"]###`donor` 表示细胞来自哪位供体
    if not set(cols) <= set(ad.obs) or ad.obs[cols].isna().any().any():
        raise ValueError("obs requires donor and age_group")###这三行检查每个细胞是否都填了供体 ID 和年龄组。
    for col in cols:
        ad.obs[col] = ad.obs[col].astype(str)###把细胞信息表里的 donor 和 age_group 两列统一转换成文字
    if set(ad.obs.age_group) != set(GROUPS) or ad.obs.groupby("donor").age_group.nunique().max() != 1:
        raise ValueError("Require cord, young, old and one cohort per donor")###检查年龄组填得对不对，比如把 young 写成 youth，就会触发报错。在任意一项检查失败时停止程序。一个供体可以有很多细胞，这些细胞的 age_group 要填成同一个值。
    ad.X = counts(ad.layers["counts"] if "counts" in ad.layers else ad.X, "RNA")
    ad.layers["counts"] = ad.X.copy()###这两行负责找到原始 RNA 计数、检查它，再留一份副本
    ad.var["mt"] = ad.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(ad, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)###算的是线粒体基因的 RNA 计数占这个细胞总计数的比例（算出的质控指标）。MT- 开头表示这个基因位于线粒体 DNA 上
    obs = ad.obs
    keep = (obs.n_genes_by_counts >= a.min_genes) & (obs.total_counts >= a.min_counts)
    keep &= obs.pct_counts_mt <= a.max_mt###根据刚算出的质控指标，给每个细胞做一个“是否保留”的标记
    if a.max_counts is not None:
        keep &= obs.total_counts <= a.max_counts###刚刚处理的是RNA总计数下限，这个处理的是上限
    if "is_doublet" in obs:
        flag = obs.is_doublet.astype(str).str.lower()###检查是不是双细胞。doublet——双细胞
        if not flag.isin(["true", "false", "0", "1"]).all():###检查标记的写法
            raise ValueError("is_doublet must be boolean or 0/1")
        keep &= flag.isin(["false", "0"])###这三行先检查双细胞标记写得对不对，再用它更新每个细胞的保留状态
    save(obs.assign(pass_qc=keep), a, "qc_cells", index=True)
    ad = ad[keep].copy()###先保存所有细胞的质控结果，再筛掉没通过的细胞
    if set(ad.obs.age_group) != set(GROUPS) or ad.n_obs < 6:###质控后，脐血、青年、老年这三组是否还都在
        raise ValueError("QC left too few cells or removed a cohort")####剩下的细胞总数是否少于6###这两行检查筛选后的数据还能不能继续分析
    sc.pp.filter_genes(ad, min_cells=1)###前面筛的是细胞，也就是行，这里开始筛基因##保存至少在1个通过质控的细胞中测到基因
    if ad.n_vars < 3:
        raise ValueError("Too few expressed genes")##检查筛完还剩下多少个基因##`ad.n_vars` 就是当前表达矩阵的列数。如果少于 3，脚本就报错停止，因为后面的降维至少需要有 3 个候选基因可用
    sc.pp.normalize_total(ad, target_sum=1e4)###做归一化，每个细胞这一行的数字按同比例缩放，让每一行的总和都变成10000（1e4 就是 10000）
    sc.pp.log1p(ad)###做对数变换。log1p 的意思是先加 1，再取自然对数。刚刚100会变成ln(101)
    return ad###把处理好的单细胞数据交回给主程序

###接下来定义给细胞打分的函数
def score(ad, up, down):###score接收三个东西`ad` 是准备好的细胞数据，`up` 是加分基因名单，`down` 是减分基因名单。
    observed = [sorted(set(map(str.upper, genes)) & set(ad.var_names)) for genes in (up, down)]###核对基因列是否存在，某个细胞在这列的表达值是多少，后面算分时才读取
    coverage = list(map(len, observed))###数出两份名单各找到了几个基因
    if not any(coverage) or any(len(found) < min(2, len(set(genes)))
                                for found, genes in zip(observed, (up, down))):###把每份找到的名单和它的原始名单配在一起检查
        return np.full(ad.n_obs, np.nan), coverage###在基因不够时，给每个细胞返回一个 NaN，也就是分数留空，同时交回 coverage，让你知道实际找到了几个基因。
    means = [np.asarray(ad[:, genes].X.mean(axis=1)).ravel() if genes else 0 for genes in observed]###先处理加分名单，再处理减分名单
    return means[0] - means[1], coverage###用加分均值 means[0] 减去减分均值 means[1]

###接下来把前面写好的score（）用到各个打分项目上
def add_scores(ad, a):###ad是处理好的细胞数据，a是本次运行的设置
    signatures = {k: (up.split(), down.split()) for k, (up, down) in MARKERS.items()}###先整理markers，把整理好的名单存入新字典。原来每份基因名单是一整段文字，比如 "IL2RA IFNG GZMB"。`.split()` 会按空格把它拆成一个个基因名，方便后面的 score() 逐个查找。
    if a.senolytic_genes:###检查你有没有提供文件
        spec = json.loads(Path(a.senolytic_genes).read_text())###先用Path(...).read_text() 读出文件里的文字，再用 json.loads(...) 把这些文字解析成 Python 字典，存进 spec
        signatures["senolytic_signature"] = spec["up"], spec.get("down", [])###取出字典里的两份名单，添加到上一段的 signatures 里
    edges = None###表示目前还没有调控关系表
    if a.regulons:
        edges = pd.read_csv(a.regulons, sep="\t").drop_duplicates()###检查你是否在终端填了 --regulons regulons.tsv。
        if edges[["module", "tf", "target", "direction"]].isna().any().any():
            raise ValueError("Regulon rows must be complete")###检查四列有没有空值
        for col in ("tf", "target"):
            edges[col] = edges[col].astype(str).str.upper()###依此处理这两列，把数值转成文字，再把文字大写
        if not edges.direction.isin([-1, 1]).all():
            raise ValueError("direction must be -1 or 1")###检查direction，1对应加分名单，-1对应减分名单
        for name, g in edges.groupby("module"):###把模块相同的行放在一起
            up = sorted(set(g.loc[g.direction == 1, "target"]))###选出方向为1的行，只取这些行的target列
            down = sorted(set(g.loc[g.direction == -1, "target"]))###做同样的操作，但选方向为-1的靶基因，结果存入down
            if set(up) & set(down):###& 找出两份名单共有的基因
                raise ValueError(f"Conflicting directions in {name}")###发现冲突就报错
            signatures[f"module_{name}"] = up, down###检查通过后，把这两份名单存入 signatures
    coverage = {}###创建一个空字典 coverage，用来记录每项打分实际找到了几个加分基因、几个减分基因。
    for name, (up, down) in signatures.items():##用循环逐个取出打分项目
        ad.obs[name], coverage[name] = score(ad, up, down)###调用之前学过的 score()
    for name, alias in [(a.reference_module, "cord_reference_score"), (a.lock_module, "old_state_score")]:###循环最多处理两项。
        if name:###检查你有没有指定这个模块
            key = "module_" + name###用 "module_" + name 拼出原来的分数列名，令为k
            if key not in coverage or ad.obs[key].isna().all():##检查两种情况，找不到打分记录或者分数都为空
                raise ValueError(f"Missing or insufficiently covered module {name}")###在任一检查失败时停止程序，并在报错里显示模块名
            ad.obs[alias] = ad.obs[key]###在检查通过后，把已有的分数复制到 alias 对应的新列
    (a.out / "signature_coverage.json").write_text(json.dumps(coverage, indent=2))###把每项打分实际找到了几个基因的记录保存下来。比如，文件里出现 "bright": [4, 0]，就表示 bright 这项找到了 4 个加分基因、0 个减分基因。
    return edges, list(signatures)###结束 add_scores()，把两样东西交回给调用它的地方


def scale_block(x):###定义scale_block()。这里x可以是一张RNA或蛋白的特征表。一行是一个细胞，一列是一个特征
    return StandardScaler().fit_transform(x) / np.sqrt(x.shape[1])###用第 15 行导入的工具做标准化

###负责整合不同的类型的数据。先从RNA降维开始
def integrate(ad, x, a):##定义函数
    k = min(30, min(x.shape) - 1)###决定PCA要出输出几个特征，也就是k。结果限制在最多30.比如 x 有 20 个细胞、100 个基因，那么 k = min(30, 20 - 1) = 19。
    blocks = [scale_block(sc.pp.pca(x, n_comps=k, random_state=a.seed))]###用 sc.pp.pca(...) 把每个细胞原来的基因表达值概括成 k 个新数字。`random_state=a.seed` 用来控制这一步涉及的随机性。
    if a.adt:
        tab = read_indexed(a.adt)##如果你在终端写了 --adt proteins.tsv，就用前面学过的 read_indexed() 读入蛋白表。如果没提供 --adt，就跳过这一段。
        if not tab.index.is_unique or not ad.obs_names.isin(tab.index).all():
            raise ValueError("ADT must match all retained cell IDs uniquely")###检查蛋白表里的细胞 ID 有没有重复，以及每个通过 RNA 质控的细胞能不能在蛋白表中找到。任何一项不满足，程序就会报错。
        tab = tab.loc[ad.obs_names].astype(float)###按 RNA 的细胞 ID 取蛋白表的行。
        if not np.isfinite(tab.values).all() or (tab.values < 0).any():
            raise ValueError("ADT requires finite nonnegative counts")###检查蛋白值有没有缺失、无穷大或负数。只要发现一个这样的值，程序就会报错。
        log = np.log1p(tab.values)###用 tab.values 取出蛋白表里的数字，再用熟悉的 log1p 做转换
        blocks.append(scale_block(log - log.mean(axis=1, keepdims=True) if log.shape[1] > 1 else log))###解读从括号里忘外读。现在 blocks 里有RNA 特征表和蛋白特征表。它们还各自存着，后面才会拼成一张表。
        for protein in tab:###遍历 tab 的列名
            ad.obs[f"ADT_{protein}"] = tab[protein].values###用 f"ADT_{protein}" 拼出新列名
    if a.atac:
        at = sc.read_h5ad(a.atac)#如果你在终端写了 --atac peaks.h5ad，就用 Scanpy 读取这个文件，并把读到的数据存进 at。如果没提供，就跳过 ATAC 部分。
        if not at.obs_names.is_unique or not ad.obs_names.isin(at.obs_names).all():
            raise ValueError("ATAC must match all retained cell IDs uniquely")###检查 ATAC 数据里的细胞 ID 有没有重复，以及每个通过 RNA 质控的细胞能不能在 ATAC 数据里找到。任何一项不满足，程序都会报错。
        x_at = counts(at[ad.obs_names].X, "ATAC")
        depth = np.asarray(x_at.sum(axis=1)).ravel()
        if np.any(depth == 0) or min(x_at.shape) < 3:
            raise ValueError("ATAC requires positive depth and at least three peaks")###取出 ATAC 计数，并检查它能不能用于后面的分析
        df = np.asarray((x_at > 0).sum(axis=0)).ravel()
        tfidf = sparse.diags(1 / depth) @ x_at @ sparse.diags(np.log1p(x_at.shape[0] / (1 + df)))##用 TF-IDF 给 ATAC 表里的计数重新加权
        k = min(31, min(tfidf.shape) - 1)
        lsi = TruncatedSVD(k, random_state=a.seed).fit_transform(tfidf)
        blocks.append(scale_block(lsi[:, 1:]))###把刚才的 ATAC 加权表降维，再把结果放进 blocks
    return np.hstack(blocks).astype(np.float32)##把 blocks 里的特征表横着拼起来，交回给调用 integrate() 的地方
###以上内容是完成了数据的预处理
###开始定义给图 VAE 建立细胞邻接矩阵的函数
def stage_graph(z, groups, a):##定义stage—_graph,它收集拼好的特征表（一行对应一个细胞）、groups记录每个细胞的分类、a是运行设置，里面有邻居数 a.neighbors 和自身权重 a.self_weight
    rows, cols, weights = [], [], []###创建三个空列表。例如后面计算第 0 行细胞的特征时，要参考第 2 行细胞，权重是 0.25，留的是这三个位置。
    for group in GROUPS:###依次取出 cord、`young`、`old`。
        ids = np.flatnonzero(groups == group)####groups == group 逐个检查每个细胞是否属于当前组。比如 groups 是 ["cord", "old", "cord"]，这一轮得到的 ids 就是 [0, 2]，因为 Python 从第 0 行开始数。
        k = min(a.neighbors, len(ids) - 1)####用 len(ids) - 1 算出每个细胞在组里最多有几个其他细胞可当邻居
        rows.extend(ids)
        cols.extend(ids)
        weights.extend(np.full(len(ids), a.self_weight if k else 1))###这三行记录每个细胞到自己的连接。比如上一段给的ids=【0，2】，这意味着0和2加到rows和cols。考虑邻居的同时也要考虑自己【0，0】，【2，2】。然后给这两条连接各放一个权重（在a.self_weight=），默认是0.5
        if k:
            nn = NearestNeighbors(n_neighbors=k).fit(z[ids]).kneighbors(return_distance=False)
            rows.extend(np.repeat(ids, k))
            cols.extend(ids[nn].ravel())
            weights.extend(np.full(len(ids) * k, (1 - a.self_weight) / k))###上一段给每个细胞记录了自己的权重。这一段找到它的邻居，再把剩下的权重平均分给这些邻居
    graph = sparse.csr_matrix((np.asarray(weights, dtype=np.float32), (rows, cols)),
                              shape=(len(z), len(z)))
    graph.eliminate_zeros()
    return graph###上一段把连接记在 rows、`cols` 和 weights 三个列表里。这一段把这些记录装进一张邻接矩阵。真正用这些权重计算细胞特征，是后面图 VAE 的工作。

###212行开始定义graph_vae()。x：选出的高变基因表达表；z：前面拼接好的特征表，可以包含RNA、蛋白和ATAC特征。graph：刚建好的邻接矩阵，决定每个细胞参与谁、参考多少。weight：基因的重建权重，决定重建哪些基因时，误差更受重视。a：本运行的设置，比如学习率和训练轮数。tag：这次训练用的名字，后面用它给结果文件命名。
def graph_vae(x, z, graph, weights, a, tag):
    import torch
    from torch import nn###导入 PyTorch 和搭建神经网络用的 nn
###接下来给VAE的训练做准备。下面这两行分别准备了随机数的起点和模型在哪里计算。
    torch.manual_seed(a.seed)###用 a.seed 设置 PyTorch 的随机种子，默认是 8848。设定种子，相当于让 PyTorch 从同一个起点生成随机数，方便你重复运行时比较结果
    device = torch.device(a.device)##读取 a.device，也就是终端里 --device 的设置，默认是 "cpu"。`torch.device(...)` 把它变成 PyTorch 能用的设备对象，存到 device。后面创建模型时，才会用这个对象把模型放到指定设备上。
    context = graph @ z###用邻接矩阵里的权重，把每个细胞自己和邻居的特征加在一起。graph 记录每个细胞该参考谁、各参考多少；@ 是矩阵乘法；z 是前面拼好的特征表，每行是一个细胞的 RNA、蛋白或 ATAC 特征。
    encoder = nn.Sequential(nn.Linear(x.shape[1] + z.shape[1], 128), nn.Softplus(),###把输入变成 128 个中间特征。nn.Softplus()是对着128个数逐个做一次非线性变换，让后面的网络可以学习更复杂的关系。
                            nn.Linear(128, 2 * a.latent)).to(device)##数字的个数仍然是128.i这两行搭建 VAE 的编码器。它的任务是把每个细胞的一长串输入数字，变成描述这个细胞的潜在分布所需的数字。nn.Linear(128, 2 * a.latent)：把这 128 个数变成最终输出。默认 a.latent = 32，所以会输出 64 个数。后面会把它们拆成两半：前 32 个是均值，后 32 个是对数方差。每个潜在维度都需要这两个数来描述它的位置和波动。
    decoder = nn.Sequential(nn.Linear(a.latent, 128), nn.Softplus(),###
                            nn.Linear(128, x.shape[1])).to(device)###这两行搭建VAE的解码器。尝试将细胞的低维表示重建这个细胞的基因表达值。（就是将上面的那个过程逆过来在进行一遍）。.to(device)` 把解码器放到和编码器相同的设备上，训练时，程序会把解码器重建的那一行与 x 中同一个细胞的表达行做比较，看它重建得有多准。
    optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=a.lr)#####创建了一个叫 Adam 的优化器。优化器的工作是：训练时根据误差带来的梯度，调整网络里的权重，让下一次重建有机会更接近原来的表达值。
    w = torch.as_tensor(weights, device=device, dtype=torch.float32)###把传进来的基因重建权重 weights 转成 PyTorch 能直接计算的格式，存到 w 里。
    rng = np.random.default_rng(a.seed)###创建 NumPy 的随机数工具，存到 rng
    history = []#创建空列表 history

    def batches(ids):##定义batches函数。它接收 ids，也就是要处理的细胞行号。
        return np.array_split(ids, int(np.ceil(len(ids) / 256)))##第 229 行把这些行号分成几批，方便模型每次只处理一部分细胞。
####接下来定义 encode()，它负责把一批细胞的数据送进编码器，拿到每个细胞的均值和对数方差。
    def encode(ids):##def encode(ids)：`ids` 是当前这批细胞的行号。比如传入 [0, 5]，就只为第 0 和第 5 行的细胞计算编码结果。
        pooled = (graph[ids] @ x).toarray()
        inp = torch.as_tensor(np.hstack([pooled, context[ids]]), device=device)
        mu, logvar = encoder(inp).chunk(2, dim=1)
        return mu, logvar.clamp(-10, 10)
###下面开始按批次训练图 VAE。先给一批细胞准备好两样东西：要重建的原始目标 target，以及交给解码器的低维向量 latent。
    for epoch in range(a.epochs):
        totals = np.zeros(2)
        for ids in batches(rng.permutation(x.shape[0])):
            target = torch.as_tensor(x[ids].toarray(), device=device)
            mu, lv = encode(ids)
            latent = mu + torch.randn_like(mu) * torch.exp(lv / 2)###用均值和对数方差抽出一个低维向量 latent，供解码器使用
            recon = ((decoder(latent) - target).square() * w).sum(1).mean() / w.sum()
            kl = (0.5 * (mu.square() + lv.exp() - 1 - lv)).mean()
            loss = recon + a.beta * kl##行算总损失 loss。 kl 衡量编码器给出的潜在分布偏离标准正态分布多少。比如，某一维的均值 mu 是 0，对数方差 lv 也是 0，它的方差就是 1，这一维的 KL 贡献就是 0
            if not torch.isfinite(loss):
                raise ValueError("Graph VAE training diverged")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()###这八行完成一批细胞的一次训练更新。先算模型表现怎么样，再根据结果调整模型参数。
            totals += len(ids) * np.array([recon.item(), kl.item()])
        history.append([epoch + 1, *(totals / x.shape[0])])
    save(pd.DataFrame(history, columns=["epoch", "mse", "kl"]), a, tag + "_training")
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
               a.out / (tag + "_weights.pt"))
    with torch.no_grad():
        return np.vstack([encode(ids)[0].cpu().numpy() for ids in batches(np.arange(x.shape[0]))])###前面每处理一批细胞，模型就更新一次。这里负责汇总每轮的训练结果、保存模型，然后输出每个细胞的低维表示。

###接下来开始用每个细胞的特征找邻居、分群，再画出可以放在平面上的坐标。
def cluster_and_order(ad, rep, a):###定义函数。`ad` 是细胞数据，`a` 是运行设置。`rep` 是要用哪张特征表的名字。主分析传入 "X_integrated"，也就是前面拼好的 RNA、蛋白或 ATAC 特征。
    sc.pp.neighbors(ad, use_rep=rep, n_neighbors=min(a.neighbors + 1, ad.n_obs - 1),
                    random_state=a.seed)###用 Scanpy 根据这张特征表找邻居
    sc.tl.leiden(ad, resolution=a.resolution, random_state=a.seed, flavor="igraph",
                 n_iterations=2, key_added="cluster")##用 Leiden 算法根据邻居图分群
    sc.tl.umap(ad, random_state=a.seed)###用 UMAP 给每个细胞算两个坐标，让你可以把它画成平面上的一个点。
    markers = [*MARKERS, *ad.obs.filter(regex="^ADT_").columns]
    save(ad.obs.groupby("cluster", observed=True)[markers].median(), a, "cluster_markers", index=True)###给每个群做一份特征汇总。
    ad.obs["subtype"] = "cluster_" + ad.obs.cluster.astype(str)##先给每个群一个临时名字。
    if a.annotations:
        labels = read_indexed(a.annotations).subtype
        if not labels.index.is_unique or set(labels.index) != set(ad.obs.cluster.astype(str)):
            raise ValueError("Annotations must map every cluster exactly once")
        if labels.isna().any() or labels.astype(str).str.strip().eq("").any():
            raise ValueError("Annotations must be nonempty")###如果你提供了审核表，就读取并检查。
        ad.obs["subtype"] = ad.obs.cluster.astype(str).map(labels).astype(str)###把审核后的名字填到每个细胞上。##上一段的 Leiden 给每个细胞分了一个群编号，比如 0、`1`。这一段先整理每个群的特征，再读取你审核过的亚群名称。比如你的审核表写着 0 → CD56bright，那么群 0 里每个细胞的 subtype 都会填上 CD56bright，它们的 cluster 仍然记录为 0
    else:
        save(pd.DataFrame({"cluster": sorted(ad.obs.cluster.unique()), "subtype": ""}), a, "annotations_template")##没提供审核表，就生成一份模板。 else 接着上一段的 if a.annotations。
    _, comp = connected_components(ad.obsp["connectivities"], directed=False)###看看哪些细胞在邻居图里能走到一起。
    ad.obs["trajectory_component"] = comp###这里用到了第 12 行导入的 connected_components。比如 A 连着 B，B 连着 C，那么 A、B、C 属于同一块。函数返回两个结果：一共有几块，以及每个细胞属于哪一块。`_` 表示这里用不到第一个结果，`comp` 接住第二个结果，再存入 trajectory_component。后面选好起点后，程序就能知道哪些细胞和起点连通。
    s = ad.obs.get("cord_reference_score", ad.obs.precursor_like)
    eligible = ad.obs.age_group.eq("cord") & s.notna()
    ad.obs["pseudotime"] = np.nan###这三行准备选伪时间的起点
    if eligible.any():
        ids = np.flatnonzero((eligible & (s >= s[eligible].quantile(0.9))).to_numpy())
        z = ad.obsm[rep][ids]
        root = ids[np.argmin(np.square(z - np.median(z, axis=0)).sum(axis=1))]
        ad.uns["iroot"] = int(root)
        ad.uns["trajectory_root"] = str(ad.obs_names[root])###这六行从有分数的脐血细胞里挑出一个细胞，作为伪时间的起点
        sc.tl.diffmap(ad, n_comps=min(15, ad.n_obs - 1), random_state=a.seed)
        sc.tl.dpt(ad, n_dcs=min(10, ad.obsm["X_diffmap"].shape[1]))
        valid = (comp == comp[root]) & np.isfinite(ad.obs.dpt_pseudotime)
        ad.obs.loc[valid, "pseudotime"] = ad.obs.loc[valid, "dpt_pseudotime"]###上一段选好了脐血起点。这四行才真正计算伪时间，并把有效结果写回细胞信息表。
##伪时间算的是每个细胞在邻居图上离参考起点的相对距离。因为这个距离可以用来给细胞状态排出一个先后顺序。
#这段定义了一个多重检验校正函数。比如，你要比较青年组和老年组的 10 种 NK 亚群比例，每比较一次就得到一个 p 值。做的比较越多，就越容易碰巧得到某个很小的 p 值。这个函数用 Benjamini–Hochberg（BH）方法把原始 p 值转换成校正后的值，代码里记作 q。
def fdr(values):
    p = np.asarray(values, dtype=float)
    q = np.full(p.shape, np.nan)
    ids = np.flatnonzero(np.isfinite(p))
    ordered = ids[np.argsort(p[ids])]
    ranks = np.arange(1, len(ordered) + 1)
    q[ordered] = np.minimum(1, np.minimum.accumulate((p[ordered] * len(ordered) / ranks)[::-1])[::-1])
    return q

###接下来开始按供体计算 NK 亚群比例
def donor_outputs(ad, names, a):
    obs = ad.obs##用 obs 简称细胞信息表 ad.obs
    prop = pd.crosstab([obs.donor, obs.age_group], obs.subtype)###用 pd.crosstab(...) 数细胞。结果的每一行对应一个供体及其年龄组，每一列对应一个亚群。
    prop = prop.div(prop.sum(axis=1), axis=0)
    save(prop, a, "donor_subtype_fractions", index=True)
    rows = []####上一段已经算出了每位供体各自的亚群比例。这十行用那些比例来比较不同年龄组。
    for subtype in prop:
        for g1, g2 in itertools.combinations(GROUPS, 2):
            x, y = (prop.loc[prop.index.get_level_values("age_group") == group, subtype]
                    for group in (g1, g2))###决定比较哪两组，取出两组的比例。
            p = stats.mannwhitneyu(x, y).pvalue if min(len(x), len(y)) >= a.min_donors else np.nan
            rows.append([subtype, g1, g2, len(x), len(y), y.median() - x.median(), p])###算检验结果和差了多少。
    tests = pd.DataFrame(rows, columns=["subtype", "group1", "group2", "n1", "n2", "median_diff", "p"])
    tests["q"] = fdr(tests.p)
    save(tests, a, "subtype_comparisons")###整理并保存。先把所有比较结果变成一张表，再用刚学过的 fdr() 给 p 值做多重检验校正，把校正值存进 q 列。最后保存为 subtype_comparisons.tsv。结果表的一行，就是“一个 NK 亚群在两个年龄组之间的一次比较”。
####代码在同一位供体、同一个 NK 亚群里，检查细胞的伪时间和某项分数是否一起变化。
    # One correlation per donor/subtype; each donor has equal weight in the sign test.
    rows = []
    for (donor, age, subtype), g in obs.groupby(["donor", "age_group", "subtype"], observed=True):
        for key in names:
            t = g[["pseudotime", key]].dropna()
            if len(t) >= a.min_cells and t.nunique().min() > 1:
                rows.append([donor, age, subtype, key, len(t), stats.spearmanr(t.iloc[:, 0], t.iloc[:, 1]).statistic])
    corr = pd.DataFrame(rows, columns=["donor", "age_group", "subtype", "score", "cells", "rho"])
    save(corr, a, "within_donor_trajectory_associations")
    summary = []###上一段给每位供体各算了一个相关系数 rho。这段接着问：同一个年龄组里，多位供体的相关方向一致吗？
    for (age, subtype, key), g in corr.groupby(["age_group", "subtype", "score"]):
        nonzero = g.rho[g.rho != 0]
        p = stats.binomtest(int((nonzero > 0).sum()), len(nonzero)).pvalue if len(nonzero) >= a.min_donors else np.nan
        summary.append([age, subtype, key, len(g), g.rho.median(), p])
    summary = pd.DataFrame(summary, columns=["age_group", "subtype", "score", "donors", "median_rho", "p"])
    summary["q"] = fdr(summary.p)
    save(summary, a, "trajectory_association_summary")###你可以把两个指标分开读：median_rho 看典型的相关方向和强弱，`p` 和 q 看多位供体的方向是否一致到超出随机波动的预期。
    grouped = obs.groupby(["donor", "age_group", "subtype"], observed=True)###前面算的是分数是否随着伪时间变化。这里开始算另一件事：每位供体的某个 NK 亚群，整体分数是多少？
    summary = grouped[names].median().join(grouped.size().rename("n_cells"))
    save(summary, a, "donor_state_scores", index=True)
    rows = []
    for subtype, g in summary.reset_index().groupby("subtype", observed=True):
        g = g[g.n_cells >= a.min_cells]###例如，D01 有 30 个 CD56bright 细胞，D02 只有 8 个。两位供体的分数都会写进汇总文件，但下一步比较 CD56bright 分数时，默认只纳入 D01。
        for name in names:###这段和前面的亚群比例比较很像。这一次问的是：同一种 NK 亚群，在不同年龄组里的状态分数有没有差异？
            for g1, g2 in itertools.combinations(GROUPS, 2):
                x, y = (g.loc[g.age_group == group, name].dropna() for group in (g1, g2))
                p = stats.mannwhitneyu(x, y).pvalue if min(len(x), len(y)) >= a.min_donors else np.nan
                rows.append([subtype, name, g1, g2, len(x), len(y), y.median() - x.median(), p])
    contrasts = pd.DataFrame(rows, columns=["subtype", "score", "group1", "group2",
                                             "n1", "n2", "median_diff", "p"])
    contrasts["q"] = fdr(contrasts.p)
    save(contrasts, a, "state_score_comparisons")##最后保存比较结果。 pd.DataFrame(...) 把每次比较整理成一行，`fdr(...)` 计算校正后的 q，然后保存为 state_score_comparisons.tsv。例如，你可以在这张表里找到**“CD56dim 的耗竭相关分数，青年组与老年组的比较”**这一行。

###下面开始定义 rewiring()，用来比较同一个 NK 亚群里，转录因子和靶基因的表达关联在青年组和老年组之间有没有变化。
def rewiring(ad, edges, a):
    cols = ["subtype", "module", "tf", "target", "prior_direction", "n_young", "n_old",
            "rho_young", "rho_old", "delta_rho", "p", "q"]
    rows = []
    if edges is not None:
        rng = np.random.default_rng(a.seed)
        genes = sorted((set(edges.tf) | set(edges.target)) & set(ad.var_names))
        for subtype, sub in ad.obs.groupby("subtype", observed=True):##这里的调控关系由你提供。接下来的代码会检查这些候选关系在两组供体中的表达相关性，给你线索判断哪些关系值得进一步验证。
            if not genes:###对每个 NK 亚群，把同一位供体的细胞计数加起来。处理前是一行一个细胞，处理后是一行一位供体。这种汇总叫 pseudobulk。
                continue
            subset = ad[sub.index, genes]
            donor_codes, donors = pd.factorize(sub.donor, sort=True)
            pool = sparse.csr_matrix(
                (np.ones(len(sub)), (donor_codes, np.arange(len(sub)))),
                shape=(len(donors), len(sub)),
            )
            pb = pd.DataFrame((pool @ subset.layers["counts"]).toarray(), index=donors, columns=genes)
            # Normalize by all-gene library size rather than the tested gene panel.###这里先做加总。供体测到的细胞数和总计数可能不同，下一段才会对这些汇总值做归一化。
            library = sub.groupby("donor").total_counts.sum()###先把各供体的基因计数换到同一个尺度，再检查青年组和老年组各有多少位供体可以参与比较
            pb = np.log1p(pb.div(library, axis=0) * 1e6)
            info = sub.groupby("donor").agg(age=("age_group", "first"), cells=("age_group", "size"))
            groups = [info.index[(info.age == cohort) & (info.cells >= a.min_cells)] for cohort in ("young", "old")]
            if min(map(len, groups)) < a.min_donors:
                continue
            n1, n2 = map(len, groups)
            for e in edges.itertuples(index=False):###检查两组供体数。 map(len, groups) 数出两份名单各有多少人。比如青年组有 6 位，老年组有 4 位，默认要求每组至少 5 位，`continue` 就会跳过这个亚群。两组都够时，把人数分别存入 n1 和 n2，供下一段比较基因相关性。
                if e.tf == e.target or e.tf not in pb or e.target not in pb:###这九行对每一对转录因子和靶基因，分别计算它们在青年供体和老年供体中的表达相关性。
                    continue
                values = pb.loc[groups[0].append(groups[1]), [e.tf, e.target]].to_numpy()
                def rho(v):
                    if np.unique(v[:, 0]).size < 2 or np.unique(v[:, 1]).size < 2:
                        return np.nan
                    return stats.spearmanr(v[:, 0], v[:, 1]).statistic
                r1, r2 = rho(values[:n1]), rho(values[n1:])
                if not np.isfinite([r1, r2]).all():###比如，在同一个 NK 亚群里，如果青年供体中转录因子表达越高的人，靶基因表达也往往越高，`r1` 就是正数。如果老年供体中这种排列关系很弱，`r2` 就可能接近 0。这里先分别算出两个相关系数，下一段才比较它们差多少。
                    continue###这段先算出真实分组下的相关性差异，再把供体的分组打乱多次，看看仅靠随机分组能不能产生同样大的差异。
                delta = r2 - r1
                extreme = valid = 0
                for _ in range(a.permutations):
                    v = values[rng.permutation(len(values))]
                    diff = rho(v[n1:]) - rho(v[:n1])
                    if np.isfinite(diff):
                        valid += 1
                        extreme += abs(diff) >= abs(delta) - 1e-12
                p = (1 + extreme) / (1 + valid) if valid >= a.permutations * 0.9 else np.nan###比如，打乱 999 次后，有 990 次能算出有效差异，其中 20 次的差异至少和真实差异一样大，那么 valid = 990，`extreme = 20`。下一行就会用这两个数计算 p 值。
                rows.append([subtype, e.module, e.tf, e.target, e.direction, n1, n2, r1, r2, delta, p])###上一段数出了多少次随机分组能算出结果，以及其中多少次的差异至少和真实差异一样大。这六行用这两个数计算 p 值，再保存结果。
    result = pd.DataFrame(rows, columns=cols[:-1])
    result["q"] = fdr(result.p)
    save(result, a, "candidate_regulatory_rewiring")
    return result##读结果时，delta_rho 告诉你相关性变强还是变弱，`q` 帮你判断这项差异在做了多次比较后是否仍有统计证据。即使 q 很小，得到的也是两组供体之间的表达相关性差异，可以作为候选调控关系进一步验证。
###以上是第一步，将细胞划分
###下面开始处理你用 --response 提供的 IL-2／IL-15 功能实验数据。
def cytokine_response(ad, a):
    columns = ["subtype", "cytokine", "endpoint", "n", "mean_delta", "ci_low", "ci_high", "evidence"]
    rows = []
    if a.response:
        if not a.annotations:
            raise ValueError("Response assays require reviewed subtype annotations")
        tab = pd.read_csv(a.response, sep="\t", dtype={"donor": str})
        keys = ["donor", "subtype", "cytokine", "endpoint"]
        if tab[keys].isna().any().any() or tab.duplicated(keys).any():###到这里，程序已经读进实验表。下一段才会检查这些记录有没有缺项或重复。
            raise ValueError("Require one matched assay row per donor/subtype/cytokine/endpoint")##这六行做三项检查。任何一项没通过，程序都会报错并停止。
        if not np.isfinite(tab[["baseline", "stimulated"]].to_numpy(dtype=float)).all():
            raise ValueError("Assay measurements must be finite")
        if not set(tab.cytokine) <= {"IL-2", "IL-15"}:
            raise ValueError("cytokine must be IL-2 or IL-15")
        pairs = set(zip(tab.donor, tab.subtype))###这一步只检查实验表本身。接下来才会核对表里的供体和亚群能不能在 NK 图谱里找到。
        known = set(zip(ad.obs.loc[ad.obs.age_group == "old", "donor"],##这七行先核对实验记录能不能和老年 NK 图谱对上，再把实验数据分组，准备算刺激后的变化
                        ad.obs.loc[ad.obs.age_group == "old", "subtype"]))
        if not pairs <= known:
            raise ValueError("Each assay donor/subtype must be present in the old cohort atlas")
        margin = a.response_margin
        grouped = list(tab.groupby(keys[1:]))
        for key, g in grouped:
            delta = g.stimulated - g.baseline###这七行先核对实验记录能不能和老年 NK 图谱对上，再把实验数据分组，准备算刺激后的变化
            n, mean = len(delta), delta.mean()
            lo = hi = np.nan
            evidence = "insufficient_donors"
            if n >= a.min_donors:
                critical = stats.t.ppf(1 - 0.05 / (2 * len(grouped)), n - 1)
                half = critical * stats.sem(delta)
                lo, hi = mean - half, mean + half
                evidence = "inconclusive"###比如，另一组实验的平均改善是 0.09，算出的半宽是 0.03，区间就是 [0.06, 0.12]。这表示平均改善的估计是 9 个百分点，区间从 6 到 12 个百分点。下一段会拿这个区间和你设定的 --response-margin 比较。
                if lo > margin:#这十行用整个置信区间和你设定的阈值 margin 比较，给实验结果贴一个标签。比如 margin = 0.05，就是把 5 个百分点设为有实际意义的变化门槛。
                    evidence = "response_supported"
                elif lo > -margin and hi < margin:
                    evidence = "small_response_supported"
                elif hi < -margin:
                    evidence = "decrease_supported"
            rows.append([*key, n, mean, lo, hi, evidence])
    response = pd.DataFrame(rows, columns=columns)
    save(response, a, "cytokine_response")
    subtypes = pd.DataFrame({"subtype": sorted(ad.obs.loc[ad.obs.age_group == "old", "subtype"].unique())})###将实验结果保存。读表时，尤其要分清 “证据暂不明确” 和 “有证据支持变化较小”。比如 [0.02, 0.10] 仍容许改善达到 10 个百分点，`[-0.02, 0.03]` 则整个区间都在小变化范围内。
    status = []###读表时，尤其要分清 “证据暂不明确” 和 “有证据支持变化较小”。比如 [0.02, 0.10] 仍容许改善达到 10 个百分点，`[-0.02, 0.03]` 则整个区间都在小变化范围内。
    for subtype in subtypes.subtype:
        evidence = response[response.subtype == subtype]
        both_low = any(set(g.cytokine) == {"IL-2", "IL-15"} and
                       g.evidence.eq("small_response_supported").all()
                       for _, g in evidence.groupby("endpoint"))
        status.append("response_supported" if evidence.evidence.eq("response_supported").any() else###比如，IL-2 和 IL-15 都测了靶细胞杀伤，两项都支持变化较小，`both_low` 就是 True。。如果 IL-2 测的是杀伤、IL-15 测的是另一项指标，即使两项变化都小，也不能凑成同一个指标下的双低响应。这里先得到一个 True 或 False，下一段才会用它来确定亚群的汇总标签。
                      "small_response_both_cytokines" if both_low else "undetermined")###这几行给每个老年 NK 亚群汇总一个功能响应标签。
    subtypes["functional_evidence"] = status
    save(subtypes, a, "old_subtype_evidence")###到这里，功能实验的分析就结束了。


###这几行给 RNA 速度分析准备数据。主程序只有在你开启 --velocity 时，才会调用这个函数。
def velocity(ad, a):
    import scvelo as scv

    if not {"spliced", "unspliced"} <= set(ad.layers):
        raise ValueError("Velocity requires raw spliced and unspliced layers aligned in the RNA h5ad")
    from anndata import AnnData

    vel = AnnData(X=ad.layers["counts"].copy(), obs=ad.obs.copy(), var=ad.var.copy())
    vel.obsm["X_umap"] = ad.obsm["X_umap"].copy()##这里先准备了 vel 的原始表达、行列信息和画图坐标。下一段才会把已剪接和未剪接的两张计数表放进去。
    for layer in ("spliced", "unspliced"):###这十行完成 RNA 速度分析的预处理和计算。可以分成四步来看。
        vel.layers[layer] = counts(ad.layers[layer], layer)
    scv.pp.filter_and_normalize(vel, min_shared_counts=20)
    if vel.n_vars < 3:
        raise ValueError("Velocity filtering left fewer than three shared genes")
    scv.pp.moments(vel, n_pcs=min(30, min(vel.shape) - 1),
                   n_neighbors=min(a.neighbors, vel.n_obs - 1))
    if a.velocity == "dynamical":
        scv.tl.recover_dynamics(vel, n_jobs=1)
    scv.tl.velocity(vel, mode=a.velocity)##到第 479 行，程序得到的是基因层面的变化估计。下一段才会把许多基因的变化合起来，判断一个细胞的整体表达状态更可能朝哪个方向移动。
    scv.tl.velocity_graph(vel, n_jobs=1)###上一段估计了每个细胞里各个基因的表达变化方向。这一段用这些变化来判断细胞整体状态更朝向哪些邻居，并准备和伪时间做比较。
    scv.tl.velocity_confidence(vel)
    scv.tl.velocity_embedding(vel, basis="umap")
    t = ad.obs.pseudotime.to_numpy()
    graph = vel.uns["velocity_graph"].tocsr()
    observed = np.isfinite(t)
    within = graph[:, observed]###到这里，两种结果已经对齐。下一段会问：RNA 速度指向的那些邻居，伪时间通常比当前细胞更大，还是更小？
    denominator = np.asarray(within.sum(axis=1)).ravel()###这七行回答一个问题：RNA 速度指向的邻居，伪时间通常比当前细胞更大，还是更小？
    expected = np.divide(within @ t[observed], denominator,
                         out=np.full(len(t), np.nan), where=denominator > 0)
    ad.obs["velocity_dpt_delta"] = expected - t
    ad.obs["velocity_confidence"] = vel.obs.velocity_confidence
    ad.obsm["velocity_umap"] = vel.obsm["velocity_umap"]
    vel.write_h5ad(a.out / "velocity.h5ad")##第 491–493 行把速度置信度和 UMAP 箭头坐标复制到主图谱 ad，再把完整的 RNA 速度分析对象 vel 保存为 velocity.h5ad。

#开始画图。接下来开始定义 plot_atlas()，负责把前面算好的结果画成图。
def plot_atlas(ad, network, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = ["age_group", "subtype", "pseudotime", "cytotoxicity_proxy",
               "senescence_proxy", "exhaustion_proxy", "cord_reference_score",
               "old_state_score", "senolytic_signature", "velocity_dpt_delta"]
    for name in columns:###这五行把上一段列出的项目逐个画成 UMAP 图。以 pseudotime 为例：
        if name in ad.obs and ad.obs[name].notna().any():
            sc.pl.umap(ad, color=name, show=False)
            plt.savefig(a.out / f"umap_{name}.png", dpi=150, bbox_inches="tight")
            plt.close()###关闭刚画完的图，释放它占用的资源。已经保存的图片仍在输出文件夹里，循环可以继续画下一张。
    top = network.dropna(subset=["q"]).sort_values("q").head(30)###前面画的是每个细胞在 UMAP 上的位置。这段开始画另一张图，用颜色比较同一对基因在青年和老年供体中的表达相关性。
    if len(top):
        labels = top.subtype + " | " + top.tf + " → " + top.target
        fig, ax = plt.subplots(figsize=(5, max(3, 0.26 * len(top))))
        image = ax.imshow(top[["rho_young", "rho_old"]], aspect="auto",
                          cmap="coolwarm", vmin=-1, vmax=1)##比如，某一行左边是红色、右边接近白色，表示这对基因在青年供体中呈正相关，在老年供体中相关性接近零。下一段会加上行列标签和色标，再把图保存下来。
        ax.set_xticks([0, 1], ["Young", "Old"])##上一段已经画出了热图里的色块。这六行让你看得懂每块颜色对应什么，再把图保存下来。
        ax.set_yticks(range(len(top)), labels)
        ax.set_title("Candidate regulatory associations")
        fig.colorbar(image, ax=ax, label="Donor Spearman rho")
        fig.savefig(a.out / "regulatory_edges.png", dpi=150, bbox_inches="tight")
        plt.close(fig)###看图时，可以横着读同一行，比较同一对基因在青年组和老年组的相关性。想查这项差异的 p 或 q 值，就看 candidate_regulatory_rewiring.tsv。


def main():##接下来开始定义 main()，它会按顺序调用前面学过的函数，把整套分析串起来。
    a = arguments()
    a.out.mkdir(parents=True, exist_ok=True)
    if any(a.out.iterdir()):
        raise ValueError("Use an empty output directory to prevent mixing runs")
    (a.out / "config.json").write_text(json.dumps(vars(a), indent=2, default=str))
    ad = prepare(a)
    edges, scores = add_scores(ad, a)##这段到第 530 行为止，数据和各项分数已经准备好。接下来才会选高变基因、整合特征和分群。
    sc.pp.highly_variable_genes(ad, n_top_genes=min(a.hvg, ad.n_vars), flavor="seurat")##这七行先选出用于降维的高变基因，再把 RNA 和可用的其他组学特征整合成一张表。
    genes = ad.var_names[ad.var.highly_variable]
    if len(genes) < 3:
        raise ValueError("Fewer than three HVGs; check RNA input")
    x = ad[:, genes].X.tocsr().astype(np.float32)
    z = integrate(ad, x, a)
    ad.obsm["X_integrated"] = z##这里最容易混的是 x 和 z。`x` 的每一列仍对应一个具体基因。`z` 的每一列是处理后用于比较细胞的特征。
    if a.embedding == "graph-vae":##这十行把前面学过的建图和训练函数接到主程序里。先训练一个基因等权的图 VAE。如果你指定了模块权重，再训练一个加权版本。
        graph = stage_graph(z, ad.obs.age_group.to_numpy(), a)
        ad.obsp["within_stage_graph"] = graph
        weights = np.ones(len(genes), dtype=np.float32)
        ad.obsm["X_graph_vae"] = graph_vae(x, z, graph, weights, a, "graph_vae")
        if a.module_weight:
            weights += a.module_weight * genes.isin(edges.target).astype(np.float32)
            if np.all(weights == 1):
                raise ValueError("No prior module targets among HVGs")
            ad.obsm["X_weighted_graph_vae"] = graph_vae(x, z, graph, weights, a, "weighted_graph_vae")##这里可以对照着记：第 539 行决定细胞参考哪些邻居，第 544 行决定模型更重视哪些基因的重建误差
    # Use unsmoothed, unweighted multimodal coordinates for primary trajectory and tests.###这十行先做主分群和伪时间，再用图 VAE 的特征另外做分群，看看两套分群有多一致。
    cluster_and_order(ad, "X_integrated", a)
    for tag in ("graph_vae", "weighted_graph_vae"):
        if "X_" + tag in ad.obsm:
            sc.pp.neighbors(ad, use_rep="X_" + tag, key_added=tag,
                            n_neighbors=min(a.neighbors + 1, ad.n_obs - 1), random_state=a.seed)
            key = tag + "_cluster"
            sc.tl.leiden(ad, neighbors_key=tag, key_added=key, resolution=a.resolution,
                         random_state=a.seed, flavor="igraph", n_iterations=2)
            save(pd.crosstab(ad.obs.cluster, ad.obs[key]), a, tag + "_cluster_overlap", index=True)#####所以，主分析给出后面统计要用的亚群和伪时间，图 VAE 的分群结果用来对照。如果主分群的某个群在图 VAE 里被拆成了好几个群，这张交叉表就能帮你找到具体是哪一群。
    donor_outputs(ad, [*scores, *[c for c in ("cord_reference_score", "old_state_score") if c in ad.obs]], a)##这十行是主程序的收尾。它调用前面学过的统计函数，再把结果存成表格、图片和完整图谱。
    network = rewiring(ad, edges, a)
    cytokine_response(ad, a)
    if a.velocity != "off":
        velocity(ad, a)
    ad.obsm["X_umap"] = ad.obsm["X_umap"].astype(np.float32)
    save(ad.obs.join(pd.DataFrame(ad.obsm["X_umap"], index=ad.obs_names, columns=["UMAP1", "UMAP2"])),
         a, "cells", index=True)
    plot_atlas(ad, network, a)
    ad.write_h5ad(a.out / "nk_atlas.h5ad", compression="gzip")###以后如果你只想查某个细胞属于哪个亚群、伪时间是多少，可以打开 cells.tsv。如果你想在 Python 里重新打开图谱，继续使用表达矩阵、细胞信息和邻居图，就读 nk_atlas.h5ad。


if __name__ == "__main__":
    main()
##这两行决定什么时候启动整套分析。
###如果你以后在另一份 Python 文件里写 from nk_aging_atlas import score，就可以单独使用打分函数，整套分析不会自动启动。