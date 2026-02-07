"""
模块: find_best_match
功能: 给定一条输入序列，查找 data/tRNAs_flat.tsv 中与之比对得分最高的 tRNA，并返回详细信息。
    可被其他脚本通过 import 调用。接口保持不变:
    find_best_match(input_seq, tsv_path=..., match=..., mismatch=..., gap_open=..., gap_extend=...)
"""
from __future__ import annotations
import csv
import os
import pickle
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from math import ceil
import time
from typing import List, Tuple, Dict, Any, Set

from pythonscript.alignment_score import alignment_score_and_str
from pythonscript.debug_config import env_flag as _env_flag, env_int as _env_int, DEBUG_MATCH as _DEBUG_MATCH

# --------------------------- 模块级缓存 ---------------------------
_CACHE = {
    "tsv_path": None,
    "records": None,        # 原始行(list[dict])
    "slim_rows": None,      # list[(seqname, flat_seq, extra)]
    "size": 0,
}

# 为快速粗筛预留的索引缓存（bitset/inverted list 思路，类似 MySQL 中的 VARBINARY 预编码）
_FAST_CACHE: Dict[str, Any] = {
    "tsv_path": None,
    "k": None,
    "inverted": None,  # list[list[int]]，下标为编码后的 k-mer
    "size": 0,
}

def _load_tsv_if_needed(tsv_path: str):
    """只在路径变化或首次调用时加载 TSV 并构建 slim_rows；如有缓存则优先读缓存。"""
    global _CACHE
    if _CACHE["tsv_path"] == tsv_path and _CACHE["records"] is not None:
        return
    cache_path = Path(tsv_path).with_suffix(Path(tsv_path).suffix + ".slim.pkl")
    tsv_mtime = Path(tsv_path).stat().st_mtime if Path(tsv_path).exists() else 0.0

    # 优先尝试读取缓存（要求缓存比 TSV 新）
    if cache_path.exists() and cache_path.stat().st_mtime >= tsv_mtime:
        try:
            with cache_path.open("rb") as fin:
                data = pickle.load(fin)
            if isinstance(data, dict) and "slim_rows" in data and "records" in data:
                _CACHE.update(data)
                _CACHE["tsv_path"] = tsv_path
                return
        except Exception:
            pass  # 读取失败则回退重建

    records: List[Dict[str, Any]] = []
    with open(tsv_path, 'r', encoding='utf-8') as fin:
        reader = csv.DictReader(fin, delimiter='\t')
        cols = [c for c in reader.fieldnames if c != 'seqname']
        for row in reader:
            # 预扁平化，后续无需再 join
            flat = ''.join(row[c] for c in cols)
            row['flat_seq'] = flat
            records.append(row)

    slim_rows: List[Tuple[str, str, Dict[str, Any]]] = []
    for row in records:
        seqname = row['seqname']
        flat_seq = row['flat_seq']
        extra = {k: v for k, v in row.items() if k not in ('seqname', 'flat_seq')}
        slim_rows.append((seqname, flat_seq, extra))

    _CACHE = {
        "tsv_path": tsv_path,
        "records": records,
        "slim_rows": slim_rows,
        "size": len(records),
    }

    # 将 slim_rows/records 落盘缓存，后续启动无需重新解析 TSV
    try:
        with cache_path.open("wb") as fout:
            pickle.dump(_CACHE, fout, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass

# --------------------------- 预筛 (k-mer) ---------------------------

def _kmerset(seq: str, k: int) -> Set[str]:
    s = seq.upper().replace('T', 'U')
    n = len(s)
    if n <= 0:
        return set()
    if n <= k:
        return {s}
    return {s[i:i+k] for i in range(n - k + 1)}

def _prefilter_params(total_records: int) -> tuple[int, int]:
    """
    返回 (k, topn)
    - k: TRNAALIGN_PREFILTER_K，默认 5
    - topn: 取前 max(200, frac*N)，但不超过 TRNAALIGN_PREFILTER_TOPN（默认 500）
    """
    k = int(os.getenv("TRNAALIGN_PREFILTER_K", "5"))
    frac = float(os.getenv("TRNAALIGN_PREFILTER_FRAC", "0.05"))  # 5%
    topn_abs = int(os.getenv("TRNAALIGN_PREFILTER_TOPN", "500"))
    topn_by_frac = int(max(1, total_records * frac))
    topn = min(topn_abs, max(200, topn_by_frac))
    return k, topn

def _prefilter_candidates(
    input_seq: str,
    slim_rows: List[Tuple[str, str, Dict[str, Any]]],
    total_records: int
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    用 k-mer overlap 粗筛，返回候选三元组列表 (seqname, flat_seq, extra)。
    如极端情况下 overlap=0，则回退取前 topn 个（按原顺序）。
    """
    k, topn = _prefilter_params(total_records)
    qset = _kmerset(input_seq, k)
    scored: List[Tuple[str, str, Dict[str, Any], int]] = []
    for seqname, flat_seq, extra in slim_rows:
        oscore = len(qset & _kmerset(flat_seq, k))
        if oscore > 0:
            scored.append((seqname, flat_seq, extra, oscore))

    if scored:
        scored.sort(key=lambda x: x[3], reverse=True)
        return [(s, f, e) for (s, f, e, _o) in scored[:topn]]

    # 没有 overlap（非常罕见），保底：取前 topn 条
    return slim_rows[:topn]

# --------------------------- 快速粗筛（bitset/倒排表） ---------------------------

_BASE2INT = {"A": 0, "C": 1, "G": 2, "U": 3}

def _dprint(msg: str):
    if _DEBUG_MATCH:
        print(msg)

def _encode_kmer(kmer: str) -> int | None:
    """把 k-mer 编码成 base4 整数，遇到非 A/C/G/U 返回 None。"""
    val = 0
    for ch in kmer:
        v = _BASE2INT.get(ch)
        if v is None:
            return None
        val = (val << 2) | v
    return val

def _ensure_fast_index(tsv_path: str,
                       slim_rows: List[Tuple[str, str, Dict[str, Any]]],
                       k: int):
    """构建/复用倒排表，避免每次调用重复预处理；优先尝试缓存。"""
    global _FAST_CACHE
    if (_FAST_CACHE["tsv_path"] == tsv_path and
            _FAST_CACHE["k"] == k and
            _FAST_CACHE.get("inverted")):
        return

    cache_path = Path(tsv_path).with_suffix(Path(tsv_path).suffix + f".fastk{k}.pkl")
    tsv_mtime = Path(tsv_path).stat().st_mtime if Path(tsv_path).exists() else 0.0
    if cache_path.exists() and cache_path.stat().st_mtime >= tsv_mtime:
        try:
            with cache_path.open("rb") as fin:
                data = pickle.load(fin)
            if isinstance(data, dict) and data.get("k") == k and data.get("inverted"):
                _FAST_CACHE = data
                _FAST_CACHE["tsv_path"] = tsv_path
                return
        except Exception:
            pass  # 读取失败则重建

    vocab = 4 ** k
    inverted: List[List[int]] = [[] for _ in range(vocab)]

    for idx, (_seqname, flat_seq, _extra) in enumerate(slim_rows):
        s = flat_seq.upper().replace('T', 'U')
        n = len(s)
        if n < k:
            continue
        seen = set()
        for i in range(n - k + 1):
            enc = _encode_kmer(s[i:i + k])
            if enc is None or enc in seen:
                continue
            seen.add(enc)
            inverted[enc].append(idx)

    _FAST_CACHE = {
        "tsv_path": tsv_path,
        "k": k,
        "inverted": inverted,
        "size": len(slim_rows),
    }

    try:
        with cache_path.open("wb") as fout:
            pickle.dump(_FAST_CACHE, fout, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass

def _fast_prefilter_candidates(
    input_seq: str,
    slim_rows: List[Tuple[str, str, Dict[str, Any]]],
    total_records: int,
    tsv_path: str
) -> List[Tuple[str, str, Dict[str, Any]]] | None:
    """
    参考数据库 bitset 的思路：k-mer 倒排表 + 命中计数。
    仅返回 topN 候选，未命中时返回 None 以便回退到旧逻辑。
    """
    k = _env_int("TRNAALIGN_FAST_K", 6)
    # 防止内存暴涨，k 约束在 [3, 8]
    k = max(3, min(8, k))
    topn = _env_int("TRNAALIGN_FAST_TOPN", 120)
    topn = max(50, min(topn, total_records))

    _ensure_fast_index(tsv_path, slim_rows, k)
    inverted: List[List[int]] = _FAST_CACHE["inverted"]  # type: ignore
    if not inverted:
        return None

    s = input_seq.upper().replace("T", "U")
    n = len(s)
    if n < k:
        return None

    qset: Set[int] = set()
    for i in range(n - k + 1):
        enc = _encode_kmer(s[i:i + k])
        if enc is not None:
            qset.add(enc)

    if not qset:
        return None

    # 计数命中次数（相当于交集大小），避免扫描全库
    counts: Dict[int, int] = defaultdict(int)
    for enc in qset:
        posting = inverted[enc]
        for tpl_id in posting:
            counts[tpl_id] += 1

    if not counts:
        return None

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_ids = [idx for idx, _hit in ranked[:topn]]
    return [slim_rows[i] for i in top_ids]

# --------------------------- 并发策略 ---------------------------

def _decide_workers(total_tasks: int) -> int:
    """
    自适应线程数：
      - TRNAALIGN_WORKERS 优先；
      - 默认不超过 CPU 核数，可用 TRNAALIGN_MAX_WORKERS 设上限。
    """
    env = os.getenv("TRNAALIGN_WORKERS")
    if env:
        try:
            v = int(env)
            if v >= 1:
                return min(v, total_tasks) if total_tasks > 0 else 1
        except Exception:
            pass
    cpu = os.cpu_count() or 1
    default_cap = min(cpu, 64)  # 默认封顶 64，避免过度调度；可用 TRNAALIGN_MAX_WORKERS 调高或调低
    try:
        cap = int(os.getenv("TRNAALIGN_MAX_WORKERS", default_cap))
    except Exception:
        cap = default_cap
    cap = max(1, cap)
    return max(1, min(cpu, total_tasks, cap))

# --------------------------- 打分包装 ---------------------------

def _score_one(pack, input_seq, match, mismatch, gap_open, gap_extend):
    seqname, flat_seq, extra = pack
    score, aln = alignment_score_and_str(
        input_seq, flat_seq, match, mismatch, gap_open, gap_extend
    )
    return score, seqname, flat_seq, extra, aln

# --------------------------- 主函数（接口不变） ---------------------------

def find_best_match(input_seq: str,
                    tsv_path: str = os.path.join('data', 'tRNAs_flat.tsv'),
                    match: float = 2.0,
                    mismatch: float = -1.0,
                    gap_open: float = -2.0,
                    gap_extend: float = -0.5):
    """
    流程：
      1) 模块级缓存加载 TSV，构建 slim_rows（只做一次）
      2) k-mer 粗筛 → 大幅缩小候选集合
      3) 对候选做准确打分（edlib/DP），线程并发精排
      4) 返回得分最高的记录（含原 TSV 其它字段）
    """
    _load_tsv_if_needed(tsv_path)
    slim_rows = _CACHE["slim_rows"]
    total = _CACHE["size"]
    if not slim_rows or total == 0:
        return None

    # 预筛：先尝试 fast 倒排表模式，失败则回退旧逻辑
    candidates = None
    used_fast = False
    if _env_flag("TRNAALIGN_FAST_PREFILTER", True):
        candidates = _fast_prefilter_candidates(input_seq, slim_rows, total, tsv_path)
        used_fast = candidates is not None
    if not candidates:
        candidates = _prefilter_candidates(input_seq, slim_rows, total)
        used_fast = False

    # DP 精排条数上限，可用 TRNAALIGN_SCORE_TOPN 控制；0/负数表示不截断
    # 默认不截断；若显式设置 TRNAALIGN_SCORE_TOPN>0 则截断
    score_cap = _env_int("TRNAALIGN_SCORE_TOPN", 0)
    if score_cap > 0:
        candidates = candidates[:min(score_cap, len(candidates))]

    n_tasks = len(candidates)
    n_workers = _decide_workers(n_tasks)
    approx = ceil(n_tasks / n_workers) if n_workers else n_tasks

    align_backend = os.getenv("TRNAALIGN_BACKEND_LOG")
    if align_backend and _DEBUG_MATCH:
        _dprint(f"[find_best_match] backend_hint={align_backend}")
    _dprint(
        f"[find_best_match] total={total}, prefilter_candidates={n_tasks}, "
        f"workers={n_workers}, approx_per_worker={approx}, fast_prefilter={used_fast}"
    )

    # 精排（线程池）
    best = None  # (score, seqname, flat_seq, extra, aln)
    if n_workers <= 1 or n_tasks <= 1:
        for pack in candidates:
            res = _score_one(pack, input_seq, match, mismatch, gap_open, gap_extend)
            if best is None or res[0] > best[0]:
                best = res
    else:
        use_proc = _env_flag("TRNAALIGN_USE_PROCESS", False)
        ExecutorCls = ProcessPoolExecutor if use_proc else ThreadPoolExecutor
        try:
            with ExecutorCls(max_workers=n_workers) as ex:
                futs = [
                    ex.submit(_score_one, pack, input_seq, match, mismatch, gap_open, gap_extend)
                    for pack in candidates
                ]
                for fut in as_completed(futs):
                    res = fut.result()
                    if best is None or res[0] > best[0]:
                        best = res
        except PermissionError:
            # 某些受限环境禁用进程池，自动回退线程池
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futs = [
                    ex.submit(_score_one, pack, input_seq, match, mismatch, gap_open, gap_extend)
                    for pack in candidates
                ]
                for fut in as_completed(futs):
                    res = fut.result()
                    if best is None or res[0] > best[0]:
                        best = res

    if best is None:
        return None

    score, seqname, flat_seq, extra, aln = best
    result = {
        'seqname': seqname,
        'flat_seq': flat_seq,
        'score': score,
        'alignment': aln,
    }
    result.update(extra)
    return result


if __name__ == '__main__':
    seq = "GGUCUCGUGGCCCAAUGGUUAAGGCGCUUGACUACGGAUCAAGAGAUUCCAGGUUCGACUCCUGGCGGGAUCG"
    t0 = time.time()
    info = find_best_match(seq, tsv_path='data/tRNAs_flat.tsv')
    elapsed = time.time() - t0
    if info is None:
        print('未找到匹配。')
    else:
        print(f"最佳匹配 seqname: {info['seqname']}")
        print(f"匹配序列    : {info['flat_seq']}")
        print(f"比对分数    : {info['score']}")
        print("对齐结果:\n", info['alignment'])
        print("其他字段:")
        for k, v in info.items():
            if k in ('seqname', 'flat_seq', 'score', 'alignment'):
                continue
            print(f"  {k}: {v}")
        print(f"耗时: {elapsed:.3f} 秒")
