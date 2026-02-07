# -*- coding: utf-8 -*- 
"""
验证/筛选模块：
- 公共词表与工具（tokenize/detokenize、GC、Hamming、结构打包）
- tRNAscan 注释 + 六区域切分
- 结构有效性判定
- 语言模型一致性分数（pseudo log-prob）
- 候选筛选与重排（filter_and_rerank）
"""

import os
import tempfile
import subprocess
from typing import List, Dict, Optional, Tuple, Iterable

import torch
import torch.nn.functional as F


from AYLM.utils.tRNAscan_SE_annotator import annotate_trna_sequences
from AYLM.extract import (
    extract_amino_acid_arm,
    extract_d_loop,
    extract_anticodon_arm,
    extract_variable_loop,
    extract_t_arm,
)

# ===== 词表 / 常量 =====
TOK2ID = {'pad':0,'A':1,'C':2,'G':3,'U':4,'T':4,'M':5}
ID2TOK = {0:'pad',1:'A',2:'C',3:'G',4:'U',5:'M'}
MASK_ID = TOK2ID['M']

REGION_ORDER = [
    'AA_Arm_5prime',
    'D_Loop',
    'Anticodon_Arm',
    'Variable_Loop',
    'T_Arm',
    'AA_Arm_3prime',
]

# ===== 基础工具 =====
def tokenize_seq(seq: str) -> List[int]:
    return [TOK2ID.get(c.upper(), 0) for c in seq]

def detokenize(ids: Iterable[int]) -> str:
    out = []
    for i in ids:
        if i == 0:
            continue
        out.append("N" if i == MASK_ID else ID2TOK.get(i, "N"))
    return "".join(out)

def tokenize_struct(struct: str) -> List[int]:
    m = {'.':1,'<':2,'>':3}
    return [m.get(c, 0) for c in struct]

def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))

def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    gc = sum(1 for c in seq.upper() if c in ("G","C"))
    return gc / len(seq)

def pairs_balance(s: str) -> Tuple[int, int]:
    return s.count('<'), s.count('>')


def _restore_3prime_tail(original_seq: str, regions: Dict[str, Dict[str, str]], verbose: bool = False) -> None:
    """
    tRNAscan-SE 偶尔会截掉 3' CCA 尾或将末端替换成模板。对比原始输入与
    StructFileSequence，若检测到截断或末端不一致，则用原始序列补齐，
    并把缺失部分归到 AA_Arm_3prime。
    """

    def _normalize(s: str) -> str:
        return (s or "").upper().replace("T", "U")

    target_seq = _normalize(original_seq)
    if not target_seq:
        return

    sf_seq_raw = regions.get("StructFileSequence", "") or ""
    sec_struct = regions.get("SecondaryStructure", "") or ""
    sf_seq_norm = _normalize(sf_seq_raw)

    if not sf_seq_norm:
        regions["StructFileSequence"] = target_seq
        return

    aa3 = regions.get("AA_Arm_3prime") or {"Seq": "", "Struct": ""}
    aa3_seq = _normalize(aa3.get("Seq", ""))
    aa3_struct = aa3.get("Struct", "") or "." * max(len(aa3_seq), 1)

    updated_seq = sf_seq_norm
    updated_struct = sec_struct

    if len(target_seq) > len(sf_seq_norm) and target_seq.startswith(sf_seq_norm):
        # 典型：CCA 被截掉
        suffix = target_seq[len(sf_seq_norm) :]
        if verbose:
            print(f"[注释] 检测到 3' 尾巴被截断，自动补齐 {len(suffix)} nt")
        updated_seq = sf_seq_norm + suffix
        updated_struct = sec_struct + "." * len(suffix)
        aa3_seq = aa3_seq + suffix
        aa3_struct = (aa3.get("Struct") or "") + "." * len(suffix)
    else:
        # 长度一致但末端不匹配（常见于 CCA 被替换）
        if target_seq != sf_seq_norm:
            mismatch = 0
            for i in range(1, min(len(target_seq), len(sf_seq_norm)) + 1):
                if target_seq[-i] != sf_seq_norm[-i]:
                    mismatch = i
                else:
                    break
            if mismatch:
                if verbose:
                    print(f"[注释] 3' 末端不一致，使用原始序列尾部 {mismatch} nt")
                updated_seq = sf_seq_norm[:-mismatch] + target_seq[-mismatch:]
            else:
                updated_seq = target_seq
        else:
            updated_seq = target_seq

        aa3_len = len(aa3_seq) if aa3_seq else min(3, len(target_seq))
        aa3_seq = target_seq[-aa3_len:]
        aa3_struct = (aa3.get("Struct") or "")[:aa3_len] or "." * aa3_len

    regions["StructFileSequence"] = updated_seq
    regions["SecondaryStructure"] = updated_struct or "." * len(updated_seq)
    regions["AA_Arm_3prime"] = {"Seq": aa3_seq, "Struct": aa3_struct}

# ===== 直接调用 tRNAscan-SE 的函数 =====
def run_trnascan_directly(seq: str, verbose: bool = False) -> Dict[str, str]:
    """直接调用 tRNAscan-SE 并解析结果"""
    # print(f"[DEBUG] 直接运行 tRNAscan-SE，序列长度: {len(seq)}")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f_fa:
        f_fa.write(f">seed\n{seq}\n")
        fa_file = f_fa.name
    
    out_file = fa_file + '.out'
    struct_file = fa_file + '.struct'
    
    try:
        cmd = ['tRNAscan-SE', '-o', out_file, '-f', struct_file, fa_file]
        if verbose:
            print(f"[DEBUG] 执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"tRNAscan-SE 运行失败: {result.stderr}")
        
        # 解析输出文件
        result_data = {}
        
        # 读取结构文件
        if os.path.exists(struct_file):
            with open(struct_file, 'r') as f:
                struct_content = f.read()
                if verbose:
                    print(f"[DEBUG] 结构文件内容:\n{struct_content}")
                
                # 解析结构文件
                lines = struct_content.strip().split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Seq:'):
                        # 提取序列行
                        seq_line = line.replace('Seq:', '').strip()
                        # 移除空格和数字
                        seq_clean = ''.join([c for c in seq_line if c.upper() in 'ACGTU'])
                        result_data['StructFileSequence'] = seq_clean.replace('T', 'U')
                        
                        # 下一行应该是结构行
                        if i + 1 < len(lines) and lines[i+1].startswith('Str:'):
                            struct_line = lines[i+1].replace('Str:', '').strip()
                            # 移除空格和数字，只保留结构字符
                            struct_clean = ''.join([c for c in struct_line if c in '.<>'])
                            result_data['SecondaryStructure'] = struct_clean
                        break
        
        # 读取输出文件获取其他信息
        if os.path.exists(out_file):
            with open(out_file, 'r') as f:
                out_content = f.read()
                if verbose:
                    print(f"[DEBUG] 输出文件内容:\n{out_content}")
                
                # 解析输出文件获取类型和反密码子信息
                lines = out_content.strip().split('\n')
                for line in lines:
                    if line.startswith('seed') and not line.startswith('Sequence'):
                        parts = line.split()
                        if len(parts) >= 5:
                            result_data['Type'] = parts[3]
                            result_data['Anticodon'] = parts[4]
                        break
        
        if 'StructFileSequence' not in result_data or 'SecondaryStructure' not in result_data:
            raise RuntimeError("无法从 tRNAscan-SE 输出中解析序列和结构")
            
        if verbose:
            print(f"[DEBUG] 解析结果: 序列长度={len(result_data['StructFileSequence'])}, 结构长度={len(result_data['SecondaryStructure'])}")
            
        return result_data
        
    except Exception as e:
        raise RuntimeError(f"tRNAscan-SE 直接调用失败: {e}")
    finally:
        # 清理临时文件
        for f in [fa_file, out_file, struct_file]:
            if os.path.exists(f):
                os.unlink(f)

# ===== tRNAscan 注释 + 六区域切分 =====
def annotate_and_extract_regions(seed_seq: str, *, verbose: bool=False) -> Dict[str, Dict[str, str]]:
    seq = (seed_seq or "").upper().replace("T","U")
    assert seq, "seed_seq 不能为空"
    if verbose:
        print(f"[注释] tRNAscan 输入长度={len(seq)}")

    # 首先尝试使用 AYLM 的注释器
    try:
        pairs = {"seed_1": seq}
        res = annotate_trna_sequences(pairs) or []
        if res:
            item = res[0]
            struct = (item.get("SecondaryStructure") or "").strip()
            sf_seq = (item.get("StructFileSequence") or "").strip()
            
            if struct and sf_seq:
                if verbose:
                    print("[注释] 使用 AYLM 注释器成功")
                # 继续原有逻辑...
            else:
                if verbose:
                    print("[注释] AYLM 注释器返回缺字段，尝试直接调用 tRNAscan-SE")
                raise RuntimeError("AYLM 注释器返回缺字段")
    except Exception as e:
        if verbose:
            print(f"[注释] AYLM 注释器失败: {e}，尝试直接调用 tRNAscan-SE")
        # AYLM 注释器失败，尝试直接调用
    
    # 直接调用 tRNAscan-SE
    try:
        result_data = run_trnascan_directly(seq, verbose=verbose)
        sf_seq = result_data['StructFileSequence']
        struct = result_data['SecondaryStructure']
        
        if verbose:
            print(f"[注释] 直接调用 tRNAscan-SE 成功")
    except Exception as e:
        raise RuntimeError(f"tRNAscan 注释失败: {e}")

    # 宽松对齐：结构长度对齐到序列长度
    if len(struct) != len(sf_seq):
        if verbose:
            print(f"[警告] 结构长({len(struct)})!=序列长({len(sf_seq)}), 对齐序列长")
        if len(struct) < len(sf_seq):
            struct = struct + "."*(len(sf_seq)-len(struct))
        else:
            struct = struct[:len(sf_seq)]

    # 六区域提取（取不到则给空）
    try:
        aa = extract_amino_acid_arm(sf_seq, struct) or ("","","","")
        d  = extract_d_loop(sf_seq, struct)          or ("","")
        ac = extract_anticodon_arm(sf_seq, struct)   or ("","")
        vl = extract_variable_loop(sf_seq, struct)   or ("","")
        t  = extract_t_arm(sf_seq, struct)           or ("","")
    except Exception as e:
        if verbose:
            print(f"[警告] 区域提取失败: {e}，使用空区域")
        aa = ("", "", "", "")
        d = ("", "")
        ac = ("", "")
        vl = ("", "")
        t = ("", "")

    aa5_seq, aa3_seq, aa5_struct, aa3_struct = aa
    d_seq, d_struct   = d
    ac_seq, ac_struct = ac
    vl_seq, vl_struct = vl
    t_seq,  t_struct  = t

    regions = {
        'AA_Arm_5prime': {'Seq': aa5_seq, 'Struct': aa5_struct},
        'D_Loop':        {'Seq': d_seq,  'Struct': d_struct},
        'Anticodon_Arm': {'Seq': ac_seq, 'Struct': ac_struct},
        'Variable_Loop': {'Seq': vl_seq, 'Struct': vl_struct},
        'T_Arm':         {'Seq': t_seq,  'Struct': t_struct},
        'AA_Arm_3prime': {'Seq': aa3_seq,'Struct': aa3_struct},
        'StructFileSequence': sf_seq,
        'SecondaryStructure': struct,
    }

    _restore_3prime_tail(seq, regions, verbose=verbose)

    if verbose:
        print("[区域长度]", {k: len(regions[k]['Seq']) for k in REGION_ORDER})
    return regions

def pack_regions_for_embedding(regions: Dict[str, Dict[str, str]], device: torch.device):
    region_tokens: Dict[int, torch.Tensor] = {}
    region_tokens_mask: Dict[int, torch.Tensor] = {}
    region_structures: Dict[int, torch.Tensor] = {}
    segment_ids: Dict[int, torch.Tensor] = {}

    seg_id = 1
    for idx, name in enumerate(REGION_ORDER):
        rseq = regions[name]['Seq']
        rstr = regions[name]['Struct']
        ids_seq  = torch.tensor(tokenize_seq(rseq), dtype=torch.long, device=device).unsqueeze(0)
        ids_mask = ids_seq.clone()
        ids_str  = torch.tensor(tokenize_struct(rstr), dtype=torch.long, device=device).unsqueeze(0)
        ids_seg  = torch.full_like(ids_seq, fill_value=int(seg_id))
        region_tokens[idx]      = ids_seq
        region_tokens_mask[idx] = ids_mask
        region_structures[idx]  = ids_str
        segment_ids[idx]        = ids_seg
        seg_id += 1

    return region_tokens, region_tokens_mask, segment_ids, region_structures

def flatten_lengths(regions: Dict[str, Dict[str, str]]) -> Tuple[List[int], Dict[int, Tuple[int,int]]]:
    lens: List[int] = [len(regions[name]['Seq']) for name in REGION_ORDER]
    posmap: Dict[int, Tuple[int,int]] = {}
    offset = 0
    for ridx, L in enumerate(lens):
        for lp in range(L):
            posmap[offset + lp] = (ridx, lp)
        offset += L
    return lens, posmap

# ===== 评分/校验/筛选 =====
@torch.no_grad()
def pseudo_logprob_via_mlm(seq: str,
                           embedding,
                           model,
                           device: torch.device,
                           verbose: bool=False) -> float:
    """逐位 mask 的平均 log-prob，越高越合理。"""
    regs = annotate_and_extract_regions(seq, verbose=False)
    reg_tok, reg_mask, seg_ids, reg_str = pack_regions_for_embedding(regs, device)
    lens, _ = flatten_lengths(regs)
    L_total = sum(lens)
    ids = torch.tensor(tokenize_seq(regs['StructFileSequence']), dtype=torch.long, device=device)

    total_logp: float = 0.0
    count = 0

    for gpos in range(L_total):
        # 重置 mask，与第 gpos 位的所在片段置为 MASK
        for ridx in range(6):
            reg_mask[ridx] = reg_tok[ridx].clone()
        cum = 0
        for ridx, Lr in enumerate(lens):
            if cum <= gpos < cum + Lr:
                lp = gpos - cum
                idx = torch.tensor([lp], dtype=torch.long, device=device)
                val = torch.tensor([int(MASK_ID)], dtype=torch.long, device=device)
                reg_mask[ridx][0] = reg_mask[ridx][0].scatter(0, idx, val)
                break
            cum += Lr

        tok_seg, msk_seg, _, _, _, _ = embedding(reg_tok, reg_mask, seg_ids, reg_str)
        logits_list, _, _, _ = model(
            tok_seg, msk_seg,
            [torch.tensor([gpos], dtype=torch.long, device=device)],
            str_tags=None, str_mask=None, use_dp=False
        )
        logits = logits_list[0][0]
        logp = float(F.log_softmax(logits, dim=-1)[ids[gpos]].item())
        total_logp += logp
        count += 1

    return float(total_logp / max(1, count))

def structure_valid_with_regions(seq: str,
                                 min_region_lens: Optional[Dict[str, int]] = None) -> bool:
    """二级结构括号配对 + 六区域长度阈值"""
    try:
        regs = annotate_and_extract_regions(seq, verbose=False)
        sf = regs['SecondaryStructure']
        lt, gt = pairs_balance(sf)
        if not (lt == gt and lt > 0):
            return False

        lens = {k: len(regs[k]['Seq']) for k in REGION_ORDER}
        default_min = {
            'AA_Arm_5prime': 5,
            'D_Loop': 8,
            'Anticodon_Arm': 8,
            'Variable_Loop': 0,
            'T_Arm': 8,
            'AA_Arm_3prime': 5,
        }
        thr = {**default_min, **(min_region_lens or {})}
        return all(lens[k] >= int(thr[k]) for k in thr)
    except Exception:
        return False

@torch.no_grad()
def filter_and_rerank(cands: List[str],
                      seed_seq: str,
                      embedding,
                      model,
                      device: torch.device,
                      min_hd: int = 6,
                      gc_low: float = 0.40,
                      gc_high: float = 0.70,
                      top_k: int = 10,
                      min_region_lens: Optional[Dict[str,int]] = None,
                      verbose: bool=False) -> List[Tuple[str, float]]:
    """先硬筛（HD/GC/结构有效），再按 pseudo log-prob 排序取前 top_k"""
    out: List[Tuple[str,float]] = []
    base = seed_seq.upper().replace("T","U")

    for s in cands:
        t = s.upper().replace("T","U")
        if hamming(base, t) < int(min_hd):
            continue
        gc = gc_content(t)
        if not (float(gc_low) <= gc <= float(gc_high)):
            continue
        if not structure_valid_with_regions(t, min_region_lens=min_region_lens):
            continue

        score = pseudo_logprob_via_mlm(t, embedding, model, device, verbose=False)
        out.append((t, float(score)))
        if verbose:
            print(f"[评分] L={len(t)} GC={gc:.3f} HD={hamming(base,t)} avgLogP={score:.4f}")

    out.sort(key=lambda x: x[1], reverse=True)
    return out[:int(top_k)]
