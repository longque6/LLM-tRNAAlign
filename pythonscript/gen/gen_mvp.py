# -*- coding: utf-8 -*-
"""
生成模块：
- 加载模型
- 单条生成（迭代遮盖重采样）
- 批量生成（持续增量直到达标）+ 小批重排筛选（调用 trna_validator 进行验证/复排）
"""

import os
import sys
import random
import tempfile
import subprocess
from typing import List, Dict, Optional, Tuple, Iterable, Any

import torch
import torch.nn.functional as F

# 路径
project_root = os.path.abspath(os.path.join(os.getcwd(), "./pythonscript"))
if project_root not in sys.argv and project_root not in sys.path:
    sys.path.insert(0, project_root)

# 依赖
from AYLM.embedding import RNATransformerEmbedding
from AYLM.TransformerModel import TransformerModel
from AYLM.modeling_rope_utils import PretrainedConfig

# ===== 从验证模块导入公共工具/常量与验证函数 =====
from pythonscript.gen.trna_validator import (
    TOK2ID, ID2TOK, MASK_ID,
    REGION_ORDER,
    tokenize_seq, detokenize, tokenize_struct,
    annotate_and_extract_regions,
    pack_regions_for_embedding, flatten_lengths,
    gc_content, hamming,
    filter_and_rerank
)

# ===== 采样器 =====
def _sample_from_logits1D(logits_1d: torch.Tensor,
                          temperature: float = 1.0,
                          top_k: int = 0,
                          top_p: float = 0.9) -> int:
    logits = logits_1d / max(1e-6, float(temperature))
    probs = F.softmax(logits, dim=-1)

    if top_k and int(top_k) > 0:
        k = min(int(top_k), probs.numel())
        topk = torch.topk(probs, k=k)
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[topk.indices] = True
        probs = torch.where(mask, probs, torch.zeros_like(probs))
        s = probs.sum()
        probs = probs/s if s.item() > 0 else F.one_hot(topk.indices[0], num_classes=int(probs.numel())).float()

    if top_p and 0.0 < float(top_p) < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=0)
        cutoff = torch.where(cum > float(top_p))[0]
        if len(cutoff) > 0:
            last = cutoff[0]
            keep = si[:last+1]
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask[keep] = True
            probs = torch.where(mask, probs, torch.zeros_like(probs))
            s = probs.sum()
            probs = probs/s if s.item() > 0 else F.one_hot(keep[0], num_classes=int(probs.numel())).float()

    return int(torch.multinomial(probs, num_samples=1).item())


def _normalize_index_list(values: Any) -> List[int]:
    """
    Convert API payloads to zero-based integer indices.
    Inputs are assumed to use 1-based sequence indexes (the way the UI exposes
    them), so we shift each valid value by -1. Invalid entries are ignored.
    """
    if values is None:
        return []
    iterable: Iterable[Any]
    if isinstance(values, (list, tuple, set)):
        iterable = values
    else:
        iterable = [values]

    out: List[int] = []
    for item in iterable:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        idx -= 1
        if idx < 0:
            continue
        out.append(idx)
    return out


def _normalize_force_map(payload: Any) -> Dict[int, str]:
    """
    Convert arbitrary JSON force-position payloads into {0-based index: base}.
    Inputs are treated as 1-based sequence indexes (matching the UI), so we
    shift by -1 when parsing. Supports dicts, list/tuple pairs, or "i:Base" strings.
    """
    if not payload:
        return {}

    pairs: List[Tuple[Any, Any]] = []

    if isinstance(payload, dict):
        pairs.extend(payload.items())
    elif isinstance(payload, (list, tuple, set)):
        for entry in payload:
            if isinstance(entry, dict):
                pairs.extend(entry.items())
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                pairs.append((entry[0], entry[1]))
            elif isinstance(entry, str) and ":" in entry:
                k, v = entry.split(":", 1)
                pairs.append((k, v))
    elif isinstance(payload, str):
        chunks = [chunk.strip() for chunk in payload.split(",")]
        for chunk in chunks:
            if ":" in chunk:
                k, v = chunk.split(":", 1)
                pairs.append((k, v))

    normalized: Dict[int, str] = {}

    for key, val in pairs:
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        idx -= 1
        if idx < 0:
            continue
        base = str(val).strip().upper() if val is not None else ""
        if base in ("A", "C", "G", "U"):
            normalized[idx] = base

    return normalized

# ===== 模型加载 =====
def load_embedding_and_model(checkpoint_path: str, device: torch.device,
                             VOCAB_SIZE=6, D_MODEL=512, N_SEGMENTS=6+1,
                             HIDDEN_SIZE=1024, FFN_HIDDEN_SIZE=2048,
                             NUM_HEADS=16, NUM_LAYERS=12, STRUCT_VOCAB_SIZE=4):
    print(f"[信息] 初始化模型")
    embedding = RNATransformerEmbedding(
        vocab_size=int(VOCAB_SIZE), d_model=int(D_MODEL),
        n_segments=int(N_SEGMENTS), mask_token_id=int(MASK_ID)
    ).to(device)

    rope = PretrainedConfig(
        rope_theta=10000.0, partial_rotary_factor=1.0,
        head_dim=int(HIDDEN_SIZE) // int(NUM_HEADS),
        max_position_embeddings=512,
        rope_scaling={"rope_type": "dynamic", "factor": 1.5},
        hidden_size=int(HIDDEN_SIZE), num_attention_heads=int(NUM_HEADS)
    )
    rope.rope_type = "dynamic"

    model = TransformerModel(
        hidden_size=int(HIDDEN_SIZE), ffn_hidden_size=int(FFN_HIDDEN_SIZE),
        num_heads=int(NUM_HEADS), num_layers=int(NUM_LAYERS),
        vocab_size=int(VOCAB_SIZE), structure_vocab_size=int(STRUCT_VOCAB_SIZE),
        dropout=0.0, rope_config=rope
    ).to(device)

    assert os.path.exists(checkpoint_path), f"找不到 checkpoint: {checkpoint_path}"
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "embedding" in ckpt and "transformer" in ckpt:
        embedding.load_state_dict(ckpt["embedding"], strict=False)
        model.load_state_dict(ckpt["transformer"], strict=False)
    elif "embedding_state_dict" in ckpt and "model_state_dict" in ckpt:
        embedding.load_state_dict(ckpt["embedding_state_dict"], strict=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        raise RuntimeError("无法识别 checkpoint 格式")

    embedding.eval(); model.eval()
    print("[信息] 权重加载完成")
    return embedding, model

# ===== 单条生成 =====
@torch.no_grad()
def generate_one_mvp_regions(seed_seq: str,
                             embedding: RNATransformerEmbedding,
                             model: TransformerModel,
                             device: torch.device,
                             rounds: int = 20,
                             mask_frac: float = 0.15,
                             temperature: float = 0.9,
                             top_p: float = 0.9,
                             top_k: int = 0,
                             freeze_positions: Optional[List[int]] = None,
                             force_positions: Optional[Dict[int, str]] = None,
                             mask_k: Optional[int] = None,
                             freeze_regions: Optional[List[str]] = None,
                             prefer_positions: Optional[List[int]] = None,
                             prefer_regions: Optional[List[str]] = None,
                             prefer_capacity_ratio: float = 0.5,
                             verbose: bool=False) -> str:
    print(f"[DEBUG] 开始生成单条序列，seed_seq: {seed_seq}")
    
    try:
        regs = annotate_and_extract_regions(seed_seq, verbose=verbose)
    except Exception as e:
        print(f"[ERROR] annotate_and_extract_regions 失败: {e}")
        print(f"[DEBUG] 尝试手动运行 tRNAscan-SE 进行调试...")
        
        # 手动运行 tRNAscan-SE 进行调试
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f_fa:
            f_fa.write(f">seed\n{seed_seq}\n")
            fa_file = f_fa.name
        
        out_file = fa_file + '.out'
        struct_file = fa_file + '.struct'
        stats_file = fa_file + '.stats'
        
        try:
            cmd = ['tRNAscan-SE', '-o', out_file, '-f', struct_file, '-m', stats_file, fa_file]
            print(f"[DEBUG] 执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(f"[DEBUG] tRNAscan-SE 返回码: {result.returncode}")
            print(f"[DEBUG] tRNAscan-SE stdout: {result.stdout}")
            print(f"[DEBUG] tRNAscan-SE stderr: {result.stderr}")
            
            if os.path.exists(out_file):
                with open(out_file, 'r') as f:
                    print(f"[DEBUG] out_file 内容:\n{f.read()}")
            if os.path.exists(struct_file):
                with open(struct_file, 'r') as f:
                    print(f"[DEBUG] struct_file 内容:\n{f.read()}")
                    
        except Exception as sub_e:
            print(f"[ERROR] 手动运行 tRNAscan-SE 失败: {sub_e}")
        finally:
            # 清理临时文件
            for f in [fa_file, out_file, struct_file, stats_file]:
                if os.path.exists(f):
                    os.unlink(f)
        
        raise RuntimeError(f"tRNAscan 注释失败: {e}")

    full_seq = regs['StructFileSequence']
    L_total  = len(full_seq)
    lens, _  = flatten_lengths(regs)

    freeze_positions = _normalize_index_list(freeze_positions)
    prefer_positions = _normalize_index_list(prefer_positions)
    force_positions = _normalize_force_map(force_positions)
    
    print(f"[DEBUG] 获取到完整序列，长度: {L_total}")
    print(f"[DEBUG] 各区域长度: {lens}")

    # 区域全局 span
    spans: Dict[str, Tuple[int,int]] = {}
    off = 0
    for name in REGION_ORDER:
        Lr = len(regs[name]['Seq'])
        spans[name] = (off, off + Lr - 1) if Lr > 0 else (off, off - 1)
        off += Lr

    reg_tok, reg_mask, seg_ids, reg_str = pack_regions_for_embedding(regs, device)
    cur_ids: List[int] = tokenize_seq(full_seq)

    # 冻结/强制/偏好位点
    frozen = set([p for p in (freeze_positions or []) if 0 <= p < L_total])
    for rg in (freeze_regions or []):
        if rg in spans:
            s, e = spans[rg]
            if s <= e:
                frozen.update(range(s, e + 1))
    frozen = sorted(frozen)

    force_positions = {p: c for p, c in (force_positions or {}).items()
                       if 0 <= p < L_total and c.upper() in ("A", "C", "G", "U")}

    preferred = set([p for p in (prefer_positions or []) if 0 <= p < L_total])
    for rg in (prefer_regions or []):
        if rg in spans:
            s, e = spans[rg]
            if s <= e:
                preferred.update(range(s, e + 1))
    preferred = sorted([p for p in preferred if p not in frozen])

    rounds_i = int(rounds)
    for _r in range(1, rounds_i + 1):
        editable = [i for i in range(L_total) if i not in frozen]
        if not editable:
            break

        K_by_frac = max(1, int(len(editable) * float(mask_frac)))
        K = min(len(editable), int(mask_k)) if (mask_k is not None and int(mask_k) > 0) else K_by_frac

        mask_global = sorted(random.sample(editable, K))

        # 倾斜到 prefer
        cap = max(1, int(K * float(prefer_capacity_ratio)))
        inject_candidates = [p for p in preferred if p in editable and p not in mask_global]
        random.shuffle(inject_candidates)
        replace_count = min(min(cap, K), len(inject_candidates))
        if replace_count > 0:
            mask_global[:replace_count] = inject_candidates[:replace_count]

        # 强制位点尽可能纳入 mask
        enforce_cap = max(1, int(0.3 * K))
        for p in force_positions:
            if p in editable and p not in mask_global and len(mask_global) < K + enforce_cap:
                mask_global.append(p)
        mask_global = sorted(set(mask_global))

        # 写入 MASK，并记录全局掩码位置
        flat_mask_positions: List[int] = []
        for ridx in range(6):
            reg_mask[ridx] = reg_tok[ridx].clone()

        cum = 0
        for ridx, Lr in enumerate(lens):
            cum_next = cum + Lr
            local_pos: List[int] = []
            for gpos in mask_global:
                if cum <= gpos < cum_next:
                    local_pos.append(gpos - cum)
                    flat_mask_positions.append(gpos)
            if local_pos:
                idx = torch.tensor(local_pos, dtype=torch.long, device=device)
                vals = torch.full((len(local_pos),), int(MASK_ID), dtype=torch.long, device=device)
                reg_mask[ridx][0] = reg_mask[ridx][0].scatter(0, idx, vals)
            cum = cum_next

        tok_seg, msk_seg, _, _, _, _ = embedding(reg_tok, reg_mask, seg_ids, reg_str)
        logits_mlm_list, _, _, _ = model(
            tok_seg, msk_seg,
            [torch.tensor(flat_mask_positions, dtype=torch.long, device=device)],
            str_tags=None, str_mask=None, use_dp=False
        )
        logits_mlm = logits_mlm_list[0]

        # 逐位采样
        for j, gpos in enumerate(flat_mask_positions):
            new_id = _sample_from_logits1D(logits_mlm[j], temperature=temperature, top_k=top_k, top_p=top_p)
            if new_id not in (1, 2, 3, 4):  # 仅 A/C/G/U
                four = logits_mlm[j][:5].clone()
                four[0] = four.min() - 1000.0  # 排除 pad
                new_id = int(torch.argmax(four).item()) or 1
            cur_ids[gpos] = new_id

        # 同步回各片段
        cur_seq_now = detokenize(cur_ids).replace("N", "A")
        pos = 0
        for ridx, Lr in enumerate(lens):
            piece = cur_seq_now[pos:pos+Lr]
            reg_tok[ridx]  = torch.tensor(tokenize_seq(piece), dtype=torch.long, device=device).unsqueeze(0)
            reg_mask[ridx] = reg_tok[ridx].clone()
            pos += Lr

    # 最后强制位点写回
    for p, ch in (force_positions or {}).items():
        cur_ids[int(p)] = TOK2ID[ch.upper()]

    new_seq = detokenize(cur_ids).replace("N", "A")
    if verbose:
        print("[生成完成] HD=", hamming(full_seq, new_seq), "GC=", f"{gc_content(new_seq):.3f}")
    return new_seq

# ========= 辅助：打印与评估不合格原因 =========
def _log_reject(seq: str, reason: str, detail: str = "", verbose: bool = False):
    if not verbose:
        return
    msg = f"[REJECT] {reason}"
    if detail:
        msg += f" | {detail}"
    # 序列可很长，默认不打印全文，避免刷屏；如需可改为打印前后若干碱基
    print(msg)

def _check_region_lengths(seq: str, min_region_lens: Optional[Dict[str, int]], verbose: bool) -> Tuple[bool, str]:
    """
    返回 (是否通过, 失败细节字符串)
    """
    if not min_region_lens:
        return True, ""
    try:
        regs = annotate_and_extract_regions(seq, verbose=False)
        lens, _ = flatten_lengths(regs)
        name2len = {name: lens[i] for i, name in enumerate(REGION_ORDER)}
        for k, v in min_region_lens.items():
            need = int(v)
            got = int(name2len.get(k, 0))
            if got < need:
                return False, f"{k} length {got} < min {need}"
        return True, ""
    except Exception as e:
        return False, f"region parse failed: {e}"

# ===== 批量生成（持续增量直到达标） =====
@torch.no_grad()
def generate_batch_mvp_regions(seed_seq: str,
                               embedding: RNATransformerEmbedding,
                               model: TransformerModel,
                               device: torch.device,
                               num_samples: int = 20,
                               rounds: int = 20,
                               mask_frac: float = 0.15,
                               temperature: float = 0.9,
                               top_p: float = 0.9,
                               top_k: int = 0,
                               freeze_positions: Optional[List[int]] = None,
                               force_positions: Optional[Dict[int,str]] = None,
                               min_hd: int = 2,
                               dedup: bool = True,
                               mask_k: Optional[int] = None,
                               freeze_regions: Optional[List[str]] = None,
                               prefer_positions: Optional[List[int]] = None,
                               prefer_regions: Optional[List[str]] = None,
                               prefer_capacity_ratio: float = 0.5,
                               oversample_factor: int = 4,
                               rerank_min_hd: int = 6,
                               rerank_top_k: Optional[int] = None,
                               gc_low: float = 0.40,
                               gc_high: float = 0.70,
                               min_region_lens: Optional[Dict[str,int]] = None,
                               ensure_reach: bool = True,
                               max_attempts: int = 5,
                               verbose: bool=False) -> List[str]:
    """
    新逻辑（与上层签名保持一致）+ 打印每条不合格原因（仅在 verbose=True 时）：
    - 持续逐条生成，直到：
        a) 达到 target；
        b) 连续失败次数 > max_attempts；
        c) 总尝试次数 >= target * oversample_factor （最大过采样倍率作为全局上限）。
    - 使用 raw_pool 小批缓存，定期做“严格预筛 + 打分重排”，并打印不合格原因。
    """
    target = int(num_samples)
    final: List[str] = []
    global_seen: set = set()

    # 解析基准序列（用于 HD 过滤）
    try:
        base = annotate_and_extract_regions(seed_seq, verbose=False)['StructFileSequence']
        if verbose:
            print(f"[DEBUG] 基础序列注释成功，长度: {len(base)}")
    except Exception as e:
        if verbose:
            print(f"[ERROR] 基础序列注释失败: {e}")
        raise

    # 统计/控制变量
    total_draws_cap = target * max(1, int(oversample_factor))   # 最大尝试数（包含失败/重复/不达标）
    total_draws = 0
    consecutive_fail = 0

    # 小批缓存，用于周期性重排挑选
    raw_pool: List[str] = []

    def pool_trigger_size() -> int:
        """触发重排的池阈值（自适应：至少10，至多50；不超过剩余需求的2倍）。"""
        remain = max(0, target - len(final))
        return max(10, min(2 * remain, 50))

    if verbose:
        print(f"[策略] 目标={target} | 最大尝试数(oversample cap)={total_draws_cap} | 最大连续失败={max_attempts}")

    # 主循环：增量生成直到触发停止条件
    while len(final) < target:
        # 保险丝1：总尝试数达到上限
        if total_draws >= total_draws_cap:
            if verbose:
                print(f"[停止] 达到最大过采样次数上限 total_draws={total_draws}/{total_draws_cap}")
            break

        # 保险丝2：连续失败次数超限
        if consecutive_fail > int(max_attempts):
            if verbose:
                print(f"[停止] 连续失败次数超过上限 consecutive_fail={consecutive_fail} > max_attempts={max_attempts}")
            break

        # 生成一条候选
        total_draws += 1
        try:
            s = generate_one_mvp_regions(
                seed_seq, embedding, model, device,
                rounds=rounds, mask_frac=mask_frac, temperature=temperature, top_p=top_p, top_k=top_k,
                freeze_positions=freeze_positions, force_positions=force_positions,
                mask_k=mask_k, freeze_regions=freeze_regions,
                prefer_positions=prefer_positions, prefer_regions=prefer_regions,
                prefer_capacity_ratio=prefer_capacity_ratio,
                verbose=False
            )
        except Exception as e:
            if verbose:
                _log_reject("<generation_error>", "single generation error", str(e), verbose=True)
            consecutive_fail += 1
            continue

        # 基础过滤：去重 & 变异度（HD）
        if dedup and (s in global_seen):
            _log_reject(s, "duplicate", "already seen in global_seen", verbose)
            consecutive_fail += 1
            continue
        hd0 = hamming(base, s)
        if hd0 < int(min_hd):
            _log_reject(s, "min_hd fail", f"HD={hd0} < min_hd={min_hd}", verbose)
            consecutive_fail += 1
            continue

        # 候选入池（记全局去重）
        raw_pool.append(s)
        global_seen.add(s)
        consecutive_fail = 0  # 有有效候选，清零失败计数

        # 判断是否触发一次重排挑选
        trigger = (len(raw_pool) >= pool_trigger_size()) \
                  or (total_draws >= total_draws_cap) \
                  or ((target - len(final)) <= len(raw_pool))
        if trigger:
            need = target - len(final)
            if need <= 0:
                break

            # ====== 严格预筛（打印失败原因），仅把合格者送入排名 ======
            pre_pass: List[str] = []
            for cand in raw_pool:
                # 二次更严格的 HD
                hd1 = hamming(base, cand)
                if hd1 < int(rerank_min_hd):
                    _log_reject(cand, "rerank_min_hd fail", f"HD={hd1} < rerank_min_hd={rerank_min_hd}", verbose)
                    continue
                # GC
                gc = gc_content(cand)
                if not (float(gc_low) <= gc <= float(gc_high)):
                    _log_reject(cand, "GC out of range", f"GC={gc:.3f} not in [{gc_low}, {gc_high}]", verbose)
                    continue
                # 区域长度
                ok_len, reason_len = _check_region_lengths(cand, min_region_lens, verbose)
                if not ok_len:
                    _log_reject(cand, "region length fail", reason_len, verbose)
                    continue
                pre_pass.append(cand)

            if verbose:
                print(f"[预筛] 通过 {len(pre_pass)}/{len(raw_pool)}，进入打分重排")

            # 本次希望挑选的上限
            topN = int(rerank_top_k) if rerank_top_k is not None else need

            ranked = []
            if pre_pass:
                # 把“预筛通过”的送进既有的 filter_and_rerank 排名
                ranked = filter_and_rerank(
                    pre_pass, seed_seq, embedding, model, device,
                    min_hd=int(rerank_min_hd),
                    gc_low=float(gc_low), gc_high=float(gc_high),
                    top_k=max(topN, need),
                    min_region_lens=min_region_lens,
                    verbose=verbose
                )

            # 选入前 need 条
            taken = 0
            for cand, _score in ranked:
                if cand not in final:
                    final.append(cand)
                    taken += 1
                    if len(final) >= target:
                        break
                    if taken >= need:
                        break

            if verbose:
                print(f"[重排] 池大小={len(raw_pool)} | 预筛后={len(pre_pass)} | 选入={taken} | 已达成={len(final)}/{target} | 尝试={total_draws}/{total_draws_cap}")

            # 清空池（避免重复重排同一批）
            raw_pool.clear()

    # 保险：若循环结束但 final 仍不足，尝试用池里残留的做最后一次填充（不再新增生成）
    if len(final) < target and raw_pool:
        need = target - len(final)
        pre_pass: List[str] = []
        for cand in raw_pool:
            hd1 = hamming(base, cand)
            if hd1 < int(rerank_min_hd):
                _log_reject(cand, "rerank_min_hd fail", f"HD={hd1} < rerank_min_hd={rerank_min_hd}", verbose)
                continue
            gc = gc_content(cand)
            if not (float(gc_low) <= gc <= float(gc_high)):
                _log_reject(cand, "GC out of range", f"GC={gc:.3f} not in [{gc_low}, {gc_high}]", verbose)
                continue
            ok_len, reason_len = _check_region_lengths(cand, min_region_lens, verbose)
            if not ok_len:
                _log_reject(cand, "region length fail", reason_len, verbose)
                continue
            pre_pass.append(cand)

        ranked = []
        if pre_pass:
            ranked = filter_and_rerank(
                pre_pass, seed_seq, embedding, model, device,
                min_hd=int(rerank_min_hd),
                gc_low=float(gc_low), gc_high=float(gc_high),
                top_k=need,
                min_region_lens=min_region_lens,
                verbose=verbose
            )
        for cand, _score in ranked:
            if cand not in final:
                final.append(cand)
                if len(final) >= target:
                    break

    if verbose:
        print(f"[完成] 返回 {len(final)} 条（请求 {target} 条） | 总尝试 {total_draws}/{total_draws_cap}")

    return final

# ===== 示例入口 =====
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[信息] 设备：", device)

    ckpt_path = "pythonscript/gen/checkpoint.pth"
    embedding, model = load_embedding_and_model(ckpt_path, device=device)

    seed_seq = "GCCUUGUUGGCGCAAUCGGUAGCGCGUAUGACUCUUAAUCAUAUGUUCAGGGAUUCGAGCCCCCUAGAGGGCU"

    freeze_positions: List[int] = []
    force_positions: Dict[int,str]  = {}
    freeze_regions: List[str]   = []
    prefer_regions: List[str]   = ["Variable_Loop"]
    prefer_positions: List[int] = []
    mask_k = 5

    print(f"[DEBUG] 开始单条生成测试...")
    
    # 单条
    try:
        one = generate_one_mvp_regions(
            seed_seq, embedding, model, device,
            rounds=3, mask_frac=0.20, temperature=0.9, top_p=0.9, top_k=0,
            freeze_positions=freeze_positions, force_positions=force_positions,
            mask_k=mask_k, freeze_regions=freeze_regions,
            prefer_positions=prefer_positions, prefer_regions=prefer_regions,
            prefer_capacity_ratio=0.6,
            verbose=False  # 关闭详细输出
        )
        print("[单条]", one)
    except Exception as e:
        print(f"[ERROR] 单条生成失败: {e}")
        import traceback
        traceback.print_exc()

    # 批量
    min_region_lens = {
        'AA_Arm_5prime': 5,
        'D_Loop': 8,
        'Anticodon_Arm': 8,
        'Variable_Loop': 0,
        'T_Arm': 8,
        'AA_Arm_3prime': 5,
    }

    print(f"[DEBUG] 开始批量生成测试...")
    
    try:
        many = generate_batch_mvp_regions(
            seed_seq, embedding, model, device,
            num_samples=3, rounds=3, mask_frac=0.20, temperature=0.95, top_p=0.9,
            freeze_positions=freeze_positions, force_positions=force_positions,
            min_hd=2, dedup=True,
            mask_k=mask_k, freeze_regions=freeze_regions,
            prefer_positions=prefer_positions, prefer_regions=prefer_regions,
            prefer_capacity_ratio=0.6,
            oversample_factor=5,   # 最大过采样倍率（总尝试上限 = target * 5）
            rerank_min_hd=6,
            gc_low=0.42, gc_high=0.66,
            min_region_lens=min_region_lens,
            ensure_reach=True,
            max_attempts=6,       # 最大连续失败次数
            verbose=True          # ✅ 开启以打印每条不合格原因
        )

        print("\n========== 最终集合 ==========")
        for i, s in enumerate(many, 1):
            print(f"{i:02d}: {s}")
    except Exception as e:
        print(f"[ERROR] 批量生成失败: {e}")
        import traceback
        traceback.print_exc()
