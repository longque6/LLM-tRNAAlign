#pythonscript/readthrough/inference.py
# -*- coding: utf-8 -*-
from typing import Dict
import os, sys, re, json, unicodedata, hashlib, functools, threading
import tempfile
import subprocess
import numpy as np
import torch
import torch.nn as nn

project_root = os.path.abspath(os.path.join(os.getcwd(), "./pythonscript"))
if project_root not in sys.argv and project_root not in sys.path:
    sys.path.insert(0, project_root)
    
# ========= 配置 =========
CKPT_PATH = "/home/ouyangzhuo/LLM-tRNAAlign/pythonscript/readthrough/best_regression_model_8layers_tunedv1.2_cv3.pth"
DEVICE = torch.device("cpu")
MC_SAMPLES_DEFAULT = 50
VERBOSE = True  # 详细输出控制
# ============================

# ------------ 项目路径 ------------
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
if project_root not in sys.argv and project_root not in sys.path:
    sys.path.insert(0, project_root)

# ------------ 上游依赖 ------------
from AYLM.utils.tRNAscan_SE_annotator import annotate_trna_sequences
from AYLM.extract import (
    extract_amino_acid_arm,
    extract_d_loop,
    extract_anticodon_arm,
    extract_variable_loop,
    extract_t_arm
)

from AYLM.embedding import RNATransformerEmbedding
from AYLM.TransformerModel import TransformerModel
from AYLM.modeling_rope_utils import PretrainedConfig

def _verbose_print(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# ---------------- 新增：直接调用 tRNAscan-SE ----------------
def run_trnascan_directly(seq: str) -> Dict[str, str]:
    """直接调用 tRNAscan-SE 并解析结果（基于你的方案）"""
    _verbose_print(f"[tRNAscan] 直接调用 tRNAscan-SE，序列长度: {len(seq)}")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f_fa:
        f_fa.write(f">query\n{seq}\n")  # 使用固定ID 'query'
        fa_file = f_fa.name
    
    out_file = fa_file + '.out'
    struct_file = fa_file + '.struct'
    
    try:
        cmd = ['tRNAscan-SE', '-o', out_file, '-f', struct_file, fa_file]
        _verbose_print(f"[tRNAscan] 执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"tRNAscan-SE 运行失败: {result.stderr}")
        
        # 解析结构文件
        result_data = {}
        if os.path.exists(struct_file):
            with open(struct_file, 'r') as f:
                struct_content = f.read()
                _verbose_print(f"[tRNAscan] 结构文件内容:\n{struct_content}")
                
                lines = struct_content.strip().split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Seq:'):
                        # 提取序列行
                        seq_line = line.replace('Seq:', '').strip()
                        # 移除空格和数字，只保留核苷酸
                        seq_clean = ''.join([c for c in seq_line if c.upper() in 'ACGTU'])
                        result_data['StructFileSequence'] = seq_clean.replace('T', 'U')
                        
                        # 下一行应该是结构行
                        if i + 1 < len(lines) and lines[i+1].startswith('Str:'):
                            struct_line = lines[i+1].replace('Str:', '').strip()
                            # 移除空格和数字，只保留结构字符
                            struct_clean = ''.join([c for c in struct_line if c in '.<>'])
                            result_data['SecondaryStructure'] = struct_clean
                        break
        
        if 'StructFileSequence' not in result_data or 'SecondaryStructure' not in result_data:
            raise RuntimeError("无法从 tRNAscan-SE 输出中解析序列和结构")
            
        _verbose_print(f"[tRNAscan] 解析成功: 序列长度={len(result_data['StructFileSequence'])}, 结构长度={len(result_data['SecondaryStructure'])}")
        return result_data
        
    except Exception as e:
        raise RuntimeError(f"tRNAscan-SE 直接调用失败: {e}")
    finally:
        # 清理临时文件
        for f in [fa_file, out_file, struct_file]:
            if os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass

# ---------------- 基础工具 ----------------
def _safe_id(s: str, max_len: int = 64) -> str:
    if s is None:
        s = ""
    s = unicodedata.normalize("NFKD", str(s))
    s = re.sub(r"\s+", "_", s.strip())
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r'[:*?"<>|]', "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    if not s:
        s = "seq"
    if len(s) > max_len:
        h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
        s = f"{s[:max_len-9]}_{h}"
    return s

def _tokenize_nt(seq: str):
    tok_map = {'pad':0,'A':1,'C':2,'G':3,'T':4,'U':4,'M':5}
    return [tok_map.get(c.upper(), 0) for c in seq]

REGION_ORDER = [
    'AA_Arm_5prime_Seq', 'D_Loop_Seq', 'Anticodon_Arm_Seq',
    'Variable_Loop_Seq', 'T_Arm_Seq', 'AA_Arm_3prime_Seq'
]

def _gc_ratio(seq: str) -> float:
    if not seq: return 0.0
    s = seq.upper()
    n = len(s)
    if n == 0: return 0.0
    gc = sum(1 for c in s if c in ('G','C'))
    return gc / n

def compute_handcrafted_features(regions: dict):
    """
    18 维手工特征，保持与训练权重对齐：
      0-5:  各区域长度
      6-11: 各区域 GC 比例
      12:   全序列长度
      13:   全序列 GC 比例
      14:   区域缺失比例（兜底，用不到则为 0）
      15:   结构-序列长度相对差
      16:   二级结构中成对碱基比例
      17:   二级结构中非配对比例
    """
    lens, gcs = [], []
    total_seq = ""
    missing_regions = 0
    for col in REGION_ORDER:
        s = regions.get(col, "") or ""
        if len(s) == 0:
            missing_regions += 1
        lens.append(len(s))
        gcs.append(_gc_ratio(s))
        total_seq += s

    total_len = len(total_seq)
    total_gc  = _gc_ratio(total_seq)

    # 结构相关特征（基于分段结构拼接）
    struct_order = [
        'AA_Arm_5prime_Struct', 'D_Loop_Struct', 'Anticodon_Arm_Struct',
        'Variable_Loop_Struct', 'T_Arm_Struct', 'AA_Arm_3prime_Struct'
    ]
    struct_cat = "".join(regions.get(k, "") or "" for k in struct_order)
    struct_len = len(struct_cat)
    paired_cnt = struct_cat.count("<") + struct_cat.count(">")
    unpaired_cnt = struct_cat.count(".")

    missing_ratio = missing_regions / len(REGION_ORDER)
    len_diff_ratio = 0.0 if total_len == 0 else abs(struct_len - total_len) / max(1, total_len)
    paired_ratio = 0.0 if struct_len == 0 else paired_cnt / struct_len
    unpaired_ratio = 0.0 if struct_len == 0 else unpaired_cnt / struct_len

    feats = (
        lens + gcs +
        [total_len, total_gc, missing_ratio, len_diff_ratio, paired_ratio, unpaired_ratio]
    )
    return np.array(feats, dtype=np.float32)

# ---------------- 模型骨架（保持不变） ----------------
class TokenAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scorer = nn.Linear(dim, 1)
    def forward(self, x, mask):
        scores = self.scorer(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        return out, attn

class RegressionTransformerModelSimple(nn.Module):
    """均值池化 + 回归头（老模型）"""
    def __init__(self, embedding, base_model, dropout=0.2, device=DEVICE):
        super().__init__()
        self.embedding = embedding
        self.base_model = base_model
        with torch.no_grad():
            dummy = {0: torch.randint(0, 6, (1, 100), device=device)}
            dmask = {0: torch.ones(1,100, dtype=torch.long, device=device)}
            dseg  = {0: torch.ones(1,100, dtype=torch.long, device=device)}
            dstr  = {0: torch.zeros(1,100, dtype=torch.long, device=device)}
            tok_seg, _, *_ = self.embedding(dummy, dmask, dseg, dstr)
            enc_out, _ = self.base_model.encoder(tok_seg.permute(1,0,2))
            feat_dim = enc_out.permute(1,0,2).size(-1)
        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),      nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),      nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,  64),      nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,    1)
        )
    def forward(self, region_tokens, region_tokens_mask, segment_ids, region_structures):
        tok_seg, _, *_ = self.embedding(region_tokens, region_tokens_mask, segment_ids, region_structures)
        enc_out, _ = self.base_model.encoder(tok_seg.permute(1,0,2))
        feat = enc_out.permute(1,0,2).mean(dim=1)  # [B, H]
        return self.regressor(feat).squeeze(-1)

class RegressionTransformerModelAttn(nn.Module):
    """注意力池化 + 区域 gate + 回归头（无手工特征）"""
    def __init__(self, embedding, base_model, dropout=0.2, n_regions=6, device=DEVICE):
        super().__init__()
        self.embedding = embedding
        self.base_model = base_model
        self.n_regions = n_regions
        with torch.no_grad():
            dummy = {0: torch.randint(0, 6, (1, 100), device=device)}
            dmask = {0: torch.ones(1,100, dtype=torch.long, device=device)}
            dseg  = {0: torch.ones(1,100, dtype=torch.long, device=device)}
            dstr  = {0: torch.zeros(1,100, dtype=torch.long, device=device)}
            tok_seg, _, *_ = self.embedding(dummy, dmask, dseg, dstr)
            enc_out, _ = self.base_model.encoder(tok_seg.permute(1,0,2))
            feat_dim = enc_out.permute(1,0,2).size(-1)
        self.token_attn = TokenAttention(feat_dim)
        self.region_gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(feat_dim//2, 1)
        )
        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),      nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),      nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,  64),      nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,    1)
        )
    def forward(self, region_tokens, region_tokens_mask, segment_ids, region_structures):
        tok_seg, _, *_ = self.embedding(region_tokens, region_tokens_mask, segment_ids, region_structures)
        enc_out, _ = self.base_model.encoder(tok_seg.permute(1,0,2))
        enc_out = enc_out.permute(1,0,2)  # [B,L,H]
        lengths = [region_tokens[r].size(1) for r in sorted(region_tokens.keys())]
        cum = np.cumsum([0] + lengths)
        region_vecs = []
        for i in range(self.n_regions):
            s, e = cum[i], cum[i+1]
            if e - s == 0:
                region_vecs.append(torch.zeros(enc_out.size(0), enc_out.size(-1), device=enc_out.device))
                continue
            chunk = enc_out[:, s:e, :]
            mask_i = (region_tokens[i] != 0).long()
            vec_i, _ = self.token_attn(chunk, mask_i)
            region_vecs.append(vec_i)
        R = torch.stack(region_vecs, dim=1)         # [B,R,H]
        w = torch.softmax(self.region_gate(R).squeeze(-1), dim=1)  # [B,R]
        feat = torch.bmm(w.unsqueeze(1), R).squeeze(1)  # [B,H]
        return self.regressor(feat).squeeze(-1)

class RegressionTransformerModelFused(nn.Module):
    """注意力池化 + 区域 gate + 18维手工特征投影融合 + 回归头（最佳模型，对齐训练权重）"""
    def __init__(self, embedding, base_model, dropout=0.2, n_regions=6, extra_feat_dim=18, device=DEVICE):
        super().__init__()
        self.embedding = embedding
        self.base_model = base_model
        self.n_regions = n_regions
        with torch.no_grad():
            dummy = {0: torch.randint(0, 6, (1, 100), device=device)}
            dmask = {0: torch.ones(1,100, dtype=torch.long, device=device)}
            dseg  = {0: torch.ones(1,100, dtype=torch.long, device=device)}
            dstr  = {0: torch.zeros(1,100, dtype=torch.long, device=device)}
            tok_seg, _, *_ = self.embedding(dummy, dmask, dseg, dstr)
            enc_out, _ = self.base_model.encoder(tok_seg.permute(1,0,2))
            feat_dim = enc_out.permute(1,0,2).size(-1)
        self.token_attn = TokenAttention(feat_dim)
        self.region_gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(feat_dim//2, 1)
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(extra_feat_dim, 128),
            nn.ReLU(), nn.Dropout(dropout)
        )
        fused_dim = feat_dim + 128
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,  64),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,    1)
        )
    def forward(self, region_tokens, region_tokens_mask, segment_ids, region_structures, extra_feats):
        tok_seg, _, *_ = self.embedding(region_tokens, region_tokens_mask, segment_ids, region_structures)
        enc_out, _ = self.base_model.encoder(tok_seg.permute(1,0,2))
        enc_out = enc_out.permute(1,0,2)  # [B,L,H]
        lengths = [region_tokens[r].size(1) for r in sorted(region_tokens.keys())]
        cum = np.cumsum([0] + lengths)
        region_vecs = []
        for i in range(self.n_regions):
            s, e = cum[i], cum[i+1]
            if e - s == 0:
                region_vecs.append(torch.zeros(enc_out.size(0), enc_out.size(-1), device=enc_out.device))
                continue
            chunk = enc_out[:, s:e, :]
            mask_i = (region_tokens[i] != 0).long()
            vec_i, _ = self.token_attn(chunk, mask_i)
            region_vecs.append(vec_i)
        R = torch.stack(region_vecs, dim=1)               # [B,R,H]
        w = torch.softmax(self.region_gate(R).squeeze(-1), dim=1)
        feat_enc = torch.bmm(w.unsqueeze(1), R).squeeze(1)  # [B,H]
        feat_aux = self.feat_proj(extra_feats)              # [B,128]
        fused = torch.cat([feat_enc, feat_aux], dim=-1)     # [B,H+128]
        return self.regressor(fused).squeeze(-1)


# ---- 缓存/并发安全 ----
_model_cache_lock = threading.Lock()

def _build_backbone(device=DEVICE):
    """构建通用 embedding + base encoder"""
    D_MODEL=512; FFN=2048; HEADS=16; LAYERS=12; VOCAB=6; STR_VOCAB=4
    n_segments = 7
    embedding = RNATransformerEmbedding(vocab_size=VOCAB, d_model=D_MODEL, n_segments=n_segments).to(device)
    rope = PretrainedConfig(
        rope_theta=10000.0, partial_rotary_factor=1.0,
        head_dim=(2*D_MODEL)//HEADS, max_position_embeddings=512,
        rope_scaling={"rope_type":"dynamic","factor":1.5},
        hidden_size=2*D_MODEL, num_attention_heads=HEADS,
    ); rope.rope_type = "dynamic"
    base = TransformerModel(
        hidden_size=2*D_MODEL, ffn_hidden_size=FFN, num_heads=HEADS,
        num_layers=LAYERS, vocab_size=VOCAB, structure_vocab_size=STR_VOCAB,
        dropout=0.2, rope_config=rope
    ).to(device)
    return embedding, base

def _detect_arch_from_state_dict(sd: dict) -> str:
    ks = set(sd.keys())
    if any(k.startswith("feat_proj.") for k in ks):
        return "fused"
    if ("token_attn.scorer.weight" in ks) or ("region_gate.0.weight" in ks):
        return "attn"
    return "simple"

@functools.lru_cache(maxsize=4)
def _cached_loader(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    with _model_cache_lock:
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        sd = ck["model_state_dict"]
        arch = _detect_arch_from_state_dict(sd)
        embedding, base = _build_backbone()
        if arch == "fused":
            model = RegressionTransformerModelFused(embedding, base, dropout=0.2, n_regions=6, extra_feat_dim=18).to(DEVICE)
            model.load_state_dict(sd, strict=False)
            _verbose_print("[inference] Loaded FUSED (attn + handcrafted feats) model.")
        elif arch == "attn":
            model = RegressionTransformerModelAttn(embedding, base, dropout=0.2, n_regions=6).to(DEVICE)
            model.load_state_dict(sd, strict=False)
            _verbose_print("[inference] Loaded ATTENTION model.")
        else:
            model = RegressionTransformerModelSimple(embedding, base, dropout=0.2).to(DEVICE)
            model.load_state_dict(sd, strict=False)
            _verbose_print("[inference] Loaded SIMPLE (mean-pool) model.")
        
        eff_stats = ck.get('eff_norm_stats', None)
        if eff_stats is None:
            raise KeyError("eff_norm_stats not found in checkpoint; cannot denormalize.")
        eff_mu = float(eff_stats['mean']); eff_std = float(eff_stats['std'])
        
        feat_stats = ck.get('feat_norm_stats', None) if arch == "fused" else None
        if arch == "fused" and feat_stats is None:
            raise KeyError("feat_norm_stats not found in fused checkpoint.")
        if feat_stats is not None:
            feat_mu = np.asarray(feat_stats['mean'], dtype=np.float32)
            feat_std= np.maximum(1e-8, np.asarray(feat_stats['std'], dtype=np.float32))
        else:
            feat_mu, feat_std = None, None
        model.eval()
    return model, arch, eff_mu, eff_std, feat_mu, feat_std

def load_trained_model_and_stats(ckpt_path: str):
    return _cached_loader(ckpt_path)

# ---------------- 修改后的区域分割函数 ----------------
def segment_regions_for_sequence(seq: str):
    """使用直接调用 tRNAscan-SE 的方法获取二级结构"""
    _verbose_print(f"[segment_regions] 开始处理序列，长度: {len(seq)}")
    
    # 首先尝试直接调用 tRNAscan-SE（基于你的方案）
    try:
        result_data = run_trnascan_directly(seq)
        sf_seq = result_data['StructFileSequence']
        structure = result_data['SecondaryStructure']
        _verbose_print(f"[segment_regions] 直接调用成功，结构长度: {len(structure)}")
    except Exception as e:
        _verbose_print(f"[segment_regions] 直接调用失败: {e}，尝试使用 AYLM 注释器")
        # 备用方案：尝试 AYLM 注释器
        sid = _safe_id("query")
        results = annotate_trna_sequences({sid: seq})
        if not results or not isinstance(results, list):
            raise RuntimeError("所有注释方法都失败了")
        entry = results[0]
        structure = entry.get("SecondaryStructure", "")
        sf_seq = entry.get("StructFileSequence", "")
        if not structure:
            # 如果 AYLM 也没有结构，尝试从 ModelPredictions 中获取
            model_preds = entry.get("ModelPredictions", {})
            rnafold = model_preds.get("RNAfold", {})
            structure = rnafold.get("SecondaryStructure", "")
            if structure:
                _verbose_print("[segment_regions] 从 RNAfold 预测获取结构")
    
    if not structure:
        raise RuntimeError("Failed to obtain SecondaryStructure for the input sequence.")

    # 结构长度对齐
    if len(structure) != len(sf_seq):
        _verbose_print(f"[segment_regions] 结构长度({len(structure)}) != 序列长度({len(sf_seq)}), 进行对齐")
        if len(structure) < len(sf_seq):
            structure = structure + "." * (len(sf_seq) - len(structure))
        else:
            structure = structure[:len(sf_seq)]

    regions = {
        'AA_Arm_5prime_Seq': '', 'AA_Arm_3prime_Seq': '',
        'D_Loop_Seq': '', 'Anticodon_Arm_Seq': '',
        'Variable_Loop_Seq': '', 'T_Arm_Seq': '',
        'AA_Arm_5prime_Struct': '', 'AA_Arm_3prime_Struct': '',
        'D_Loop_Struct': '', 'Anticodon_Arm_Struct': '',
        'Variable_Loop_Struct': '', 'T_Arm_Struct': ''
    }

    # 提取各个区域
    try:
        aa_arms = extract_amino_acid_arm(sf_seq, structure)
        if aa_arms:
            regions['AA_Arm_5prime_Seq'], regions['AA_Arm_3prime_Seq'], \
            regions['AA_Arm_5prime_Struct'], regions['AA_Arm_3prime_Struct'] = aa_arms
            _verbose_print(f"[segment_regions] AA Arm 提取成功")
    except Exception as e:
        _verbose_print(f"[segment_regions] AA Arm 提取失败: {e}")

    try:
        d_loop = extract_d_loop(sf_seq, structure)
        if d_loop:
            regions['D_Loop_Seq'], regions['D_Loop_Struct'] = d_loop
            _verbose_print(f"[segment_regions] D Loop 提取成功")
    except Exception as e:
        _verbose_print(f"[segment_regions] D Loop 提取失败: {e}")

    try:
        anticodon = extract_anticodon_arm(sf_seq, structure)
        if anticodon:
            regions['Anticodon_Arm_Seq'], regions['Anticodon_Arm_Struct'] = anticodon
            _verbose_print(f"[segment_regions] Anticodon Arm 提取成功")
    except Exception as e:
        _verbose_print(f"[segment_regions] Anticodon Arm 提取失败: {e}")

    try:
        var_loop = extract_variable_loop(sf_seq, structure)
        if var_loop:
            regions['Variable_Loop_Seq'], regions['Variable_Loop_Struct'] = var_loop
            _verbose_print(f"[segment_regions] Variable Loop 提取成功")
    except Exception as e:
        _verbose_print(f"[segment_regions] Variable Loop 提取失败: {e}")

    try:
        t_arm = extract_t_arm(sf_seq, structure)
        if t_arm:
            regions['T_Arm_Seq'], regions['T_Arm_Struct'] = t_arm
            _verbose_print(f"[segment_regions] T Arm 提取成功")
    except Exception as e:
        _verbose_print(f"[segment_regions] T Arm 提取失败: {e}")

    # 检查是否有区域被成功提取
    extracted_count = sum(1 for k in REGION_ORDER if regions.get(k, ""))
    _verbose_print(f"[segment_regions] 成功提取 {extracted_count}/6 个区域")
    
    if extracted_count == 0:
        _verbose_print("[segment_regions] 警告：所有区域提取失败，使用全序列作为 AA_Arm_5prime_Seq 兜底")
        regions['AA_Arm_5prime_Seq'] = sf_seq

    return regions, structure


# ---------------- 构造单样本 batch ----------------
def make_single_batch_from_regions(regions: dict):
    region_tokens, region_tokens_mask, segment_ids, region_structures = {}, {}, {}, {}
    for i, key in enumerate(REGION_ORDER):
        s = regions.get(key, "") or ""
        toks = _tokenize_nt(s)
        if len(toks) == 0:
            region_tokens[i] = torch.empty(1, 0, dtype=torch.long, device=DEVICE)
            region_tokens_mask[i] = torch.empty(1, 0, dtype=torch.long, device=DEVICE)
            segment_ids[i] = torch.empty(1, 0, dtype=torch.long, device=DEVICE)
            region_structures[i] = torch.empty(1, 0, dtype=torch.long, device=DEVICE)
        else:
            t = torch.tensor(toks, dtype=torch.long, device=DEVICE).unsqueeze(0)  # [1,L]
            region_tokens[i] = t
            region_tokens_mask[i] = torch.ones_like(t, dtype=torch.long, device=DEVICE)
            segment_ids[i] = torch.full_like(t, fill_value=i+1, dtype=torch.long, device=DEVICE)
            region_structures[i] = torch.zeros_like(t, dtype=torch.long, device=DEVICE)
    return region_tokens, region_tokens_mask, segment_ids, region_structures


# ---------------- MC Dropout 开启（仅让 Dropout 生效） ----------------
def _enable_mc_dropout(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.train()


# ---------------- 主推理函数（对外） ----------------
def predict_suptrna_efficiency(seq: str, mc_samples: int = MC_SAMPLES_DEFAULT, ckpt_path: str = CKPT_PATH):
    """
    输入：单条 sup-tRNA 序列（ACGTU...）
    返回：
      {
        'pred_raw': float,              # 原始尺度预测（通读效率）
        'pred_log': float,              # log 标准化空间预测
        'std_raw': float,               # 原始尺度标准差（MC Dropout）
        'ci95_raw': [low, high],        # 95% 置信区间（正态近似）
        'confidence_pct': float,        # 0-100 的可信度（由变异系数映射）
        'structure': str,               # 注释得到的二级结构
        'region_seqs': {...}            # 六个区域的序列（审计用途）
      }
    """
    if not isinstance(seq, str) or not re.search(r"[ACGTUacgtu]", seq or ""):
        raise ValueError("请输入包含 A/C/G/T/U 的核苷酸序列。")

    # 1) 区域序列与结构
    regions, structure = segment_regions_for_sequence(seq)

    # 2) 载入模型与标准化统计（自动识别骨架）
    model, arch, eff_mu, eff_std, feat_mu, feat_std = load_trained_model_and_stats(ckpt_path)

    # 3) 构造单样本 batch
    rt, rmask, rseg, rstr = make_single_batch_from_regions(regions)

    # 4) 准备融合模型所需的 14 维手工特征（并标准化）
    extra_feat_t = None
    if arch == "fused":
        feats = compute_handcrafted_features(regions)        # (14,)
        feats_norm = (feats - feat_mu) / np.maximum(1e-8, feat_std)
        extra_feat_t = torch.tensor(feats_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1,14]

    # 5) MC Dropout 前向（对“有 Dropout 的模块”打开随机性）
    model.eval()
    if hasattr(model, "regressor"):   _enable_mc_dropout(model.regressor)
    if hasattr(model, "region_gate"): _enable_mc_dropout(model.region_gate)
    if hasattr(model, "feat_proj"):   _enable_mc_dropout(model.feat_proj)

    preds_log = []
    with torch.no_grad():
        for _ in range(max(1, int(mc_samples))):
            if arch == "fused":
                yhat_norm = model(rt, rmask, rseg, rstr, extra_feat_t)   # [1]
            else:
                yhat_norm = model(rt, rmask, rseg, rstr)                 # [1]
            preds_log.append(float(yhat_norm.item()))
    preds_log = np.array(preds_log, dtype=np.float64)

    # 6) 反标准化到原始尺度： y_raw = expm1( y_log*std + mean )
    ylog = preds_log * max(1e-8, eff_std) + eff_mu
    preds_raw = np.expm1(ylog)

    # 7) 统计量 + 可信度
    mean_log = float(np.mean(preds_log))
    mean_raw = float(np.mean(preds_raw))
    std_raw  = float(np.std(preds_raw, ddof=1)) if len(preds_raw) > 1 else 0.0
    ci_low   = float(mean_raw - 1.96 * std_raw)
    ci_high  = float(mean_raw + 1.96 * std_raw)
    cv = std_raw / (abs(mean_raw) + 1e-8)
    confidence_pct = float(max(0.0, min(100.0, 100.0 / (1.0 + cv))))

    return {
        "pred_raw": mean_raw,
        "pred_log": mean_log,
        "std_raw": std_raw,
        "ci95_raw": [ci_low, ci_high],
        "confidence_pct": confidence_pct,
        "structure": structure,
        "region_seqs": {
            "AA_Arm_5prime_Seq": regions.get("AA_Arm_5prime_Seq", ""),
            "D_Loop_Seq": regions.get("D_Loop_Seq", ""),
            "Anticodon_Arm_Seq": regions.get("Anticodon_Arm_Seq", ""),
            "Variable_Loop_Seq": regions.get("Variable_Loop_Seq", ""),
            "T_Arm_Seq": regions.get("T_Arm_Seq", ""),
            "AA_Arm_3prime_Seq": regions.get("AA_Arm_3prime_Seq", "")
        }
    }


# ---------------- 本地快速测试（可删） ----------------
if __name__ == "__main__":
    print("\n=== Single-seq Inference Quick Test ===")
    try:
        user_seq = input("请输入一条 sup-tRNA 序列 (A/C/G/T/U)，直接回车使用示例：\n> ").strip()
    except Exception:
        user_seq = ""
    default_seq = "GAGAAGGUCAUAGAGGUUAUGGGAUUGGCUCUAAACCAGUCUCUGGGGGGUUCGAUUCCCUCCUUUUUCA"
    if not user_seq:
        print("[提示] 使用内置示例序列。")
        user_seq = default_seq
    try:
        out = predict_suptrna_efficiency(
            seq=user_seq,
            mc_samples=50,
            ckpt_path=CKPT_PATH
        )
        print("\n--- 推理结果 ---")
        print(json.dumps(out, ensure_ascii=False, indent=2))
    except Exception as e:
        print("\n[错误] 推理失败：", repr(e))
        import traceback
        print("详细错误信息：")
        traceback.print_exc()
        print("请检查：1) 序列仅含 A/C/G/T/U；2) 模型权重路径；3) 依赖可用（AYLM 组件）。")
