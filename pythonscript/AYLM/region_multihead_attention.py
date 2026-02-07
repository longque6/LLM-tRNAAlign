"""
region_multihead_attention.py

本文件实现了一个多头注意力模块 MultiheadAttention，
支持增量解码（incremental state）、相对位置编码（Relative Position Encoding）和区域注意力掩码（Region Attention Mask）。
它依赖于以下模块：
  - fairseq_incremental_state.py：提供 FairseqIncrementalState 和 with_incremental_state
  - attention_utils.py：提供 utils_softmax 和 apply_sparse_mask
  - relative_position.py：提供 RelativePositionEmbedding 和 compute_relative_position_scores
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

import uuid

# 引入增量状态管理工具
from AYLM.fairseq_incremental_state import FairseqIncrementalState, with_incremental_state
# 引入注意力相关辅助函数
from AYLM.attention_utils import utils_softmax, apply_sparse_mask
# 引入相对位置编码相关模块
from AYLM.relative_position import RelativePositionEmbedding, compute_relative_position_scores
# 引入 RoPE 相关功能
from AYLM.modeling_rope_utils import PretrainedConfig, ROPE_INIT_FUNCTIONS

@with_incremental_state
class MultiheadAttention(nn.Module):
    """
    Multi-headed attention with incremental decoding support.
    
    Extended Features:
      - Relative Position Encoding
      - Region Attention Mask
      
    该模块支持：
        * 增量状态管理（用于序列生成时缓存历史计算结果）
        * add_bias_kv, add_zero_attn 等选项（来自 Fairseq 的实现）
        * 相对位置编码：使用 RelativePositionEmbedding 和 compute_relative_position_scores
        * 区域注意力掩码：通过 region_mask 参数对不同区域的注意力进行加权调整
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        self_attention: bool = False,
        encoder_decoder_attention: bool = False,
        max_relative_positions: int = 0,   # 0 表示不使用相对位置编码
        region_attention: bool = False,     # 是否使用区域注意力掩码
        max_position: int = 256,           # 用于限制序列最大长度
        rope_config: Optional[PretrainedConfig] = PretrainedConfig(),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and value to be of the same size"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

        self.enable_torch_version = hasattr(F, "multi_head_attention_forward")

        # ---------- Relative Position Encoding -----------
        self.use_relative_positions = (max_relative_positions > 0)
        self.max_relative_positions = max_relative_positions
        if self.use_relative_positions:
            # 构造相对位置嵌入层，词汇表大小为 2*max_relative_positions + 1
            self.relative_position_embedding = RelativePositionEmbedding(max_relative_positions, self.head_dim)
        self.max_position = max_position
        self.region_attention = region_attention

        # 添加 RoPE 的初始化
        # —— RoPE：绑定 config + 算一次最长 inv_freq 放 buffer —— 
        self.config    = rope_config
        self.rope_type = rope_config.rope_type if rope_config else "default"
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type] # type: ignore
        inv_freq, _ = self.rope_init_fn(self.config, device="cpu", seq_len=max_position) # type: ignore
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        

        # 记录初始缓存长度
        self.max_seq_len_cached = max_position
        self.original_max_seq_len = max_position
        # --------------------------------------------------

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)


    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        region_mask: Optional[Tensor] = None,  # 形状 [batch_size, tgt_len, src_len]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            query: (tgt_len, batch_size, embed_dim)
            key: (src_len, batch_size, embed_dim) 或 None
            value: (src_len, batch_size, embed_dim) 或 None
            key_padding_mask: (batch_size, src_len)
            attn_mask: (tgt_len, src_len)
            region_mask: (batch_size, tgt_len, src_len)，用于对特定区域加权调整
            incremental_state: 用于增量解码的状态缓存
            need_weights: 是否返回注意力权重
            need_head_weights: 是否返回每个头的注意力权重
            static_kv: 若 True，表示 Key 和 Value 在增量解码过程中保持不变

        Returns:
            attn: (tgt_len, batch_size, embed_dim)
            attn_weights_out: (Optional) 注意力权重
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        # ----- 1. Torch默认多头实现条件下的快速分支 -----
        # only use the Torch fast‐path if *not* doing any RoPE / relative‐pos / custom logic
        if (self.enable_torch_version
            and self.rope_type == "none"            # <- skip if any RoPE configured
            and not self.onnx_trace
            and incremental_state is None
            and not static_kv
            and not torch.jit.is_scripting() # type: ignore
            and not need_head_weights):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )

        # ----- 2. 增量解码：读写缓存 -----
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        # ----- 3. 计算 Q, K, V -----
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling
        # ======= RoPE 旋转位置编码 =======
        # 1) 先把 q,k 从 [L, B, H*D] → [L, B, H, D]
        Lq, B, HD = q.size()
        H, D = self.num_heads, self.head_dim
        q = q.view(Lq, B, H, D)
        Lk, _, _  = k.size() # type: ignore
        k = k.view(Lk, B, H, D) # type: ignore

        # 2) 转成 [B, H, L, D] 方便按时间步算 sin/cos
        q = q.permute(1,2,0,3)  # [B,H,Lq,D]
        k = k.permute(1,2,0,3)  # [B,H,Lk,D]

        # —— 动态 RoPE：如果 rope_type == "dynamic" 并且序列超出初始 max_position，就重算 inv_freq —— 
        if self.rope_type == "dynamic":
            L = q.size(2)
        # 只有当 L 比 buffer 里存的还大时，才真正触发重算
            if L > self.max_seq_len_cached:
                new_inv, new_scale = self.rope_init_fn(self.config, device=q.device, seq_len=L)
                self.inv_freq = new_inv.to(self.inv_freq.device)
                self.attention_scaling = new_scale
                self.max_seq_len_cached = L
            elif L < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
                # 回落到最初
                self.inv_freq = self.original_inv_freq.to(self.inv_freq.device)
                self.max_seq_len_cached = self.original_max_seq_len
        # —— RoPE: 用 buffer 只做一次切片 —— 
        inv = self.inv_freq.to(q.device)                 # [D/2]
        pos = torch.arange(q.size(2), device=q.device).unsqueeze(1)
        freqs = pos * inv.unsqueeze(0)                    # [L, D/2]

        # 4) 拼出 cos,sin：shape → [L, D]
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        # expand 到 [B,H,L,D]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # 在做 q,k * cos + rotate(q,k)*sin 之前
        if hasattr(self, "attention_scaling"):
            q = q * self.attention_scaling
            k = k * self.attention_scaling

        # 5) 定义“每两个维度旋转一次”的 helper
        def rotate_every_two(x):
            # x: [B,H,L,D]
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            # stack → [..., 2, D/2] → flatten 回 D
            return torch.stack((-x2, x1), dim=-1).flatten(-2)

        # 6) 应用 rotary
        q = q * cos + rotate_every_two(q) * sin
        k = k * cos + rotate_every_two(k) * sin

        # 7) 再还原回 [L, B, H*D]
        q = q.permute(2,0,1,3).reshape(Lq, B, H*D)
        k = k.permute(2,0,1,3).reshape(Lk, B, H*D)
        # ======= RoPE 插入结束 =======
        # ----- 4. 如果设置 add_bias_kv, 追加 bias_k / bias_v -----
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)]) # type: ignore
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)],
                    dim=1,
                )

        # ----- 5. reshape Q, K, V to (B * num_heads, seq, head_dim) -----
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # ----- 6. 如果有缓存状态，则拼接到历史 Key, Value 后面 -----
        if saved_state is not None:
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )
            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        assert k is not None
        src_len = k.size(1)

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        # ----- 7. add_zero_attn -----
        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)],
                    dim=1,
                )

        # ----- 8. 计算注意力 logits: Q x K^T -----
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = MultiheadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz) # type: ignore

        # ----- 9. 相对位置编码 (Relative Position) -----
        if self.use_relative_positions:
            BH = bsz * self.num_heads
            Q_4d = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
            range_vec = torch.arange(tgt_len, device=q.device)
            rel_pos_mat = range_vec.unsqueeze(-1) - range_vec.unsqueeze(0)
            rel_pos_mat.clamp_(-self.max_relative_positions, self.max_relative_positions)
            rel_pos_mat = rel_pos_mat + self.max_relative_positions
            rel_emb = self.relative_position_embedding(tgt_len)
            rel_logits = self._relative_position_scores(Q_4d, rel_emb)
            rel_logits = rel_logits.view(BH, tgt_len, tgt_len)
            attn_weights += rel_logits

        # ----- 10. 加入 attn_mask / key_padding_mask -----
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # ----- 11. 区域注意力掩码 region_mask -----
        if self.region_attention and (region_mask is not None):
            if region_mask.dim() == 3:
                region_mask = region_mask.unsqueeze(1)  # [B, 1, tgt_len, src_len]
                region_mask = region_mask.repeat(1, self.num_heads, 1, 1)  # [B, num_heads, tgt_len, src_len]
                region_mask = region_mask.view(bsz * self.num_heads, tgt_len, src_len)
                attn_weights += region_mask

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils_softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
        assert v is not None
        attn = torch.bmm(attn_probs, v)

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights_out: Optional[Tensor] = None
        if need_weights:
            attn_weights_out = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                attn_weights_out = attn_weights_out.mean(dim=0)

        return attn, attn_weights_out
   
    # -------------- Relative Position Function --------------
    def _relative_position_scores(self, Q_4d: Tensor, rel_emb: Tensor) -> Tensor:
        """
        计算相对位置得分:
          Q_4d: [B, num_heads, seq, head_dim]
          rel_emb: [seq, seq, head_dim]，通过相对位置嵌入获得
        返回 shape: [B, num_heads, seq, seq]
        """
        return torch.einsum('bhid,ijd->bhij', Q_4d, rel_emb)

    # ----------------------------------------------------------

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), key_padding_mask.float()], dim=1)
        elif prev_key_padding_mask is not None:
            filler = torch.zeros((batch_size, src_len - prev_key_padding_mask.size(1)), device=prev_key_padding_mask.device)
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), filler.float()], dim=1)
        elif key_padding_mask is not None:
            filler = torch.zeros((batch_size, src_len - key_padding_mask.size(1)), device=key_padding_mask.device)
            new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(0) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        return result if result is not None else {}

    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], buffer: Dict[str, Optional[Tensor]]) -> Dict[str, Dict[str, Optional[Tensor]]]:
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(attn_weights: Tensor, tgt_len: int, src_len: int, bsz: int) -> Tensor: # type: ignore
        return attn_weights

    def upgrade_state_dict_named(self, state_dict: Dict, name: str):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in list(state_dict.keys()):
            if k.endswith(prefix + "in_proj_weight"):
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim:2*dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2*dim:]
                keys_to_remove.append(k)
                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict:
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][dim:2*dim]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2*dim:]
                    keys_to_remove.append(k_bias)
        for k in keys_to_remove:
            del state_dict[k]
        for key, value in items_to_add.items():
            state_dict[key] = value

# 运行测试
if __name__ == "__main__":
    import torch

    def run_basic_test(rope_type: str, max_pos: int, seq_len: int):
        """
        测试 RoPE 类型、最大位置和序列长度，并打印调试信息
        """
        print(f"\n--- Testing RoPE type={rope_type!r}, max_pos={max_pos}, seq_len={seq_len} ---")

        # 构造配置
        config = PretrainedConfig(
            rope_theta=10000.0,
            partial_rotary_factor=1.0,
            head_dim=16 // 4,
            max_position_embeddings=max_pos,
            rope_scaling={"rope_type": rope_type, "factor": 1.5},
            hidden_size=16,
            num_attention_heads=4,
        )
        config.rope_type = rope_type

        # 初始化注意力模块
        m = MultiheadAttention(
            embed_dim=16,
            num_heads=4,
            max_position=max_pos,
            rope_config=config,
        )
        m.to(torch.device("cpu")).eval()

        # 随机输入
        q = torch.rand(seq_len, 2, 16)

        # 记录调用前状态
        print(f"Before forward: m.rope_type={m.rope_type}, m.max_seq_len_cached={m.max_seq_len_cached}")

        out, weights = m(q, q, q)

        # 记录调用后状态
        print(f"After forward:  m.rope_type={m.rope_type}, m.max_seq_len_cached={m.max_seq_len_cached}")

        # 检查输出形状
        print(f"Output shape: {out.shape}, Weights shape: {weights.shape}")
        assert out.shape == (seq_len, 2, 16), f"Output shape mismatch: {out.shape}"
        assert weights.shape == (2, seq_len, seq_len), f"Weights shape mismatch: {weights.shape}"

        # 预期缓存
        expected_cache = seq_len if (rope_type == "dynamic" and seq_len > max_pos) else max_pos
        print(f"Expected cache length: {expected_cache}")
        print(f"Actual cache length:   {m.max_seq_len_cached}")

        assert m.max_seq_len_cached == expected_cache, (
            f"Cache length mismatch: expected {expected_cache}, but got {m.max_seq_len_cached}"
        )

        print("  ✔ forward pass and cache check successful")


    # 1) 静态 RoPE，不超过 max_position
    run_basic_test(rope_type="default", max_pos=32, seq_len=16)

    # 2) 动态 RoPE，不超过 max_position
    run_basic_test(rope_type="dynamic", max_pos=32, seq_len=16)

    # 3) 动态 RoPE，超过 max_position（触发扩容）
    run_basic_test(rope_type="dynamic", max_pos=32, seq_len=64)

    print("\nAll tests passed!")