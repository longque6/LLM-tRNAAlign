import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from AYLM.modeling_rope_utils import PretrainedConfig
from AYLM.modules import TransformerEncoder
from torchcrf import CRF

class TransformerModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_heads: int,
        num_layers: int,
        vocab_size: int,
        structure_vocab_size: int,
        rope_config: PretrainedConfig,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            rope_config=rope_config,
        )
        self.fc_mlm = nn.Linear(hidden_size, vocab_size)
        self.fc_str = nn.Linear(hidden_size, structure_vocab_size)
        self.crf    = CRF(structure_vocab_size)

    def forward(
        self,
        x_original: torch.Tensor,       # [B, L, H]
        x_masked:   torch.Tensor,       # [B, L, H]
        mask_positions: list,           # List[Tensor([M_b])], M_b = number of masked positions in batch element b
        str_tags:    torch.LongTensor = None,  # [L, B] # type: ignore
        str_mask:    torch.BoolTensor  = None, # [L, B] # type: ignore
        use_dp:      bool = False,      # 推理时用 Nussinov DP
    ):
        # —— 保证 CRF 最首 timestep 不被 mask 掉 —— 
        if str_mask is not None:
            str_mask = str_mask.clone() # type: ignore
            str_mask[0].fill_(True)

        # 转成 [L, B, H]
        x_ori = x_original.permute(1, 0, 2)
        x_msk = x_masked.permute(1, 0, 2)

        # 编码
        x_ori, _ = self.encoder(x_ori)
        x_msk, _ = self.encoder(x_msk)

        # MLM & 结构打头
        logits_mlm_all = self.fc_mlm(x_msk)   # [L, B, V]
        emissions_all  = self.fc_str(x_ori)   # [L, B, S]
        emissions = emissions_all            # [L, B, S]

        # 抽出 MLM 位置的 logits
        B = logits_mlm_all.size(1)
        masked_output = []
        for b in range(B):
            pos = mask_positions[b].to(logits_mlm_all.device)
            masked_output.append(logits_mlm_all[pos, b, :])

        # —— 训练：CRF loss —— 
        if str_tags is not None and str_mask is not None:
            ll = self.crf(emissions, str_tags, mask=str_mask, reduction='mean')
            return masked_output, emissions.permute(1, 0, 2), -ll

        # —— 推理 —— 
        # 先计算每位置的后验概率 [B, L, S]
        probs_all = F.softmax(emissions.permute(1, 0, 2), dim=-1)

        if use_dp:
            # Nussinov DP 解码（带 score_dot）
            best_paths = []
            for b in range(B):
                score_dot   = emissions[:, b, 1].cpu().tolist()  # 索引1 对应 "."
                score_open  = emissions[:, b, 2].cpu().tolist()  # 索引2 对应 "<"
                score_close = emissions[:, b, 3].cpu().tolist()  # 索引3 对应 ">"
                mask_b      = str_mask[:, b].cpu().tolist()
                best_paths.append(
                    self._nussinov_decode(score_dot, score_open, score_close, mask_b)
                )
        else:
            # CRF Viterbi 解码
            best_paths = self.crf.decode(emissions, str_mask)

        return masked_output, emissions.permute(1, 0, 2), best_paths, probs_all

    @staticmethod
    def _nussinov_decode(score_dot, score_open, score_close, mask):
        """
        改造版 Nussinov DP：
        - score_dot[i]   : 位置 i 做 "." 的发射分数
        - score_open[i]  : 位置 i 做 "<" 的发射分数
        - score_close[j] : 位置 j 做 ">" 的发射分数
        - mask[k]        : True 表示 k 位有效（非 pad）
        返回长度 L 的标签列表（0=pad, 1='.', 2='<', 3='>'）
        """
        L = len(score_dot)
        dp   = [[-float('inf')] * L for _ in range(L)]
        back = {}

        # 子区间长度=1 时，只能选单点 '.'
        for i in range(L):
            if mask[i]:
                dp[i][i] = score_dot[i]
            else:
                dp[i][i] = 0.0

        # 子区间长度从 2 到 L 逐步递推
        for length in range(2, L+1):
            for i in range(0, L-length+1):
                j = i + length - 1
                best, op = -float('inf'), None

                # 情形1：i 留作 "." → score_dot[i] + dp[i+1][j]
                if mask[i]:
                    v = score_dot[i] + dp[i+1][j]
                else:
                    v = dp[i+1][j]
                if v > best:
                    best, op = v, ('i_dot', None)

                # 情形2：j 留作 "." → dp[i][j-1] + score_dot[j]
                if mask[j]:
                    v = dp[i][j-1] + score_dot[j]
                else:
                    v = dp[i][j-1]
                if v > best:
                    best, op = v, ('j_dot', None)

                # 情形3：i/j 配对 "<" + ">"
                if mask[i] and mask[j]:
                    v = dp[i+1][j-1] + score_open[i] + score_close[j]
                    if v > best:
                        best, op = v, ('pair', None)

                # 情形4：拆分子区间
                for k in range(i, j):
                    v = dp[i][k] + dp[k+1][j]
                    if v > best:
                        best, op = v, ('split', k)

                dp[i][j] = best
                back[(i, j)] = op

        # 回溯
        labels = [0] * L
        def trace(i, j):
            # 处理越界或 i>j 的情况
            if i > j:
                return
            # 如果 i == j，只有单点可能
            if i == j:
                labels[i] = 1 if mask[i] else 0
                return

            op, k = back[(i, j)]
            if op == 'i_dot':
                labels[i] = 1  # "."
                trace(i+1, j)
            elif op == 'j_dot':
                labels[j] = 1  # "."
                trace(i, j-1)
            elif op == 'pair':
                labels[i], labels[j] = 2, 3  # i="<", j=">"
                trace(i+1, j-1)
            else:  # 'split'
                trace(i, k)
                trace(k+1, j)

        trace(0, L-1)
        return labels


# === 模块自测 ===
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    B, L, H = 2, 30, 128
    V_SZ, S_SZ = 6, 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 准备模型
    rope_cfg = PretrainedConfig(
        rope_theta=1e4,
        partial_rotary_factor=1.0,
        head_dim=H // 8,
        max_position_embeddings=512,
        rope_scaling={"rope_type": "dynamic", "factor": 1.5},
        hidden_size=H,
        num_attention_heads=8,
    )
    model = TransformerModel(
        hidden_size=H,
        ffn_hidden_size=512,
        num_heads=8,
        num_layers=4,
        vocab_size=V_SZ,
        structure_vocab_size=S_SZ,
        dropout=0.1,
        rope_config=rope_cfg
    ).to(device).eval()

    # 假数据
    x = torch.randn(B, L, H, device=device)
    mask_pos = [torch.tensor([1, 3, 5], device=device) for _ in range(B)]
    # 造标签 & 掩码
    tags = torch.randint(1, S_SZ, (L, B), device=device)
    mask = torch.rand(L, B, device=device) > 0.1
    mask[0, :] = True  # CRF 要求首行全 True

    # 1. 训练分支检查 loss
    m_out, em, loss = model(x, x, mask_pos, str_tags=tags, str_mask=mask)
    print("CRF loss:", loss.item())

    # 2. CRF 解码（并拿概率）
    m_out, em, crf_paths, crf_probs = model(x, x, mask_pos, str_tags=None, str_mask=mask, use_dp=False)
    print("CRF decode:", crf_paths)
    print("CRF probs shape:", crf_probs.shape)  # [B, L, S]

    # 3. DP 解码（并拿概率）
    m_out, em, dp_paths, dp_probs = model(x, x, mask_pos, str_tags=None, str_mask=mask, use_dp=True)
    print("DP  decode:", dp_paths)
    print("DP  probs shape:", dp_probs.shape)   # [B, L, S]