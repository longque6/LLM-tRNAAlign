import logging
import math
from functools import wraps
from typing import Optional, Tuple
import torch

logger = logging.getLogger(__name__)

class PretrainedConfig:
    """
    简易 RoPE 配置类，用于保存 RoPE 相关超参数。
    """
    def __init__(
        self,
        rope_theta: float = 10000.0,
        partial_rotary_factor: float = 1.0,
        head_dim: int = 64,
        max_position_embeddings: int = 512,
        rope_scaling: Optional[dict] = None,
        hidden_size: Optional[int] = None,
        num_attention_heads: Optional[int] = None
    ):
        # RoPE 公式中的 theta 基数
        self.rope_theta = rope_theta
        # RoPE 中的 partial factor，默认 1.0
        self.partial_rotary_factor = partial_rotary_factor
        # 每个注意力头的维度
        self.head_dim = head_dim
        # 最大支持的序列长度
        self.max_position_embeddings = max_position_embeddings
        # 如果没有传入 rope_scaling，就用一个只含 factor=1.0 的字典
        self.rope_scaling = rope_scaling if rope_scaling is not None else {'factor': 1.0}
        # 记录整层 hidden size 与头数，方便默认 RoPE 计算
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        # 从 rope_scaling 中读取 rope_type，默认 "default"
        self.rope_type = self.rope_scaling.get("rope_type", "default")


def dynamic_rope_update(rope_forward):
    """
    装饰器：给动态 RoPE 层在 forward 前自动检查、更新 inv_freq。
    """
    def dynamic_frequency_update(self, position_ids, device):
        # 实际上要处理两种情况：序列变长时扩容、序列回到较短时还原
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len
            )
            # 注册新的 inv_freq buffer
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
            # 恢复到最初的 inv_freq
            orig = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", orig, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @wraps(rope_forward)
    def wrapper(self, x, position_ids):
        # 只有 rope_type 包含 "dynamic" 时才做动态更新
        if "dynamic" in getattr(self, "rope_type", ""):
            dynamic_frequency_update(self, position_ids, device=x.device)
        return rope_forward(self, x, position_ids)

    return wrapper


def _compute_default_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional[torch.device] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple[torch.Tensor, float]:
    """
    计算“静态”RoPE 的 inv_freq，与原始论文一致。
    """
    if config is not None and rope_kwargs:
        raise ValueError(
            "在 _compute_default_rope_parameters 中，"
            "`config` 与 `rope_kwargs` 不能同时使用。"
        )
    if rope_kwargs:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    else:
        base = config.rope_theta # type: ignore
        partial = config.partial_rotary_factor # type: ignore
        # 如果传入 hidden_size 和头数，就用它们计算 head_dim
        if hasattr(config, "head_dim") and config.head_dim is not None: # type: ignore
            dim = int(config.head_dim * partial) # type: ignore
        else:
            # fallback：hidden_size / num_attention_heads
            dim = int((config.hidden_size // config.num_attention_heads) * partial) # type: ignore

    attention_factor = 1.0  # 静态版不需要 scaling

    inv_freq = 1.0 / (
        base ** (
            torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim
        )
    )
    return inv_freq, attention_factor


def _compute_dynamic_ntk_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional[torch.device] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple[torch.Tensor, float]:
    """
    计算带 NTK scaling 的动态 RoPE inv_freq。
    """
    if config is None:
        raise ValueError("动态 RoPE 需要传入 config")
    # 从 config 里拿 factor 和 max_position_embeddings
    base = config.rope_theta
    partial = config.partial_rotary_factor
    # 头维度
    head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads) # type: ignore
    dim = int(head_dim * partial)
    max_pos = config.max_position_embeddings
    factor = config.rope_scaling["factor"]

    # 如果当前序列更长，就按 NTK 公式放大 base
    effective_len = seq_len if (seq_len and seq_len > max_pos) else max_pos
    base = base * ((factor * effective_len / max_pos) - (factor - 1)) ** (dim / (dim - 2))

    inv_freq = 1.0 / (
        base ** (
            torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim
        )
    )
    attention_factor = 1.0
    return inv_freq, attention_factor


# 注册所有计算函数，包括“静态”和“动态”两种
ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
}


def _check_received_keys(
    rope_type: str,
    received_keys: set,
    required_keys: set,
    optional_keys: Optional[set] = None,
    ignore_keys: Optional[set] = None,
):
    """
    用于校验 config.rope_scaling 字典中的必需/可选 key。
    """
    if "type" in received_keys:
        received_keys.remove("type")
        required_keys.add("rope_type")
    if ignore_keys:
        received_keys -= ignore_keys
    missing = required_keys - received_keys
    if missing:
        raise KeyError(f"RoPE 缩放配置缺少字段: {missing}")
    if optional_keys:
        unused = received_keys - required_keys - optional_keys
    else:
        unused = received_keys - required_keys
    if unused:
        logger.warning(f"未识别的 RoPE 缩放字段: {unused}")


def _validate_dynamic_scaling_rope_parameters(
    config: PretrainedConfig, ignore_keys: Optional[set] = None
):
    """
    校验动态 RoPE 特有的缩放参数。
    """
    rope_scaling = config.rope_scaling
    required = {"rope_type", "factor"}
    optional = {"original_max_position_embeddings"}
    keys = set(rope_scaling.keys())
    _check_received_keys(config.rope_type, keys, required, optional, ignore_keys) # type: ignore
    factor = rope_scaling["factor"]
    if not isinstance(factor, float) or factor < 1.0:
        logger.warning(f"`rope_scaling.factor` 应该是 >=1 的浮点数，当前: {factor}")


# RoPE 配置校验映射
ROPE_VALIDATION_FUNCTIONS = {
    "dynamic": _validate_dynamic_scaling_rope_parameters,
}


def rope_config_validation(
    config: PretrainedConfig, ignore_keys: Optional[set] = None
):
    """
    对传入的 PretrainedConfig 中的 rope_scaling 做整体校验。
    """
    rs = getattr(config, "rope_scaling", None)
    if rs is None:
        return
    rt = rs.get("rope_type", rs.get("type", "default"))
    fn = ROPE_VALIDATION_FUNCTIONS.get(rt)
    if fn:
        fn(config, ignore_keys)
    else:
        logger.warning(f"缺少 RoPE 校验函数: rope_type={rt}")