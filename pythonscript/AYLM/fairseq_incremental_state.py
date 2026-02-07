"""
fairseq_incremental_state.py

这个模块提供了管理增量状态的工具，
用于在序列生成任务中缓存和重用之前计算的 Key/Value，
从而避免重复计算，提升推理效率。

包含内容：
  - FairseqIncrementalState 类：封装了增量状态的初始化、读取和写入逻辑。
  - with_incremental_state 装饰器：将其它模块的基类中插入 FairseqIncrementalState，
    使它们可以方便地使用增量状态管理。
"""

import uuid
from typing import Dict, Optional
from torch import Tensor

class FairseqIncrementalState(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        # 为每个模块生成唯一的增量状态 ID
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        # 使用模块的增量状态 ID 和指定的 key 生成全局唯一的 key
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        """辅助方法：从 incremental_state 中获取指定 key 的状态。
        
        Args:
            incremental_state: 一个字典，保存了多个模块的增量状态。
            key: 需要读取的状态键。
        
        Returns:
            对应的增量状态字典，或者如果不存在返回 None。
        """
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """辅助方法：将指定 key 的状态保存到 incremental_state 中。
        
        Args:
            incremental_state: 一个字典，用于存储模块的增量状态。
            key: 要保存的状态键。
            value: 要保存的状态字典。
        
        Returns:
            更新后的 incremental_state 字典。
        """
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state

def with_incremental_state(cls):
    """
    装饰器：将目标类的基类中插入 FairseqIncrementalState，
    这样目标类就自动具备增量状态管理的功能。
    """
    cls.__bases__ = (FairseqIncrementalState,) + tuple(b for b in cls.__bases__ if b != FairseqIncrementalState)
    return cls