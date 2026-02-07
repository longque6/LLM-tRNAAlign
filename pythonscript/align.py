# ==============================================
# File: pythonscript/dp_align.py
# ----------------------------------------------
# Pure DP-based alignment workflow.
# Exposes: perform_full_alignment_dp(...)
# ==============================================
from __future__ import annotations
import io
import csv
from typing import List, Dict, Tuple, Optional

from pythonscript.alignment_score import alignment_score_and_str
from pythonscript.dataloader import compare_input_template
from pythonscript.llm_postcheck import postcheck_alignment
from pythonscript.debug_config import (
    DEBUG_ALIGN as _ALIGN_DEBUG,
    DEBUG_ALIGN_FULL as _ALIGN_DEBUG_FULL,
    DEBUG_ALIGN_MAXLEN as _ALIGN_DEBUG_MAXLEN,
)

__all__ = [
    "perform_full_alignment",
    "_align_with_dp",
]

def _short(s: str, max_len: int = _ALIGN_DEBUG_MAXLEN) -> str:
    if s is None:
        return ""
    if len(s) <= max_len:
        return s
    return f"{s[:max_len]}...(len={len(s)})"

def _dprint(msg: str):
    if _ALIGN_DEBUG:
        print(msg)


def validate_alignment(target_seq: str, aligned_seq) -> bool:
    valid_bases = set("AUGCN")
    # 统一大写后再比较，避免大小写造成误判
    tgt = "".join(ch.upper() for ch in target_seq if ch.upper() in valid_bases)
    if isinstance(aligned_seq, (list, tuple)):
        aligned_str = "".join(aligned_seq)
    else:
        aligned_str = aligned_seq
    ali = "".join(ch.upper() for ch in aligned_str if ch.upper() in valid_bases)

    if tgt != ali:
        _dprint("[align][DEBUG] Base-preservation check failed!")
        if _ALIGN_DEBUG_FULL:
            _dprint(f"[align][DEBUG] target_bases : {tgt}")
            _dprint(f"[align][DEBUG] aligned_bases: {ali}")
        else:
            _dprint(f"[align][DEBUG] target_bases : {_short(tgt)}")
            _dprint(f"[align][DEBUG] aligned_bases: {_short(ali)}")
        return False
    return True


def _count_tail_slots(template_num: List[str], start_idx: int, tail_set: set[str]) -> int:
    """从 start_idx 开始，统计剩余可消费的 3' 尾部标准位（如 74/75/76）的数量。"""
    cnt = 0
    for n in template_num[start_idx:]:
        if n in tail_set:
            cnt += 1
    return cnt


def _align_with_dp(
    template_num: List[str],
    template_seq: str,
    target_seq: str,
    match: float = 2.0,
    mismatch: float = -1.0,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
    *,
    region: Optional[str] = None,  # 用于变量环/3'端的特殊消费策略
    prev_num: Optional[str] = None,  # 用于跨区段的插入编号锚点
) -> Dict[str, List[str]]:
    """
    Global alignment using PairwiseAligner (via alignment_score_and_str). Converts to
    aligned_numbering/aligned_sequence while consuming numbered vacant slots.

    特殊规则：
      - Variable loop：模板列是 '-' 时，若当前模板编号以 'V' 开头，直接消费该编号（承接目标碱基或 '-')
      - 3' 端：模板列是 '-' 时，只有当“目标剩余非缺口数” <= “可消费的 74/75/76 数”，才消费 74/75/76；
              否则将该碱基记为 73i1/73i2... 等插入，从而把最后三个保留给 74/75/76（典型 CCA）。
    """
    _dprint(f"=== [align][DEBUG] _align_with_dp called (region={region!r}) ===")
    _dprint(f"[align][DEBUG] template_seq : {_short(template_seq)}")
    _dprint(f"[align][DEBUG] target_seq   : {_short(target_seq)}")
    _dprint(f"[align][DEBUG] template_num : {template_num[:20]} ... len={len(template_num)}")

    score, flat = alignment_score_and_str(
        target_seq, template_seq, match, mismatch, gap_open, gap_extend, use_edlib=False
    )
    _dprint(f"[align][DEBUG] DP score     : {score}")
    if _ALIGN_DEBUG_FULL:
        _dprint(f"[align][DEBUG] DP flat:\n{flat}")
    else:
        lines = flat.splitlines()
        if lines:
            shown = "\n".join(_short(ln) for ln in lines[:3])
            _dprint(f"[align][DEBUG] DP flat (trunc):\n{shown}")

    # 解析成两行
    flat_target = ''
    flat_template = ''
    cols: List[tuple[str, str]] = []
    for line in flat.splitlines():
        parts = line.split()
        if not parts:
            continue
        if line.startswith('target'):
            flat_target = parts[1]
        elif line.startswith('query'):
            flat_template = parts[1]
    if not flat_target or not flat_template:
        raise RuntimeError("alignment_score_and_str returned unexpected format")

    # 逐列数组与“目标后缀非缺口计数”
    L = min(len(flat_target), len(flat_template))
    t_list = list(flat_target[:L])
    q_list = list(flat_template[:L])
    cols = list(zip(q_list, t_list))

    rem_non_gap = [0] * L
    running = 0
    for i in range(L - 1, -1, -1):
        if t_list[i] != '-':
            running += 1
        rem_non_gap[i] = running

    aligned_nums: List[str] = []
    aligned_seqs: List[str] = []
    aligned_templ: List[str] = []
    idx_template = 0
    def _normalize_anchor(n: Optional[str]) -> Optional[str]:
        if not n:
            return None
        if n.startswith("-") or n.startswith("V"):
            return None
        # 43i1 -> 43
        if "i" in n:
            base = n.split("i", 1)[0]
            return base if base and not base.startswith("V") else None
        return n

    last_num: str | None = _normalize_anchor(prev_num)
    ins_count = 1

    TAIL_CANON = {"74", "75", "76"}  # 3' 端 CCA 的标准编号集合

    for col_idx, (templ_char, targ_char) in enumerate(cols):
        # 关键：template_seq 本身可能包含 '-'（表示该编号位在该模板中缺失/空位），
        # 这类 '-' 仍然应当“消耗一个模板编号”，否则会导致后续编号整体左移。
        # 只有当对齐结果中的 '-' 来自“对齐插入的缺口（模板侧 gap）”时，才视作插入列（不消耗模板编号）。
        templ_is_placeholder = (
            templ_char == '-' and
            idx_template < len(template_seq) and
            template_seq[idx_template] == '-'
        )

        # 情况A：模板列是对齐产生的 gap（模板侧 '-' 且不是占位符）
        if templ_char == '-' and not templ_is_placeholder:
            curr_num = template_num[idx_template] if idx_template < len(template_num) else None

            # A-1) 若模板编号本身是占位（如 '-1'），优先用它来承接
            if curr_num is not None and curr_num.startswith('-'):
                aligned_nums.append(curr_num)
                aligned_seqs.append(targ_char if targ_char != '-' else '-')
                aligned_templ.append('-')
                last_num = curr_num
                idx_template += 1
                ins_count = 1
                continue

            # A-2) 变量环：模板编号形如 'Vxx'，直接消费（把插入对到 Vxx）
            if (
                curr_num is not None and
                region == "Variable loop" and
                curr_num.startswith('V')
            ):
                aligned_nums.append(curr_num)
                aligned_seqs.append(targ_char if targ_char != '-' else '-')
                aligned_templ.append('-')
                last_num = curr_num
                idx_template += 1
                ins_count = 1
                if targ_char != '-':
                    _dprint(f"[align][DEBUG] Variable loop consume: {curr_num} <- {targ_char}")
                continue

            # A-3) 3' 端：只在需要时才消费 74/75/76（保留最后三个给 CCA）
            if (
                curr_num is not None and
                region == "Aminoacyl arm 3' end" and
                curr_num in TAIL_CANON and
                targ_char != '-'
            ):
                tail_slots_remaining = _count_tail_slots(template_num, idx_template, TAIL_CANON)
                tg_remaining = rem_non_gap[col_idx]  # 从当前列到末尾，目标非缺口数
                # 只有当剩余目标碱基数 <= 可消费尾部位数 时，才消费 74/75/76
                if tg_remaining <= tail_slots_remaining:
                    aligned_nums.append(curr_num)
                    aligned_seqs.append(targ_char)
                    aligned_templ.append('-')
                    last_num = curr_num
                    idx_template += 1
                    ins_count = 1
                    _dprint(f"[align][DEBUG] Tail consume: {curr_num} <- {targ_char} (tg_rem={tg_remaining}, tail_slots={tail_slots_remaining})")
                    continue
                # 否则，记为插入，保留尾部位给真正的最后三个
                base = last_num or (template_num[0] if template_num else 'X')
                num = f"{base}i{ins_count}"
                aligned_nums.append(num)
                aligned_seqs.append(targ_char)
                aligned_templ.append('-')
                ins_count += 1
                _dprint(f"[align][DEBUG] Tail defer as insert: {num} <- {targ_char} (tg_rem>{tail_slots_remaining})")
                continue

            # A-4) 其余情况：真正插入（不消耗 idx_template）
            if targ_char != '-':
                base = last_num or (template_num[0] if template_num else 'X')
                num = f"{base}i{ins_count}"
                aligned_nums.append(num)
                aligned_seqs.append(targ_char)
                aligned_templ.append('-')
                ins_count += 1
            # 双缺口则跳过
            continue

        # 情况B：模板列不是 gap，应该消耗一个模板编号
        if idx_template >= len(template_num):
            # 兜底：模板编号耗尽但目标仍有碱基 → 作为插入附着在最后编号后
            if targ_char != '-':
                base = last_num or (template_num[-1] if template_num else 'X')
                num = f"{base}i{ins_count}"
                aligned_nums.append(num)
                aligned_seqs.append(targ_char)
                aligned_templ.append('-')
                ins_count += 1
            continue

        num = template_num[idx_template]

        # B-1) 即便这里出现了占位编号（理论上少见），照旧消耗
        if num.startswith('-'):
            aligned_nums.append(num)
            aligned_seqs.append('-' if targ_char == '-' else targ_char)
            aligned_templ.append(template_seq[idx_template] if idx_template < len(template_seq) else '-')
            last_num = num
            idx_template += 1
            ins_count = 1
            continue

        # B-2) 正常消耗模板编号
        aligned_nums.append(num)
        last_num = num
        aligned_templ.append(template_seq[idx_template] if idx_template < len(template_seq) else '-')
        idx_template += 1
        ins_count = 1
        aligned_seqs.append('-' if targ_char == '-' else targ_char)

    # 把剩余没消耗完的模板编号补齐成尾部缺口
    while idx_template < len(template_num):
        num = template_num[idx_template]
        aligned_nums.append(num)
        aligned_seqs.append('-')
        aligned_templ.append(template_seq[idx_template] if idx_template < len(template_seq) else '-')
        idx_template += 1

    if not validate_alignment(target_seq, aligned_seqs):
        raise ValueError(
            "DP alignment failed base-preservation check "
            "(set TRNAALIGN_DEBUG_ALIGN=1 for details)"
        )

    return {
        "aligned_numbering": aligned_nums,
        "aligned_sequence": aligned_seqs,
        "aligned_template": aligned_templ,
    }


def perform_full_alignment(
    target_seq: str,
    output_csv_path: str,
    anticode: str = "",
    use_llm: bool = True,
) -> Tuple[str, str]:
    """
    DP-only pipeline across the six regions returned by compare_input_template.
    在写出 CSV 之前，构造 6 个区域的“三行对象”并调用（可选的）LLM 后校对钩子：
      block = {
        "region": <str>,
        "lines": [
          <List[str]>  # 第1行：编号（对齐后的标准编号数组）
          <List[str]>  # 第2行：模板序列（与编号等长；若无法获取则用 '-' 占位）
          <List[str]>  # 第3行：目标序列（对齐后的目标序列，含 '-')
        ]
      }
    最终仍只写两行（编号 + 目标序列）到 CSV。
    """
    # 选模板
    cmp = compare_input_template(target_seq)
    if not cmp:
        raise RuntimeError("compare_input_template failed")
    template_name = cmp['template_name']

    # 标准 6 区段
    regions = [
        "Aminoacyl arm 5' end",
        "D loop + D stem",
        "Anticodon loop + Anticodon stem",
        "Variable loop",
        "T loop + T stem",
        "Aminoacyl arm 3' end",
    ]

    # 先逐段 DP 对齐，收集为 blocks（供 LLM 后校对）
    def _anchor_from_aligned(nums: List[str]) -> Optional[str]:
        # 取上一段对齐结果里的最后一个编号；插入号取其基号（如 43i1 -> 43）
        for n in reversed(nums):
            if not n or n.startswith("-") or n.startswith("V"):
                continue
            if "i" in n:
                base = n.split("i", 1)[0]
                return base if base else None
            return n
        return None

    def _fix_boundary_fill(prev_blk: dict, cur_blk: dict) -> None:
        # 若上一段最后编号对应目标为 '-'，且下一段开头是该编号的插入列，
        # 则把该插入碱基“回填”到上一段最后位置，避免出现 43i1 有碱基而 43 为 '-'。
        prev_nums, prev_templ, prev_targ = prev_blk["lines"]
        cur_nums, cur_templ, cur_targ = cur_blk["lines"]
        if not prev_nums or not cur_nums:
            return
        prev_last_num = prev_nums[-1]
        if not prev_last_num or prev_last_num.startswith("-") or prev_last_num.startswith("V"):
            return
        if prev_targ[-1] != "-":
            return
        first_num = cur_nums[0]
        if not first_num or not first_num.startswith(f"{prev_last_num}i"):
            return
        if cur_targ[0] == "-":
            return
        # 回填
        prev_targ[-1] = cur_targ[0]
        # 删除当前段首列
        cur_nums.pop(0)
        cur_templ.pop(0)
        cur_targ.pop(0)
        # 连续插入编号依次前移（i2->i1, i3->i2）
        base = prev_last_num
        for idx, n in enumerate(cur_nums):
            if not n.startswith(base + "i"):
                break
            suffix = n[len(base) + 1 :]
            if not suffix.isdigit():
                break
            new_i = int(suffix) - 1
            if new_i <= 0:
                cur_nums[idx] = f"{base}i1"
            else:
                cur_nums[idx] = f"{base}i{new_i}"

    prev_anchor_num: Optional[str] = None
    blocks = []  # 每个元素: {"region": name, "lines": [nums, templ, target]}
    for region in regions:
        data = cmp["regions"].get(region)
        if not data:
            raise RuntimeError(f"No data for region: {region}")
        tpl_num = data["template_numbering"]
        tpl_seq = data["template_seq"]
        inp_seq = data["input_seq"]

        out = _align_with_dp(tpl_num, tpl_seq, inp_seq, region=region, prev_num=prev_anchor_num)

        # 第1行：编号
        nums = list(out.get("aligned_numbering", []))
        # 第3行：目标序列（对齐后）
        targ = list(out.get("aligned_sequence", []))

        # 第2行：模板序列（优先从 _align_with_dp 的返回里拿；否则用 '-' 占位到等长）
        templ = list(out.get("aligned_template", []))
        if not templ or len(templ) != len(nums):
            templ = ['-'] * len(nums)

        cur_block = {
            "region": region,
            "lines": [nums, templ, targ],
        }

        if blocks:
            _fix_boundary_fill(blocks[-1], cur_block)

        blocks.append(cur_block)

        prev_anchor_num = _anchor_from_aligned(nums)

    # 可选 LLM 校对
    if use_llm:
        blocks = postcheck_alignment(blocks)

    def _fix_cross_block_insertion(prev_blk: dict, cur_blk: dict) -> None:
        # 处理跨区段：若上一段末尾有基号 i1，但下一段的“后一个基号”为空，
        # 且模板碱基与 i1 一致，则把 i1 移到后一个基号，并将剩余插入改挂到后一个基号。
        prev_nums, prev_templ, prev_targ = prev_blk["lines"]
        cur_nums, cur_templ, cur_targ = cur_blk["lines"]
        if not prev_nums or not cur_nums:
            return
        # 找上一段最后一个“非插入、非占位、非 V”的基号
        prev_base = None
        for n in reversed(prev_nums):
            if not n or n.startswith("-") or n.startswith("V") or "i" in n:
                continue
            prev_base = n
            break
        if not prev_base or not prev_base.isdigit():
            return
        # 收集上一段末尾的插入列（必须是连续尾部）
        ins_bases: List[str] = []
        while prev_nums and prev_nums[-1].startswith(prev_base + "i"):
            ins_bases.append(prev_targ.pop())
            prev_nums.pop()
            prev_templ.pop()
        if not ins_bases:
            return
        ins_bases.reverse()  # i1, i2, ...
        # 找下一段第一个基号
        cur_base = None
        cur_base_idx = None
        for i, n in enumerate(cur_nums):
            if not n or n.startswith("-") or n.startswith("V") or "i" in n:
                continue
            cur_base = n
            cur_base_idx = i
            break
        if not cur_base or not cur_base.isdigit():
            return
        # 仅处理相邻基号（例如 26 -> 27）
        if int(cur_base) != int(prev_base) + 1:
            return
        if cur_base_idx is None:
            return
        # 规则：下一基号位为空，且模板碱基与 i1 一致，才允许搬移
        if cur_targ[cur_base_idx] != "-":
            return
        ins1 = ins_bases[0]
        if not ins1 or ins1 == "-":
            return
        if cur_templ[cur_base_idx].upper() != ins1.upper():
            return
        # 把 i1 移到下一基号
        cur_targ[cur_base_idx] = ins1
        # 把剩余插入改挂到下一基号（27i1, 27i2...）
        remain = ins_bases[1:]
        if remain:
            insert_pos = cur_base_idx + 1
            for k, base in enumerate(remain, start=1):
                cur_nums.insert(insert_pos, f"{cur_base}i{k}")
                cur_templ.insert(insert_pos, "-")
                cur_targ.insert(insert_pos, base)
                insert_pos += 1
            # 若后面已有 27i*，把编号整体后移
            shift = len(remain)
            for i in range(cur_base_idx + 1 + shift, len(cur_nums)):
                n = cur_nums[i]
                if not n.startswith(cur_base + "i"):
                    break
                suffix = n[len(cur_base) + 1 :]
                if suffix.isdigit():
                    cur_nums[i] = f"{cur_base}i{int(suffix) + shift}"

    def _normalize_insertions_in_block(block: dict) -> None:
        nums, templ, targ = block["lines"]
        i = 0
        while i < len(nums):
            n = nums[i]
            if not n or n.startswith("-") or n.startswith("V") or "i" in n:
                i += 1
                continue
            base = n
            # 收集该基号后的连续插入列
            j = i + 1
            while j < len(nums) and nums[j].startswith(base + "i"):
                j += 1
            if j == i + 1:
                i += 1
                continue
            ins_idx = i + 1
            ins_base = targ[ins_idx]
            if ins_base == "-":
                i += 1
                continue
            # 规则1：若基号位为空，则回填 i1 到基号位，并左移剩余插入编号
            if targ[i] == "-":
                targ[i] = ins_base
                del nums[ins_idx]
                del templ[ins_idx]
                del targ[ins_idx]
                k = i + 1
                while k < len(nums) and nums[k].startswith(base + "i"):
                    suffix = nums[k][len(base) + 1 :]
                    if suffix.isdigit():
                        new_i = int(suffix) - 1
                        nums[k] = f"{base}i{new_i}"
                    k += 1
                continue
            # 规则2：若后一个基号位为空且模板碱基与 i1 一致，则把 i1 移到后一个基号位
            next_idx = None
            for k in range(j, len(nums)):
                nn = nums[k]
                if not nn or nn.startswith("-") or nn.startswith("V") or "i" in nn:
                    continue
                next_idx = k
                break
            if next_idx is not None and targ[next_idx] == "-" and templ[next_idx].upper() == ins_base.upper():
                targ[next_idx] = ins_base
                del nums[ins_idx]
                del templ[ins_idx]
                del targ[ins_idx]
                if ins_idx < next_idx:
                    next_idx -= 1
                new_base = nums[next_idx]
                k = i + 1
                new_i = 1
                while k < len(nums) and nums[k].startswith(base + "i"):
                    nums[k] = f"{new_base}i{new_i}"
                    new_i += 1
                    k += 1
            i += 1

    if len(blocks) > 1:
        for i in range(1, len(blocks)):
            _fix_cross_block_insertion(blocks[i - 1], blocks[i])

    for blk in blocks:
        _normalize_insertions_in_block(blk)

    # 展平成整分子的两行（编号 + 目标）。模板行仅用于 LLM，不写入 CSV。
    full_nums: List[str] = []
    full_seqs: List[str] = []
    for blk in blocks:
        nums, templ, targ = blk["lines"]
        full_nums.extend(nums)
        full_seqs.extend(targ)

    # 写两行 CSV
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(full_nums)  # 第一行：编号
    writer.writerow(full_seqs)  # 第二行：对齐后的目标序列
    csv_content = buffer.getvalue()

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(csv_content)

    return template_name, csv_content
