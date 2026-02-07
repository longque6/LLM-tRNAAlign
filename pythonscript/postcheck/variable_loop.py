# -*- coding: utf-8 -*-
# ==============================================
# File: pythonscript/postcheck/variable_loop.py
# ----------------------------------------------
# Variable loop/arm post-check (TARGET-ONLY over variable block):
# - NEVER move columns; frontend order is fixed.
# - KEEP LINE1 (labels/numbering) and LINE2 (template) EXACTLY as-is.
# - ONLY rewrite LINE3 (target) within the VARIABLE BLOCK:
#     * Preserve the exact left-to-right order of all non-gap bases
#       (same sequence, same multiplicity, same order).
#     * Stem pairing rule:
#         - If BOTH labels {V1k, V2k} exist, they must be BOTH non-gaps or BOTH '-'.
#         - If ONLY ONE side label exists (mate label missing), that side MUST be '-'.
#     * Loop labels (V1..V5) take at most one base each (or '-').
# - LLM first: propose "var_target" (variable-block tokens). Validate strictly.
#   On ANY violation, feed explicit FEEDBACK and retry.
# - If LLM fails, use a deterministic allocator (order-preserving):
#   * Compute x pairs and y loops so 2*x + y == T (T = #non-gap tokens in variable block),
#     y <= loop capacity, x <= #available pairs, and x as large as capacity allows.
#   * Build the exact ENABLED column set = {both ends of the chosen x pairs} ∪ {y loop columns}.
#   * Single left-to-right pass: for each ENABLED column, consume the next token and write it;
#     all other variable columns write '-'. This guarantees order-preserving.
# ==============================================

from __future__ import annotations
import os
import json
import re
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple

import requests
from requests.exceptions import RequestException, Timeout, SSLError, ConnectionError, ReadTimeout

# ===== Env =====
MODEL_NAME        = os.getenv("LLM_MODEL", "qwen3:32b")
# 默认走本地 Ollama（可用 LLM_BASE_URL 覆盖）
BASE_URL          = os.getenv("LLM_BASE_URL", "http://localhost:11434").rstrip("/")
POSTCHECK_ON      = os.getenv("LLM_POSTCHECK", "1") not in ("0", "false", "False")
VERIFY_SSL        = os.getenv("LLM_VERIFY_SSL", "1") not in ("0", "false", "False")

TIMEOUT_CONNECT   = float(os.getenv("LLM_CONNECT_TIMEOUT", "8"))
TIMEOUT_READ      = float(os.getenv("LLM_READ_TIMEOUT", "20"))
HARD_BUDGET_S     = float(os.getenv("LLM_HARD_BUDGET", "30"))
# 默认开启流式（可用 LLM_STREAM=0 关闭）
STREAMING_ENV     = os.getenv("LLM_STREAM", "1") in ("1", "true", "True")
MAX_LOG_CHARS     = int(os.getenv("LLM_MAX_LOG_CHARS", "4000"))
MAX_RETRIES       = int(os.getenv("LLM_RETRIES", "2"))
BACKOFF_BASE_S    = float(os.getenv("LLM_BACKOFF_BASE", "0.5"))

# ===== Logging helper =====
def _log_trunc(s: str, limit: int = MAX_LOG_CHARS) -> str:
    if s is None:
        return ""
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... [truncated {len(s)-limit} chars]"

print(f"[POSTCHECK][CFG] BASE_URL={BASE_URL}, MODEL={MODEL_NAME}, "
      f"CONNECT_TIMEOUT={TIMEOUT_CONNECT}s, READ_TIMEOUT={TIMEOUT_READ}s, "
      f"HARD_BUDGET={HARD_BUDGET_S}s, VERIFY_SSL={VERIFY_SSL}, "
      f"STREAMING={STREAMING_ENV}, POSTCHECK_ON={POSTCHECK_ON}, "
      f"RETRIES={MAX_RETRIES}")

# -----------------------------
# Variable-label helpers
# -----------------------------
_RE_VAR_5 = re.compile(r'^V1([1-7])$')   # V11..V17 (5' stem)
_RE_VAR_L = re.compile(r'^V([1-5])$')    # V1..V5   (loop)
_RE_VAR_3 = re.compile(r'^V2([1-7])$')   # V21..V27 (3' stem)

def _classify_var_position(num: str) -> Optional[str]:
    """Return 'stem5', 'stem3', or 'loop' (None for non-variable)."""
    if not isinstance(num, str):
        return None
    if _RE_VAR_5.match(num): return "stem5"
    if _RE_VAR_L.match(num): return "loop"
    if _RE_VAR_3.match(num): return "stem3"
    return None

def _var_indices(nums: List[str]) -> List[int]:
    return [i for i, n in enumerate(nums) if _classify_var_position(n or "") is not None]

def _stem_k_from_label(lbl: str) -> Optional[int]:
    m5 = _RE_VAR_5.match(lbl)
    if m5: return int(m5.group(1))
    m3 = _RE_VAR_3.match(lbl)
    if m3: return int(m3.group(1))
    return None

def _pairs_map(nums: List[str]) -> Dict[int, Tuple[int, int]]:
    """
    Return {k: (i_left, i_right)} for every k where BOTH V1k and V2k exist.
    i_left < i_right are absolute column indices.
    """
    pos = {}
    for i, n in enumerate(nums):
        k = _stem_k_from_label(n or "")
        if k is None:
            continue
        if n.startswith("V1"):
            pos.setdefault(k, {})["L"] = i
        elif n.startswith("V2"):
            pos.setdefault(k, {})["R"] = i
    pairs = {}
    for k, d in pos.items():
        if "L" in d and "R" in d:
            iL, iR = d["L"], d["R"]
            pairs[k] = (iL, iR) if iL < iR else (iR, iL)
    return pairs

def _loop_indices(nums: List[str]) -> List[int]:
    return [i for i, n in enumerate(nums) if _classify_var_position(n or "") == "loop"]

def _singletons_indices(nums: List[str]) -> List[int]:
    """
    Indices of stem labels where ONLY ONE side exists (mate label missing).
    These MUST be '-' by rule.
    """
    seenL: Dict[int, int] = {}
    seenR: Dict[int, int] = {}
    for i, n in enumerate(nums):
        k = _stem_k_from_label(n or "")
        if k is None:
            continue
        if n.startswith("V1"): seenL[k] = i
        elif n.startswith("V2"): seenR[k] = i
    singles = []
    for k, iL in seenL.items():
        if k not in seenR:
            singles.append(iL)
    for k, iR in seenR.items():
        if k not in seenL:
            singles.append(iR)
    return sorted(singles)

# -----------------------------
# Validation on target
# -----------------------------
_VALID_BASES = set(list("AUGCN"))

def _check_pairing(nums: List[str], targ: List[str]) -> Tuple[bool, List[str]]:
    """
    Pairing rule on stems:
      - If both labels exist: targ must have BOTH non-gap or BOTH '-'.
      - If only one label exists: that side MUST be '-'.
    """
    pairs = _pairs_map(nums)
    errs = []

    # both exist → must be both gaps or both bases
    for k, (iL, iR) in pairs.items():
        a, b = targ[iL], targ[iR]
        if (a == '-') != (b == '-'):
            errs.append(f"Unpaired stem: V1{k}↔V2{k} got '{a}' vs '{b}'")

    # only one side exists → that side must be '-'
    singles = _singletons_indices(nums)
    for idx in singles:
        if targ[idx] != '-':
            lbl = nums[idx]
            mate = f"V2{_stem_k_from_label(lbl)}" if lbl.startswith("V1") else f"V1{_stem_k_from_label(lbl)}"
            errs.append(f"Singleton stem must be '-': have {lbl}='{targ[idx]}', missing mate {mate}")

    return (len(errs) == 0, errs)

def _non_gap_sequence_in_var(nums: List[str], targ: List[str]) -> List[str]:
    vi = _var_indices(nums)
    return [targ[i] for i in vi if targ[i] != '-']

def _check_order_preserved(nums: List[str], targ_before: List[str], targ_after: List[str]) -> bool:
    before = _non_gap_sequence_in_var(nums, targ_before)
    after  = _non_gap_sequence_in_var(nums, targ_after)
    ok = (before == after)
    if not ok:
        print("[POSTCHECK][CHK] Variable-block token order changed! before:",
              ''.join(before), "after:", ''.join(after))
    return ok

def _check_var_target_shape_and_order(var_target: List[str], nums: List[str],
                                      targ_before: List[str]) -> Tuple[bool, str]:
    vi = _var_indices(nums)
    if not isinstance(var_target, list) or len(var_target) != len(vi):
        return False, f"length mismatch (expected {len(vi)}, got {len(var_target) if isinstance(var_target,list) else 'non-list'})"
    for ch in var_target:
        if ch != '-' and ch.upper() not in _VALID_BASES:
            return False, f"invalid token '{ch}'"

    # order of non-gap tokens must be preserved
    before_bases = [targ_before[i] for i in vi if targ_before[i] != '-']
    after_bases  = [x for x in var_target if x != '-']
    if before_bases != after_bases:
        return False, f"base order mismatch: expected {''.join(before_bases)}, got {''.join(after_bases)}"

    # singleton stems must be '-'
    singles = _singletons_indices(nums)
    for j, i in enumerate(vi):
        if i in singles and var_target[j] != '-':
            return False, f"singleton at index {i} ({nums[i]}) must be '-', got '{var_target[j]}'"

    return True, ""

# -----------------------------
# Deterministic allocator (order-preserving; enable-set + single pass)
# -----------------------------
def _alloc_deterministic(nums: List[str], targ_before: List[str]) -> List[str]:
    """
    Order-preserving deterministic allocator:

      1) Collect tokens T = original non-gap bases in variable block.
      2) Compute pairs (both labels exist) and loop capacity (V1..V5).
      3) Choose x,y so 2*x + y == T, with x <= #pairs, y <= loop_capacity,
         and x as large as possible (maximize pairing under capacity).
      4) ENABLED columns = both ends of the first x pairs (by left-index order)
                           ∪ first y loop columns (by index).
         NOTE: any singleton stems are NOT enabled (must remain '-').
      5) Single left-to-right pass over all columns:
         if idx ∈ ENABLED → take next token and write; else write '-' for variable columns.
         Non-variable columns unchanged.

      This preserves the non-gap order exactly and prevents orphans.
    """
    vi = _var_indices(nums)
    new_targ = list(targ_before)
    # Zero out variable block
    for i in vi:
        new_targ[i] = '-'

    tokens = _non_gap_sequence_in_var(nums, targ_before)
    T = len(tokens)
    if T == 0:
        return new_targ

    pairs_dict = _pairs_map(nums)  # {k: (iL,iR)}
    pairs_sorted = sorted(pairs_dict.items(), key=lambda kv: kv[1][0])  # by left index
    P = len(pairs_sorted)

    loops_sorted = sorted(_loop_indices(nums))
    L = len(loops_sorted)

    # choose x,y
    x = min(P, T // 2)
    while x >= 0 and (T - 2*x) > L:
        x -= 1
    if x < 0:
        # capacity infeasible; safest: keep original var-block
        print("[POSTCHECK][FALLBACK] capacity infeasible; keep var-block as-is.")
        return list(targ_before)

    y = T - 2*x  # 0..L

    # Build ENABLED columns
    chosen_pairs = pairs_sorted[:x]
    enabled: set[int] = set()
    for _, (iL, iR) in chosen_pairs:
        enabled.add(iL); enabled.add(iR)
    enabled_loops = loops_sorted[:y]
    for idx in enabled_loops:
        enabled.add(idx)

    # Enforce singleton stems MUST be '-' (do not enable them)
    for idx in _singletons_indices(nums):
        if idx in enabled:
            enabled.remove(idx)

    # Fill in order
    p = 0
    for idx, lbl in enumerate(nums):
        if _classify_var_position(lbl or "") is None:
            continue
        if idx in enabled:
            if p < T:
                new_targ[idx] = tokens[p]; p += 1
        # else remains '-'

    # Safety checks
    if not _check_order_preserved(nums, targ_before, new_targ):
        before = ''.join(_non_gap_sequence_in_var(nums, targ_before))
        after  = ''.join(_non_gap_sequence_in_var(nums, new_targ))
        print(f"[POSTCHECK][FATAL] deterministic allocator order changed (must never happen). before={before} after={after}")
        return list(targ_before)

    ok_pair, errs = _check_pairing(nums, new_targ)
    if not ok_pair:
        print("[POSTCHECK][WARN] deterministic allocator pairing failed:", "; ".join(errs))
        return list(targ_before)

    return new_targ

# -----------------------------
# LLM proposer with FEEDBACK loop (returns var_target list)
# -----------------------------
def _headers_with_auth() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers

def _http_post_json(url: str, payload: dict) -> Tuple[bool, str]:
    print("\n[POSTCHECK][HTTP] POST", url)
    headers = _headers_with_auth()
    log_headers = dict(headers)
    if "Authorization" in log_headers:
        log_headers["Authorization"] = "Bearer ****(hidden)****"
    print("[POSTCHECK][HTTP] headers:", log_headers)
    print("[POSTCHECK][HTTP] payload:", _log_trunc(json.dumps(payload, ensure_ascii=False)))
    start = time.monotonic()
    stream_opt = payload.get("stream", STREAMING_ENV)
    try:
        resp = requests.post(
            url,
            data=json.dumps(payload),
            headers=headers,
            timeout=(TIMEOUT_CONNECT, TIMEOUT_READ),
            verify=VERIFY_SSL,
            stream=stream_opt
        )
    except (ReadTimeout, Timeout, SSLError, ConnectionError, RequestException) as e:
        print(f"[POSTCHECK][HTTP] request exception after {time.monotonic()-start:.2f}s:", repr(e))
        print(traceback.format_exc())
        return False, ""
    elapsed = time.monotonic() - start
    print(f"[POSTCHECK][HTTP] status: {resp.status_code} (elapsed {elapsed:.2f}s)")
    if resp.status_code != 200 or elapsed > HARD_BUDGET_S:
        if not stream_opt:
            print("[POSTCHECK][HTTP] raw text:", _log_trunc(resp.text))
        return False, ""

    if stream_opt:
        chunks: List[str] = []
        display = ""
        content_parts: List[str] = []
        try:
            for line in resp.iter_lines(decode_unicode=True, chunk_size=1024):
                if not line:
                    continue
                raw_line = line
                if raw_line.startswith("data:"):
                    raw_line = raw_line[len("data:"):].strip()
                if raw_line in ("[DONE]", ""):
                    continue

                # 尝试从流式 JSON 中提取 content 片段（忽略 reasoning/role）
                piece = ""
                try:
                    parsed = json.loads(raw_line)
                    # OpenAI 样式
                    choices = parsed.get("choices")
                    if isinstance(choices, list) and choices:
                        delta = choices[0].get("delta") or choices[0].get("message") or {}
                        piece = str(delta.get("content") or "")
                    # Ollama 样式
                    if not piece and isinstance(parsed.get("message"), dict):
                        piece = str(parsed["message"].get("content") or "")
                except Exception:
                    piece = ""

                # 只在有内容时输出，避免刷屏
                if not piece:
                    continue

                content_parts.append(piece)
                display = (display + piece)[-400:]  # 只保留末尾展示
                print("\r[POSTCHECK][STREAM] " + _log_trunc(display, 400), end="", flush=True)
            print()  # 换行
        except Exception as e:
            print("\n[POSTCHECK][STREAM] error while reading:", repr(e))
        content = "".join(content_parts).strip()
        if not content:
            content = "\n".join(chunks)
        return True, content
    else:
        text = resp.text or ""
        # 非流模式才打印最终文本，避免重复
        print("[POSTCHECK][HTTP] raw text:", _log_trunc(text))
        return True, text

def _is_ollama(url: str) -> bool:
    u = url.lower()
    return ("ollama" in u) or ("/api" in u)

def _build_llm_messages(nums: List[str], targ_before: List[str]) -> List[Dict[str, str]]:
    vi = _var_indices(nums)
    var_labels = [nums[i] for i in vi]
    orig_bases = [targ_before[i] for i in vi if targ_before[i] != '-']

    pairs = _pairs_map(nums)
    pairs_lr = [f"V1{k}↔V2{k}" for k, _ in sorted(pairs.items(), key=lambda kv: kv[1][0])]
    loops = [nums[i] for i in sorted(_loop_indices(nums))]
    singles_idx = _singletons_indices(nums)
    singles_desc = [f"{nums[i]}(idx={i})" for i in singles_idx]

    sys_prompt = (
        "You are a strict RNA alignment post-editor.\n"
        "CRITICAL RULES:\n"
        "1) NEVER move columns; NEVER change numbering (LINE1) or template (LINE2).\n"
        "2) ONLY produce a new TARGET token list for the VARIABLE BLOCK.\n"
        "3) Preserve the EXACT left-to-right order of ALL non-gap bases from the original variable block.\n"
        "4) Stem pairing:\n"
        "   - If BOTH labels V1k and V2k exist, return EITHER two bases (both non-gaps) OR two '-' (no orphan).\n"
        "   - If ONLY ONE side label exists (mate missing), that side MUST be '-' (do not place a base there).\n"
        "5) Loop labels (V1..V5) take at most one base each; otherwise '-'.\n"
        "6) Use only A/U/G/C/N or '-' (dash). Return ONLY JSON with key 'var_target' (array of strings)."
    )
    user_prompt = (
        "VARIABLE LABELS (left-to-right; fixed):\n"
        f"{' '.join(var_labels)}\n\n"
        "ORIGINAL NON-GAP BASES in this variable block (order to preserve):\n"
        f"{' '.join(orig_bases) if orig_bases else '(empty)'}\n\n"
        "PRESENT STEM PAIRS (must be both bases or both dashes):\n"
        f"{', '.join(pairs_lr) if pairs_lr else '(none)'}\n"
        "PRESENT LOOP LABELS:\n"
        f"{' '.join(loops) if loops else '(none)'}\n"
        "SINGLETON STEM POSITIONS (MUST be '-'):\n"
        f"{', '.join(singles_desc) if singles_desc else '(none)'}\n\n"
        f"Variable-block length = {len(vi)}. Return exactly this many tokens.\n"
        "Return only JSON: {\"var_target\": [\"A\", \"-\", \"U\", ...]}"
    )
    return [{"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_prompt}]

def _llm_propose_var_target_with_feedback(nums: List[str], targ_before: List[str]) -> Optional[List[str]]:
    messages = _build_llm_messages(nums, targ_before)
    attempts = 0
    start = time.monotonic()

    while attempts <= MAX_RETRIES:
        attempts += 1
        print(f"[POSTCHECK][LLM] Variable block TARGET proposal attempt #{attempts} ...")

        # Call
        if _is_ollama(BASE_URL):
            url = f"{BASE_URL}/api/chat"
            payload = {"model": MODEL_NAME, "messages": messages,
                       "options": {"temperature": 0}, "format": "json", "stream": STREAMING_ENV}
            ok, content = _http_post_json(url, payload)
            if not ok:
                if (time.monotonic() - start) >= HARD_BUDGET_S: return None
                delay = min(BACKOFF_BASE_S * (2 ** (attempts - 1)), 2.0)
                print(f"[POSTCHECK][LLM] call failed; backoff {delay:.2f}s")
                time.sleep(delay)
                continue
            if not content:
                try:
                    data = json.loads(content) if content else {}
                    content = ((data.get("message") or {}).get("content") or "").strip()
                except Exception:
                    content = ""
        else:
            url = f"{BASE_URL}/v1/chat/completions"
            payload = {"model": MODEL_NAME, "messages": messages,
                       "temperature": 0.0, "max_tokens": 1024, "stream": STREAMING_ENV}
            ok, content = _http_post_json(url, payload)
            if not ok:
                if (time.monotonic() - start) >= HARD_BUDGET_S: return None
                delay = min(BACKOFF_BASE_S * (2 ** (attempts - 1)), 2.0)
                print(f"[POSTCHECK][LLM] call failed; backoff {delay:.2f}s")
                time.sleep(delay)
                continue
            if not content:
                try:
                    data = json.loads(content) if content else {}
                    content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                    content = content.strip()
                except Exception:
                    content = ""

        m = re.search(r"\{[\s\S]*\}", content)
        if not m:
            messages.append({"role": "user",
                             "content": "FEEDBACK: JSON object not found. Return only {\"var_target\": [...]}."})
            continue

        try:
            obj = json.loads(m.group(0))
            var_target = obj.get("var_target", None)
        except Exception:
            messages.append({"role": "user",
                             "content": "FEEDBACK: JSON parse error. Return only {\"var_target\": [...]}."})
            continue

        if not isinstance(var_target, list):
            messages.append({"role": "user",
                             "content": "FEEDBACK: 'var_target' must be an array of tokens."})
            continue

        # Shape & order & singleton checks
        ok_shape, err = _check_var_target_shape_and_order(var_target, nums, targ_before)
        if not ok_shape:
            vi = _var_indices(nums)
            before_bases = [targ_before[i] for i in vi if targ_before[i] != '-']
            after_bases  = [x for x in var_target if x != '-']
            singles = _singletons_indices(nums)
            must_dash = [nums[i] for i in singles]
            fb = (
                "FEEDBACK: Rejected.\n"
                f"- Reason: {err}\n"
                f"- Expected non-gap order: {' '.join(before_bases) if before_bases else '(empty)'}\n"
                f"- You returned order     : {' '.join(after_bases) if after_bases else '(empty)'}\n"
                f"- These singleton labels MUST be '-': {' '.join(must_dash) if must_dash else '(none)'}\n"
                "Please regenerate 'var_target' strictly preserving the non-gap order and length, "
                "and NEVER place a base at any singleton stem."
            )
            messages.append({"role": "user", "content": fb})
            continue

        # Build candidate targ and validate pairing + final order
        vi = _var_indices(nums)
        targ_llm = list(targ_before)
        for j, i in enumerate(vi):
            tok = str(var_target[j]).upper()
            targ_llm[i] = tok

        ok_pair, errs = _check_pairing(nums, targ_llm)
        if not ok_pair:
            messages.append({"role": "user",
                             "content": "FEEDBACK: Pairing violation: " + "; ".join(errs)})
            continue

        if not _check_order_preserved(nums, targ_before, targ_llm):
            before = ''.join(_non_gap_sequence_in_var(nums, targ_before))
            after  = ''.join(_non_gap_sequence_in_var(nums, targ_llm))
            messages.append({"role": "user", "content":
                f"FEEDBACK: Non-gap order changed! before='{before}', after='{after}'. "
                "Regenerate so the order is IDENTICAL."})
            continue

        print("[POSTCHECK][LLM] var_target accepted.")
        return [targ_llm[i] for i in vi]  # normalized accepted var-target

    print("[POSTCHECK][LLM] give up after retries/budget.")
    return None

# -----------------------------
# Public handler
# -----------------------------
def postcheck_variable_loop_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """
    Variable loop/arm post-check (TARGET-ONLY inside variable columns).
    """
    if not POSTCHECK_ON:
        return block

    lines = block.get("lines", [])
    if not (isinstance(lines, list) and len(lines) == 3):
        return block

    nums  = list(lines[0])  # KEEP
    templ = list(lines[1])  # KEEP
    targ  = list(lines[2])  # MODIFY in var-block only

    vi = _var_indices(nums)
    if not vi:
        print("[POSTCHECK] Variable loop block has no V positions; pass-through.")
        return {"region": block.get("region", "Variable loop"),
                "lines": [nums, templ, targ]}

    # ---- Try LLM with feedback ----
    print("[POSTCHECK][LLM] Variable loop TARGET proposer...")
    var_target = _llm_propose_var_target_with_feedback(nums, targ)

    if var_target is not None:
        # Apply accepted var_target
        targ_new = list(targ)
        for j, i in enumerate(vi):
            targ_new[i] = str(var_target[j]).upper()
        # Final checks
        ok_pair, errs = _check_pairing(nums, targ_new)
        if not ok_pair:
            print("[POSTCHECK][LLM] unexpected pairing failure after accept:", "; ".join(errs))
        if not _check_order_preserved(nums, targ, targ_new):
            print("[POSTCHECK][LLM] unexpected order failure after accept (should not).")
        targ = targ_new
    else:
        # ---- Deterministic fallback (order-preserving) ----
        print("[POSTCHECK] LLM unavailable/invalid; using deterministic allocator.")
        targ_d = _alloc_deterministic(nums, targ)
        targ = targ_d  # allocator内部已做顺序/配对/单侧禁填校验；失败则回退原值

    return {"region": block.get("region", "Variable loop"),
            "lines": [nums, templ, targ]}

# ===== Backward-compat alias =====
def postcheck_variable_loop(block: Dict[str, Any]) -> Dict[str, Any]:
    return postcheck_variable_loop_block(block)
