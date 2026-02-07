import type { Ref } from "vue";

interface GeneratorDeps {
  mode: Ref<"one" | "batch">;
  seedSeq: Ref<string>;
  numSamples: Ref<number>;
  loading: Ref<boolean>;
  error: Ref<string>;
  elapsedSec: Ref<number | null>;
  oneResult: Ref<string>;
  batchResult: Ref<string[]>;
  generateEnabled: Ref<boolean>;
  seqSelectMode: Ref<"select" | "custom">;
  selectedSequenceName: Ref<string>;
  selectedSpecies: Ref<string>;
  adv: Ref<any>;
  freezeRegions: Ref<string[]>;
  preferRegions: Ref<string[]>;
  freezePositionsText: Ref<string>;
  preferPositionsText: Ref<string>;
  forcePositionsText: Ref<string>;
  styleLevel: Ref<number>;
  hiddenParams: Ref<any>;
  REGION_OPTIONS: string[];
}

const DEFAULT_OVERSAMPLE_FACTOR = 5;

export function useGeneratorLogic(deps: GeneratorDeps) {
  const {
    mode,
    seedSeq,
    numSamples,
    loading,
    error,
    elapsedSec,
    oneResult,
    batchResult,
    generateEnabled,
    adv,
    freezeRegions,
    preferRegions,
    freezePositionsText,
    preferPositionsText,
    forcePositionsText,
    styleLevel,
    hiddenParams,
    REGION_OPTIONS,
  } = deps;

  /* 滑块调参映射 */
  function applyStyle() {
    const lv = styleLevel.value;
    const temps = [0.85, 1.5, 1.95, 2.4, 2.85];
    const topps = [0.98, 0.95, 0.92, 0.88, 0.85];
    const topks = [0, 8, 16, 32, 64];
    const mfrac = [0.1, 0.14, 0.2, 0.28, 0.35];
    const mk = [3, 4, 5, 7, 9];
    const rounds = [2, 3, 4, 6, 8];
    const minhd = [1, 2, 3, 4, 6];
    const rminhd = [2, 4, 6, 8, 10];

    adv.value = {
      ...adv.value,
      temperature: temps[lv],
      top_p: topps[lv],
      rounds: rounds[lv],
      min_hd: minhd[lv],
      rerank_min_hd: rminhd[lv],
    };
    hiddenParams.value = {
      top_k: topks[lv],
      mask_frac: mfrac[lv],
      mask_k: mk[lv],
    };
  }

  /* 序列规范化 */
  function normalizeSeq(raw: string) {
    return raw
      .toUpperCase()
      .replace(/T/g, "U")
      .replace(/[^AUGC]/g, "N");
  }

  /* 解析工具 */
  function parseIntList(s: string): number[] {
    return (s || "")
      .split(/[^0-9]+/g)
      .map((x) => parseInt(x.trim(), 10))
      .filter((n) => !Number.isNaN(n));
  }

  function parseForceMap(s: string): Record<number, string> {
    const out: Record<number, string> = {};
    if (!s) return out;
    if (s.trim().startsWith("{")) {
      try {
        const obj = JSON.parse(s);
        for (const k in obj) {
          const n = parseInt(k, 10);
          const v = String(obj[k]).toUpperCase();
          if (!Number.isNaN(n) && /^[ACGTU]$/.test(v)) out[n] = v;
        }
        return out;
      } catch {}
    }
    s.split(/[;,]+/g).forEach((pair) => {
      const [k, v] = pair.split(":").map((x) => x.trim());
      const n = parseInt(k, 10);
      const base = (v || "").toUpperCase();
      if (!Number.isNaN(n) && /^[ACGTU]$/.test(base)) out[n] = base;
    });
    return out;
  }

  /* 区域冲突检测 */
  function validateRegionSelections(): string | null {
    const all = REGION_OPTIONS.length;
    if (freezeRegions.value.length === all)
      return "Freeze regions cannot include all regions.";
    const overlap = freezeRegions.value.filter((r) =>
      preferRegions.value.includes(r)
    );
    return overlap.length
      ? `Freeze and Prefer cannot overlap: ${overlap.join(", ")}`
      : null;
  }

  /* 通用 POST 封装 */
  async function postJSON(url: string, payload: any) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  /* 执行生成 */
  async function onRun() {
    error.value = "";
    elapsedSec.value = null;
    oneResult.value = "";
    batchResult.value = [];
    const seq = normalizeSeq(seedSeq.value || "");
    if (!seq) {
      error.value = "Please input a seed sequence.";
      return;
    }

    // 检查序列长度要求
    if (seq.length <= 50) {
      error.value = "Sequence must be longer than 50 bases to generate.";
      return;
    }

    const n = Math.max(1, Math.min(numSamples.value, 10));
    numSamples.value = n;

    const regionErr = validateRegionSelections();
    if (regionErr) {
      error.value = regionErr;
      return;
    }

    const freeze_positions = parseIntList(freezePositionsText.value);
    const prefer_positions = parseIntList(preferPositionsText.value);
    const force_positions = parseForceMap(forcePositionsText.value);

    loading.value = true;
    try {
      const basePayload = {
        seed_seq: seq,
        temperature: adv.value.temperature,
        top_p: adv.value.top_p,
        rounds: adv.value.rounds,
        top_k: hiddenParams.value.top_k,
        mask_frac: hiddenParams.value.mask_frac,
        mask_k: hiddenParams.value.mask_k,
        prefer_regions: preferRegions.value.length
          ? [...preferRegions.value]
          : ["Variable_Loop"],
        freeze_regions: [...freezeRegions.value],
        freeze_positions,
        prefer_positions,
        force_positions,
      };

      if (mode.value === "one") {
        const data = await postJSON("/trnagen/one", basePayload);
        oneResult.value = data.sequence || "";
        elapsedSec.value = data.elapsed_sec ?? null;
      } else {
        const data = await postJSON("/trnagen/batch", {
          ...basePayload,
          num_samples: n,
          oversample_factor: DEFAULT_OVERSAMPLE_FACTOR,
          min_hd: adv.value.min_hd,
          rerank_min_hd: adv.value.rerank_min_hd,
          gc_low: adv.value.gc_low,
          gc_high: adv.value.gc_high,
        });
        batchResult.value = data.sequences || [];
        elapsedSec.value = data.elapsed_sec ?? null;
      }
    } catch (e: any) {
      error.value = e.message || String(e);
    } finally {
      loading.value = false;
    }
  }

  /* 下载与复制 */
  function downloadFasta() {
    const txt = batchResult.value
      .map((s, i) => `>seq_${i + 1}\n${s}`)
      .join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([txt], { type: "text/plain" }));
    a.download = `generated_${new Date()
      .toISOString()
      .replace(/[:.]/g, "-")}.fa`;
    a.click();
  }

  async function copy(t: string) {
    try {
      await navigator.clipboard.writeText(t);
      alert("Copied!");
    } catch {
      alert("Copy failed.");
    }
  }

  /* 重置 */
  function resetAll() {
    seedSeq.value =
      "GGUCUCGUGGCCCAAUGGUUAAGGCGCUUGACUACGGAUCAAGAGAUUCCAGGUUCGACUCCUGGCGGGAUCG";
    numSamples.value = 3;
    styleLevel.value = 2;
    applyStyle();
    oneResult.value = "";
    batchResult.value = [];
    error.value = "";
    elapsedSec.value = null;
    generateEnabled.value = false;

    // 高级控制复位
    freezeRegions.value = [];
    preferRegions.value = ["Variable_Loop"];
    freezePositionsText.value = "";
    preferPositionsText.value = "";
    forcePositionsText.value = "";

    // 序列选择状态复位
    deps.seqSelectMode.value = "select";
    deps.selectedSequenceName.value = "";
    deps.selectedSpecies.value = "";
  }

  function clearResults() {
    oneResult.value = "";
    batchResult.value = [];
    error.value = "";
    elapsedSec.value = null;
    loading.value = false;
  }

  return {
    applyStyle,
    onRun,
    resetAll,
    clearResults,
    copy,
    downloadFasta,
  };
}
