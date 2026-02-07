# src/views/generate/GeneratorResults.vue
<template>
  <section v-if="rows.length" class="card">
    <h2 class="card-title">
      {{ mode === "one" ? "Single result" : `Batch results (${rows.length})` }}
    </h2>

    <div class="table-wrap">
      <table class="result-table">
        <thead>
          <tr>
            <th style="width: 64px">#</th>
            <th>Sequence (compared to seed)</th>
            <th class="sortable" @click="toggleSort('pred')">
              Readthrough
              <span class="sort-ico" :class="sortIcon('pred')"></span>
            </th>
            <th class="sortable" @click="toggleSort('conf')">
              Confidence (%)
              <span class="sort-ico" :class="sortIcon('conf')"></span>
            </th>
            <th style="width: 120px">Copy Seq</th>
          </tr>
        </thead>

        <tbody>
          <tr v-for="r in sortedRows" :key="r._key">
            <td>Seq {{ r.idx + 1 }}</td>
            <td class="sequence-cell">
              <div class="sequence-comparison">
                <span
                  v-for="(char, index) in r.seq"
                  :key="index"
                  class="nucleotide"
                  :class="{
                    changed: seedSeq && char !== seedSeq[index],
                    unchanged: seedSeq && char === seedSeq[index],
                  }"
                  :title="
                    seedSeq
                      ? `Seed: ${seedSeq[index]}, Generated: ${char}`
                      : char
                  "
                >
                  {{ char }}
                </span>
              </div>
              <div v-if="r.loading" class="loading-indicator">
                <div class="spinner"></div>
                <span class="loading-text">Calculating readthrough...</span>
              </div>
              <div v-if="r.error" class="error-indicator">
                <span class="error-text">Calculation failed</span>
                <button class="btn-retry" @click="retryPrediction(r.seq)">
                  Retry
                </button>
              </div>
            </td>
            <td>
              <span v-if="r.pred_raw != null">{{
                formatNum(r.pred_raw, 3)
              }}</span>
              <span v-else-if="r.loading" class="muted">–</span>
              <span v-else class="muted">–</span>
            </td>
            <td>
              <span v-if="r.confidence_pct != null">{{
                formatNum(r.confidence_pct, 2)
              }}</span>
              <span v-else-if="r.loading" class="muted">–</span>
              <span v-else class="muted">–</span>
            </td>
            <td>
              <button
                class="btn btn-plain btn-xs"
                @click="copySeq(r.seq)"
                :disabled="r.loading"
              >
                Copy
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- 批量处理状态指示器 -->
    <div v-if="isProcessingBatch" class="batch-status">
      <div class="batch-spinner"></div>
      <span class="batch-text">
        Calculating readthrough... ({{ processedCount }}/{{ rows.length }})
      </span>
    </div>

    <!-- 仅 batch 显示下载按钮 -->
    <div v-if="mode === 'batch' && batchResult.length" class="btn-row">
      <button
        class="btn btn-outline"
        @click="handleDownload"
        title="Export as FASTA"
      >
        Download FASTA
      </button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";

const props = defineProps<{
  mode: "one" | "batch";
  oneResult: string;
  batchResult: string[];
  seedSeq: string;
}>();

const emit = defineEmits<{
  (e: "copy", text: string): void;
  (e: "download"): void;
}>();

// 处理下载事件
const handleDownload = () => {
  emit("download");
};

/* ===================== 行数据与预测缓存 ===================== */
type PredLite = {
  pred_raw: number | null;
  confidence_pct: number | null;
  loading: boolean;
  error: boolean;
  retryCount: number;
};

const STORAGE_KEY = "aylm-readthrough-cache-v1";

function loadPersistedPredictions(): Record<string, PredLite> {
  if (typeof window === "undefined") return {};
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    const hydrated: Record<string, PredLite> = {};
    Object.entries(parsed).forEach(([seq, val]) => {
      hydrated[seq] = {
        pred_raw:
          typeof (val as any)?.pred_raw === "number"
            ? (val as any).pred_raw
            : null,
        confidence_pct:
          typeof (val as any)?.confidence_pct === "number"
            ? (val as any).confidence_pct
            : null,
        loading: false,
        error: false,
        retryCount: 0,
      };
    });
    return hydrated;
  } catch (err) {
    console.warn("Failed to load readthrough cache", err);
    return {};
  }
}

function persistPredictions(cache: Record<string, PredLite>) {
  if (typeof window === "undefined") return;
  try {
    const payload: Record<string, { pred_raw: number | null; confidence_pct: number | null }> =
      {};
    Object.entries(cache).forEach(([seq, info]) => {
      if (info.pred_raw == null && info.confidence_pct == null) return;
      payload[seq] = {
        pred_raw: info.pred_raw,
        confidence_pct: info.confidence_pct,
      };
    });
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch (err) {
    console.warn("Failed to persist readthrough cache", err);
  }
}

const predCache = ref<Record<string, PredLite>>(loadPersistedPredictions());
const processingQueue = ref<string[]>([]);
const isProcessingBatch = ref(false);
const processedCount = ref(0);
const currentProcessingIndex = ref(0);

const rows = computed(() => {
  const seqs =
    props.mode === "one"
      ? props.oneResult
        ? [props.oneResult]
        : []
      : props.batchResult || [];
  return seqs.map((seq, i) => ({
    _key: `${i}-${seq.slice(0, 12)}`,
    idx: i,
    seq,
    pred_raw: predCache.value[seq]?.pred_raw ?? null,
    confidence_pct: predCache.value[seq]?.confidence_pct ?? null,
    loading: predCache.value[seq]?.loading ?? false,
    error: predCache.value[seq]?.error ?? false,
    retryCount: predCache.value[seq]?.retryCount ?? 0,
  }));
});

// 监听序列变化，开始批量处理
watch(
  rows,
  (newRows) => {
    if (newRows.length > 0) {
      startBatchProcessing();
    }
  },
  { immediate: true }
);

// 开始批量处理
function startBatchProcessing() {
  if (isProcessingBatch.value) return;

  const sequences = rows.value.map((r) => r.seq);
  processingQueue.value = [...sequences];
  isProcessingBatch.value = true;
  processedCount.value = 0;
  currentProcessingIndex.value = 0;

  processNextSequence();
}

// 处理下一个序列
async function processNextSequence() {
  if (currentProcessingIndex.value >= processingQueue.value.length) {
    isProcessingBatch.value = false;
    return;
  }

  const currentSeq = processingQueue.value[currentProcessingIndex.value];

  if (predCache.value[currentSeq] && !predCache.value[currentSeq].error) {
    currentProcessingIndex.value++;
    processedCount.value++;
    processNextSequence();
    return;
  }

  await fetchPredictionWithRetry(currentSeq);

  currentProcessingIndex.value++;
  processedCount.value++;
  processNextSequence();
}

// 带重试的预测请求
async function fetchPredictionWithRetry(seq: string, retryCount = 0) {
  if (!seq) return;

  const current = predCache.value[seq] || {
    pred_raw: null,
    confidence_pct: null,
    loading: false,
    error: false,
    retryCount: 0,
  };

  predCache.value = {
    ...predCache.value,
    [seq]: {
      ...current,
      loading: true,
      error: false,
      retryCount,
    },
  };

  try {
    const res = await fetch("/readthrough/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        accept: "application/json",
      },
      body: JSON.stringify({
        sequence: seq,
        mc_samples: 50,
        ckpt_name: "readthrough",
      }),
    });

    if (!res.ok) throw new Error(await res.text());

    const j = await res.json();
    predCache.value = {
      ...predCache.value,
      [seq]: {
        pred_raw: typeof j?.pred_raw === "number" ? j.pred_raw : null,
        confidence_pct:
          typeof j?.confidence_pct === "number" ? j.confidence_pct : null,
        loading: false,
        error: false,
        retryCount: 0,
      },
    };
    persistPredictions(predCache.value);
  } catch (e) {
    if (retryCount < 3) {
      console.warn(`Prediction failed, retrying (${retryCount + 1}/3)...`, e);
      await new Promise((resolve) =>
        setTimeout(resolve, 1000 * (retryCount + 1))
      );
      return fetchPredictionWithRetry(seq, retryCount + 1);
    } else {
      predCache.value = {
        ...predCache.value,
        [seq]: {
          pred_raw: null,
          confidence_pct: null,
          loading: false,
          error: true,
          retryCount,
        },
      };
      persistPredictions(predCache.value);
      console.error("Prediction failed after 3 retries:", e);
    }
  }
}

// 手动重试预测
function retryPrediction(seq: string) {
  const index = rows.value.findIndex((r) => r.seq === seq);
  if (index !== -1) {
    if (isProcessingBatch.value) {
      processingQueue.value.splice(currentProcessingIndex.value + 1, 0, seq);
    } else {
      processingQueue.value = [seq];
      isProcessingBatch.value = true;
      processedCount.value = 0;
      currentProcessingIndex.value = 0;
      processNextSequence();
    }
  }
}

/* ===================== 排序 ===================== */
const sortKey = ref<"none" | "pred" | "conf">("none");
const sortDir = ref<"asc" | "desc">("desc");

function toggleSort(key: "pred" | "conf") {
  if (sortKey.value !== key) {
    sortKey.value = key;
    sortDir.value = "desc";
  } else {
    sortDir.value = sortDir.value === "desc" ? "asc" : "desc";
  }
}
function sortIcon(key: "pred" | "conf") {
  if (sortKey.value !== key) return "none";
  return sortDir.value === "desc" ? "desc" : "asc";
}

const sortedRows = computed(() => {
  const data = rows.value.slice();
  if (sortKey.value === "none") return data;
  const field = sortKey.value === "pred" ? "pred_raw" : "confidence_pct";
  data.sort((a: any, b: any) => {
    const va = a[field] ?? -Infinity;
    const vb = b[field] ?? -Infinity;
    if (va === vb) return 0;
    return sortDir.value === "asc" ? va - vb : vb - va;
  });
  return data;
});

/* ===================== 复制相关 ===================== */
async function copyToClipboard(t: string) {
  try {
    await navigator.clipboard.writeText(t);
  } finally {
    emit("copy", t);
  }
}
function copySeq(seq: string) {
  copyToClipboard(seq);
}

/* ===================== 小工具 ===================== */
function formatNum(n: number, d = 2) {
  return Number.isFinite(n) ? n.toFixed(d) : "–";
}
</script>

<style scoped>
.card {
  background: #ffffff;
  border: 1px solid #e0e0e0;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}
.card-title {
  font-weight: 800;
  font-size: 1.3rem;
  margin-bottom: 16px;
  color: #333;
}

/* 表格样式 */
.table-wrap {
  overflow-x: auto;
}
.result-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  background: #ffffff;
}
.result-table th,
.result-table td {
  padding: 12px 10px;
  border-bottom: 1px solid #e0e0e0;
  color: #333;
  text-align: left;
}
.result-table thead th {
  background: #f8f9fa;
  position: sticky;
  top: 0;
  z-index: 1;
  font-weight: 600;
}
.muted {
  color: #666;
}

/* 序列单元格样式 */
.sequence-cell {
  position: relative;
  min-height: 50px;
}

.sequence-comparison {
  display: flex;
  flex-wrap: wrap;
  gap: 1px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
    "Liberation Mono", "Courier New", monospace;
  font-size: 13px;
  line-height: 1.4;
}

.nucleotide {
  padding: 2px 3px;
  border-radius: 2px;
  transition: all 0.2s;
}

.nucleotide.changed {
  background-color: #ff6b6b;
  color: white;
  font-weight: bold;
}

.nucleotide.unchanged {
  background-color: #f8f9fa;
  color: #666;
}

.nucleotide:hover {
  transform: scale(1.1);
  z-index: 1;
  position: relative;
}

/* 加载指示器 */
.loading-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
  font-size: 12px;
}

.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid #e0e0e0;
  border-radius: 50%;
  border-top: 2px solid #4f46e5;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

.loading-text {
  color: #666;
}

/* 错误指示器 */
.error-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
  font-size: 12px;
}

.error-text {
  color: #dc3545;
}

.btn-retry {
  background: rgba(220, 53, 69, 0.1);
  color: #dc3545;
  border: 1px solid rgba(220, 53, 69, 0.3);
  border-radius: 4px;
  padding: 4px 8px;
  font-size: 11px;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-retry:hover {
  background: rgba(220, 53, 69, 0.2);
}

/* 批量处理状态 */
.batch-status {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 16px;
  padding: 12px 16px;
  background: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e0e0e0;
}

.batch-spinner {
  width: 20px;
  height: 20px;
  border: 2px solid #e0e0e0;
  border-radius: 50%;
  border-top: 2px solid #4f46e5;
  animation: spin 1s linear infinite;
}

.batch-text {
  color: #333;
  font-size: 14px;
  font-weight: 500;
}

/* 按钮 */
.btn-row {
  margin-top: 16px;
}
.btn {
  cursor: pointer;
  border: none;
  border-radius: 6px;
  transition: all 0.2s;
  font-weight: 500;
}
.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.btn-outline {
  background: transparent;
  color: #4f46e5;
  border: 1px solid #4f46e5;
  padding: 8px 16px;
}
.btn-outline:hover:not(:disabled) {
  background: #4f46e5;
  color: white;
}
.btn-plain {
  background: #4f46e5;
  color: #fff;
  padding: 6px 12px;
}
.btn-plain:hover:not(:disabled) {
  background: #4338ca;
}
.btn-xs {
  font-size: 12px;
}

/* 排序图标 */
.sortable {
  cursor: pointer;
  user-select: none;
}
.sort-ico {
  display: inline-block;
  margin-left: 6px;
  width: 0;
  height: 0;
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
}
.sort-ico.none {
  border-top: 0;
  border-bottom: 0;
}
.sort-ico.asc {
  border-bottom: 7px solid #4f46e5;
}
.sort-ico.desc {
  border-top: 7px solid #4f46e5;
}
</style>
