<!-- src/components/ResultsPanel.vue -->
<template>
  <div>
    <!-- ==== 汇总下载按钮（仅有 ≥2 条成功结果时显示） ==== -->
    <div v-if="successResults.length > 1" class="merge-row">
      <button class="btn-download-big" @click="downloadMerged">
        ⬇︎ Download All ({{ successResults.length }} seqs)
      </button>
    </div>

    <!-- ==== 合并可视化图像 ==== -->
    <div v-if="imageUrl" class="clustermap-preview">
      <!-- 缩略图，点击打开 Lightbox -->
      <img
        :src="imageUrl"
        alt="Clustermap"
        class="preview-thumb"
        @click="lightboxVisible = true"
      />
      <!-- Lightbox 组件 -->
      <teleport to="body">
        <vue-easy-lightbox
          :visible="lightboxVisible"
          :imgs="[imageUrl]"
          :index="0"
          @hide="lightboxVisible = false"
        />
      </teleport>
    </div>

    <!-- ==== 单条结果循环 ==== -->
    <div v-for="(res, idx) in results" :key="idx" class="result-block">
      <!-- ✨ 错误条目 -->
      <template v-if="res.error">
        <div class="error-card">
          <strong>Seq {{ idx + 1 }}</strong> — {{ res.error }}
        </div>
      </template>

      <!-- ✨ 正常条目 -->
      <template v-else>
        <!-- 标题 + 下载按钮 -->
        <div class="title-row">
          <h3 class="res-title">
            Seq {{ idx + 1 }} → Template: {{ res.template_name }}
          </h3>
          <button class="btn-download" @click="downloadResult(res, idx)">
            ⬇︎ Download
          </button>
        </div>

        <!-- 三行表格 -->
        <AlignmentTable
          :alignment-data="buildNodes(res)"
          class="alignment-tables"
        />

        <!-- 径向示意图 -->
        <sup-trna-radial class="radial" :data="buildNodes(res)" :height="520" />
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
// 1️⃣ 一定要先 import 样式，否则看不到任何 overlay
// import "vue-easy-lightbox/lib/style.css";

import { computed, ref, watch } from "vue";
import AlignmentTable from "@/components/AlignmentTable.vue";
import SupTrnaRadial from "@/components/supTrnaRadial.vue";
import VueEasyLightbox from "vue-easy-lightbox"; // 组件本身

interface AlignmentResult {
  template_name?: string;
  nums?: string[];
  seq?: string[];
  refNums?: string[];
  ref?: string[];
  error?: string;
}

const props = defineProps<{ results: AlignmentResult[] }>();

/* ---------- 成功条目 ---------- */
const successResults = computed(() =>
  props.results.filter(
    (r) => !r.error && r.nums && r.seq && r.nums.length === r.seq.length
  )
);

/* ---------- 单条节点组装 ---------- */
function buildNodes(res: AlignmentResult) {
  if (!res.nums || !res.seq) return [];
  const supMap = new Map(res.nums.map((id, i) => [id, res.seq![i] || "-"]));
  const baseMap = res.refNums
    ? new Map(res.refNums.map((id, i) => [id, res.ref![i] || "-"]))
    : new Map<string, string>();
  return res.nums.map((id) => {
    const sup = supMap.get(id)!;
    const base = baseMap.get(id) ?? "-";
    let type: "match" | "mismatch" | "insertion" | "deletion" = "match";
    if (base === "-" && sup !== "-") type = "insertion";
    else if (base !== "-" && sup === "-") type = "deletion";
    else if (base !== sup) type = "mismatch";
    return { id, base, sup_base: sup, type };
  });
}

/* ---------- 单条下载 ---------- */
function downloadResult(res: AlignmentResult, idx: number) {
  if (!res.nums || !res.seq) return;
  const nodes = buildNodes(res);
  const idRow = ["ID", ...nodes.map((n) => n.id)].join(",");
  const baseRow = ["Origin", ...nodes.map((n) => n.base)].join(",");
  const supRow = ["Target", ...nodes.map((n) => n.sup_base)].join(",");
  triggerCsvDownload(
    [idRow, baseRow, supRow].join("\n"),
    `seq${idx + 1}_${res.template_name || "alignment"}.csv`
  );
}

/* ---------- 合并下载 & 可视化 ---------- */
const imageUrl = ref<string | null>(null);
const lightboxVisible = ref(false);

function makeMergedCsv(): string {
  const succ = successResults.value;
  if (succ.length < 1) return "";
  const firstNums = succ[0].nums!;
  const canonical = firstNums.filter((id) => !/i\d+$/.test(id));
  const insertionsByParent: Record<string, string[]> = {};
  succ.forEach((r) => {
    r.nums!.filter((id) => /i\d+$/.test(id)).forEach((id) => {
      const parent = id.split("i")[0];
      insertionsByParent[parent] ??= [];
      if (!insertionsByParent[parent].includes(id))
        insertionsByParent[parent].push(id);
    });
  });
  Object.values(insertionsByParent).forEach((arr) =>
    arr.sort(
      (a, b) => parseInt(a.split("i")[1], 10) - parseInt(b.split("i")[1], 10)
    )
  );
  const allIds: string[] = [];
  canonical.forEach((id) => {
    allIds.push(id);
    if (insertionsByParent[id]) {
      allIds.push(...insertionsByParent[id]);
      delete insertionsByParent[id];
    }
  });
  for (const [parent, insArr] of Object.entries(insertionsByParent)) {
    allIds.push(parent, ...insArr);
  }
  const rows: string[][] = [];
  rows.push(["ID", ...allIds]);
  succ.forEach((r, i) => {
    const map = new Map(r.nums!.map((id, j) => [id, r.seq![j] || "-"]));
    const row = ["Seq" + (i + 1)];
    allIds.forEach((id) => row.push(map.get(id) ?? "-"));
    rows.push(row);
  });
  return rows.map((r) => r.join(",")).join("\n");
}

function downloadMerged() {
  const csvText = makeMergedCsv();
  triggerCsvDownload(csvText, `merged_${successResults.value.length}seqs.csv`);
}

// 完成 ≥2 条后自动去画 clustermap
watch(
  () => successResults.value.length,
  (len) => {
    if (len > 1) {
      const fd = new FormData();
      const blob = new Blob([makeMergedCsv()], { type: "text/csv" });
      fd.append("file", blob, "merged.csv");
      fetch("/plot/clustermap", { method: "POST", body: fd })
        .then((res) => res.blob())
        .then((b) => {
          imageUrl.value = URL.createObjectURL(b);
        })
        .catch(console.error);
    }
  }
);

/* ---------- 浏览器下载助手 ---------- */
function triggerCsvDownload(csv: string, filename: string) {
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
</script>

<style scoped>
.merge-row {
  text-align: right;
  margin-bottom: 1rem;
}
.btn-download-big {
  background: #3b82f6;
  color: #fff;
  border: none;
  border-radius: 5px;
  padding: 6px 14px;
  font-size: 0.85rem;
  cursor: pointer;
}
.btn-download-big:hover {
  background: #2563eb;
}

/* clustermap 预览 */
.clustermap-preview img.preview-thumb {
  display: block;
  max-width: 100%;
  margin: 1rem auto;
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 4px;
  cursor: zoom-in;
}
/* 确保 Lightbox 在最顶层 */
:global(.vue-easy-lightbox__container) {
  z-index: 2000;
}

.result-block {
  margin-bottom: 1.5rem;
}
.title-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.res-title {
  font-size: 1.1rem;
  color: #d1fae5;
  margin: 0.5rem 0;
}
.btn-download {
  background: #34d399;
  color: #fff;
  border: none;
  border-radius: 4px;
  padding: 4px 12px;
  font-size: 0.8rem;
  cursor: pointer;
}
.btn-download:hover {
  background: #2ebd8a;
}
.radial {
  max-width: 620px;
  margin: 0 auto 1rem;
}
.error-card {
  padding: 0.75rem 1rem;
  border-left: 4px solid #ff6b6b;
  background: rgba(255, 107, 107, 0.12);
  color: #ffbdbd;
  border-radius: 4px;
  font-size: 0.9rem;
}

.alignment-tables {
  margin-bottom: 1rem;
}
</style>
