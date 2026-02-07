<!-- src/views/sprinzl/SprinzlNormal.vue -->
<template>
  <div class="home-container">
    <div class="card">
      <!-- ------------ 标题 ------------ -->
      <h1 class="title">LLM-tRNA Sequence Alignment Tool</h1>

      <!-- ------------ 输入区 ------------ -->
      <SeqInputList v-model="seqList" :default-seq="DEFAULT_SEQ" />

      <!-- ------------ 全局选项 ------------ -->
      <div class="options">
        <div class="opt-row">
          <span class="switch-label" :class="{ active: !use_llm }"
            >Only Use DP</span
          >
          <el-switch v-model="use_llm" />
          <span class="switch-label" :class="{ active: use_llm }">Use LLM</span>
        </div>
      </div>

      <!-- ------------ 连续进度条 ------------ -->
      <div class="progress-wrapper" v-show="running || progress > 0">
        <div class="progress-bar">
          <div class="progress-inner" :style="{ width: progress + '%' }"></div>
        </div>
        <span class="progress-text">{{ progress }}%</span>
      </div>

      <!-- ------------ 运行按钮 ------------ -->
      <div class="actions">
        <button class="btn-run" :disabled="running" @click="runAlign">
          {{ running ? "Aligning…" : "Run Alignment" }}
        </button>
      </div>

      <!-- ------------ 结果实时输出 ------------ -->
      <ResultsSection :results="results" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onBeforeUnmount, onMounted, nextTick } from "vue";
import SeqInputList from "./SeqInputList.vue";
import ResultsSection from "./ResultsPanel.vue";

/* ---------- 数据类型 ---------- */
interface SeqEntry {
  seq: string;
  anticode: string;
}
interface AlignmentResult {
  template_name?: string;
  nums?: string[];
  seq?: string[];
  refNums?: string[];
  ref?: string[];
  error?: string;
}

/* ---------- 基础状态 ---------- */
const DEFAULT_SEQ =
  "GGUCUCGUGGCCCAAUGGUUAAGGCGCUUGACUACGGAUCAAGAGAUUCCAGGUUCGACUCCUGGCGGGAUCG";
const seqList = ref<SeqEntry[]>([{ seq: DEFAULT_SEQ, anticode: "ACG" }]);
const use_llm = ref(true);
const running = ref(false);
const results = ref<AlignmentResult[]>([]);

/* ---------- 进度条逻辑 ---------- */
const progress = ref(0); // UI 显示的 %
let fakeTimer: any = null; // setInterval 句柄
let doneCount = 0; // 已完成条数
let totalCount = 0; // 总条数

/* ---------- 辅助 ---------- */
function pushResult(res: AlignmentResult) {
  results.value.push(res);
  doneCount += 1;

  /* 真实百分比 —— 立刻跳 */
  const realPercent = Math.floor((doneCount / totalCount) * 100);
  progress.value = Math.max(progress.value, realPercent);
}

/* ---------- 规范化 & 读取 URL ---------- */
function normalizeSeq(raw: string): string {
  return (raw || "")
    .toUpperCase()
    .replace(/%20/gi, " ") // 保险处理
    .replace(/\s+/g, "")
    .replace(/T/g, "U")
    .replace(/[^AUGC]/g, ""); // 移除非 A/U/G/C
}

function getSequenceFromURL(): string | null {
  try {
    const params = new URLSearchParams(window.location.search);
    const q = params.get("sequence");
    if (!q) return null;
    // decodeURIComponent 安全解码
    const decoded = decodeURIComponent(q);
    const norm = normalizeSeq(decoded);
    return norm.length ? norm : null;
  } catch {
    return null;
  }
}

async function setFromURLIfAny(autoRun: boolean = true) {
  const urlSeq = getSequenceFromURL();
  if (!urlSeq) return;
  // 设置到输入列表（默认 anticode 保持 ACG，如需从 URL 读取可自行扩展 ?anticode=XXX）
  seqList.value = [{ seq: urlSeq, anticode: "ACG" }];
  // 下一拍触发子组件 v-model 更新后再运行
  if (autoRun) {
    await nextTick();
    if (!running.value) {
      runAlign();
    }
  }
}

/* ---------- 主流程（顺序执行） ---------- */
async function runAlign() {
  // 初始化
  results.value = [];
  running.value = true;
  progress.value = 0;
  doneCount = 0;
  totalCount = seqList.value.length;

  // 启动连续伪增长定时器
  clearInterval(fakeTimer);
  fakeTimer = setInterval(() => {
    if (running.value) {
      /* 允许上涨到 "下一格 - 1%" */
      const nextLimit = Math.min(
        99,
        Math.floor(((doneCount + 1) / totalCount) * 100) - 1
      );
      if (progress.value < nextLimit) progress.value += 1;
    }
  }, 300);

  /* 顺序对齐 —— 一个完成再下一个 */
  for (let i = 0; i < seqList.value.length; i++) {
    const item = seqList.value[i];
    try {
      /* 1️⃣ 对齐 */
      const res1 = await fetch("/align/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_seq: item.seq,
          anticode: item.anticode,
          use_llm: use_llm.value,
        }),
      });
      if (!res1.ok) throw new Error(await res1.text());
      const j1 = await res1.json();
      const [h, b] = j1.csv_content.trim().split(/\r?\n/);
      const aline: AlignmentResult = {
        template_name: j1.template_name,
        nums: h.split(","),
        seq: b.split(","),
      };

      /* 2️⃣ 模板（可忽略失败） */
      try {
        const res2 = await fetch(
          `/align/export/?template_name=${encodeURIComponent(
            aline.template_name!
          )}`
        );
        if (res2.ok) {
          const j2 = await res2.json();
          const [hh, bb] = j2.csv_content.trim().split(/\r?\n/);
          aline.refNums = hh.split(",");
          aline.ref = bb.split(",");
        }
      } catch {
        /* ignore */
      }

      pushResult(aline);
    } catch (err: any) {
      pushResult({ error: err.message || `Unknown error (#${i + 1})` });
    }
  }

  /* 全部完成 —— 填满条、1s 后归零 */
  progress.value = 100;
  running.value = false;
  clearInterval(fakeTimer);
  setTimeout(() => (progress.value = 0), 1000);
}

/* ---------- 挂载时：读取 URL 并自动运行 ---------- */
onMounted(() => {
  setFromURLIfAny(true);

  // 监听浏览器前进/后退导致的 query 变化，自动加载并可选自动执行
  window.addEventListener("popstate", () => setFromURLIfAny(true));
});

/* 组件销毁时停表 & 清理监听 */
onBeforeUnmount(() => {
  clearInterval(fakeTimer);
  window.removeEventListener("popstate", () => setFromURLIfAny(true));
});
</script>

<style scoped>
/* ------------ 布局 ------------ */
.home-container {
  display: flex;
  justify-content: center;
  padding: 2rem;
}
.card {
  width: 100%;
  max-width: 900px;
  background: rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(6px);
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.25);
}

/* ------------ 标题 ------------ */
.title {
  font-size: 2rem;
  color: #fff;
  margin-bottom: 1.5rem;
  text-align: center;
}

/* ------------ 选项 ------------ */
.options {
  margin-top: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
.opt-row {
  display: flex;
  align-items: center;
}
.switch-label {
  margin: 0 0.5rem;
  color: rgba(255, 255, 255, 0.6);
}
.switch-label.active {
  color: #17ff00;
}
.opt-number-label {
  color: #eee;
  margin: 0 0.5rem;
}

/* ------------ 进度条 ------------ */
.progress-wrapper {
  margin: 1.5rem 0;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.progress-bar {
  flex: 1;
  height: 8px;
  background: rgba(255, 255, 255, 0.25);
  border-radius: 4px;
  overflow: hidden;
}
.progress-inner {
  height: 100%;
  background: linear-gradient(90deg, #34d399, #3b82f6);
  transition: width 0.18s linear;
}
.progress-text {
  width: 48px;
  font-size: 0.8rem;
  color: #fff;
  text-align: right;
}

/* ------------ 按钮 ------------ */
.actions {
  text-align: center;
  margin-top: 1.5rem;
}
.btn-run {
  background: #3b82f6;
  color: #fff;
  border: none;
  border-radius: 6px;
  padding: 0.8rem 2.2rem;
  cursor: pointer;
}

/* ------------ element-plus 适配 ------------ */
:deep(.el-input-number .el-input__inner) {
  border-color: rgba(255, 255, 255, 0.5) !important;
  color: #000 !important;
}
:deep(.el-checkbox__label),
:deep(.el-checkbox .el-checkbox__inner) {
  color: #fff !important;
  border-color: rgba(255, 255, 255, 0.6) !important;
}
:deep(.el-checkbox.is-checked .el-checkbox__inner) {
  border-color: #42b983 !important;
  background: #42b983 !important;
}
</style>
