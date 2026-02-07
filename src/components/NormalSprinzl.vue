<!-- src/views/generate/TRNAOneSequenceViewer.vue -->
<template>
  <div class="trna-container" style="position: relative">
    <!-- Main SVG chart -->
    <svg :width="width" :height="height">
      <g
        font-size="12"
        text-anchor="middle"
        stroke="#333"
        stroke-width="1"
        :transform="groupMatrix"
      >
        <g
          v-for="node in nodes"
          :key="node.id"
          class="base-group"
          :class="{
            interactive: !!interactiveMode && !node.isGap,
            frozen: isFrozen(node.id),
            preferred: isPreferred(node.id),
            forced: isForced(node.id),
          }"
          @mouseenter="hoverNode = node"
          @mouseleave="hoverNode = null"
          @click="handleNodeClick(node)"
        >
          <circle
            :cx="node.x"
            :cy="node.y"
            :r="r"
            :class="{
              'base-hovered': hoverNode && hoverNode.id === node.id,
              'gap-node': node.isGap,
              'nucleotide-node': !node.isGap,
            }"
            fill="#fff"
          />
          <text :x="node.x" :y="node.y + 4">
            {{ node.base }}
          </text>
        </g>
      </g>
    </svg>

    <!-- Tooltip -->
    <div v-if="hoverNode" class="tooltip" :style="tooltipStyle">
      <div><strong>Template:</strong> {{ payload?.template_name || "—" }}</div>
      <div><strong>Sprinzl Number:</strong> {{ hoverNode.id }}</div>
      <div>
        <strong>Sequence Index:</strong>
        {{ formatSequenceIndex(hoverNode.id) }}
      </div>
      <div>
        <strong>Region:</strong>
        {{ hoverNode.region || "—" }}
      </div>
      <div><strong>Base:</strong> {{ hoverNode.base }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from "vue";
import type { PropType } from "vue";

interface ParsedCsv {
  baseById: Record<string, string>;
  indexById: Record<string, number>;
}
type RegionName =
  | "AA_Arm_5prime"
  | "D_Loop"
  | "Anticodon_Arm"
  | "Variable_Loop"
  | "T_Arm"
  | "AA_Arm_3prime";

interface SprinzlNode {
  id: string;
  base: string;
  originalBase: string;
  sequenceIndex: string | null;
  region: RegionName | null;
  isGap: boolean;
  x: number;
  y: number;
}

/**
 * 传入单条序列：
 * {
 *   template_name: string,
 *   csv_content: string // 两行：第一行header(位点)，第二行bases
 * }
 */
const props = defineProps({
  payload: { type: Object, required: true },
  width: { type: Number, default: 300 },
  height: { type: Number, default: 300 },
  r: { type: Number, default: 12 },
  yOffset: { type: Number, default: 40 }, // 兼容保留
  freezePositions: {
    type: Array as PropType<string[]>,
    default: () => [],
  },
  preferPositions: {
    type: Array as PropType<string[]>,
    default: () => [],
  },
  forcePositions: {
    type: Array as PropType<string[]>,
    default: () => [],
  },
  forceBaseMap: {
    type: Object as PropType<Record<string, string>>,
    default: () => ({}),
  },
  interactiveMode: {
    type: String as PropType<"freeze" | "prefer" | "force" | null>,
    default: null,
  },
});

const emit = defineEmits<{
  (e: "node-click", payload: { sprinzlId: string; sequenceIndex: string; base: string; originalBase: string }): void;
  (e: "mapping", payload: Array<{ sprinzlId: string; sequenceIndex: string | null; region: RegionName | null; base: string; originalBase: string }>): void;
}>();

// ----------------- 固定的 canonical 坐标（Sprinzl 风格） -----------------
const positions = {
  "-1": { x: 220, y: 20 },
  1: { x: 245, y: 20 },
  2: { x: 245, y: 45 },
  3: { x: 245, y: 70 },
  4: { x: 245, y: 95 },
  5: { x: 245, y: 120 },
  6: { x: 245, y: 145 },
  7: { x: 245, y: 170 },
  8: { x: 225, y: 185 },
  9: { x: 205, y: 200 },
  10: { x: 185, y: 215 },
  11: { x: 160, y: 215 },
  12: { x: 135, y: 215 },
  13: { x: 110, y: 215 },
  14: { x: 92, y: 198 },
  15: { x: 76, y: 179 },
  16: { x: 51, y: 179 },
  17: { x: 33, y: 196 },
  "17a": { x: 18, y: 215 },
  18: { x: 18, y: 240 },
  19: { x: 18, y: 265 },
  20: { x: 33, y: 285 },
  "20a": { x: 51, y: 302 },
  "20b": { x: 76, y: 302 },
  21: { x: 94, y: 285 },
  22: { x: 110, y: 265 },
  23: { x: 135, y: 265 },
  24: { x: 160, y: 265 },
  25: { x: 185, y: 265 },
  26: { x: 205, y: 280 },
  27: { x: 205, y: 305 },
  28: { x: 205, y: 330 },
  29: { x: 205, y: 355 },
  30: { x: 205, y: 380 },
  31: { x: 205, y: 405 },
  32: { x: 195, y: 428 },
  33: { x: 195, y: 453 },
  34: { x: 210, y: 473 },
  35: { x: 232, y: 485 },
  36: { x: 254, y: 473 },
  37: { x: 268, y: 453 },
  38: { x: 268, y: 428 },
  39: { x: 258, y: 405 },
  40: { x: 258, y: 380 },
  41: { x: 258, y: 355 },
  42: { x: 258, y: 330 },
  43: { x: 258, y: 305 },
  44: { x: 258, y: 280 },
  45: { x: 283, y: 280 },
  V11: { x: 290, y: 304 },
  V12: { x: 310, y: 319 },
  V13: { x: 330, y: 334 },
  V14: { x: 350, y: 349 },
  V15: { x: 370, y: 364 },
  V16: { x: 390, y: 379 },
  V17: { x: 410, y: 394 },
  V1: { x: 430, y: 409 },
  V2: { x: 450, y: 424 },
  V3: { x: 472, y: 437 },
  V4: { x: 490, y: 420 },
  V5: { x: 500, y: 397 },
  V27: { x: 480, y: 382 },
  V26: { x: 460, y: 367 },
  V25: { x: 440, y: 352 },
  V24: { x: 420, y: 337 },
  V23: { x: 400, y: 322 },
  V22: { x: 380, y: 307 },
  V21: { x: 360, y: 292 },
  46: { x: 340, y: 277 },
  47: { x: 320, y: 262 },
  48: { x: 300, y: 247 },
  49: { x: 320, y: 232 },
  50: { x: 345, y: 232 },
  51: { x: 370, y: 232 },
  52: { x: 395, y: 232 },
  53: { x: 420, y: 232 },
  54: { x: 440, y: 246 },
  55: { x: 464, y: 249 },
  56: { x: 480, y: 230 },
  57: { x: 490, y: 207 },
  58: { x: 480, y: 184 },
  59: { x: 464, y: 165 },
  60: { x: 440, y: 168 },
  61: { x: 420, y: 182 },
  62: { x: 395, y: 182 },
  63: { x: 370, y: 182 },
  64: { x: 345, y: 182 },
  65: { x: 320, y: 182 },
  66: { x: 298, y: 170 },
  67: { x: 298, y: 145 },
  68: { x: 298, y: 120 },
  69: { x: 298, y: 95 },
  70: { x: 298, y: 70 },
  71: { x: 298, y: 45 },
  72: { x: 298, y: 20 },
  73: { x: 323, y: 20 },
  74: { x: 348, y: 20 },
  75: { x: 373, y: 20 },
  76: { x: 398, y: 20 },
};

const regionLookup = ref<Record<string, RegionName | null>>({});

async function fetchRegionMapping() {
  const numbers = Object.keys(positions);
  try {
    const resp = await fetch("/regions/lookup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ numbers }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    const map: Record<string, RegionName | null> = {};
    data.results.forEach((item: any) => {
      map[item.input] = normalizeRegionName(item.region);
    });
    regionLookup.value = map;
  } catch (err) {
    console.warn("Failed to fetch region mapping, fallback to local rules:", err);
    regionLookup.value = {};
  }
}

onMounted(fetchRegionMapping);

function determineRegion(identifier: string): RegionName | null {
  const mapped = regionLookup.value[identifier];
  if (mapped !== undefined) return mapped;

  let clean = identifier.toLowerCase();
  clean = clean.replace(/i\d+$/, "");

  if (/^-?\d+$/.test(clean)) {
    const num = parseInt(clean, 10);
    if (-1 <= num && num <= 9) return "AA_Arm_5prime";
    if (10 <= num && num <= 26) return "D_Loop";
    if (27 <= num && num <= 43) return "Anticodon_Arm";
    if (44 <= num && num <= 48) return "Variable_Loop";
    if (49 <= num && num <= 65) return "T_Arm";
    if (66 <= num && num <= 76) return "AA_Arm_3prime";
    return null;
  }

  if (["17a", "20a", "20b"].includes(clean)) {
    return "D_Loop";
  }

  if (clean.startsWith("v")) {
    const vnum = clean.slice(1);
    if (/^\d+$/.test(vnum)) {
      const n = parseInt(vnum, 10);
      if (
        (1 <= n && n <= 5) ||
        (11 <= n && n <= 17) ||
        (21 <= n && n <= 27)
      ) {
        return "Variable_Loop";
      }
    }
  }

  return null;
}

interface ParsedCsv {
  baseById: Record<string, string>;
  indexById: Record<string, number>;
}

interface SprinzlNode {
  id: string;
  base: string;
  originalBase: string;
  sequenceIndex: string | null;
  isGap: boolean;
  x: number;
  y: number;
}

// ----------------- 解析 CSV：header -> base + 原序号 -----------------
const parsedCsv = computed<ParsedCsv>(() => {
  const text = props.payload?.csv_content || "";
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return { baseById: {}, indexById: {} };

  const headers = lines[0].split(",").map((s: string) => s.trim());
  const row = lines[1].split(",").map((s: string) => s.trim());
  const baseById: Record<string, string> = {};
  const indexById: Record<string, number> = {};
  let cursor = 0;

  headers.forEach((h: string, i: number) => {
    const raw = (row[i] || "").toUpperCase();
    const normalized = raw === "" ? "-" : raw;
    baseById[h] = normalized;
    if (normalized && normalized !== "-") {
      cursor += 1;
      indexById[h] = cursor;
    }
  });

  return { baseById, indexById };
});

const id2base = computed<Record<string, string>>(
  () => parsedCsv.value.baseById
);
const sequenceIndexMap = computed<Record<string, number>>(
  () => parsedCsv.value.indexById
);

// ----------------- 自适应缩放与平移 -----------------
const fit = computed(() => {
  if (!Object.keys(positions).length) return { s: 1, tx: 0, ty: 0 };
  const pad = 24;
  const r = props.r || 12;

  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (const pos of Object.values(positions)) {
    minX = Math.min(minX, pos.x - r);
    minY = Math.min(minY, pos.y - r);
    maxX = Math.max(maxX, pos.x + r);
    maxY = Math.max(maxY, pos.y + r);
  }
  const rawW = Math.max(1, maxX - minX);
  const rawH = Math.max(1, maxY - minY);
  const s = Math.min(
    1,
    (props.width - 2 * pad) / rawW,
    (props.height - 2 * pad) / rawH
  );
  const contentW = rawW * s;
  const contentH = rawH * s;
  const tx = (props.width - contentW) / 2 - s * minX;
  const ty = (props.height - contentH) / 2 - s * minY;
  return { s, tx, ty };
});

const groupMatrix = computed(() => {
  const { s, tx, ty } = fit.value;
  return `matrix(${s},0,0,${s},${tx},${ty})`;
});

// 转屏幕坐标（tooltip 用）
function toScreen(n: { x: number; y: number }) {
  const { s, tx, ty } = fit.value;
  return { x: n.x * s + tx, y: n.y * s + ty };
}

// ----------------- 生成节点：仅单序列 -----------------
const nodes = computed<SprinzlNode[]>(() => {
  const list: SprinzlNode[] = [];
  const map = id2base.value;
  for (const [id, pos] of Object.entries(positions)) {
    const base = (map[id] ?? "-").toUpperCase();
    const isGap = base === "-" || base === "";
    const sequenceIndex = getSequenceIndex(id);
    const forcedBase =
      sequenceIndex && props.forceBaseMap[sequenceIndex]
        ? props.forceBaseMap[sequenceIndex].toUpperCase()
        : null;
    const region = determineRegion(id);
    const displayBase = forcedBase || base;
    list.push({
      id,
      base: isGap ? "-" : displayBase,
      originalBase: isGap ? "-" : base,
      sequenceIndex,
      region,
      isGap,
      x: pos.x,
      y: pos.y,
    });
  }
  return list;
});

watch(
  () => nodes.value,
  (list) => {
    const payload = list.map((node) => ({
      sprinzlId: node.id,
      sequenceIndex: node.sequenceIndex,
      region: node.region,
      base: node.base,
      originalBase: node.originalBase,
    }));
    emit("mapping", payload);
  },
  { immediate: true }
);

const hoverNode = ref<SprinzlNode | null>(null);
const freezeIndexSet = computed(
  () =>
    new Set(
      (props.freezePositions ?? [])
        .map((pos) => pos.trim())
        .filter((pos) => pos.length > 0)
    )
);
const preferIndexSet = computed(
  () =>
    new Set(
      (props.preferPositions ?? [])
        .map((pos) => pos.trim())
        .filter((pos) => pos.length > 0)
    )
);
const forceIndexSet = computed(() => {
  const set = new Set<string>();
  (props.forcePositions ?? []).forEach((entry) => {
    const index = extractForceIndex(entry);
    if (index) set.add(index);
  });
  return set;
});

function handleNodeClick(node: SprinzlNode) {
  if (!props.interactiveMode || node.isGap) return;
  const seqIndex = getSequenceIndex(node.id);
  if (!seqIndex) return;
  emit("node-click", {
    sprinzlId: node.id,
    sequenceIndex: seqIndex,
    base: node.base,
    originalBase: node.originalBase,
  });
}

function formatSequenceIndex(id: string) {
  const idx = sequenceIndexMap.value[id];
  return typeof idx === "number" ? idx : "—";
}

function getSequenceIndex(id: string): string | null {
  const idx = sequenceIndexMap.value[id];
  return typeof idx === "number" ? idx.toString() : null;
}

function isFrozen(id: string) {
  const seq = getSequenceIndex(id);
  return !!seq && freezeIndexSet.value.has(seq);
}
function isPreferred(id: string) {
  const seq = getSequenceIndex(id);
  return !!seq && preferIndexSet.value.has(seq);
}
function isForced(id: string) {
  const seq = getSequenceIndex(id);
  return !!seq && forceIndexSet.value.has(seq);
}

function extractForceIndex(entry: string): string | null {
  const trimmed = entry.trim();
  if (!trimmed) return null;
  if (trimmed.startsWith("{")) {
    try {
      const obj = JSON.parse(trimmed);
      const key = Object.keys(obj)[0];
      return key?.trim() || null;
    } catch {
      return null;
    }
  }
  const idx = trimmed.split(":")[0]?.trim();
  return idx || null;
}

function normalizeRegionName(name: string | null): RegionName | null {
  if (!name) return null;
  const normalized = name.trim().toLowerCase();
  switch (normalized) {
    case "aminoacyl arm 5' end":
      return "AA_Arm_5prime";
    case "d loop + d stem":
      return "D_Loop";
    case "anticodon loop + anticodon stem":
      return "Anticodon_Arm";
    case "variable loop":
      return "Variable_Loop";
    case "t loop + t stem":
      return "T_Arm";
    case "aminoacyl arm 3' end":
      return "AA_Arm_3prime";
    default:
      return null;
  }
}

const tooltipStyle = computed(() => {
  if (!hoverNode.value) return {};
  const p = toScreen(hoverNode.value);
  const tooltipWidth = 200;
  const tooltipHeight = 90;
  const offset = 12;
  let left = p.x + offset;
  let top = p.y - (tooltipHeight + offset);

  if (left + tooltipWidth > props.width) {
    left = p.x - tooltipWidth - offset;
  }
  if (left < 0) left = 0;

  if (top < 0) {
    top = p.y + offset;
  }
  const maxTop = props.height - tooltipHeight - offset;
  if (top > maxTop) {
    top = Math.max(offset, maxTop);
  }

  return {
    top: `${top}px`,
    left: `${left}px`,
  };
});

const payload = computed(() => props.payload);
</script>

<style scoped>
.trna-container {
  background: rgba(255, 255, 255, 0.12);
  border-radius: 8px;
  padding: 4px;
}

/* SVG 本身保持居中 */
svg {
  display: block;
  margin: 0 auto;
  background: transparent;
}

.base-group {
  cursor: default;
}
.base-hovered {
  stroke: #ff9800;
  stroke-width: 2;
  fill: #fff5e6;
}
.base-group.interactive {
  cursor: pointer;
}
.base-group.frozen .nucleotide-node {
  fill: rgba(241, 245, 249, 0.85);
  stroke: #94a3b8;
}
.base-group.preferred .nucleotide-node {
  fill: rgba(250, 204, 21, 0.35);
  stroke: #f59e0b;
}
.base-group.forced .nucleotide-node {
  fill: rgba(147, 197, 253, 0.45);
  stroke: #2563eb;
}

/* 仅两种状态：gap / nucleotide */
.gap-node {
  /* '-' */
  fill: #b0bec5; /* 蓝灰 */
  stroke: #607d8b;
}
.nucleotide-node {
  /* A/U/G/C */
  fill: #c8e6c9; /* 绿色 */
  stroke: #4caf50;
}

.tooltip {
  position: absolute;
  background: rgba(0, 0, 0, 0.7);
  color: #fff;
  padding: 6px 10px;
  pointer-events: none;
  font-size: 14px;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  white-space: nowrap;
  z-index: 10;
  min-width: 180px;
  line-height: 1.4;
}
</style>
