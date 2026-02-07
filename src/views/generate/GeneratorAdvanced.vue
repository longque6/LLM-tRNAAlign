<template>
  <section v-if="show" class="card">
    <h2 class="card-title">Advanced (overrides slider)</h2>

    <!-- ===================== 基础参数 ===================== -->
    <div class="form grid-3">
      <div>
        <label class="label">temperature</label>
        <input
          type="number"
          step="0.01"
          :value="localAdv.temperature"
          @input="handleAdvInput('temperature', $event)"
          class="input"
        />
      </div>
      <div>
        <label class="label">top_p</label>
        <input
          type="number"
          step="0.01"
          min="0"
          max="1"
          :value="localAdv.top_p"
          @input="handleAdvInput('top_p', $event)"
          class="input"
        />
      </div>
      <div>
        <label class="label">rounds</label>
        <input
          type="number"
          min="1"
          :value="localAdv.rounds"
          @input="handleAdvInput('rounds', $event)"
          class="input"
        />
      </div>
      <div>
        <label class="label">min_hd (batch)</label>
        <input
          type="number"
          min="0"
          :value="localAdv.min_hd"
          @input="handleAdvInput('min_hd', $event)"
          class="input"
        />
      </div>
      <div>
        <label class="label">rerank_min_hd (batch)</label>
        <input
          type="number"
          min="0"
          :value="localAdv.rerank_min_hd"
          @input="handleAdvInput('rerank_min_hd', $event)"
          class="input"
        />
      </div>
      <div>
        <label class="label">GC range (batch)</label>
        <div class="gc">
          <input
            type="number"
            step="0.01"
            min="0"
            max="1"
            :value="localAdv.gc_low"
            @input="handleAdvInput('gc_low', $event)"
            class="input"
          />
          <span>~</span>
          <input
            type="number"
            step="0.01"
            min="0"
            max="1"
            :value="localAdv.gc_high"
            @input="handleAdvInput('gc_high', $event)"
            class="input"
          />
        </div>
      </div>
    </div>

    <!-- 调试信息 -->
    <div class="debug-info" v-if="showDebug">
      <h4>Debug Info:</h4>
      <p>Style Level Applied: {{ lastAppliedStyleLevel }}</p>
      <p>Temperature: {{ props.adv.temperature }}</p>
      <p>Top_p: {{ props.adv.top_p }}</p>
      <p>Rounds: {{ props.adv.rounds }}</p>
      <p>Min_hd: {{ props.adv.min_hd }}</p>
      <p>Rerank_min_hd: {{ props.adv.rerank_min_hd }}</p>
    </div>

    <!-- ===================== tRNA 序列预览 ===================== -->
    <div class="form" v-if="showPreview">
      <label class="label">Sprinzl Reference Preview</label>

      <!-- 选择模式下的预览 -->
      <div v-if="isSelectMode && effectivePayload" class="viewer-panel">
        <div class="viewer-layer-toolbar">
          <span>Interactive layer:</span>
          <div class="toolbar-buttons">
            <button
              class="btn-xs"
              :class="{ active: viewerLayer === 'freeze' }"
              @click="toggleViewerLayer('freeze')"
              type="button"
            >
              Freeze
            </button>
            <button
              class="btn-xs"
              :class="{ active: viewerLayer === 'prefer' }"
              @click="toggleViewerLayer('prefer')"
              type="button"
            >
              Prefer
            </button>
            <button
              class="btn-xs"
              :class="{ active: viewerLayer === 'force' }"
              @click="toggleViewerLayer('force')"
              type="button"
            >
              Force
            </button>
            <button
              class="btn-xs"
              :disabled="!viewerLayer"
              @click="toggleViewerLayer(null)"
              type="button"
            >
              Off
            </button>
          </div>
          <div v-if="viewerLayer === 'force'" class="force-select-group">
            <label class="muted" for="force-base-select">Target base</label>
            <select
              id="force-base-select"
              class="force-select"
              v-model="forceBase"
            >
              <option v-for="base in FORCE_BASE_OPTIONS" :key="base" :value="base">
                {{ base }}
              </option>
            </select>
          </div>
          <small class="muted"
            >Click nucleotides to toggle the selected layer.</small
          >
        </div>
        <TRNAOneSequenceViewer
          :payload="effectivePayload"
          :width="viewerWidth"
          :height="viewerHeight"
          :r="viewerR"
          :freeze-positions="freezePositionsList"
          :prefer-positions="preferPositionsList"
          :force-positions="forcePositionsList"
          :force-base-map="forceBaseMap"
          :interactive-mode="viewerLayer"
          @node-click="handleViewerNodeSelect"
          @mapping="handleViewerMapping"
        />
        <small class="muted"
          >Showing: {{ effectivePayload.template_name || "—" }}</small
        >
      </div>

      <!-- 自定义模式下的占位提示 -->
      <div v-else-if="isCustomMode" class="preview-placeholder">
        <div class="placeholder-content">
          <span class="placeholder-icon">🔍</span>
          <p>No preview available for custom input sequences</p>
          <small class="muted">
            Custom sequences are not from the database, so structural preview is
            not available.
          </small>
        </div>
      </div>

      <!-- 加载状态 -->
      <div v-else-if="isLoadingPreview" class="preview-loading">
        <div class="loading-spinner"></div>
        <span>Loading preview...</span>
      </div>
    </div>

    <!-- ===================== 约束与偏好 ===================== -->
    <div class="constraints-panel">
      <div class="constraints-heading">
        <h3 class="card-subtitle">Constraints & Preferences</h3>
        <small class="muted">
          Refine edits after you inspect the Sprinzl map above.
        </small>
      </div>
      <div class="constraints-columns">
        <div class="constraints-block">
          <p class="label">Regions</p>
          <div class="region-grid">
            <div class="region-card">
              <div class="muted">Freeze regions (no change)</div>
              <div class="chips">
                <label v-for="r in REGION_OPTIONS" :key="'fr-' + r" class="chip">
                  <input
                    type="checkbox"
                    :value="r"
                    :checked="freezeRegionState(r) === 'full'"
                    :indeterminate="freezeRegionState(r) === 'partial'"
                    @change="handleFreezeRegionChange(r, $event)"
                  />
                  <span>{{ r }}</span>
                </label>
              </div>
            </div>
            <div class="region-card">
              <div class="muted">Prefer regions (edit first)</div>
              <div class="chips">
                <label v-for="r in REGION_OPTIONS" :key="'pr-' + r" class="chip">
                  <input
                    type="checkbox"
                    :value="r"
                    :checked="preferRegionState(r) === 'full'"
                    :indeterminate="preferRegionState(r) === 'partial'"
                    @change="handlePreferRegionChange(r, $event)"
                  />
                  <span>{{ r }}</span>
                </label>
              </div>
            </div>
          </div>
          <small class="muted available-note"
            >Available: {{ REGION_OPTIONS.join(", ") }}</small
          >
        </div>
        <div class="constraints-block">
          <p class="label">Position lists</p>
          <div class="positions-grid">
            <div>
              <label class="label">Freeze positions</label>
              <input
                :value="localFreezePositionsText"
                @input="handleFreezePositionsInput($event)"
                class="input"
                placeholder="e.g. 1, 2, 3, 73"
              />
            </div>
            <div>
              <label class="label">Prefer positions</label>
              <input
                :value="localPreferPositionsText"
                @input="handlePreferPositionsInput($event)"
                class="input"
                placeholder="e.g. 34, 35, 36"
              />
            </div>
            <div>
              <label class="label">Force positions</label>
              <input
                :value="localForcePositionsText"
                @input="handleForcePositionsInput($event)"
                class="input"
                placeholder='37:A or {"37":"A"}'
              />
            </div>
          </div>
          <small class="muted positions-note">
            Positions use sequence indexes; freeze / prefer / force lists are
            mutually exclusive.
          </small>
        </div>
      </div>
      <div v-if="localUiNote" class="note">{{ localUiNote }}</div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref, watch, computed, onMounted } from "vue";
import TRNAOneSequenceViewer from "@/components/NormalSprinzl.vue";

interface AdvParams {
  temperature: number;
  top_p: number;
  rounds: number;
  min_hd: number;
  rerank_min_hd: number;
  gc_low: number;
  gc_high: number;
}
interface ViewerPayload {
  template_name: string;
  csv_content: string;
}

interface ViewerNodeMapping {
  sprinzlId: string;
  sequenceIndex: string | null;
  region: string | null;
  base: string;
  originalBase: string;
}

const props = defineProps<{
  show: boolean;
  adv: AdvParams;
  REGION_OPTIONS: string[];
  freezeRegions: string[];
  preferRegions: string[];
  freezePositionsText: string;
  preferPositionsText: string;
  forcePositionsText: string;
  uiNote: string;
  isSelectMode: boolean;
  isCustomMode: boolean;
  selectedSequenceName?: string;
  selectedSpecies?: string;
  viewerWidth?: number;
  viewerHeight?: number;
  viewerR?: number;
  showDebug?: boolean;
}>();

const emit = defineEmits([
  "update:adv",
  "update:freezeRegions",
  "update:preferRegions",
  "update:freezePositionsText",
  "update:preferPositionsText",
  "update:forcePositionsText",
  "update:uiNote",
]);

// 本地副本 - 使用浅拷贝避免深度响应式
const localAdv = ref({ ...props.adv });
const localFreezeRegions = ref([...props.freezeRegions]);
const localPreferRegions = ref([...props.preferRegions]);
const localFreezePositionsText = ref(props.freezePositionsText);
const localPreferPositionsText = ref(props.preferPositionsText);
const localForcePositionsText = ref(props.forcePositionsText);
const localUiNote = ref(props.uiNote);
const viewerLayer = ref<null | "freeze" | "prefer" | "force">(null);
const forceBase = ref<"A" | "U" | "G" | "C">("A");
const FORCE_BASE_OPTIONS: Array<"A" | "U" | "G" | "C"> = ["A", "U", "G", "C"];
const viewerNodes = ref<ViewerNodeMapping[]>([]);

// 防止循环更新的标志
let isExternalUpdate = false;

// 预览相关状态
const viewerWidth = props.viewerWidth ?? 720;
const viewerHeight = props.viewerHeight ?? 400;
const viewerR = props.viewerR ?? 12;
const effectivePayload = ref<ViewerPayload | undefined>(undefined);
const isLoadingPreview = ref(false);
const lastAppliedStyleLevel = ref(-1);

const showPreview = computed(() => {
  return props.isSelectMode || props.isCustomMode;
});

const freezePositionsList = computed(() =>
  parsePositionList(localFreezePositionsText.value)
);
const preferPositionsList = computed(() =>
  parsePositionList(localPreferPositionsText.value)
);
const forcePositionsList = computed(() =>
  normalizeForceEntries(parsePositionList(localForcePositionsText.value))
);
const forceBaseMap = computed<Record<string, string>>(() => {
  const map: Record<string, string> = {};
  forcePositionsList.value.forEach((entry) => {
    const idx = extractIndexFromForceEntry(entry);
    const base = extractBaseFromForceEntry(entry);
    if (idx && base) map[idx] = base;
  });
  return map;
});
const regionSequenceMap = computed<Record<string, string[]>>(() => {
  const map: Record<string, string[]> = {};
  props.REGION_OPTIONS.forEach((region) => {
    map[region] = [];
  });
  viewerNodes.value.forEach((node) => {
    if (node.region && node.sequenceIndex) {
      if (!map[node.region]) map[node.region] = [];
      map[node.region].push(node.sequenceIndex);
    }
  });
  return map;
});

function getRegionSequenceList(region: string): string[] {
  const list = regionSequenceMap.value[region];
  return list ? [...list] : [];
}

// 监听props变化 - 单向同步：父组件 -> 子组件
watch(
  () => props.adv,
  (newAdv) => {
    if (isExternalUpdate) {
      isExternalUpdate = false;
      return;
    }
    localAdv.value = { ...newAdv };
  },
  { deep: true }
);

watch(
  () => props.freezeRegions,
  (newRegions) => {
    setFreezeRegionsInternal([...newRegions]);
    syncRegionSelections();
  }
);

watch(
  () => props.preferRegions,
  (newRegions) => {
    setPreferRegionsInternal([...newRegions]);
    syncRegionSelections();
  }
);

watch(
  () => props.freezePositionsText,
  (newText) => {
    localFreezePositionsText.value = newText;
  }
);

watch(
  () => props.preferPositionsText,
  (newText) => {
    localPreferPositionsText.value = newText;
  }
);

watch(
  () => props.forcePositionsText,
  (newText) => {
    localForcePositionsText.value = newText;
  }
);

watch(
  () => props.uiNote,
  (newNote) => {
    localUiNote.value = newNote;
  }
);

// 监听选中的序列变化
watch(
  () => props.selectedSequenceName,
  async (newSeqName) => {
    if (props.isSelectMode && newSeqName) {
      await loadSequencePreview(newSeqName, props.selectedSpecies);
    } else {
      effectivePayload.value = undefined;
    }
  }
);

// 监听模式变化
watch(
  () => props.isSelectMode,
  (newVal) => {
    if (!newVal) {
      effectivePayload.value = undefined;
    } else if (props.selectedSequenceName) {
      loadSequencePreview(props.selectedSequenceName, props.selectedSpecies);
    }
  }
);

// 加载序列预览数据
async function loadSequencePreview(sequenceName: string, species?: string) {
  if (!sequenceName) return;

  isLoadingPreview.value = true;
  try {
    const params = new URLSearchParams({ template_name: sequenceName });
    if (species) params.append("species", species);

    const response = await fetch(`/align/export/?${params.toString()}`);
    if (!response.ok)
      throw new Error(`Failed to load preview: ${response.status}`);

    const data = await response.json();
    effectivePayload.value = data;
  } catch (error) {
    console.error("Failed to load sequence preview:", error);
    showNote("Failed to load sequence preview. See console for details.");
    effectivePayload.value = undefined;
  } finally {
    isLoadingPreview.value = false;
  }
}

// 输入处理函数 - 单向同步：子组件 -> 父组件
function handleAdvInput(field: keyof AdvParams, event: Event) {
  const target = event.target as HTMLInputElement;
  const value = parseFloat(target.value);

  if (!isNaN(value)) {
    localAdv.value[field] = value as any;
    isExternalUpdate = true;
    emit("update:adv", { ...localAdv.value });
  }
}

function handleFreezeRegionChange(region: string, event: Event) {
  const target = event.target as HTMLInputElement;
  const checked = target.checked;

  let newRegions = [...localFreezeRegions.value];

  if (checked) {
    if (!newRegions.includes(region)) newRegions.push(region);
  } else {
    newRegions = newRegions.filter((r) => r !== region);
  }

  if (checked && localPreferRegions.value.includes(region)) {
    const newPrefer = localPreferRegions.value.filter((r) => r !== region);
    setPreferRegionsInternal(newPrefer);
    showNote(`Removed "${region}" from Prefer to avoid overlap.`);
  }

  if (newRegions.length === props.REGION_OPTIONS.length) {
    showNote("Freeze regions cannot include all regions.");
    return;
  }

  setFreezeRegionsInternal(newRegions);

  const regionSeq = getRegionSequenceList(region);
  if (!regionSeq.length) {
    showNote("Region mapping unavailable in current preview.");
    return;
  }
  if (checked) {
    applyFreezePositions([...freezePositionsList.value, ...regionSeq]);
  } else {
    const seqSet = new Set(regionSeq);
    applyFreezePositions(
      freezePositionsList.value.filter((pos) => !seqSet.has(pos))
    );
  }
}

function handlePreferRegionChange(region: string, event: Event) {
  const target = event.target as HTMLInputElement;
  const checked = target.checked;

  let newRegions = [...localPreferRegions.value];

  if (checked) {
    if (!newRegions.includes(region)) newRegions.push(region);
  } else {
    newRegions = newRegions.filter((r) => r !== region);
  }

  if (checked && localFreezeRegions.value.includes(region)) {
    const newFreeze = localFreezeRegions.value.filter((r) => r !== region);
    setFreezeRegionsInternal(newFreeze);
    showNote(`Removed "${region}" from Freeze to avoid overlap.`);
  }

  setPreferRegionsInternal(newRegions);

  const regionSeq = getRegionSequenceList(region);
  if (!regionSeq.length) {
    showNote("Region mapping unavailable in current preview.");
    return;
  }
  if (checked) {
    applyPreferPositions([...preferPositionsList.value, ...regionSeq]);
  } else {
    const seqSet = new Set(regionSeq);
    applyPreferPositions(
      preferPositionsList.value.filter((pos) => !seqSet.has(pos))
    );
  }
}

function handleFreezePositionsInput(event: Event) {
  const target = event.target as HTMLInputElement;
  const list = parsePositionList(target.value);
  applyFreezePositions(list);
}

function handlePreferPositionsInput(event: Event) {
  const target = event.target as HTMLInputElement;
  const list = parsePositionList(target.value);
  applyPreferPositions(list);
}

function handleForcePositionsInput(event: Event) {
  const target = event.target as HTMLInputElement;
  const list = parsePositionList(target.value);
  applyForcePositions(list);
}

function showNote(msg: string) {
  localUiNote.value = msg;
  emit("update:uiNote", msg);
  setTimeout(() => {
    if (localUiNote.value === msg) {
      localUiNote.value = "";
      emit("update:uiNote", "");
    }
  }, 3000);
}

function toggleViewerLayer(mode: "freeze" | "prefer" | "force" | null) {
  if (!mode) {
    viewerLayer.value = null;
    return;
  }
  viewerLayer.value = viewerLayer.value === mode ? null : mode;
}

function handleViewerMapping(nodes: ViewerNodeMapping[]) {
  viewerNodes.value = nodes;
  syncRegionSelections();
}

function handleViewerNodeSelect(payload: {
  sprinzlId: string;
  sequenceIndex: string;
  base: string;
  originalBase: string;
}) {
  if (!viewerLayer.value) return;
  const seqIndex = payload.sequenceIndex;
  if (!seqIndex) {
    showNote("Sequence index unavailable for this position.");
    return;
  }
  if (viewerLayer.value === "freeze") {
    const next = togglePosition(freezePositionsList.value, seqIndex);
    applyFreezePositions(next);
  } else if (viewerLayer.value === "prefer") {
    const next = togglePosition(preferPositionsList.value, seqIndex);
    applyPreferPositions(next);
  } else if (viewerLayer.value === "force") {
    const entries = [...forcePositionsList.value];
    const existingIdx = entries.findIndex(
      (entry) => extractIndexFromForceEntry(entry) === seqIndex
    );
    if (existingIdx >= 0) {
      entries.splice(existingIdx, 1);
    } else {
      const targetBase =
        (forceBase.value || payload.base || "A").toUpperCase();
      const original = (payload.originalBase || "").toUpperCase();
      if (targetBase === original) {
        return;
      }
      entries.push(`${seqIndex}:${targetBase}`);
    }
    applyForcePositions(entries);
  }
}

function parsePositionList(text: string): string[] {
  return text
    .split(/[,，\s]+/)
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
}

function togglePosition(list: string[], id: string) {
  const exists = list.includes(id);
  if (exists) {
    return list.filter((item) => item !== id);
  }
  return [...list, id];
}

function applyFreezePositions(list: string[]) {
  const unique = dedupeSimpleList(list);
  setFreezeText(unique);
  if (unique.length) {
    const indexSet = new Set(unique);
    const filteredPrefer = preferPositionsList.value.filter(
      (pos) => !indexSet.has(pos)
    );
    if (filteredPrefer.length !== preferPositionsList.value.length) {
      setPreferText(filteredPrefer);
    }
    const filteredForce = forcePositionsList.value.filter((entry) => {
      const idx = extractIndexFromForceEntry(entry);
      return !idx || !indexSet.has(idx);
    });
    if (filteredForce.length !== forcePositionsList.value.length) {
      setForceText(filteredForce);
    }
  }
}

function applyPreferPositions(list: string[]) {
  const unique = dedupeSimpleList(list);
  setPreferText(unique);
  if (unique.length) {
    const indexSet = new Set(unique);
    const filteredFreeze = freezePositionsList.value.filter(
      (pos) => !indexSet.has(pos)
    );
    if (filteredFreeze.length !== freezePositionsList.value.length) {
      setFreezeText(filteredFreeze);
    }
    const filteredForce = forcePositionsList.value.filter((entry) => {
      const idx = extractIndexFromForceEntry(entry);
      return !idx || !indexSet.has(idx);
    });
    if (filteredForce.length !== forcePositionsList.value.length) {
      setForceText(filteredForce);
    }
  }
}

function applyForcePositions(list: string[]) {
  const normalized = normalizeForceEntries(list);
  setForceText(normalized);
  if (normalized.length) {
    const indexSet = new Set(
      normalized
        .map((entry) => extractIndexFromForceEntry(entry))
        .filter((idx): idx is string => !!idx)
    );
    const filteredFreeze = freezePositionsList.value.filter(
      (pos) => !indexSet.has(pos)
    );
    if (filteredFreeze.length !== freezePositionsList.value.length) {
      setFreezeText(filteredFreeze);
    }
    const filteredPrefer = preferPositionsList.value.filter(
      (pos) => !indexSet.has(pos)
    );
    if (filteredPrefer.length !== preferPositionsList.value.length) {
      setPreferText(filteredPrefer);
    }
  }
}

function setFreezeText(list: string[]) {
  const text = list.join(", ");
  localFreezePositionsText.value = text;
  emit("update:freezePositionsText", text);
}

function setPreferText(list: string[]) {
  const text = list.join(", ");
  localPreferPositionsText.value = text;
  emit("update:preferPositionsText", text);
}

function setForceText(list: string[]) {
  const text = list.join(", ");
  localForcePositionsText.value = text;
  emit("update:forcePositionsText", text);
}

function dedupeSimpleList(list: string[]) {
  const seen = new Set<string>();
  const result: string[] = [];
  list.forEach((item) => {
    const trimmed = item.trim();
    if (!trimmed || seen.has(trimmed)) return;
    seen.add(trimmed);
    result.push(trimmed);
  });
  return result;
}

function normalizeForceEntries(list: string[]) {
  const normalized: string[] = [];
  const map = new Map<string, string>();
  list.forEach((entry) => {
    const index = extractIndexFromForceEntry(entry);
    const base = extractBaseFromForceEntry(entry);
    if (!index || !base) return;
    map.set(index, `${index}:${base}`);
  });
  map.forEach((value) => normalized.push(value));
  return normalized;
}

function extractIndexFromForceEntry(entry: string): string | null {
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

function extractBaseFromForceEntry(entry: string): string | null {
  const trimmed = entry.trim();
  if (!trimmed) return null;
  if (trimmed.startsWith("{")) {
    try {
      const obj = JSON.parse(trimmed);
      const key = Object.keys(obj)[0];
      const value = obj[key];
      return normalizeBase(value);
    } catch {
      return null;
    }
  }
  const parts = trimmed.split(":");
  const value = parts[1];
  return normalizeBase(value);
}

function normalizeBase(value: unknown): string | null {
  if (value === undefined || value === null) return null;
  const base = String(value).trim().toUpperCase();
  if (!base) return null;
  return base;
}

function canonicalizeRegionList(list: string[]) {
  const seen = new Set<string>();
  const result: string[] = [];
  props.REGION_OPTIONS.forEach((region) => {
    if (list.includes(region) && !seen.has(region)) {
      seen.add(region);
      result.push(region);
    }
  });
  return result;
}

function areArraysEqual(a: string[], b: string[]) {
  return a.length === b.length && a.every((val, idx) => val === b[idx]);
}

function setFreezeRegionsInternal(list: string[]) {
  const canonical = canonicalizeRegionList(list);
  if (!areArraysEqual(localFreezeRegions.value, canonical)) {
    localFreezeRegions.value = canonical;
    emit("update:freezeRegions", canonical);
  }
}

function setPreferRegionsInternal(list: string[]) {
  const canonical = canonicalizeRegionList(list);
  if (!areArraysEqual(localPreferRegions.value, canonical)) {
    localPreferRegions.value = canonical;
    emit("update:preferRegions", canonical);
  }
}

function freezeRegionState(region: string): "none" | "partial" | "full" {
  const seqs = regionSequenceMap.value[region] || [];
  if (!seqs.length) {
    return localFreezeRegions.value.includes(region) ? "full" : "none";
  }
  const freezeSet = new Set(freezePositionsList.value);
  const matched = seqs.filter((seq) => freezeSet.has(seq)).length;
  if (matched === 0) return "none";
  if (matched === seqs.length) return "full";
  return "partial";
}

function preferRegionState(region: string): "none" | "partial" | "full" {
  const seqs = regionSequenceMap.value[region] || [];
  if (!seqs.length) {
    return localPreferRegions.value.includes(region) ? "full" : "none";
  }
  const preferSet = new Set(preferPositionsList.value);
  const matched = seqs.filter((seq) => preferSet.has(seq)).length;
  if (matched === 0) return "none";
  if (matched === seqs.length) return "full";
  return "partial";
}

function syncRegionSelections() {
  const map = regionSequenceMap.value;
  const freezeSet = new Set(freezePositionsList.value);
  const preferSet = new Set(preferPositionsList.value);

  const autoFreeze: string[] = [];
  const autoPrefer: string[] = [];

  props.REGION_OPTIONS.forEach((region) => {
    const seqs = map[region] || [];
    if (seqs.length && seqs.every((seq) => freezeSet.has(seq))) {
      autoFreeze.push(region);
    }
    if (seqs.length && seqs.every((seq) => preferSet.has(seq))) {
      autoPrefer.push(region);
    }
  });

  const manualFreeze = localFreezeRegions.value.filter(
    (region) => !(map[region] && map[region].length)
  );
  const manualPrefer = localPreferRegions.value.filter(
    (region) => !(map[region] && map[region].length)
  );

  setFreezeRegionsInternal([
    ...new Set([...manualFreeze, ...autoFreeze]),
  ]);
  setPreferRegionsInternal([
    ...new Set([...manualPrefer, ...autoPrefer]),
  ]);
}

watch(regionSequenceMap, syncRegionSelections, { immediate: true });

// 组件挂载时加载预览
onMounted(() => {
  if (props.isSelectMode && props.selectedSequenceName) {
    loadSequencePreview(props.selectedSequenceName, props.selectedSpecies);
  }
});
</script>

<style scoped>
.gc {
  display: flex;
  gap: 8px;
  align-items: center;
}
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border: 1px solid var(--chip-border, #e5e7eb);
  border-radius: 999px;
  background: var(--chip, rgba(238, 242, 255, 0.8));
  color: var(--text-strong, #0f172a);
  user-select: none;
}
.note {
  margin-top: 8px;
  padding: 8px 12px;
  background: rgba(255, 247, 237, 0.85);
  border: 1px solid rgba(254, 215, 170, 0.9);
  color: #9a3412;
  border-radius: 6px;
}
.positions-note {
  display: block;
  margin-top: 6px;
}
.constraints-panel {
  margin-top: 24px;
  padding: 16px;
  border: 1px solid var(--card-border, rgba(226, 232, 240, 0.8));
  border-radius: 12px;
  background: color-mix(in srgb, var(--card-bg, rgba(255, 255, 255, 0.92)) 92%, transparent);
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.constraints-heading {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}
.constraints-columns {
  display: flex;
  flex-wrap: wrap;
  gap: 24px;
}
.constraints-block {
  flex: 1 1 300px;
  min-width: 260px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.region-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}
.region-card {
  border: 1px solid var(--card-border, rgba(226, 232, 240, 0.9));
  border-radius: 12px;
  padding: 12px;
  background: color-mix(in srgb, var(--field-bg, #ffffff) 90%, transparent);
  display: flex;
  flex-direction: column;
  gap: 8px;
  box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
}
.positions-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}
.available-note {
  margin-top: -4px;
}
.constraints-columns,
.region-grid,
.positions-grid {
  width: 100%;
}
@media (prefers-color-scheme: dark) {
  .note {
    background: rgba(41, 37, 36, 0.9);
    border-color: rgba(248, 113, 113, 0.5);
    color: #f8fafc;
  }
  .constraints-panel {
    box-shadow: 0 12px 32px rgba(2, 6, 23, 0.4);
  }
  .region-card {
    box-shadow: 0 8px 20px rgba(2, 6, 23, 0.45);
  }
}
@media (max-width: 640px) {
  .constraints-columns {
    flex-direction: column;
  }
}
.viewer-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.viewer-layer-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
}
.toolbar-buttons {
  display: flex;
  gap: 8px;
}
.force-select-group {
  display: flex;
  align-items: center;
  gap: 8px;
}
.force-select {
  border: 1px solid rgba(148, 163, 184, 0.8);
  border-radius: 6px;
  padding: 4px 8px;
  background: rgba(255, 255, 255, 0.9);
  color: #0f172a;
  font-size: 12px;
}
.btn-xs {
  border: 1px solid rgba(148, 163, 184, 0.8);
  background: rgba(255, 255, 255, 0.85);
  color: #0f172a;
  padding: 4px 10px;
  border-radius: 8px;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.15s ease, color 0.15s ease;
}
.btn-xs.active {
  background: #2563eb;
  border-color: #1d4ed8;
  color: #fff;
}
.btn-xs:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.debug-info {
  background: #f0f8ff;
  border: 1px solid #b3d9ff;
  border-radius: 8px;
  padding: 12px;
  margin: 12px 0;
  font-size: 12px;
}
.debug-info h4 {
  margin: 0 0 8px 0;
  color: #0066cc;
}
.debug-info p {
  margin: 4px 0;
  font-family: monospace;
}
.preview-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 200px;
  border: 2px dashed #e5e7eb;
  border-radius: 8px;
  background-color: #f9fafb;
}
.placeholder-content {
  text-align: center;
  color: #6b7280;
}
.placeholder-icon {
  font-size: 2rem;
  margin-bottom: 12px;
  display: block;
}
.preview-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 200px;
  gap: 12px;
}
.loading-spinner {
  width: 24px;
  height: 24px;
  border: 3px solid #e5e7eb;
  border-top: 3px solid #2563eb;
  border-radius: 50%;
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
</style>
