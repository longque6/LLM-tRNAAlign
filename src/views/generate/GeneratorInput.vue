<template>
  <section class="card">
    <!-- 序列选择方式切换（滑动块） -->
    <div class="form">
      <div class="tabs" role="tablist" aria-label="Sequence selection mode">
        <button
          class="tab"
          :class="{ active: seqSelectMode === 'select' }"
          @click="setSeqSelectMode('select')"
          role="tab"
          :aria-selected="seqSelectMode === 'select'"
          title="Select from candidate sequences"
        >
          <span class="tab-text">Select from candidates</span>
        </button>
        <button
          class="tab"
          :class="{ active: seqSelectMode === 'custom' }"
          @click="setSeqSelectMode('custom')"
          role="tab"
          :aria-selected="seqSelectMode === 'custom'"
          title="Custom input sequence"
        >
          <span class="tab-text">Custom input</span>
        </button>
        <div
          class="tab-indicator"
          :style="seqSelectIndicatorStyle"
          aria-hidden="true"
        ></div>
      </div>
    </div>

    <!-- 候选序列选择区域（仅选择模式下显示） -->
    <div v-if="seqSelectMode === 'select'" class="form">
      <!-- 物种选择框 - 一行显示 -->
      <div class="form">
        <div class="row-group">
          <div class="select-group">
            <label class="label">Domain</label>
            <div class="select-wrapper">
              <select
                v-model="selectedDomain"
                @change="onDomainChange"
                class="styled-select"
              >
                <option value="2157">Archaea</option>
                <option value="2">Bacteria</option>
                <option value="2759">Eukarya</option>
              </select>
              <div class="select-arrow">▼</div>
            </div>
          </div>

          <div class="select-group">
            <label class="label">Species</label>
            <div class="select-with-button">
              <div class="select-wrapper">
                <select
                  v-model="selectedSpecies"
                  class="styled-select"
                  :disabled="filteredSpecies.length === 0"
                  @change="onSpeciesChange"
                >
                  <option value="" disabled>Select a species...</option>
                  <option
                    v-for="species in filteredSpecies"
                    :key="species.dbname"
                    :value="species.dbname"
                  >
                    {{ species.name }}
                  </option>
                </select>
                <div class="select-arrow">▼</div>
              </div>
              <button
                class="random-button"
                @click="selectRandomSpecies"
                :disabled="filteredSpecies.length === 0"
                title="Select random species"
              >
                🎲 Random
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- isotype和anticodon选择框 - 一行显示 -->
      <div class="form">
        <div class="row-group">
          <div class="select-group">
            <label class="label">
              Isotype
              <small class="muted">(from original sequences)</small>
            </label>
            <div class="select-wrapper">
              <select
                v-model="selectedIsotype"
                class="styled-select"
                :disabled="!isotypeOptions.length"
                @change="onIsotypeChange"
              >
                <option value="" disabled>Select an isotype...</option>
                <option
                  v-for="isotype in isotypeOptions"
                  :key="isotype"
                  :value="isotype"
                >
                  {{ isotype }}
                </option>
              </select>
              <div class="select-arrow">▼</div>
            </div>
          </div>

          <div class="select-group">
            <label class="label">
              Anticodon
              <small class="muted">(from original sequences)</small>
            </label>
            <div class="select-wrapper">
              <select
                v-model="selectedAnticodon"
                class="styled-select"
                :disabled="!selectedIsotype || !anticodonOptions.length"
                @change="onAnticodonChange"
              >
                <option value="" disabled>Select an anticodon...</option>
                <option
                  v-for="anticodon in anticodonOptions"
                  :key="anticodon"
                  :value="anticodon"
                >
                  {{ anticodon }}
                </option>
              </select>
              <div class="select-arrow">▼</div>
            </div>
          </div>
        </div>
      </div>

      <!-- 候选序列表格 -->
      <div
        v-if="candidateSequences.length > 0"
        class="candidate-table-container"
      >
        <label class="label">Candidate Sequence</label>
        <div class="table-wrapper">
          <table class="candidate-table">
            <thead>
              <tr>
                <th class="table-checkbox">Select</th>
                <th class="table-seqname">Sequence Name</th>
                <th class="table-length">Length</th>
                <th class="table-gc">GC Content</th>
                <th class="table-sequence">Sequence</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(candidate, index) in candidateSequences"
                :key="index"
                :class="{ selected: selectedCandidateIndex === index }"
                data-guardable="true"
                @click="
                  selectCandidateSequence(
                    index,
                    candidate.sequence,
                    candidate.seqname
                  )
                "
              >
                <td class="table-checkbox">
                  <div class="checkbox-wrapper">
                    <input
                      type="radio"
                      :name="`candidate-seq-${selectedAnticodon}`"
                      :value="index"
                      v-model="selectedCandidateIndex"
                      @change="
                        selectCandidateSequence(
                          index,
                          candidate.sequence,
                          candidate.seqname
                        )
                      "
                      class="styled-radio"
                    />
                    <span class="checkmark"></span>
                  </div>
                </td>
                <td class="table-seqname">
                  {{ candidate.seqname || `Sequence ${index + 1}` }}
                </td>
                <td class="table-length">{{ candidate.length }}</td>
                <td class="table-gc">{{ candidate.gcContent }}%</td>
                <td class="table-sequence">
                  <span class="sequence-full" :title="candidate.sequence">{{
                    candidate.sequence
                  }}</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- 自定义序列输入区域（仅自定义模式下显示） -->
    <div v-if="seqSelectMode === 'custom'" class="form">
      <label class="label">Seed sequence (A/U/G/C only)</label>
      <textarea
        :value="seedSeq"
        @input="onSeedInput"
        class="textarea"
        rows="4"
        placeholder="e.g. GGUCUCUGGGCCCAAUGG..."
        title="Sequence to initialize the generator"
      ></textarea>
      <small class="muted">
        Input is auto-normalized: <code>T → U</code>, non-AUGCT becomes
        <code>N</code>. Sequence must be longer than 50 bases to generate.
      </small>
    </div>

    <!-- 参数表格和Sprinzl链接（当有选中序列时显示） -->
    <div
      v-if="selectedCandidateSeq || (seqSelectMode === 'custom' && seedSeq)"
      class="preview-section"
    >
      <div v-if="seqSelectMode === 'custom'" class="form">
        <div class="parameters-table">
          <table class="params-table">
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Sequence Length</td>
                <td>{{ currentSequence.length }}</td>
              </tr>
              <tr>
                <td>GC Content</td>
                <td>{{ calculateGCContent(currentSequence) }}%</td>
              </tr>
              <tr v-if="currentSequence.length <= 50" class="warning-row">
                <td colspan="2" class="warning-text">
                  ⚠️ Sequence must be longer than 50 bases to generate
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- <div class="form">
        <div class="sprinzl-link-container">
          <button
            class="sprinzl-link-button"
            @click="openSprinzlInNewTab"
            title="Copy current sequence and open Sprinzl numbering in new tab"
          >
            📊 View Sprinzl Numbering
          </button>
          <small class="muted"
            >Copies the current sequence and opens Sprinzl numbering
            visualization in a new tab</small
          >
        </div>
      </div> -->
    </div>

  </section>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from "vue";

// ========== Props & Emits ==========
const props = defineProps({
  seedSeq: { type: String, required: true }, // 种子序列
});
const emit = defineEmits([
  "update:seedSeq",
  "update:generateEnabled",
  // 新增：序列选择相关事件
  "update:seqSelectMode",
  "update:selectedSequenceName",
  "update:selectedSpecies",
]);

// ========== 物种信息 ==========
const selectedDomain = ref("2157");
const selectedSpecies = ref("");
const allSpecies = ref<{ dbname: string; name: string; domain: number }[]>([]);
const isLoading = ref(false);

// ========== 序列选择相关状态 ==========
const seqSelectMode = ref<"select" | "custom">("select");
const isotypeOptions = ref<string[]>([]);
const anticodonOptions = ref<string[]>([]);
const candidateSequences = ref<any[]>([]);
const selectedIsotype = ref("");
const selectedAnticodon = ref("");
const selectedCandidateIndex = ref(-1);
const selectedCandidateSeq = ref("");
const selectedSequenceName = ref("");

// ========== 本地存储键名 ==========
const STORAGE_KEYS = {
  DOMAIN: "trna-domain",
  SPECIES: "trna-species",
  ISOTYPE: "trna-isotype",
  ANTICODON: "trna-anticodon",
  SELECT_MODE: "trna-select-mode",
  SEED_SEQ: "trna-seed-seq",
  SEED_NAME: "trna-seed-name",
};

let restoringSelections = false;

// 当前序列
const currentSequence = computed(() =>
  seqSelectMode.value === "select" ? selectedCandidateSeq.value : props.seedSeq
);

// Generate 可用性
const isGenerateEnabled = computed(() => {
  if (seqSelectMode.value === "select") {
    return selectedCandidateSeq.value.length > 0;
  }
  return props.seedSeq.length > 50;
});
function updateGenerateButtonState() {
  emit("update:generateEnabled", isGenerateEnabled.value);
}

// 规范化/反规范化
function normalizeSeq(raw: string): string {
  if (!raw) return "";
  return raw.toUpperCase().replace(/T/g, "U");
}
function denormalizeSeq(raw: string): string {
  if (!raw) return "";
  return raw.toUpperCase().replace(/U/g, "T");
}
function calculateGCContent(sequence: string): string {
  if (!sequence) return "0.00";
  const gcCount = (sequence.match(/[GC]/g) || []).length;
  return ((gcCount / sequence.length) * 100).toFixed(2);
}

// Sprinzl
// function openSprinzlInNewTab() {
//   const sequence = currentSequence.value;
//   if (!sequence) {
//     alert("Please select or enter a sequence first");
//     return;
//   }
//   navigator.clipboard.writeText(sequence).finally(() => {
//     const encodedSequence = encodeURIComponent(sequence);
//     const url = `/sprinzl?sequence=${encodedSequence}`;
//     window.open(url, "_blank");
//   });
// }

// 拉全量物种
async function fetchAllSpecies() {
  isLoading.value = true;
  try {
    const response = await fetch(`/mysql/get_species`);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const data = await response.json();
    allSpecies.value = data.species || [];

    await restoreUserSelections();

    // 若未有保存的 species，则自动随机一次（会链式触发）
    if (!selectedSpecies.value && filteredSpecies.value.length > 0) {
      selectRandomSpecies();
    }
  } catch (e) {
    console.error("Failed to fetch species:", e);
    allSpecies.value = [];
    selectedSpecies.value = "";
  } finally {
    isLoading.value = false;
    // 保险：species 有值但 isotype 还没装载时，自动拉取
    ensureFiltersForSpecies();
  }
}

// 过滤物种
const filteredSpecies = computed(() =>
  allSpecies.value.filter((s) => s.domain.toString() === selectedDomain.value)
);

// 模式切换
function setSeqSelectMode(mode: "select" | "custom") {
  seqSelectMode.value = mode;
  localStorage.setItem(STORAGE_KEYS.SELECT_MODE, mode);

  // 通知父组件模式变化
  emit("update:seqSelectMode", mode);

  if (mode === "custom") {
    // 只清空下游（不要清 isotypeOptions），避免回到 select 时 isotype 变灰
    resetForCustomMode();
    // 清空序列名称
    selectedSequenceName.value = "";
    emit("update:selectedSequenceName", "");
  } else {
    // 回到 select：确保 isotype 列表可用
    ensureFiltersForSpecies();
  }
  updateGenerateButtonState();
}

// 随机物种
function selectRandomSpecies() {
  if (!filteredSpecies.value.length) return;
  const i = Math.floor(Math.random() * filteredSpecies.value.length);
  selectedSpecies.value = filteredSpecies.value[i].dbname;
  emit("update:selectedSpecies", selectedSpecies.value);
  onSpeciesChange();
}

// Domain 改变（全量重置）
function onDomainChange() {
  saveToStorage(STORAGE_KEYS.DOMAIN, selectedDomain.value);
  resetAfterSpeciesChange(); // 全量清空
  if (filteredSpecies.value.length > 0) {
    selectedSpecies.value = filteredSpecies.value[0].dbname;
    emit("update:selectedSpecies", selectedSpecies.value);
    onSpeciesChange();
  } else {
    selectedSpecies.value = "";
    emit("update:selectedSpecies", "");
  }
}

// Species 改变（UI事件）
function onSpeciesChange() {
  saveToStorage(STORAGE_KEYS.SPECIES, selectedSpecies.value);
  emit("update:selectedSpecies", selectedSpecies.value);
  fetchTrnaFilters();
}

// 保障：当 species 有值但 isotypeOptions 为空/丢失时自动拉取
async function ensureFiltersForSpecies() {
  if (restoringSelections) return;
  if (
    seqSelectMode.value === "select" &&
    selectedSpecies.value &&
    !isotypeOptions.value.length
  ) {
    await fetchTrnaFilters();
  }
}

// Isotype 改变
function onIsotypeChange() {
  saveToStorage(STORAGE_KEYS.ISOTYPE, selectedIsotype.value);
  fetchAnticodonsByIsotype();
}

// Anticodon 改变
function onAnticodonChange() {
  saveToStorage(STORAGE_KEYS.ANTICODON, selectedAnticodon.value);
  fetchCandidateSequences();
}

// 存储
function saveToStorage(key: string, value: string) {
  if (value !== undefined && value !== null) {
    localStorage.setItem(key, value);
  }
}

function clearStoredSeed() {
  localStorage.removeItem(STORAGE_KEYS.SEED_SEQ);
  localStorage.removeItem(STORAGE_KEYS.SEED_NAME);
}

// 恢复选择
async function restoreUserSelections() {
  restoringSelections = true;
  try {
    const savedSelectMode = localStorage.getItem(STORAGE_KEYS.SELECT_MODE);
    if (savedSelectMode === "select" || savedSelectMode === "custom") {
      seqSelectMode.value = savedSelectMode as "select" | "custom";
      emit("update:seqSelectMode", seqSelectMode.value);
    }

    const savedDomain = localStorage.getItem(STORAGE_KEYS.DOMAIN);
    if (savedDomain) selectedDomain.value = savedDomain;

    const savedSpecies = localStorage.getItem(STORAGE_KEYS.SPECIES);
    if (
      savedSpecies &&
      filteredSpecies.value.some((s) => s.dbname === savedSpecies)
    ) {
      selectedSpecies.value = savedSpecies;
      emit("update:selectedSpecies", selectedSpecies.value);
    } else if (filteredSpecies.value.length > 0) {
      selectedSpecies.value = filteredSpecies.value[0].dbname;
      emit("update:selectedSpecies", selectedSpecies.value);
    }

    if (selectedSpecies.value) {
      await fetchTrnaFilters({ resetSelections: false });
      const savedIsotype = localStorage.getItem(STORAGE_KEYS.ISOTYPE);
      if (savedIsotype && isotypeOptions.value.includes(savedIsotype)) {
        selectedIsotype.value = savedIsotype;
        await fetchAnticodonsByIsotype({ resetSelections: false });
        const savedAnticodon = localStorage.getItem(STORAGE_KEYS.ANTICODON);
        if (savedAnticodon && anticodonOptions.value.includes(savedAnticodon)) {
          selectedAnticodon.value = savedAnticodon;
          await fetchCandidateSequences({ resetSelections: false });
        }
      }
    }
  } finally {
    restoringSelections = false;
  }
}

// 拉 isotype (by species)
async function fetchTrnaFilters(options: { resetSelections?: boolean } = {}) {
  const { resetSelections = true } = options;
  if (!selectedSpecies.value) {
    resetAfterSpeciesChange();
    return;
  }
  try {
    const response = await fetch(
      `/mysql/get_trna_filters?species=${selectedSpecies.value}`
    );
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const data = await response.json();
    isotypeOptions.value = data.isotypes || [];

    if (resetSelections) {
      // 清空下游
      anticodonOptions.value = [];
      candidateSequences.value = [];
      selectedIsotype.value = "";
      selectedAnticodon.value = "";
      selectedCandidateIndex.value = -1;
      selectedCandidateSeq.value = "";
      selectedSequenceName.value = "";
      emit("update:seedSeq", "");
      emit("update:selectedSequenceName", "");
      clearStoredSeed();
      updateGenerateButtonState();
    } else if (
      selectedIsotype.value &&
      !isotypeOptions.value.includes(selectedIsotype.value)
    ) {
      selectedIsotype.value = "";
    }
  } catch (e) {
    console.error("Failed to fetch tRNA filters:", e);
    resetAfterSpeciesChange();
  }
}

// 拉 anticodon (by isotype)
async function fetchAnticodonsByIsotype(
  options: { resetSelections?: boolean } = {}
) {
  const { resetSelections = true } = options;
    if (!selectedSpecies.value || !selectedIsotype.value) {
      anticodonOptions.value = [];
      candidateSequences.value = [];
      selectedAnticodon.value = "";
      selectedCandidateIndex.value = -1;
      selectedCandidateSeq.value = "";
      selectedSequenceName.value = "";
      emit("update:seedSeq", "");
      emit("update:selectedSequenceName", "");
      clearStoredSeed();
      updateGenerateButtonState();
      return;
    }
  try {
    const response = await fetch("/mysql/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        table: "trna_records",
        filters: {
          dbname: selectedSpecies.value,
          isotype: denormalizeSeq(selectedIsotype.value),
        },
        limit: 1000,
      }),
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const data = await response.json();

    anticodonOptions.value = Array.from(
      new Set(
        (data.rows || [])
          .map((r: any) => r.anticodon)
          .filter(Boolean)
          .map((ac: string) => normalizeSeq(ac))
      )
    );

    if (resetSelections) {
      candidateSequences.value = [];
      selectedAnticodon.value = "";
      selectedCandidateIndex.value = -1;
      selectedCandidateSeq.value = "";
      selectedSequenceName.value = "";
      emit("update:seedSeq", "");
      emit("update:selectedSequenceName", "");
      clearStoredSeed();
      updateGenerateButtonState();
    } else if (
      selectedAnticodon.value &&
      !anticodonOptions.value.includes(selectedAnticodon.value)
    ) {
      selectedAnticodon.value = "";
    }
  } catch (e) {
    console.error("Failed to fetch anticodons by isotype:", e);
    anticodonOptions.value = [];
    candidateSequences.value = [];
    selectedAnticodon.value = "";
    selectedCandidateIndex.value = -1;
    selectedCandidateSeq.value = "";
    selectedSequenceName.value = "";
    emit("update:seedSeq", "");
    emit("update:selectedSequenceName", "");
    clearStoredSeed();
    updateGenerateButtonState();
  }
}

// 拉候选序列
async function fetchCandidateSequences(options: { resetSelections?: boolean } = {}) {
  const { resetSelections = true } = options;
  if (
    !selectedSpecies.value ||
    !selectedIsotype.value ||
    !selectedAnticodon.value
  ) {
    candidateSequences.value = [];
    selectedCandidateIndex.value = -1;
    selectedCandidateSeq.value = "";
    selectedSequenceName.value = "";
    emit("update:seedSeq", "");
    emit("update:selectedSequenceName", "");
    clearStoredSeed();
    updateGenerateButtonState();
    return;
  }
  try {
    const response = await fetch("/mysql/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        table: "trna_records",
        filters: {
          dbname: selectedSpecies.value,
          isotype: denormalizeSeq(selectedIsotype.value),
          anticodon: denormalizeSeq(selectedAnticodon.value),
        },
        limit: 50,
      }),
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const data = await response.json();

    const mapped = (data.rows || [])
      .filter((item: any) => item.sequence)
      .map((item: any) => {
        const seqU = normalizeSeq(item.sequence);
        return {
          ...item,
          sequence: seqU,
          length: seqU.length,
          gcContent: calculateGCContent(seqU),
        };
      });
    candidateSequences.value = mapped;

    if (resetSelections) {
      selectedCandidateIndex.value = -1;
      selectedCandidateSeq.value = "";
      selectedSequenceName.value = "";
      emit("update:seedSeq", "");
      emit("update:selectedSequenceName", "");
      clearStoredSeed();
      updateGenerateButtonState();
    } else {
      restoreCandidateSelectionFromStorage();
    }
  } catch (e) {
    console.error("Failed to fetch candidate sequences:", e);
    candidateSequences.value = [];
    selectedCandidateIndex.value = -1;
    selectedCandidateSeq.value = "";
    selectedSequenceName.value = "";
    emit("update:seedSeq", "");
    emit("update:selectedSequenceName", "");
    clearStoredSeed();
    updateGenerateButtonState();
  }
}

// 选择候选序列
function selectCandidateSequence(
  index: number,
  sequence: string,
  seqname: string
) {
  selectedCandidateIndex.value = index;
  const normalized = normalizeSeq(sequence);
  selectedCandidateSeq.value = normalized;
  selectedSequenceName.value = seqname;
  emit("update:seedSeq", normalized);
  emit("update:selectedSequenceName", seqname);
  saveToStorage(STORAGE_KEYS.SEED_SEQ, normalized);
  saveToStorage(STORAGE_KEYS.SEED_NAME, seqname || "");
  updateGenerateButtonState();
}

function restoreCandidateSelectionFromStorage() {
  const savedSeq = localStorage.getItem(STORAGE_KEYS.SEED_SEQ);
  if (!savedSeq) {
    updateGenerateButtonState();
    return;
  }
  const idx = candidateSequences.value.findIndex(
    (candidate) => candidate.sequence === savedSeq
  );
  if (idx === -1) {
    updateGenerateButtonState();
    return;
  }
  selectedCandidateIndex.value = idx;
  selectedCandidateSeq.value = savedSeq;
  const savedName =
    localStorage.getItem(STORAGE_KEYS.SEED_NAME) ||
    candidateSequences.value[idx].seqname ||
    `Sequence ${idx + 1}`;
  selectedSequenceName.value = savedName;
  emit("update:seedSeq", savedSeq);
  emit("update:selectedSequenceName", savedName);
  updateGenerateButtonState();
}

// —— 重置工具函数 —— //
// 切换物种/域后的"全量重置"（会清空 isotype）
function resetAfterSpeciesChange() {
  isotypeOptions.value = [];
  anticodonOptions.value = [];
  candidateSequences.value = [];
  selectedIsotype.value = "";
  selectedAnticodon.value = "";
  selectedCandidateIndex.value = -1;
  selectedCandidateSeq.value = "";
  selectedSequenceName.value = "";
  if (seqSelectMode.value === "select") {
    emit("update:seedSeq", "");
    emit("update:selectedSequenceName", "");
  }
  clearStoredSeed();
  updateGenerateButtonState();
}
// 切到 custom 时的"轻量重置"（保留 isotypeOptions，避免回到 select 变灰）
function resetForCustomMode() {
  anticodonOptions.value = [];
  candidateSequences.value = [];
  selectedAnticodon.value = "";
  selectedCandidateIndex.value = -1;
  selectedCandidateSeq.value = "";
  selectedSequenceName.value = "";
  // 不清空 isotypeOptions / selectedIsotype
  emit("update:seedSeq", "");
  emit("update:selectedSequenceName", "");
  clearStoredSeed();
  updateGenerateButtonState();
}

// 监听：模式切换
watch(seqSelectMode, (newMode) => {
  if (!restoringSelections && newMode === "select") {
    ensureFiltersForSpecies(); // 回来就保证 isotype 可用
  }
  updateGenerateButtonState();
});

// 监听：species，无论来源如何
watch(selectedSpecies, () => {
  if (restoringSelections) return;
  ensureFiltersForSpecies();
});

// 监听：自定义输入
watch(
  () => props.seedSeq,
  () => {
    if (seqSelectMode.value === "custom") updateGenerateButtonState();
  }
);

// 监听：候选序列选择
watch(selectedCandidateSeq, () => {
  if (seqSelectMode.value === "select") updateGenerateButtonState();
});

// 初始化
onMounted(() => {
  fetchAllSpecies();
});

const seqSelectIndicatorStyle = computed(() => ({
  transform:
    seqSelectMode.value === "select" ? "translateX(0%)" : "translateX(100%)",
}));

// 自定义输入标准化
function onSeedInput(e: Event) {
  const raw = (e.target as HTMLTextAreaElement).value;
  const norm = raw
    .toUpperCase()
    .replace(/T/g, "U")
    .replace(/[^AUGC]/g, "N");
  emit("update:seedSeq", norm);
  if (norm) {
    saveToStorage(STORAGE_KEYS.SEED_SEQ, norm);
    saveToStorage(STORAGE_KEYS.SEED_NAME, "Custom input");
  } else {
    clearStoredSeed();
  }
  updateGenerateButtonState();
}
</script>

<style scoped>
/* 保持原有的所有样式不变 */
.tab.active {
  color: #2563eb;
  font-weight: 500;
}

.tab-indicator {
  position: absolute;
  top: 4px;
  left: 4px;
  height: calc(100% - 8px);
  width: calc(50% - 8px);
  background: white;
  border-radius: 4px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  transition: transform 0.2s;
}

.form {
  margin-bottom: 1.5rem;
}

.label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 800;
  font-size: 1.12rem;
  color: var(--text-soft);
}

.input,
.textarea {
  width: 80%;
  padding: 8px 12px;
  border: 1px solid var(--card-border);
  border-radius: 6px;
  font-size: 14px;
  transition: border-color 0.2s;
  background: var(--field-bg);
  color: var(--field-text);
}
.input:focus,
.textarea:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent);
}
.textarea {
  resize: vertical;
  min-height: 80px;
  font-family: monospace;
}

.muted {
  color: var(--muted);
  font-size: 12px;
  margin-top: 0.25rem;
  display: inline-block;
  margin-left: 0.5rem;
}

code {
  background: color-mix(in srgb, var(--chip) 70%, transparent);
  padding: 2px 4px;
  border-radius: 4px;
  font-size: 0.875em;
  border: 1px solid var(--chip-border);
}

.sequence-info-preview {
  margin-top: 1rem;
  padding: 1rem;
  background-color: var(--chip);
  border-radius: 6px;
  border: 1px solid var(--chip-border);
}

.warning-row {
  background-color: color-mix(in srgb, var(--danger) 10%, transparent);
}
.warning-text {
  color: var(--danger);
  font-weight: 500;
  text-align: center;
  padding: 8px;
}

.row-group {
  display: flex;
  gap: 1rem;
  width: 100%;
}
.row-group .select-group {
  flex: 1;
  min-width: 0;
}

.select-group {
  margin-bottom: 1rem;
}
.select-wrapper {
  position: relative;
  width: 100%;
  display: inline-block;
}
.styled-select {
  width: 100%;
  padding: 10px 12px;
  padding-right: 40px;
  border: 1px solid var(--card-border);
  border-radius: 6px;
  background-color: var(--field-bg);
  color: var(--field-text);
  font-size: 14px;
  appearance: none;
  cursor: pointer;
  transition: border-color 0.2s;
}
.styled-select:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent);
}
.styled-select:disabled {
  background-color: color-mix(in srgb, var(--field-bg) 50%, transparent);
  cursor: not-allowed;
  opacity: 0.6;
}
.select-arrow {
  position: absolute;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  pointer-events: none;
  color: var(--muted);
  font-size: 12px;
}

.select-with-button {
  display: flex;
  gap: 0.5rem;
  width: 100%;
}
.random-button {
  padding: 10px 16px;
  background-color: var(--chip);
  border: 1px solid var(--chip-border);
  border-radius: 6px;
  color: var(--text-strong);
  cursor: pointer;
  font-size: 14px;
  white-space: nowrap;
  transition: all 0.2s;
}
.random-button:hover:not(:disabled) {
  background-color: color-mix(in srgb, var(--chip) 80%, transparent);
  border-color: var(--accent);
}
.random-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.checkbox-wrapper {
  position: relative;
  display: inline-block;
  width: 20px;
  height: 20px;
}
.styled-radio {
  position: absolute;
  opacity: 0;
  cursor: pointer;
  width: 100%;
  height: 100%;
  z-index: 2;
}
.checkmark {
  position: absolute;
  top: 0;
  left: 0;
  height: 20px;
  width: 20px;
  background-color: var(--field-bg);
  border: 2px solid var(--card-border);
  border-radius: 50%;
  transition: all 0.2s;
}
.styled-radio:checked ~ .checkmark {
  background-color: var(--accent);
  border-color: var(--accent);
}
.checkmark:after {
  content: "";
  position: absolute;
  display: none;
}
.styled-radio:checked ~ .checkmark:after {
  display: block;
  left: 6px;
  top: 2px;
  width: 5px;
  height: 10px;
  border: solid white;
  border-width: 0 2px 2px 0;
  transform: rotate(45deg);
}

.candidate-table-container {
  margin-top: 1rem;
}
.table-wrapper {
  border: 1px solid var(--card-border);
  border-radius: 6px;
  overflow: hidden;
  max-height: 400px;
  overflow-y: auto;
  background-color: var(--field-bg);
}
.candidate-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  background-color: var(--field-bg);
  color: var(--field-text);
}
.candidate-table th {
  background-color: var(--chip);
  padding: 12px 8px;
  text-align: left;
  font-weight: 600;
  border-bottom: 1px solid var(--card-border);
  position: sticky;
  top: 0;
  color: var(--text-soft);
}
.candidate-table td {
  padding: 10px 8px;
  border-bottom: 1px solid var(--chip-border);
}
.candidate-table tbody tr:hover {
  background-color: var(--chip);
  cursor: pointer;
}
.candidate-table tbody tr.selected {
  background-color: color-mix(in srgb, var(--accent) 15%, transparent);
}

.table-checkbox {
  width: 50px;
  text-align: center;
}
.table-seqname {
  width: 180px;
  font-weight: 500;
}
.table-length {
  width: 90px;
}
.table-gc {
  width: 110px;
}
.table-sequence {
  width: auto;
  min-width: 240px;
}
.sequence-full {
  font-family: monospace;
  font-size: 12px;
  word-break: break-all;
}

.preview-section {
  margin-top: 1.5rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--chip-border);
}
.parameters-table {
  margin-bottom: 1.5rem;
}
.params-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  background-color: var(--field-bg);
  border-radius: 6px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}
.params-table th {
  background-color: var(--chip);
  padding: 12px 16px;
  text-align: left;
  font-weight: 600;
  border-bottom: 1px solid var(--card-border);
  color: var(--text-soft);
}
.params-table td {
  padding: 12px 16px;
  border-bottom: 1px solid var(--chip-border);
}
.params-table tbody tr:last-child td {
  border-bottom: none;
}
.params-table tbody tr:hover {
  background-color: var(--chip);
}

.sprinzl-link-container {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.sprinzl-link-button {
  padding: 12px 20px;
  background: var(--accent-grad);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px color-mix(in srgb, var(--accent) 30%, transparent);
  width: fit-content;
}
.sprinzl-link-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px color-mix(in srgb, var(--accent) 40%, transparent);
}

/* 适配黑暗模式的选择框选项 */
.styled-select option {
  background-color: var(--field-bg);
  color: var(--field-text);
}

/* 适配黑暗模式的表格滚动条 */
.table-wrapper::-webkit-scrollbar {
  width: 8px;
}
.table-wrapper::-webkit-scrollbar-track {
  background: var(--chip);
}
.table-wrapper::-webkit-scrollbar-thumb {
  background: var(--chip-border);
  border-radius: 4px;
}
.table-wrapper::-webkit-scrollbar-thumb:hover {
  background: var(--accent);
}
</style>
