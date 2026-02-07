<template>
  <div>
    <NavBar />
    <main class="page">
      <header class="hero">
        <h1>tRNA Generator</h1>
        <p>Generate candidate sequences with AYLM-tRNA-LLM.</p>
      </header>

      <!-- Wizard 组件 -->
      <div class="wizard-container">
        <GeneratorWizard
          :current-step="currentStep"
          :steps="wizardSteps"
          @step-change="handleStepChange"
        >
          <!-- 步骤1: 序列输入 -->
          <template #step1>
            <StepInput
              :seedSeq="seedSeq"
              :generateEnabled="generateEnabled"
              :hasResults="hasResults"
              @update:seedSeq="handleUpdateSeedSeq"
              @update:generateEnabled="handleUpdateGenerateEnabled"
              @update:seqSelectMode="handleSeqSelectModeUpdate"
              @update:selectedSequenceName="handleSelectedSequenceNameUpdate"
              @update:selectedSpecies="handleSelectedSpeciesUpdate"
              @request-clear-results="handleClearGeneratedResults"
              @next="handleStep1Next"
            />
          </template>

          <!-- 步骤2: 参数配置 -->
          <template #step2>
            <StepConfiguration
              :mode="mode"
              :showAdvanced="showAdvanced"
              :styleLevel="styleLevel"
              :styleLabels="styleLabels"
              :adv="adv"
              :freezeRegions="freezeRegions"
              :preferRegions="preferRegions"
              :freezePositionsText="freezePositionsText"
              :preferPositionsText="preferPositionsText"
              :forcePositionsText="forcePositionsText"
              :uiNote="uiNote"
              :REGION_OPTIONS="REGION_OPTIONS"
              :isSelectMode="seqSelectMode === 'select'"
              :isCustomMode="seqSelectMode === 'custom'"
              :selectedSequenceName="selectedSequenceName"
              :selectedSpecies="selectedSpecies"
              @update:styleLevel="handleStyleLevelUpdate"
              @toggleAdvanced="showAdvanced = !showAdvanced"
              @update:adv="handleAdvUpdate"
              @update:freezeRegions="freezeRegions = $event"
              @update:preferRegions="preferRegions = $event"
              @update:freezePositionsText="freezePositionsText = $event"
              @update:preferPositionsText="preferPositionsText = $event"
              @update:forcePositionsText="forcePositionsText = $event"
              @update:uiNote="uiNote = $event"
              @run="onRun"
              @reset="handleReset"
              @back="handleBackToStep1"
              @next="handleStep2Next"
            />
          </template>

          <!-- 步骤3: 生成结果 -->
          <template #step3>
            <StepResults
              :mode="mode"
              :loading="loading"
              :generateEnabled="generateEnabled"
              :error="error"
              :elapsedSec="elapsedSec"
              :oneResult="oneResult"
              :batchResult="batchResult"
              :seedSeq="seedSeq"
              :showAdvanced="showAdvanced"
              :styleLevel="styleLevel"
              :styleLabels="styleLabels"
              :numSamples="numSamples"
              :adv="adv"
              :freezeRegions="freezeRegions"
              :preferRegions="preferRegions"
              @run="onRun"
              @reset="handleReset"
              @back="handleBackToStep2"
              @update:mode="handleUpdateMode"
              @update:numSamples="handleUpdateNumSamples"
              @copy="copy"
              @download="downloadFasta"
            />
          </template>
        </GeneratorWizard>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import NavBar from "../../components/NavBar.vue";
import GeneratorWizard from "./GeneratorWizard.vue";
import StepInput from "./steps/StepInput.vue";
import StepConfiguration from "./steps/StepConfiguration.vue";
import StepResults from "./steps/StepResults.vue";
import { useGeneratorLogic } from "@/composables/generate/useGeneratorLogic";

// 定义 WizardStep 接口
interface WizardStep {
  id: number;
  title: string;
  description: string;
  completed: boolean;
}

// Wizard 步骤管理
const currentStep = ref(1);
const step2Visited = ref(false);
const step3Visited = ref(false);

const wizardSteps = computed((): WizardStep[] => [
  {
    id: 1,
    title: "Sequence Input",
    description: "Select or input seed sequence",
    completed: !!(seedSeq.value && seedSeq.value.length > 0),
  },
  {
    id: 2,
    title: "Configuration",
    description: "Adjust generation parameters",
    completed: !!(
      step2Visited.value &&
      seedSeq.value &&
      seedSeq.value.length > 0
    ),
  },
  {
    id: 3,
    title: "Generation Results",
    description: "View and analyze results",
    completed: !!(
      step3Visited.value &&
      (oneResult.value || batchResult.value.length > 0)
    ),
  },
]);

function handleStepChange(step: number) {
  // 验证是否可以跳转到该步骤
  if (step === 1) {
    currentStep.value = 1;
  } else if (step === 2) {
    // 跳转到步骤2需要验证序列
    if (!seedSeq.value || seedSeq.value.length === 0) {
      alert("Please input or select a seed sequence first");
      return;
    }
    step2Visited.value = true;
    currentStep.value = step;
  } else if (step === 3) {
    // 跳转到步骤3需要验证序列和步骤2
    if (!seedSeq.value || seedSeq.value.length === 0) {
      alert("Please complete sequence input first");
      return;
    }
    if (!step2Visited.value) {
      alert("Please complete configuration step first");
      return;
    }
    step3Visited.value = true;
    currentStep.value = step;
  }
}

function handleStep1Next() {
  if (seedSeq.value && seedSeq.value.length > 0) {
    step2Visited.value = true;
    currentStep.value = 2;
  } else {
    alert("Please input or select a seed sequence first");
  }
}

function handleStep2Next() {
  step3Visited.value = true;
  currentStep.value = 3;
}

// 新增：处理回到步骤1的逻辑
function handleBackToStep1() {
  currentStep.value = 1;
}

// 新增：处理回到步骤2的逻辑
function handleBackToStep2() {
  // 保留步骤1和2的完成状态，但重置步骤3的访问状态
  step3Visited.value = false;
  currentStep.value = 2;
}

// 状态定义
const mode = ref<"one" | "batch">("one");
const seedSeq = ref("");
const numSamples = ref(3);
const loading = ref(false);
const error = ref("");
const elapsedSec = ref<number | null>(null);
const oneResult = ref("");
const batchResult = ref<string[]>([]);
const showAdvanced = ref(false);
const generateEnabled = ref(false);

// 序列选择相关状态
const seqSelectMode = ref<"select" | "custom">("select");
const selectedSequenceName = ref("");
const selectedSpecies = ref("");

// 样式参数
const styleLevel = ref(2);
const styleLabels = [
  "Ultra conservative",
  "Conservative",
  "Balanced",
  "Aggressive",
  "Ultra aggressive",
];

// 高级参数
const adv = ref({
  temperature: 1.9,
  top_p: 0.9,
  rounds: 3,
  min_hd: 2,
  rerank_min_hd: 6,
  gc_low: 0.42,
  gc_high: 0.66,
});

// 区域控制
const REGION_OPTIONS = [
  "AA_Arm_5prime",
  "D_Loop",
  "Anticodon_Arm",
  "Variable_Loop",
  "T_Arm",
  "AA_Arm_3prime",
];
const freezeRegions = ref<string[]>([]);
const preferRegions = ref<string[]>(["Variable_Loop"]);
const freezePositionsText = ref("");
const preferPositionsText = ref("");
const forcePositionsText = ref("");
const uiNote = ref("");

// 隐藏参数
const hiddenParams = ref({
  top_k: 0,
  mask_frac: 0.2,
  mask_k: 5,
});

// 使用组合式函数处理业务逻辑
const { applyStyle, onRun, resetAll, clearResults, copy, downloadFasta } =
  useGeneratorLogic({
    mode,
    seedSeq,
    numSamples,
    loading,
  error,
  elapsedSec,
  oneResult,
  batchResult,
  generateEnabled,
  seqSelectMode,
  selectedSequenceName,
  selectedSpecies,
  adv,
  freezeRegions,
  preferRegions,
  freezePositionsText,
  preferPositionsText,
  forcePositionsText,
  styleLevel,
  hiddenParams,
    REGION_OPTIONS,
  });

const hasResults = computed(
  () => !!oneResult.value || batchResult.value.length > 0
);

// 重置处理
function handleReset() {
  resetAll();
  currentStep.value = 1;
  step2Visited.value = false;
  step3Visited.value = false;
}

function handleClearGeneratedResults() {
  clearResults();
  step3Visited.value = false;
}

// 事件处理函数
function handleUpdateMode(value: "one" | "batch") {
  mode.value = value;
}

function handleUpdateSeedSeq(value: string) {
  seedSeq.value = value;
  // 仅在没有现有生成结果时才重置访问状态，避免加载缓存时触发
  if ((!value || value.length === 0) && !hasResults.value) {
    step2Visited.value = false;
    step3Visited.value = false;
  }
}

function handleUpdateNumSamples(value: number) {
  numSamples.value = value;
}

function handleUpdateGenerateEnabled(value: boolean) {
  generateEnabled.value = value;
}

function handleSeqSelectModeUpdate(newMode: "select" | "custom") {
  seqSelectMode.value = newMode;
  if (newMode === "custom") {
    selectedSequenceName.value = "";
  }
}

function handleSelectedSequenceNameUpdate(name: string) {
  selectedSequenceName.value = name;
}

function handleSelectedSpeciesUpdate(species: string) {
  selectedSpecies.value = species;
}

function handleStyleLevelUpdate(level: number) {
  styleLevel.value = level;
  applyStyle();
}

function handleAdvUpdate(newAdv: any) {
  adv.value = { ...adv.value, ...newAdv };
}

// 初始化应用样式
applyStyle();
</script>

<style>
/* 全局样式 */
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  border: 1px solid var(--border, #ddd);
  border-radius: 999px;
  font-size: 12px;
}
.card-subtitle {
  margin: 16px 0 8px;
  font-size: 14px;
  font-weight: 600;
}
.muted {
  color: #777;
  font-size: 12px;
}
.grid-3 {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}
@media (max-width: 900px) {
  .grid-3 {
    grid-template-columns: 1fr;
  }
}
.note {
  margin-top: 8px;
  font-size: 12px;
  color: #2563eb;
}

:root {
  --bg-a: #6f63e6;
  --bg-b: #6a4fb1;
  --text-strong: #0f172a;
  --text-soft: #1f2937;
  --muted: #475569;

  --card-bg: rgba(255, 255, 255, 0.92);
  --card-border: rgba(17, 24, 39, 0.12);

  --field-bg: #ffffff;
  --field-text: #0f172a;
  --placeholder: #6b7280;

  --accent: #5b7cfa;
  --accent-2: #3b5bff;
  --accent-grad: linear-gradient(135deg, #7c9bff, #5b7cfa);
  --chip: #eef2ff;
  --chip-border: #c7d2fe;
  --danger: #d93025;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg-a: #4635a8;
    --bg-b: #322470;
    --text-strong: #e5e7eb;
    --text-soft: #e2e8f0;
    --muted: #cbd5e1;

    --card-bg: rgba(18, 22, 40, 0.94);
    --card-border: rgba(148, 163, 184, 0.18);

    --field-bg: #0f1220;
    --field-text: #e5e7eb;
    --placeholder: #a3b0c3;

    --accent: #9aaaff;
    --accent-2: #7f8fff;
    --accent-grad: linear-gradient(135deg, #a9b7ff, #8b9bff);
    --chip: #1f2545;
    --chip-border: #334155;
  }
}

.page {
  min-height: 100vh;
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 96px 20px 40px;
  color: var(--text-strong);
  box-sizing: border-box;
}

.hero {
  max-width: 1000px;
  margin: 0 auto 16px;
}
.hero h1 {
  color: #fff;
  font-size: 2rem;
  font-weight: 800;
  margin: 0 0 6px;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.25);
}
.hero p {
  color: rgba(255, 255, 255, 0.92);
  margin: 0;
}

.wizard-container {
  max-width: 1200px;
  margin: 0 auto;
}

/* 原有的卡片样式保持不变，用于步骤内组件 */
.card {
  width: 100%;
  max-width: 1000px;
  margin: 16px auto 0;
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  backdrop-filter: blur(4px);
  border-radius: 14px;
  padding: 16px;
  box-shadow: 0 8px 26px rgba(0, 0, 0, 0.08);
  box-sizing: border-box;
}
.card-title {
  margin: 0 0 12px;
  font-size: 1.12rem;
  font-weight: 800;
  color: var(--text-soft);
}

.tabs {
  position: relative;
  display: flex;
  gap: 6px;
  background: var(--chip);
  border: 1px solid var(--chip-border);
  border-radius: 12px;
  padding: 6px;
  overflow: hidden;
  user-select: none;
}
.tab {
  position: relative;
  flex: 1;
  border: none;
  background: transparent;
  color: var(--text-soft);
  font-weight: 800;
  padding: 10px 16px;
  border-radius: 10px;
  cursor: pointer;
  z-index: 2;
  transition: color 0.2s ease;
}
.tab:hover {
  color: var(--accent);
}
.tab.active {
  color: #fff;
}
.tab-text {
  position: relative;
  z-index: 2;
}
.tab-indicator {
  position: absolute;
  top: 4px;
  left: 4px;
  width: calc(50% - 8px);
  height: calc(100% - 8px);
  background: var(--accent-grad);
  border-radius: 10px;
  z-index: 1;
  transition: transform 0.3s ease;
  box-shadow: 0 6px 18px rgba(60, 80, 255, 0.25);
}
@media (prefers-color-scheme: dark) {
  .tabs {
    background: rgba(30, 30, 50, 0.6);
  }
}

.form {
  margin-top: 12px;
}
.label {
  font-weight: 800;
  margin-bottom: 6px;
  display: block;
  color: var(--text-soft);
}
.input,
.textarea {
  width: 90%;
  background: var(--field-bg);
  border: 2px solid var(--card-border);
  color: var(--field-text);
  border-radius: 10px;
  padding: 12px 14px;
  font-size: 0.95rem;
  outline: none;
}
.input::placeholder,
.textarea::placeholder {
  color: var(--placeholder);
}
.input:focus,
.textarea:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 25%, transparent);
}
.textarea {
  resize: vertical;
}
.muted {
  color: var(--muted);
  font-size: 12px;
  display: block;
  margin-top: 6px;
}

.grid-3 {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}
.gc {
  display: flex;
  align-items: center;
  gap: 6px;
}

.macro {
  margin-top: 6px;
  background: color-mix(in srgb, var(--chip) 70%, transparent);
  border: 1px solid var(--chip-border);
  padding: 12px;
  border-radius: 12px;
}
.macro-head {
  display: flex;
  justify-content: space-between;
  font-weight: 800;
  margin-bottom: 8px;
}
.macro-val {
  color: var(--accent);
}
.macro input[type="range"] {
  width: 100%;
}

.btn-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 12px;
}
.btn {
  position: relative;
  border-radius: 12px;
  padding: 12px 18px;
  font-weight: 700;
  letter-spacing: 0.2px;
  cursor: pointer;
  transition: transform 0.06s ease, box-shadow 0.2s ease, filter 0.15s ease;
  user-select: none;
  -webkit-tap-highlight-color: transparent;
}
.btn:active {
  transform: translateY(1px);
}

.btn-primary {
  background: var(--accent-grad);
  color: #fff;
  border: 0;
  box-shadow: 0 8px 20px rgba(60, 80, 255, 0.25);
}
.btn-primary:hover {
  filter: brightness(1.03);
}
.btn-primary:disabled {
  opacity: 0.65;
  cursor: not-allowed;
  box-shadow: none;
}

.btn-outline {
  background: transparent;
  border: 2px solid var(--accent);
  color: var(--accent);
}
.btn-outline:hover {
  background: color-mix(in srgb, var(--accent) 15%, transparent);
}

.btn-plain {
  background: transparent;
  border: 1px dashed var(--chip-border);
  color: var(--text-strong);
}
.btn-xs {
  padding: 6px 10px;
  font-size: 12px;
  border-radius: 8px;
}

.loading {
  color: var(--muted);
  margin-top: 8px;
}
.error {
  color: var(--danger);
  margin-top: 8px;
}
.elapsed {
  color: var(--muted);
  margin-top: 6px;
}
.note {
  color: #2563eb;
  margin-top: 6px;
}

.list {
  list-style: none;
  margin: 0;
  padding: 0;
}
.mono-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  padding: 8px;
  border: 1px solid var(--card-border);
  border-radius: 10px;
}
.mono {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 8px;
  border: 1px solid var(--card-border);
  border-radius: 10px;
}
.seq {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
    "Liberation Mono", "Courier New", monospace;
  word-break: break-all;
}

@media (max-width: 860px) {
  .grid-3 {
    grid-template-columns: 1fr;
  }
}
</style>
