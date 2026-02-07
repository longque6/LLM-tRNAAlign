<template>
  <div class="step-results">
    <div class="step-card">
      <h3 class="step-card-title">Generation Results</h3>

      <!-- Mode toggle + batch controls -->
      <div class="mode-controls">
        <div class="tabs mode-tabs" role="tablist" aria-label="Generation mode">
          <button
            class="tab"
            :class="{ active: mode === 'one' }"
            @click="handleModeChange('one')"
            role="tab"
            :aria-selected="mode === 'one'"
            title="Single: generate one sequence"
          >
            <span class="tab-text">Single</span>
          </button>
          <button
            class="tab"
            :class="{ active: mode === 'batch' }"
            @click="handleModeChange('batch')"
            role="tab"
            :aria-selected="mode === 'batch'"
            title="Batch: generate multiple sequences"
          >
            <span class="tab-text">Batch</span>
          </button>
          <div
            class="tab-indicator"
            :style="modeIndicatorStyle"
            aria-hidden="true"
          ></div>
        </div>

        <div v-if="mode === 'batch'" class="mode-batch" aria-live="polite">
          <label class="num-label" for="num-samples-slider">
            Number of sequences (1–10)
          </label>
          <div class="slider-input-group">
            <input
              id="num-samples-slider"
              type="range"
              min="1"
              max="10"
              step="1"
              :value="numSamplesValue"
              @input="onNumSamplesInput($event)"
              class="slider"
              title="Number of sequences to generate (max 10)"
            />
            <input
              type="number"
              min="1"
              max="10"
              step="1"
              :value="numSamplesValue"
              @input="onNumSamplesInput($event)"
              class="number-input"
              title="Number of sequences to generate (max 10)"
            />
          </div>
          <small class="muted exponential-note">
            Increasing the batch size will slow generation dramatically (roughly
            exponential).
          </small>
        </div>
      </div>

      <!-- Parameter summary -->
      <div class="params-summary" v-if="showParamsSummary">
        <details>
          <summary class="params-summary-header">
            <span>Current configuration</span>
            <span class="toggle-icon">▼</span>
          </summary>
          <div class="params-content">
            <div class="param-group">
              <h4>Base settings</h4>
              <div class="param-grid">
                <div class="param-item">
                  <span class="param-label">Mode:</span>
                  <span class="param-value">{{ formatMode(mode) }}</span>
                </div>
                <div class="param-item">
                  <span class="param-label">Style level:</span>
                  <span class="param-value">{{
                    getStyleLabel(styleLevel)
                  }}</span>
                </div>
                <div class="param-item">
                  <span class="param-label">Samples:</span>
                  <span class="param-value">{{ numSamples }}</span>
                </div>
              </div>
            </div>

            <div class="param-group" v-if="showAdvanced && adv">
              <h4>Advanced parameters</h4>
              <div class="param-grid">
                <div class="param-item" v-for="(value, key) in adv" :key="key">
                  <span class="param-label">{{ formatParamLabel(key) }}:</span>
                  <span class="param-value">{{ value }}</span>
                </div>
              </div>
            </div>

            <div
              class="param-group"
              v-if="
                (freezeRegions && freezeRegions.length > 0) ||
                (preferRegions && preferRegions.length > 0)
              "
            >
              <h4>Region preferences</h4>
              <div class="param-grid">
                <div
                  class="param-item"
                  v-if="freezeRegions && freezeRegions.length > 0"
                >
                  <span class="param-label">Frozen regions:</span>
                  <span class="param-value">{{
                    freezeRegions.join(", ")
                  }}</span>
                </div>
                <div
                  class="param-item"
                  v-if="preferRegions && preferRegions.length > 0"
                >
                  <span class="param-label">Preferred regions:</span>
                  <span class="param-value">{{
                    preferRegions.join(", ")
                  }}</span>
                </div>
              </div>
            </div>
          </div>
        </details>
      </div>

      <!-- Action buttons -->
      <GeneratorActionButtons
        :mode="mode"
        :loading="loading"
        :generateEnabled="generateEnabled"
        :error="error"
        :elapsedSec="elapsedSec"
        @run="$emit('run')"
        @reset="$emit('reset')"
      />

      <!-- 结果显示 -->
      <GeneratorResults
        v-if="showResults"
        :mode="mode"
        :oneResult="oneResult"
        :batchResult="batchResult"
        :seedSeq="seedSeq"
        @copy="$emit('copy', $event)"
        @download="onDownload"
      />
    </div>

    <div class="wizard-actions">
      <button class="btn btn-outline" @click="$emit('back')">Back</button>
      <button class="btn btn-primary" @click="$emit('reset')">
        Start Over
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import GeneratorActionButtons from "../GeneratorActionButtons.vue";
import GeneratorResults from "../GeneratorResults.vue";

interface Props {
  mode: "one" | "batch";
  loading: boolean;
  generateEnabled: boolean;
  error: string;
  elapsedSec: number | null;
  oneResult: string;
  batchResult: string[];
  seedSeq: string;
  // 参数相关的props - 设为可选，避免递归更新
  showAdvanced?: boolean;
  styleLevel?: number;
  styleLabels?: string[];
  numSamples?: number;
  adv?: Record<string, any>;
  freezeRegions?: string[];
  preferRegions?: string[];
}

interface Emits {
  (e: "run"): void;
  (e: "reset"): void;
  (e: "back"): void;
  (e: "update:mode", value: "one" | "batch"): void;
  (e: "update:numSamples", value: number): void;
  (e: "copy", value: string): void;
  (e: "download", value: { sequences: string[]; filename: string }): void;
}

const props = withDefaults(defineProps<Props>(), {
  showAdvanced: false,
  styleLevel: 2,
  styleLabels: () => [
    "Ultra conservative",
    "Conservative",
    "Balanced",
    "Aggressive",
    "Ultra aggressive",
  ],
  numSamples: 3,
  adv: () => ({}),
  freezeRegions: () => [],
  preferRegions: () => [],
});

const emit = defineEmits<Emits>();

const numSamplesValue = computed(() => props.numSamples ?? 1);
const modeIndicatorStyle = computed(() => ({
  transform: props.mode === "one" ? "translateX(0%)" : "translateX(100%)",
}));

// 计算是否显示参数摘要
const showParamsSummary = computed(() => {
  return (
    props.showAdvanced ||
    props.freezeRegions?.length > 0 ||
    props.preferRegions?.length > 0
  );
});

// 计算是否显示结果
const showResults = computed(() => {
  return !props.loading && (props.oneResult || props.batchResult.length > 0);
});

// 获取样式标签
function getStyleLabel(level?: number): string {
  if (level === undefined) return "Unknown";
  return props.styleLabels?.[level] || `Level ${level}`;
}

// 格式化参数标签
function formatParamLabel(key: string): string {
  const labels: Record<string, string> = {
    temperature: "Temperature",
    top_p: "Top P",
    rounds: "Rounds",
    min_hd: "Min Hamming distance",
    rerank_min_hd: "Rerank min Hamming distance",
    gc_low: "GC lower bound",
    gc_high: "GC upper bound",
  };
  return labels[key] || key;
}

function formatMode(mode: "one" | "batch"): string {
  return mode === "batch" ? "Batch" : "Single";
}

function onNumSamplesInput(event: Event) {
  const nextValue = Number((event.target as HTMLInputElement).value);
  if (Number.isNaN(nextValue)) return;
  const clamped = Math.max(1, Math.min(10, nextValue));
  emit("update:numSamples", clamped);
}

function handleModeChange(nextMode: "one" | "batch") {
  if (props.mode === nextMode) return;
  emit("update:mode", nextMode);
}

function onDownload(payload?: { sequences: string[]; filename: string }) {
  if (!payload) return;
  emit("download", payload);
}
</script>

<style scoped>
.step-card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 16px;
}

.step-card-title {
  margin: 0 0 16px;
  font-size: 1.12rem;
  font-weight: 800;
  color: var(--text-soft);
}

/* 参数摘要样式 */
.params-summary {
  margin-bottom: 20px;
  border: 1px solid var(--card-border);
  border-radius: 8px;
  overflow: hidden;
}

.params-summary-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: var(--chip);
  cursor: pointer;
  font-weight: 600;
  color: var(--text-soft);
}

.toggle-icon {
  transition: transform 0.2s;
}

details[open] .toggle-icon {
  transform: rotate(180deg);
}

.params-content {
  padding: 16px;
  background: var(--card-bg);
}

.param-group {
  margin-bottom: 16px;
}

.param-group:last-child {
  margin-bottom: 0;
}

.param-group h4 {
  margin: 0 0 8px 0;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-soft);
}

.param-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 8px;
}

.param-item {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
  border-bottom: 1px solid var(--card-border);
}

.param-label {
  font-weight: 500;
  color: var(--text-soft);
}

.param-value {
  color: var(--accent);
  font-weight: 600;
}

.mode-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  align-items: flex-start;
  margin-bottom: 16px;
}

.mode-tabs {
  flex: 1 1 220px;
  margin-bottom: 0;
}

.mode-batch {
  flex: 1 1 280px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.num-label {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-soft);
  display: block;
}

.exponential-note {
  color: #f97316;
  font-weight: 500;
}

.wizard-actions {
  display: flex;
  justify-content: space-between;
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid var(--card-border);
}

.slider-input-group {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  width: 100%;
  max-width: none;
}

.slider {
  flex: 1;
  height: 6px;
  border-radius: 3px;
  background: var(--chip-border);
  outline: none;
  -webkit-appearance: none;
  appearance: none;
}
.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--accent);
  cursor: pointer;
  border: 2px solid var(--field-bg);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}
.slider::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--accent);
  cursor: pointer;
  border: 2px solid var(--field-bg);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}

.number-input {
  width: 80px;
  padding: 8px 10px;
  border-radius: 10px;
  border: 1px solid var(--card-border);
  background: var(--field-bg);
  color: var(--field-text);
  font-weight: 600;
}
.number-input:focus {
  outline: 2px solid color-mix(in srgb, var(--accent) 30%, transparent);
  border-color: transparent;
}
</style>
