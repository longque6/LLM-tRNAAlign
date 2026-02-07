<template>
  <div class="step-input">
    <div
      class="input-guard-wrapper"
      ref="inputGuardRef"
      @pointerdown.capture="handlePointerIntent"
      @focusin.capture="handleFocusIntent"
    >
      <GeneratorInput
        :seedSeq="seedSeq"
        :generateEnabled="generateEnabled"
        @update:seedSeq="handleUpdateSeedSeq"
        @update:generateEnabled="handleUpdateGenerateEnabled"
        @update:seqSelectMode="$emit('update:seqSelectMode', $event)"
        @update:selectedSequenceName="
          $emit('update:selectedSequenceName', $event)
        "
        @update:selectedSpecies="$emit('update:selectedSpecies', $event)"
      />
    </div>

    <transition name="fade">
      <div v-if="showGuard" class="input-guard-overlay">
        <div class="guard-card">
          <h3>Hold on—results are ready</h3>
          <p>
            Editing the seed sequence will clear the generated candidates and
            their readthrough predictions. Save or download them first, then
            continue.
          </p>
          <div class="guard-actions">
            <button class="btn ghost" type="button" @click="closeGuard">
              Cancel
            </button>
            <button class="btn danger" type="button" @click="confirmClear">
              Clear &amp; edit
            </button>
          </div>
        </div>
      </div>
    </transition>

    <div class="wizard-actions">
      <!-- 第一步没有上一步，用SVG图标替代 -->
      <div class="step-placeholder">
        <svg
          class="dna-icon"
          width="32"
          height="32"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M12 3C12 3 8 7 8 12C8 17 12 21 12 21"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
          />
          <path
            d="M12 3C12 3 16 7 16 12C16 17 12 21 12 21"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
          />
          <circle cx="12" cy="12" r="2" fill="currentColor" />
          <circle cx="8" cy="12" r="1" fill="currentColor" />
          <circle cx="16" cy="12" r="1" fill="currentColor" />
        </svg>
        <span class="placeholder-text">Sequence Input</span>
      </div>

      <button
        class="btn btn-primary"
        @click="$emit('next')"
        :disabled="!generateEnabled"
      >
        Next: Configuration
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from "vue";
import GeneratorInput from "../GeneratorInput.vue";

interface Props {
  seedSeq: string;
  generateEnabled: boolean;
  hasResults: boolean;
}

interface Emits {
  (e: "update:seedSeq", value: string): void;
  (e: "update:generateEnabled", value: boolean): void;
  (e: "update:seqSelectMode", value: "select" | "custom"): void;
  (e: "update:selectedSequenceName", value: string): void;
  (e: "update:selectedSpecies", value: string): void;
  (e: "request-clear-results"): void;
  (e: "next"): void;
}

const props = defineProps<Props>();
const emit = defineEmits<Emits>();

// 处理双向绑定更新
const handleUpdateSeedSeq = (value: string) => {
  emit("update:seedSeq", value);
};

const handleUpdateGenerateEnabled = (value: boolean) => {
  emit("update:generateEnabled", value);
};

const inputGuardRef = ref<HTMLElement | null>(null);
const showGuard = ref(false);
const awaitingClear = ref(false);

const interactiveSelector =
  "button, input, select, textarea, [role='tab'], [data-guardable='true']";

function targetInsideGuardArea(el: EventTarget | null) {
  if (!el || !(el instanceof HTMLElement)) return false;
  return inputGuardRef.value?.contains(el) ?? false;
}

function handlePointerIntent(event: PointerEvent) {
  if (!props.hasResults || showGuard.value) return;
  if (!targetInsideGuardArea(event.target)) return;
  const actionable = (event.target as HTMLElement).closest(interactiveSelector);
  if (!actionable) return;
  event.preventDefault();
  event.stopPropagation();
  showGuard.value = true;
}

function handleFocusIntent(event: FocusEvent) {
  if (!props.hasResults || showGuard.value) return;
  if (!targetInsideGuardArea(event.target)) return;
  event.preventDefault();
  (event.target as HTMLElement).blur();
  showGuard.value = true;
}

function closeGuard() {
  showGuard.value = false;
  awaitingClear.value = false;
}

function confirmClear() {
  awaitingClear.value = true;
  emit("request-clear-results");
}

watch(
  () => props.hasResults,
  (val) => {
    if (val) {
      showGuard.value = false;
      awaitingClear.value = false;
      return;
    }
    if (!val && awaitingClear.value) {
      showGuard.value = false;
      awaitingClear.value = false;
    }
  }
);
</script>

<style scoped>
.step-input {
  display: flex;
  flex-direction: column;
  height: 100%;
}

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

.wizard-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: auto;
  padding-top: 20px;
}

.step-placeholder {
  display: flex;
  align-items: center;
  gap: 12px;
  color: var(--text-soft);
  opacity: 0.7;
}

.dna-icon {
  color: var(--accent);
  opacity: 0.6;
}

.placeholder-text {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-soft);
}

.btn {
  position: relative;
  border-radius: 12px;
  padding: 12px 24px;
  font-weight: 700;
  letter-spacing: 0.2px;
  cursor: pointer;
  transition: all 0.2s ease;
  user-select: none;
  border: none;
  font-size: 14px;
}

.btn:active {
  transform: translateY(1px);
}

.btn-primary {
  background: var(--accent-grad);
  color: #fff;
  box-shadow: 0 4px 12px rgba(60, 80, 255, 0.25);
}

.btn-primary:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(60, 80, 255, 0.35);
}

.btn-primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
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

/* 响应式设计 */
@media (max-width: 768px) {
  .wizard-actions {
    flex-direction: column;
    gap: 16px;
  }

  .step-placeholder {
    order: 2;
  }

  .btn {
    order: 1;
    width: 100%;
  }

  .placeholder-text {
    display: none; /* 在小屏幕上隐藏文字，只显示图标 */
  }
}

.input-guard-wrapper {
  position: relative;
}
.input-guard-overlay {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.55);
  backdrop-filter: blur(6px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  border-radius: 14px;
  z-index: 10;
}
.guard-card {
  max-width: 520px;
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 24px;
  text-align: center;
  box-shadow: 0 20px 45px rgba(2, 6, 23, 0.35);
}
.guard-card h3 {
  margin: 0 0 12px;
  font-size: 1.2rem;
  color: var(--text-soft);
}
.guard-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.5;
}
.guard-actions {
  margin-top: 20px;
  display: flex;
  justify-content: center;
  gap: 12px;
}
.btn.ghost {
  background: transparent;
  color: var(--text-soft);
  border: 1px solid var(--card-border);
  padding: 10px 20px;
  border-radius: 999px;
}
.btn.ghost:hover {
  border-color: var(--accent);
}
.btn.danger {
  background: linear-gradient(135deg, #ef4444, #dc2626);
  color: #fff;
  border: none;
  padding: 11px 20px;
  border-radius: 999px;
  font-weight: 700;
  box-shadow: 0 8px 18px rgba(220, 38, 38, 0.35);
}
.btn.danger:hover {
  transform: translateY(-1px);
  box-shadow: 0 12px 24px rgba(220, 38, 38, 0.45);
}
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
