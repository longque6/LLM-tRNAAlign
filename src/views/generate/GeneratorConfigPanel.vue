<template>
  <section class="card">
    <h2 class="card-title">Style</h2>
    <div class="macro">
      <div class="macro-head">
        <span>Conservative → Aggressive</span>
        <span class="macro-val">{{ styleLabels[styleLevel] }}</span>
      </div>
      <input
        type="range"
        min="0"
        max="4"
        step="1"
        :value="styleLevel"
        @input="handleStyleInput"
        :title="`Overall style: ${styleLabels[styleLevel]}`"
      />
      <small class="muted">
        The slider adjusts generation aggressiveness while keeping sensible
        defaults.
      </small>
    </div>

    <div class="btn-row">
      <button
        class="btn btn-plain"
        @click="$emit('toggleAdvanced')"
        :aria-expanded="showAdvanced"
      >
        {{ showAdvanced ? "Hide Advanced" : "Advanced" }}
      </button>
    </div>
  </section>
</template>

<script setup lang="ts">
defineProps<{
  showAdvanced: boolean;
  styleLevel: number;
  styleLabels: string[];
}>();

const emit = defineEmits<{
  toggleAdvanced: [];
  "update:styleLevel": [value: number];
}>();

function handleStyleInput(event: Event) {
  const target = event.target as HTMLInputElement;
  emit("update:styleLevel", parseInt(target.value, 10));
}
</script>

<style scoped>
.card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 14px;
  padding: 16px;
  box-shadow: 0 8px 26px rgba(0, 0, 0, 0.08);
}
.card-title {
  margin: 0 0 12px;
  font-size: 1.12rem;
  font-weight: 800;
  color: var(--text-soft);
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

.muted {
  color: var(--muted);
  font-size: 12px;
  display: block;
  margin-top: 6px;
}
</style>
