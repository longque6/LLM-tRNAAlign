<template>
  <section class="card">
    <h2 class="card-title">Actions</h2>

    <div class="btn-row">
      <button
        class="btn btn-primary"
        :disabled="loading || !generateEnabled"
        @click="$emit('run')"
        :title="mode === 'one' ? 'Generate one' : 'Generate batch and re-rank'"
      >
        Generate
      </button>
      <button
        class="btn btn-outline"
        :disabled="loading"
        @click="$emit('reset')"
      >
        Reset
      </button>
    </div>

    <div v-if="loading" class="loading">Generating…</div>
    <div v-if="error" class="error">Error: {{ error }}</div>
    <div v-if="elapsedSec !== null" class="elapsed">
      Elapsed: {{ elapsedSec }}s
    </div>
  </section>
</template>

<script setup lang="ts">
defineProps<{
  mode: "one" | "batch";
  loading: boolean;
  generateEnabled: boolean;
  error: string;
  elapsedSec: number | null;
}>();

defineEmits<{
  run: [];
  reset: [];
}>();
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
</style>
