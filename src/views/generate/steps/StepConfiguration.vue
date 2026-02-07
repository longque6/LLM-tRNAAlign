<template>
  <div class="step-configuration">
    <!-- 配置面板 -->
    <GeneratorConfigPanel
      :showAdvanced="showAdvanced"
      :styleLevel="styleLevel"
      :styleLabels="styleLabels"
      @toggleAdvanced="$emit('toggleAdvanced')"
      @update:styleLevel="$emit('update:styleLevel', $event)"
    />

    <!-- 高级配置 -->
    <GeneratorAdvanced
      v-if="showAdvanced"
      :show="showAdvanced"
      :adv="adv"
      :freezeRegions="freezeRegions"
      :preferRegions="preferRegions"
      :freezePositionsText="freezePositionsText"
      :preferPositionsText="preferPositionsText"
      :forcePositionsText="forcePositionsText"
      :uiNote="uiNote"
      :REGION_OPTIONS="REGION_OPTIONS"
      :isSelectMode="isSelectMode"
      :isCustomMode="isCustomMode"
      :selectedSequenceName="selectedSequenceName"
      :selectedSpecies="selectedSpecies"
      @update:adv="$emit('update:adv', $event)"
      @update:freezeRegions="$emit('update:freezeRegions', $event)"
      @update:preferRegions="$emit('update:preferRegions', $event)"
      @update:freezePositionsText="$emit('update:freezePositionsText', $event)"
      @update:preferPositionsText="$emit('update:preferPositionsText', $event)"
      @update:forcePositionsText="$emit('update:forcePositionsText', $event)"
      @update:uiNote="$emit('update:uiNote', $event)"
    />
  </div>

  <div class="wizard-actions">
    <button class="btn btn-outline" @click="$emit('back')">Back</button>
    <button class="btn btn-primary" @click="$emit('next')">
      Next: Generation Results
    </button>
  </div>
</template>

<script setup lang="ts">
import GeneratorConfigPanel from "../GeneratorConfigPanel.vue";
import GeneratorAdvanced from "../GeneratorAdvanced.vue";

interface Props {
  mode: "one" | "batch";
  showAdvanced: boolean;
  styleLevel: number;
  styleLabels: string[];
  adv: any;
  freezeRegions: string[];
  preferRegions: string[];
  freezePositionsText: string;
  preferPositionsText: string;
  forcePositionsText: string;
  uiNote: string;
  REGION_OPTIONS: string[];
  isSelectMode: boolean;
  isCustomMode: boolean;
  selectedSequenceName: string;
  selectedSpecies: string;
}

interface Emits {
  (e: "update:styleLevel", value: number): void;
  (e: "toggleAdvanced"): void;
  (e: "update:adv", value: any): void;
  (e: "update:freezeRegions", value: string[]): void;
  (e: "update:preferRegions", value: string[]): void;
  (e: "update:freezePositionsText", value: string): void;
  (e: "update:preferPositionsText", value: string): void;
  (e: "update:forcePositionsText", value: string): void;
  (e: "update:uiNote", value: string): void;
  (e: "back"): void;
  (e: "next"): void;
}

defineProps<Props>();
defineEmits<Emits>();
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

.wizard-actions {
  display: flex;
  justify-content: space-between;
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid var(--card-border);
}
</style>
