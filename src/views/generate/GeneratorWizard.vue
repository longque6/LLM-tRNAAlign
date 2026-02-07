# src/views/generate/GeneratorWizard.vue
<template>
  <div class="wizard">
    <div class="wizard-header">
      <div class="wizard-steps">
        <div
          v-for="step in steps"
          :key="step.id"
          class="wizard-step"
          :class="{
            active: currentStep === step.id,
            completed: step.completed,
          }"
          @click="handleStepClick(step.id)"
        >
          <div class="step-indicator">
            <span v-if="step.completed">✓</span>
            <span v-else>{{ step.id }}</span>
          </div>
          <div class="step-info">
            <div class="step-title">{{ step.title }}</div>
            <div class="step-description">{{ step.description }}</div>
          </div>
        </div>
      </div>
    </div>

    <div class="wizard-content">
      <slot :name="`step${currentStep}`"></slot>
    </div>
  </div>
</template>

<script setup lang="ts">
interface WizardStep {
  id: number;
  title: string;
  description: string;
  completed: boolean;
}

interface Props {
  currentStep: number;
  steps: WizardStep[];
}

defineProps<Props>();
const emit = defineEmits<{
  stepChange: [step: number];
}>();

function handleStepClick(stepId: number) {
  emit("stepChange", stepId);
}
</script>

<style scoped>
.wizard {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  backdrop-filter: blur(4px);
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 8px 26px rgba(0, 0, 0, 0.08);
}

.wizard-header {
  padding: 24px 24px 0;
}

.wizard-steps {
  display: flex;
  justify-content: space-between;
  position: relative;
  margin-bottom: 32px;
}

.wizard-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  z-index: 2;
  flex: 1;
  cursor: pointer;
}

.wizard-step:not(:last-child)::after {
  content: "";
  position: absolute;
  top: 16px;
  left: 60%;
  right: -40%;
  height: 2px;
  background: var(--chip-border);
  z-index: 1;
}

.wizard-step.completed:not(:last-child)::after {
  background: var(--accent);
}

.step-indicator {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 14px;
  margin-bottom: 8px;
  background: var(--chip);
  border: 2px solid var(--chip-border);
  color: var(--muted);
  transition: all 0.3s ease;
}

.step-indicator.active {
  background: var(--accent-grad);
  border-color: var(--accent);
  color: white;
}

.step-indicator.completed {
  background: var(--accent);
  border-color: var(--accent);
  color: white;
}

.step-info {
  text-align: center;
}

.step-title {
  font-weight: 700;
  font-size: 14px;
  color: var(--text-soft);
  margin-bottom: 2px;
}

.step-description {
  font-size: 12px;
  color: var(--muted);
}

.wizard-content {
  padding: 0 24px 24px;
}

@media (max-width: 768px) {
  .wizard-steps {
    flex-direction: column;
    gap: 16px;
  }

  .wizard-step:not(:last-child)::after {
    display: none;
  }

  .step-indicator {
    width: 28px;
    height: 28px;
    font-size: 12px;
  }

  .wizard-content {
    padding: 0 16px 16px;
  }
}
</style>
