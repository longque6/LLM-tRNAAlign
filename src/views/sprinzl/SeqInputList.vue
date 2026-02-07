<template>
  <div class="seq-list">
    <div v-for="(item, i) in localList" :key="i" class="seq-item">
      <div class="seq-header">
        <span>Sequence {{ i + 1 }}</span>
        <button class="btn-remove" @click="removeSeq(i)">✕</button>
      </div>

      <div class="seq-row">
        <textarea
          v-model="item.seq"
          class="seq-input"
          :placeholder="`Enter tRNA sequence #${i + 1}`"
          @input="onSeqInput(i)"
        />

        <!-- 右侧小警告，不占行高 -->
        <span v-if="warnFlags[i]" class="warn-msg">⚠︎ invalid → N</span>
      </div>
    </div>

    <button class="btn-add-seq" @click="addSeq">＋ Add Sequence</button>
  </div>
</template>

<script lang="ts" setup>
import { reactive, watch, onMounted } from "vue";

interface SeqEntry {
  seq: string;
  anticode: string;
}

const props = defineProps<{
  modelValue: SeqEntry[];
  defaultSeq: string;
}>();
const emit = defineEmits<{
  (e: "update:modelValue", value: SeqEntry[]): void;
}>();

/* ========= 本地状态 ========= */
const localList = reactive<SeqEntry[]>(props.modelValue.map((s) => ({ ...s })));
const warnFlags = reactive<boolean[]>(Array(localList.length).fill(false));

/* ========= 序列规范化函数（与主组件保持一致） ========= */
function normalizeSeq(raw: string): string {
  return (raw || "")
    .toUpperCase()
    .replace(/\s+/g, "")
    .replace(/T/g, "U")
    .replace(/[^AUGC]/g, ""); // 移除非 A/U/G/C
}

/* ========= 单向同步：子 → 父 ========= */
watch(
  localList,
  (v) =>
    emit(
      "update:modelValue",
      v.map((s) => ({ ...s }))
    ),
  { deep: true }
);

/* ========= 事件处理 ========= */
function onSeqInput(idx: number) {
  const raw = localList[idx].seq;
  // 使用与主组件相同的规范化逻辑
  const sanitized = normalizeSeq(raw);
  warnFlags[idx] = raw !== sanitized;
  localList[idx].seq = sanitized;
}

function addSeq() {
  localList.push({ seq: props.defaultSeq, anticode: "ACG" });
  warnFlags.push(false); // 对齐 warn 数组
}

function removeSeq(i: number) {
  localList.splice(i, 1);
  warnFlags.splice(i, 1);
}

/* ========= 组件挂载时检查是否需要初始化URL序列 ========= */
onMounted(() => {
  // 如果从父组件传递了序列，确保它们被规范化
  localList.forEach((item, idx) => {
    const sanitized = normalizeSeq(item.seq);
    if (item.seq !== sanitized) {
      item.seq = sanitized;
      warnFlags[idx] = true;
    }
  });
});
</script>

<style scoped>
.seq-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.seq-item {
  background: rgba(255, 255, 255, 0.1);
  padding: 1rem;
  border-radius: 8px;
}
.seq-header {
  display: flex;
  justify-content: space-between;
  color: #eee;
  margin-bottom: 0.75rem;
}
.btn-remove {
  background: transparent;
  border: none;
  color: #f88;
  cursor: pointer;
}

.seq-row {
  display: flex;
  gap: 0.5rem;
  align-items: flex-start;
}

.seq-input {
  flex: 1 1 auto;
  min-height: 40px;
  resize: vertical;
  padding: 0.75rem;
  border-radius: 6px;
  border: 1px solid rgba(255, 255, 255, 0.5);
  background: rgba(0, 0, 0, 0.3);
  color: #fff;
  font-family: monospace;
}

.warn-msg {
  flex: 0 0 auto;
  color: #ffbdbd;
  font-size: 0.75rem;
  line-height: 1.4;
  margin-top: 4px;
  user-select: none;
}

.btn-add-seq {
  margin-top: 0.5rem;
  background: #10b981;
  color: #fff;
  padding: 0.6rem 1.2rem;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}
</style>
