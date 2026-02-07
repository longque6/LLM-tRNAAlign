<template>
  <div class="alignment-wrappers">
    <table
      v-for="(chunk, index) in chunks"
      :key="index"
      class="alignment-table"
    >
      <tbody>
        <tr>
          <th class="label-cell" scope="row">ID</th>
          <td v-for="item in chunk" :key="item.id">
            {{ item.id }}
          </td>
        </tr>

        <tr>
          <th class="label-cell" scope="row">Origin</th>
          <td
            v-for="item in chunk"
            :key="item.id + '_base'"
            :class="getType(item)"
          >
            {{ item.base }}
          </td>
        </tr>

        <tr>
          <th class="label-cell" scope="row">Target-tRNA</th>
          <td
            v-for="item in chunk"
            :key="item.id + '_sup'"
            :class="getType(item)"
          >
            {{ item.sup_base }}
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script lang="ts" setup>
import { computed } from "vue";

interface AlignmentItem {
  id: string;
  base: string;
  sup_base: string;
}

const props = defineProps<{
  alignmentData: AlignmentItem[];
  chunkSize?: number;
}>();

// 默认每块显示 20 列，可通过 chunkSize 调整
const size = props.chunkSize || 20;
const chunks = computed(() => {
  const result: AlignmentItem[][] = [];
  for (let i = 0; i < props.alignmentData.length; i += size) {
    result.push(props.alignmentData.slice(i, i + size));
  }
  return result;
});

/**
 * Determine cell type based on base and sup_base values
 */
const getType = (
  item: AlignmentItem
): "match" | "mismatch" | "insertion" | "deletion" => {
  return item.base === "-" && item.sup_base !== "-"
    ? "insertion"
    : item.base !== "-" && item.sup_base === "-"
    ? "deletion"
    : item.base === item.sup_base
    ? "match"
    : "mismatch";
};
</script>

<style scoped>
/* —— 布局 —— */
.alignment-wrappers {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

/* —— 小表 —— */
.alignment-table {
  border-collapse: collapse;
  font-family: "Consolas", monospace;
  table-layout: fixed;
  backdrop-filter: blur(4px);
  background: rgba(255, 255, 255, 0.08);
  border-radius: 6px;
  overflow: hidden;
}

/* —— 单元格 —— */
.alignment-table th,
.alignment-table td {
  border: 1px solid rgba(255, 255, 255, 0.25);
  padding: 3px 6px;
  text-align: center;
  white-space: nowrap;
  font-size: 13px;
  color: #fff;
}

/* 行首标签 */
.label-cell {
  background: rgba(255, 255, 255, 0.18);
  font-weight: 600;
}

/* —— 配色 —— */
.alignment-table td.match {
  background: rgba(76, 175, 80, 0.55);   /* 绿 */
}
.alignment-table td.mismatch {
  background: rgba(244, 67, 54, 0.55);   /* 红 */
}
.alignment-table td.insertion {
  background: rgba(33, 150, 243, 0.55);  /* 蓝 */
}
.alignment-table td.deletion {
  background: rgba(255, 167, 38, 0.55);  /* 橙 */
}
</style>
