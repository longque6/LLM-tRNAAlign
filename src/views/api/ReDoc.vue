<!-- 文件：src/views/Api.vue -->
<template>
  <div id="swagger-container"></div>
</template>

<script setup lang="ts">
import { onMounted } from "vue";
import SwaggerUI from "swagger-ui-dist/swagger-ui-bundle";
import "swagger-ui-dist/swagger-ui.css";

onMounted(() => {
  SwaggerUI({
    dom_id: "#swagger-container",
    url: "/swagger.json",
    docExpansion: "none",
    defaultModelsExpandDepth: -1,
  });
});
</script>

<style>
/* 1. 容器（减掉顶部 56px 导航高度） */
#swagger-container {
  position: absolute;
  top: 56px;
  left: 0;
  right: 0;
  bottom: 0;
  overflow: auto;
  /* 可选：左右加点内边距 */
  padding: 1rem;
  box-sizing: border-box;
  background: transparent; /* 让下方渐变显现 */
}

/* 2. 强制 swagger-ui 主框架居中并铺满 */
#swagger-container .swagger-ui {
  max-width: 100%;
  width: 100%;
  margin: 0 auto;
  background: transparent;
}

/* 3. 顶部条透明化 */
#swagger-container .swagger-ui .topbar {
  background-color: transparent !important;
  box-shadow: none !important;
}

/* 4. Info 区域透明，文字白 */
#swagger-container .swagger-ui .info {
  background: transparent !important;
  color: #fff !important;
}

/* 5. 接口块半透明背景 */
#swagger-container .swagger-ui .opblock {
  background: rgba(255, 255, 255, 0.1) !important;
}

/* 6. 方法按钮用高亮色 */
#swagger-container .swagger-ui .opblock-summary-method {
  color: #00ff8b !important;
  background: rgba(255, 255, 255, 0.2) !important;
}

/* 让 “Parameter content type” 这个 select 的文字变回黑色 */
#swagger-container .swagger-ui .body-param-content-type .content-type,
#swagger-container .swagger-ui .body-param-content-type .content-type option {
  color: #3b4151 !important;
  background-color: transparent !important;
}

/* 7. 文字/链接一律白色 */
#swagger-container .swagger-ui,
#swagger-container .swagger-ui a,
#swagger-container .swagger-ui .info .title,
#swagger-container .swagger-ui .info .description,
#swagger-container .swagger-ui .opblock-summary-path,
#swagger-container .swagger-ui .responses-wrapper {
  color: #fff !important;
}

/* 8. 去掉 schema 白底 */
#swagger-container .swagger-ui .model-box,
#swagger-container .swagger-ui .models {
  background: transparent !important;
  color: #fff !important;
}

/* 9. 滚动条细且半透明 */
#swagger-container::-webkit-scrollbar {
  width: 6px;
}
#swagger-container::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.3);
  border-radius: 3px;
}

#swagger-container .swagger-ui,
#swagger-container .swagger-ui * {
  /* 一次性把所有文字都强制成白色 */
  color: #fff !important;
}
#swagger-container .swagger-ui .opblock .opblock-section-header h4 {
  color: #3b4151 !important;
}

#swagger-container .swagger-ui .btn {
  color: #3b4151 !important;
}

/* 让 “Response content type” 这个 select 变成黑字白底 */
#swagger-container .swagger-ui .execute-content-type .content-type,
#swagger-container .swagger-ui .execute-content-type .content-type option {
  color: #3b4151 !important;
  background-color: #fff !important;
}

/* 如果你还想让那个 label 文本也变黑 */
#swagger-container .swagger-ui .opblock-section-header label > span {
  color: #3b4151 !important;
}

#swagger-container .swagger-ui textarea {
  color: #3b4151 !important;
}

/* 针对带有 cyxy-ignore-trs 类的 input */
.cyxy-ignore-trs {
  /* 文本颜色 */
  color: #fff !important;
  /* 背景透明，让下层渐变透出来 */
  background-color: transparent !important;
  /* 可选：给个白色边框 */
  border: 1px solid rgba(255, 255, 255, 0.6) !important;
  /* 根据需要调整内边距、高度之类的 */
  padding: 0.5em;
  border-radius: 4px;
}

/* placeholder 颜色 */
.cyxy-ignore-trs::placeholder {
  color: rgba(255, 255, 255, 0.6) !important;
}

/* focus 状态下的边框高亮 */
.cyxy-ignore-trs:focus {
  outline: none;
  border-color: #42b983 !important;
  box-shadow: 0 0 0 2px rgba(66, 185, 131, 0.3);
}
</style>
