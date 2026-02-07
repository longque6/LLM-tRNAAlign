<!-- src/App.vue -->
<template>
  <div id="app">
    <NavBar />
    <!-- 为了让内容不被固定导航栏遮住，加一个占位 -->
    <div style="height: 56px"></div>

    <main class="view">
      <router-view />
    </main>
  </div>
</template>

<script setup lang="ts">
import NavBar from "./components/NavBar.vue";
</script>

<style>
/* —— 全局基础，保证 100% 宽度 —— */
html,
body {
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  overscroll-behavior: none;
}
#app {
  width: 100%;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* 背景和文字 */
body {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #fff;
  font-family: "Helvetica Neue", Arial, sans-serif;
}

/* 主内容区：铺满、scroll、水平居中子组件 */
.view {
  flex: 1;
  display: flex;
  justify-content: center; /* 子组件水平居中 */
  align-items: flex-start;
  overflow-y: auto;
  box-sizing: border-box;
  padding: 1rem;
}

/* 不再在这里限制 max-width！ */
.view > * {
  width: 100%;
}
</style>
