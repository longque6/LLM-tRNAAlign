<template>
  <nav class="navbar">
    <div class="container">
      <!-- 品牌 / Logo -->
      <a href="/" class="brand">
        AYLM-<span class="brand-highlight">tRNA</span>
      </a>

      <!-- 大屏时显示链接，小屏隐藏 -->
      <ul class="nav-links" :class="{ open: menuOpen }">
        <li>
          <a href="/" :class="{ active: isActive('/') }"> Home </a>
        </li>

        <!-- generate 为多页面入口，直链到 /generate.html -->
        <li>
          <a
            href="/generate.html"
            :class="{ active: isActive('/generate.html') }"
          >
            Generate
          </a>
        </li>
        <li>
          <a href="/sprinzl" :class="{ active: startsWith('/sprinzl') }">
            Sprinzl
          </a>
        </li>
        <!-- 下列两个通常属于首页 SPA 的子路由，如果你仍想用：
             - 这里用 a 标签直链到 /api /about（会整页跳转）
             - 如果以后改回 SPA，再换回 <router-link> 即可 -->
        <li>
          <a href="/api" :class="{ active: startsWith('/api') }"> API </a>
        </li>
        <li>
          <a href="/about" :class="{ active: startsWith('/about') }"> About </a>
        </li>
      </ul>

      <!-- 汉堡按钮：只在小屏显示 -->
      <button
        class="hamburger"
        @click="menuOpen = !menuOpen"
        aria-label="Toggle navigation"
      >
        <span :class="{ active: menuOpen }"></span>
        <span :class="{ active: menuOpen }"></span>
        <span :class="{ active: menuOpen }"></span>
      </button>
    </div>
  </nav>
</template>

<script setup lang="ts">
import { ref } from "vue";

const menuOpen = ref(false);

// 当前路径（不含 query/hash）
const pathname = typeof window !== "undefined" ? window.location.pathname : "/";

// 精确匹配首页（/ 或 /index.html）
const isActive = (path: string) => {
  if (path === "/") {
    return pathname === "/" || pathname === "/index.html";
  }
  return pathname === path;
};

// 以某个前缀开始（给 /api、/about 这种 SPA 子路由用）
const startsWith = (prefix: string) => pathname.startsWith(prefix);
</script>

<style scoped>
.navbar {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  z-index: 1000;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}
.container {
  max-width: 1024px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 1rem;
  height: 56px;
}
/* 品牌 */
.brand {
  font-size: 1.4rem;
  font-weight: bold;
  text-decoration: none;
  color: white;
}
.brand-highlight {
  color: #ffdd57;
}

/* 导航链接 */
.nav-links {
  display: flex;
  list-style: none;
  margin: 0;
  padding: 0;
}
.nav-links li + li {
  margin-left: 1.5rem;
}
.nav-links a {
  text-decoration: none;
  color: white;
  font-weight: 500;
  position: relative;
}
.nav-links a::after {
  content: "";
  position: absolute;
  left: 0;
  bottom: -4px;
  width: 0;
  height: 2px;
  background: #ffdd57;
  transition: width 0.2s ease;
}
.nav-links a:hover::after,
.nav-links a.active::after {
  width: 100%;
}

/* 汉堡菜单按钮（移动端） */
.hamburger {
  display: none;
  flex-direction: column;
  justify-content: space-between;
  width: 24px;
  height: 18px;
  background: none;
  border: none;
  cursor: pointer;
  padding: 0;
}
.hamburger span {
  display: block;
  height: 2px;
  background: white;
  transition: transform 0.3s ease, opacity 0.3s ease;
}
/* 打开状态下的动画 */
.hamburger span:nth-child(1).active {
  transform: translateY(8px) rotate(45deg);
}
.hamburger span:nth-child(2).active {
  opacity: 0;
}
.hamburger span:nth-child(3).active {
  transform: translateY(-8px) rotate(-45deg);
}

/* 小屏(Media < 768px) 适配 */
@media (max-width: 767px) {
  .nav-links {
    position: absolute;
    top: 56px;
    right: 0;
    background: rgba(0, 0, 0, 0.8);
    flex-direction: column;
    width: 200px;
    transform: translateX(100%);
    transition: transform 0.3s ease;
  }
  .nav-links.open {
    transform: translateX(0);
  }
  .nav-links li + li {
    margin: 0;
  }
  .nav-links a {
    padding: 0.75rem 1rem;
  }
  .hamburger {
    display: flex;
  }
}
</style>
