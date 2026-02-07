// src/router/index.ts
import { createRouter, createWebHistory } from "vue-router";
import type { RouteRecordRaw } from "vue-router";
import Home from "../views/home/Home.vue";
import About from "../views/about/About.vue";
import SprinzlNormal from "@/views/sprinzl/SprinzlNormal.vue";
import PageNotFound from "../404.vue"; // 引入404组件
import ReDoc from "@/views/api/ReDoc.vue";
import "../utils/prevent-bounce";

const routes: Array<RouteRecordRaw> = [
  { path: "/", name: "Home", component: Home },
  { path: "/about", name: "About", component: About },
  { path: "/api", name: "Api", component: ReDoc }, // 重定向到Swagger文档
  { path: "/sprinzl", name: "SprinzlNormal", component: SprinzlNormal }, // 新增SprinzlNormal路由
  { path: "/:pathMatch(.*)*", name: "NotFound", component: PageNotFound }, // 404路由
];

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
});

export default router;
