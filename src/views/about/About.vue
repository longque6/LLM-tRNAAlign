<!-- src/views/ContactPage.vue -->
<template>
  <div class="site--main">
    <div class="container-fluid">
      <!-- ---------- Contact ---------- -->
      <el-card shadow="hover" class="box-card glass-card contact-card">
        <h2>Contact&nbsp;Us</h2>

        <p class="info-text">
          LLM-tRNA-Align&nbsp;v1.0 is actively maintained and steadily evolving
          to deliver more accurate alignments and richer visualisations.
          Questions, feedback or feature ideas are always welcome — every
          suggestion helps us improve.
        </p>

        <div class="contact-info">
          <p class="mail">
            <strong>Lingling&nbsp;Zheng:</strong>
            <el-link href="mailto:zhengll33@mail.sysu.edu.cn" target="_blank">
              zhengll33@mail.sysu.edu.cn
            </el-link>
            <br />
            RNA Information Center, State Key Laboratory for Biocontrol, Sun
            Yat-sen University, Guangzhou 510275, P. R. China
          </p>

          <p class="mail">
            <strong>Zhuo&nbsp;Ouyang:</strong>
            <el-link href="mailto:shendekoudai@lovelumine.com" target="_blank">
              shendekoudai@lovelumine.com
            </el-link>
            <br />
            RNA Information Center, State Key Laboratory for Biocontrol, Sun
            Yat-sen University, Guangzhou 510275, P. R. China
          </p>
        </div>
      </el-card>

      <!-- ---------- Tools ---------- -->
      <el-card shadow="hover" class="box-card glass-card tools-card">
        <h2>Our&nbsp;Tools</h2>

        <p class="info-text">
          Besides AYLM-tRNA, we have developed several databases and software to
          investigate RNA functions and mechanisms in biological processes.
        </p>

        <div class="tools-grid">
          <!-- RMBase -->
          <el-card shadow="always" class="tool-card">
            <a href="https://bioinformaticsscience.cn/rmbase/" target="_blank">
              <el-image
                src="https://minio.lumoxuan.cn/ensure/about/logoRMBase3.png"
                fit="contain"
                alt="RMBase Logo"
              />
            </a>
            <h4>RMBase V3.0</h4>
            <p class="info-text">
              Deciphering the map of RNA modifications from epitranscriptome
              data (Xuan et al., Nucleic Acids Res 2018).
            </p>
          </el-card>

          <!-- ENCORI -->
          <el-card shadow="always" class="tool-card">
            <a href="https://rnasysu.com/encori/" target="_blank">
              <el-image
                src="https://minio.lumoxuan.cn/ensure/about/starBase_logo.png"
                fit="contain"
                alt="ENCORI Logo"
              />
            </a>
            <h4>ENCORI</h4>
            <p class="info-text">
              A database for miRNA–mRNA / miRNA–ncRNA interaction maps (Li et
              al., Nucleic Acids Res 2014).
            </p>
          </el-card>

          <!-- ChIPBase -->
          <el-card shadow="always" class="tool-card">
            <a href="http://rna.sysu.edu.cn/chipbase" target="_blank">
              <el-image
                src="https://minio.lumoxuan.cn/ensure/about/ChIPBase Logo.png"
                fit="contain"
                alt="ChIPBase Logo"
              />
            </a>
            <h4>ChIPBase v3.0</h4>
            <p class="info-text">
              Transcriptional regulatory networks of ncRNAs and genes (Zhou et
              al., Nucleic Acids Res 2017).
            </p>
          </el-card>

          <!-- ENSURE -->
          <el-card shadow="always" class="tool-card">
            <a href="https://trna.lumoxuan.cn/" target="_blank">
              <el-image
                src="https://minio.lumoxuan.cn/ensure/bot/logo.webp"
                fit="contain"
                alt="ENSURE Logo"
              />
            </a>
            <h4>ENSURE</h4>
            <p class="info-text">
              Encyclopedia of suppressor tRNA with an AI assistant (Ouyang
              et&nbsp;al., Nucleic Acids Res 2025, doi:10.1093/nar/gkaf1062).
            </p>
          </el-card>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup>
/* 仅展示静态内容，不需要逻辑 */
</script>

<style scoped>
/* ---------- 外层背景：柔和渐变 ---------- */
.site--main {
  /* 全屏渐变背景 */
  min-height: 100vh;
  padding: 40px 0;
  background: linear-gradient(135deg, #8e9dfa 0%, #9d7efc 50%, #b38cf8 100%);
  display: flex;
  justify-content: center;
}

/* ---------- 中心容器 ---------- */
.container-fluid {
  width: 92%;
  max-width: 1100px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* ---------- 共用卡片样式（磨砂玻璃） ---------- */
.box-card {
  border-radius: 16px;
  overflow: hidden;
}

/* 玻璃拟态效果 */
.glass-card {
  background: rgba(255, 255, 255, 0.28);
  backdrop-filter: blur(12px) saturate(180%);
  -webkit-backdrop-filter: blur(12px) saturate(180%);
  border: 1px solid rgba(255, 255, 255, 0.4);
  color: #0d1b42;
  padding: 24px;
}

/* ---------- 联系方式 ---------- */
.contact-info .mail {
  margin-bottom: 12px;
  line-height: 1.5;
}

.mail a {
  color: #0d47a1;
  font-weight: 600;
  text-decoration: none;
}

.mail a:hover {
  text-decoration: underline;
}

/* ---------- 工具网格 ---------- */
.tools-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  justify-content: space-between;
  margin-top: 10px;
}

.tool-card {
  width: calc(33.333% - 60px);
  min-width: 240px;
  background: rgba(255, 255, 255, 0.45);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.35);
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 16px;
  text-align: center;
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.tool-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 10px 18px rgba(0, 0, 0, 0.18);
}

.tool-card h4 {
  margin: 10px 0 6px;
  font-size: 15px;
  color: #0d1b42;
}

.tool-card :deep(img) {
  max-height: 140px;
  object-fit: contain;
}

.info-text {
  font-size: 14px;
  color: #0d1b42;
}

@media (max-width: 768px) {
  .tool-card {
    width: 100%;
  }
}
</style>
