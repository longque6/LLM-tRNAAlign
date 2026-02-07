<template>
  <main class="home">
    <section class="hero">
      <div class="hero-text">
        <p class="eyebrow">AYLM · Academic tRNA Workspace</p>
        <h1>AYLM-tRNA for reproducible tRNA design and analysis</h1>
        <p>
          Built on the AYLM foundation model, AYLM‑tRNA unifies reference
          retrieval, LLM-constrained generation, sup‑tRNA augmentation, and
          visualization in a single environment. It explains 75.51% of variance
          in readthrough efficiency (log space) and can be extended to sequence
          tasks such as site-level modification prediction.
        </p>
        <div class="hero-actions">
          <a class="btn btn-primary" href="/generate.html">Start generator</a>
          <a class="btn btn-outline" href="/sprinzl">Open Sprinzl</a>
        </div>
      </div>
      <dl class="hero-stats">
        <div class="stat-with-note">
          <dt>
            Reference templates
            <span
              class="stat-question"
              aria-label="Template definition"
              title="Template definition"
              >?</span
            >
          </dt>
          <dd>169,256</dd>
          <div class="stat-tooltip">
            Alignment templates used for Sprinzl numbering or as seeds for new
            tRNA/sup-tRNA designs.
          </div>
        </div>
        <div>
          <dt>Training corpus</dt>
          <dd>2,037,394 sequences</dd>
        </div>
        <div>
          <dt>AYLM parameters</dt>
          <dd>100.5 million</dd>
        </div>
      </dl>
    </section>

    <section class="section">
      <header>
        <h2>Capabilities</h2>
        <p>
          Tools for research teams to plan, edit, and document tRNA experiments.
        </p>
      </header>
      <div class="feature-list">
        <article v-for="feature in features" :key="feature.title">
          <div class="feature-icon">{{ feature.icon }}</div>
          <div>
            <h3>{{ feature.title }}</h3>
            <p>{{ feature.description }}</p>
            <a v-if="feature.link" :href="feature.link">
              {{ feature.linkText ?? "Learn more" }} →
            </a>
          </div>
        </article>
      </div>
    </section>

    <section class="section">
      <header>
        <h2>Recommended workflow</h2>
        <p>
          Three steps that keep data and documentation consistent across
          collaborators.
        </p>
      </header>
      <ol class="workflow">
        <li v-for="step in workflowSteps" :key="step.step">
          <span class="step-number">{{ step.step }}</span>
          <div>
            <h4>{{ step.title }}</h4>
            <p>{{ step.description }}</p>
          </div>
        </li>
      </ol>
    </section>

    <section class="section plain-cta">
      <div>
        <p class="eyebrow">Continue exploring</p>
        <h2>
          Switch between alignment, generation, and visualization without
          leaving the workspace.
        </h2>
        <p>
          Every run exports FASTA/CSV plus alignment traces for lab notebooks
          and peer review.
        </p>
      </div>
      <div class="cta-actions">
        <a class="btn btn-primary" href="/generate.html">Run a batch</a>
        <a class="btn btn-outline" href="/api">View API</a>
      </div>
    </section>
  </main>
</template>

<script setup lang="ts">
const features = [
  {
    title: "LLM-constrained & sup-tRNA design",
    description:
      "Control style gradients, GC targets, freeze regions, and extend edits to sup-tRNA scaffolds for readthrough studies.",
    icon: "🧬",
    link: "/generate.html",
    linkText: "Open generator",
  },
  {
    title: "Sprinzl & sequence diagnostics",
    description:
      "Inspect template numbering, original coordinates, and multi-modification predictions directly from sequence.",
    icon: "🧭",
    link: "/sprinzl",
    linkText: "View visualization",
  },
  {
    title: "Reproducible exports & API",
    description:
      "FASTA, CSV, and alignment logs are ready for lab notebooks; REST APIs keep automated QC pipelines in sync.",
    icon: "📎",
    link: "/api",
    linkText: "API reference",
  },
];

const workflowSteps = [
  {
    step: "01",
    title: "Curate references",
    description:
      "Filter by domain, species, isotype or import custom sequences to anchor downstream edits.",
  },
  {
    step: "02",
    title: "Apply constraints",
    description:
      "Adjust style, HD thresholds, GC bands, or region freezes to match the experimental hypothesis.",
  },
  {
    step: "03",
    title: "Validate & record",
    description:
      "Use Sprinzl previews plus exportable logs to capture figures and shareable datasets.",
  },
];
</script>

<style scoped>
:global(body) {
  margin: 0;
}

.home {
  max-width: 1100px;
  margin: 0 auto;
  padding: 120px 28px 80px;
  display: flex;
  flex-direction: column;
  gap: 56px;
  color: #f5f5ff;
}

.hero {
  display: flex;
  flex-wrap: wrap;
  gap: 32px;
  align-items: flex-start;
}

.hero-text {
  flex: 1 1 420px;
}

.hero h1 {
  margin: 0 0 16px;
  font-size: clamp(2.2rem, 4vw, 3rem);
  font-weight: 600;
}

.hero p {
  margin: 0 0 20px;
  color: rgba(247, 247, 255, 0.86);
  line-height: 1.6;
}

.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.hero-stats {
  flex: 1 1 320px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 16px;
  margin: 0;
  padding: 0;
}
.hero-stats dt {
  font-size: 0.8rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.65);
}
.hero-stats dd {
  margin: 6px 0 0;
  font-size: 1.9rem;
  font-weight: 600;
}

.stat-with-note {
  position: relative;
}

.stat-question {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  margin-left: 6px;
  border-radius: 50%;
  border: 1px solid rgba(255, 255, 255, 0.5);
  font-size: 0.7rem;
  cursor: default;
}

.stat-tooltip {
  position: absolute;
  top: 110%;
  left: 0;
  width: 240px;
  padding: 10px;
  background: rgba(9, 12, 36, 0.9);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 10px;
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.85);
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.15s ease;
  z-index: 2;
}

.stat-with-note:hover .stat-tooltip,
.stat-with-note:focus-within .stat-tooltip {
  opacity: 1;
}

.section {
  display: flex;
  flex-direction: column;
  gap: 24px;
  border-top: 1px solid rgba(255, 255, 255, 0.15);
  padding-top: 28px;
}

.section header h2 {
  margin: 0 0 8px;
  font-size: 1.6rem;
}

.section header p {
  margin: 0;
  color: rgba(255, 255, 255, 0.7);
}

.feature-list {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.feature-list article {
  display: flex;
  gap: 16px;
  padding-bottom: 18px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.feature-list article:last-child {
  border-bottom: none;
  padding-bottom: 0;
}

.feature-icon {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.15);
  display: grid;
  place-items: center;
  font-size: 1.2rem;
}

.feature-list h3 {
  margin: 0 0 6px;
  font-size: 1.1rem;
  font-weight: 600;
}

.feature-list p {
  margin: 0 0 4px;
  color: rgba(255, 255, 255, 0.78);
}

.feature-list a {
  color: #dbe4ff;
  text-decoration: none;
  font-size: 0.9rem;
}

.workflow {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.workflow li {
  display: flex;
  gap: 16px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 14px;
  padding: 16px;
  background: rgba(8, 10, 30, 0.35);
}

.step-number {
  font-weight: 700;
  color: #a5b4fc;
  font-size: 0.95rem;
}

.workflow h4 {
  margin: 0 0 4px;
}
.workflow p {
  margin: 0;
  color: rgba(255, 255, 255, 0.75);
}

.plain-cta {
  flex-direction: row;
  justify-content: space-between;
  align-items: center;
  gap: 20px;
}

.cta-actions {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.18em;
  font-size: 0.75rem;
  color: #b4bef6;
  margin-bottom: 10px;
}

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  padding: 11px 24px;
  font-weight: 600;
  text-decoration: none;
  transition: transform 0.15s ease, background 0.2s ease;
}

.btn-primary {
  background: rgba(255, 255, 255, 0.96);
  color: #201a41;
}

.btn-outline {
  border: 1px solid rgba(255, 255, 255, 0.55);
  color: #f7f8ff;
}

.btn-outline:hover,
.btn-primary:hover {
  transform: translateY(-1px);
}

@media (max-width: 720px) {
  .hero {
    flex-direction: column;
  }

  .hero-actions,
  .cta-actions {
    flex-direction: column;
    width: 100%;
  }

  .btn {
    width: 100%;
  }
}
</style>
