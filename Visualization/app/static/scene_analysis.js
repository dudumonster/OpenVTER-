const CATEGORY_STYLES = {
  car: { color: "#2563eb" },
  van: { color: "#0891b2" },
  truck: { color: "#b91c1c" },
  bus: { color: "#7c3aed" },
  freight_car: { color: "#92400e" },
  motor: { color: "#f97316" },
};

const els = {
  status: document.getElementById("scenePageStatus"),
  refresh: document.getElementById("sceneRefreshButton"),
  cards: document.getElementById("sceneOverviewCards"),
  risks: document.getElementById("sceneRiskList"),
  charts: document.getElementById("sceneLengthCharts"),
  table: document.getElementById("sceneVideoTableBody"),
};

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
}

function fmt(value, digits = 1) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "";
  return Number(value).toFixed(digits);
}

function percentText(value) {
  return `${((Number(value) || 0) * 100).toFixed(1)}%`;
}

function api(path, options) {
  return fetch(path, options).then(async (res) => {
    const contentType = res.headers.get("content-type") || "";
    const payload = contentType.includes("application/json") ? await res.json() : await res.text();
    if (!res.ok) throw new Error(payload.error || payload || `HTTP ${res.status}`);
    return payload;
  });
}

function sceneRisk(video) {
  if (!video || video.status !== "converted") {
    return { level: "high", label: "缺少结果", score: 1000 };
  }
  const carRatio = Number(video.car_ge_5_4_ratio) || 0;
  const vanRatio = Number(video.van_lt_5_4_ratio) || 0;
  const vanShare = video.vehicle_count ? (Number(video.van_count) || 0) / video.vehicle_count : 0;
  let score = carRatio * 100 + vanRatio * 100 + Math.max(0, vanShare - 0.25) * 60;
  if ((video.car_ge_5_4 || 0) >= 5) score += 8;
  if ((video.van_lt_5_4 || 0) >= 3) score += 8;
  if (carRatio >= 0.12 || vanRatio >= 0.12 || score >= 18) return { level: "high", label: "重点检查", score };
  if (carRatio >= 0.04 || vanRatio >= 0.04 || score >= 7) return { level: "medium", label: "建议查看", score };
  return { level: "low", label: "正常", score };
}

function sortVideosByRisk(videos) {
  return [...(videos || [])].sort((a, b) => {
    const ra = sceneRisk(a);
    const rb = sceneRisk(b);
    return rb.score - ra.score || String(a.dataset_id).localeCompare(String(b.dataset_id));
  });
}

function renderCards(summary) {
  const cards = [
    ["视频", `${summary.converted_count}/${summary.video_count}`, summary.issue_count ? `${summary.issue_count} 个需检查` : "全部可读"],
    ["车辆", summary.vehicle_count, `${summary.track_count} 条轨迹`],
    ["car >= 5.4m", summary.car_ge_5_4, percentText(summary.car_ge_5_4_ratio)],
    ["van < 5.4m", summary.van_lt_5_4, percentText(summary.van_lt_5_4_ratio)],
  ];
  els.cards.innerHTML = cards
    .map(([label, value, note]) => `
      <div class="scene-metric-card">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(value)}</strong>
        <small>${escapeHtml(note)}</small>
      </div>
    `)
    .join("");
}

function renderRisks(summary) {
  const ranked = sortVideosByRisk(summary.videos || []);
  const top = ranked.filter((video) => sceneRisk(video).level !== "low").slice(0, 12);
  els.risks.innerHTML = top.length
    ? top.map((video) => {
        const risk = sceneRisk(video);
        return `
          <a class="scene-risk-item ${risk.level}" href="/?dataset=${encodeURIComponent(video.dataset_id)}">
            <span class="scene-risk-video">${escapeHtml(video.dataset_id)}</span>
            <span class="scene-risk-badge">${escapeHtml(risk.label)}</span>
            <span class="scene-risk-note">car>=5.4m ${video.car_ge_5_4} (${percentText(video.car_ge_5_4_ratio)}) · van<5.4m ${video.van_lt_5_4} (${percentText(video.van_lt_5_4_ratio)})</span>
          </a>
        `;
      }).join("")
    : '<div class="scene-risk-empty">当前规则下没有高风险视频。</div>';
}

function renderLengthChart(group) {
  const histogram = group.histogram || [];
  const maxCount = Math.max(1, ...histogram.map((item) => item.count || 0));
  const color = CATEGORY_STYLES[group.class_name]?.color || "#64748b";
  const bars = histogram.map((item) => {
    const height = Math.max(2, ((item.count || 0) / maxCount) * 120);
    return `
      <div class="scene-bar-slot" title="${escapeHtml(item.label)}: ${item.count}">
        <div class="scene-bar" style="height:${height}px;background:${color}"></div>
        <span>${item.count || ""}</span>
      </div>
    `;
  }).join("");
  const labels = histogram.map((item) => `<span>${escapeHtml(item.label)}</span>`).join("");
  return `
    <article class="scene-length-card">
      <div class="scene-length-card-head">
        <div>
          <strong>${escapeHtml(group.class_name)}</strong>
          <span>${group.count} 辆 · 峰值 ${escapeHtml(group.peak_label || "")}: ${group.peak_count || 0} 辆</span>
        </div>
        <div class="scene-length-stat">中位 ${fmt(group.length?.median, 2)} m · P95 ${fmt(group.length?.p95, 2)} m</div>
      </div>
      <div class="scene-bars">${bars}</div>
      <div class="scene-bar-labels">${labels}</div>
    </article>
  `;
}

function renderCharts(summary) {
  const classes = ["car", "van", "truck", "bus", "freight_car", "motor"];
  const groups = (summary.class_summaries || []).filter((item) => classes.includes(item.class_name));
  els.charts.innerHTML = groups.map(renderLengthChart).join("");
}

function renderTable(summary) {
  const ranked = sortVideosByRisk(summary.videos || []);
  els.table.innerHTML = ranked.map((video) => {
    const statusClass = video.status === "converted" ? "ok" : "bad";
    const risk = sceneRisk(video);
    const motorCount = (video.class_counts && video.class_counts.motor) || 0;
    return `
      <tr>
        <td><a class="scene-video-link" href="/?dataset=${encodeURIComponent(video.dataset_id)}">${escapeHtml(video.dataset_id)}</a></td>
        <td><span class="scene-status ${statusClass}">${escapeHtml(video.status)}</span></td>
        <td>${video.vehicle_count}</td>
        <td>${video.car_count}</td>
        <td>${video.van_count}</td>
        <td>${motorCount}</td>
        <td>${video.car_ge_5_4} <span class="muted">(${percentText(video.car_ge_5_4_ratio)})</span></td>
        <td>${video.van_lt_5_4} <span class="muted">(${percentText(video.van_lt_5_4_ratio)})</span></td>
        <td><span class="scene-status ${risk.level}">${escapeHtml(risk.label)}</span></td>
      </tr>
    `;
  }).join("");
}

async function loadSceneSummary() {
  els.status.textContent = "正在汇总所有视频...";
  try {
    const summary = await api("/api/scene-summary");
    els.status.textContent = `${summary.converted_count}/${summary.video_count} 个视频已转换 · ${summary.vehicle_count} 辆车 · Final Data: ${summary.final_root}`;
    renderCards(summary);
    renderRisks(summary);
    renderCharts(summary);
    renderTable(summary);
  } catch (err) {
    els.status.textContent = `场景汇总失败：${err.message || err}`;
    els.cards.innerHTML = "";
    els.risks.innerHTML = "";
    els.charts.innerHTML = "";
    els.table.innerHTML = "";
  }
}

els.refresh.addEventListener("click", loadSceneSummary);
loadSceneSummary();
