const SUGGESTIONS = [
  "Who is the CEO of the company that acquired DeepMind?",
  "Where is the parent company of Google headquartered?",
  "Who founded the company that developed AlphaGo?",
  "Which company partnered with OpenAI?",
];

const COLORS = {
  PERSON: "#d7a04b",
  ORG: "#6fc3b8",
  LOCATION: "#8fbf7a",
  CONCEPT: "#8a9384",
  UNKNOWN: "#8a9384",
};

const graphEl = document.getElementById("graph");
let network = null;
let lastHighlight = new Set();

function $(id) {
  return document.getElementById(id);
}

async function api(path, options) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body.detail || res.statusText);
  }
  return body;
}

function setStatus(id, text) {
  $(id).textContent = text || "";
}

async function refreshStats() {
  const health = await api("/health");
  $("stat-backend").textContent = health.backend || "—";
  $("stat-nodes").textContent = health.nodes ?? 0;
  $("stat-edges").textContent = health.edges ?? 0;
  $("stat-chunks").textContent = health.chunks ?? 0;
}

function formatPath(path) {
  if (!path.nodes || !path.nodes.length) return "";
  const parts = [path.nodes[0].name];
  path.edges.forEach((edge, i) => {
    const next = path.nodes[i + 1] ? path.nodes[i + 1].name : "?";
    const forward = edge.source_id === path.nodes[i].id;
    parts.push(`${forward ? "-[" + edge.type + "]->" : "<-[" + edge.type + "]-"} ${next}`);
  });
  return parts.join(" ");
}

function renderQuery(result) {
  $("answer").textContent = result.answer || "No answer.";
  const paths = $("paths");
  paths.innerHTML = "";
  (result.paths || []).slice(0, 8).forEach((path) => {
    const li = document.createElement("li");
    li.innerHTML = `<div class="path-meta">${path.hops} hop · score ${Number(path.score || 0).toFixed(2)}</div>${formatPath(path)}`;
    paths.appendChild(li);
  });
  if (!paths.children.length) {
    paths.innerHTML = "<li>No connecting paths for this question.</li>";
  }

  const evidence = $("evidence");
  evidence.innerHTML = "";
  (result.evidence || []).slice(0, 6).forEach((item) => {
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `<header><span class="badge">${item.source}</span><span>vec ${item.vector_score.toFixed(2)} · graph ${item.graph_score.toFixed(2)} · fused ${item.fused_score.toFixed(2)}</span></header><div>${item.chunk.text}</div>`;
    evidence.appendChild(card);
  });

  lastHighlight = new Set();
  (result.paths || []).forEach((path) => {
    (path.nodes || []).forEach((n) => lastHighlight.add(n.id));
  });
  (result.seed_entities || []).forEach((n) => lastHighlight.add(n.id));
}

function drawGraph(snapshot) {
  const nodes = new vis.DataSet(
    (snapshot.nodes || []).map((n) => ({
      id: n.id,
      label: n.name,
      group: n.type,
      color: {
        background: lastHighlight.has(n.id) ? "#d9897a" : COLORS[n.type] || COLORS.UNKNOWN,
        border: lastHighlight.has(n.id) ? "#f2c7bf" : "#10150f",
      },
      font: { color: "#e8e1cf", face: "Source Sans 3" },
      borderWidth: lastHighlight.has(n.id) ? 3 : 1,
    }))
  );
  const edges = new vis.DataSet(
    (snapshot.edges || []).map((e, i) => ({
      id: `${e.source_id}-${e.type}-${e.target_id}-${i}`,
      from: e.source_id,
      to: e.target_id,
      label: e.type,
      arrows: "to",
      color: { color: "#4a5b46" },
      font: { color: "#9aa58e", size: 10, strokeWidth: 0, face: "IBM Plex Mono" },
    }))
  );
  const data = { nodes, edges };
  const options = {
    physics: { stabilization: true, barnesHut: { gravitationalConstant: -2800, springLength: 140 } },
    interaction: { hover: true, tooltipDelay: 80 },
    nodes: { shape: "dot", size: 18 },
    edges: { smooth: { type: "cubicBezier" } },
  };
  if (network) {
    network.setData(data);
  } else {
    network = new vis.Network(graphEl, data, options);
  }
}

async function refreshGraph() {
  const snap = await api("/graph");
  drawGraph(snap);
  $("stat-nodes").textContent = (snap.nodes || []).length;
  $("stat-edges").textContent = (snap.edges || []).length;
  $("stat-chunks").textContent = snap.chunks ?? $("stat-chunks").textContent;
  $("stat-backend").textContent = snap.backend || $("stat-backend").textContent;
}

async function runQuery() {
  const question = $("question").value.trim();
  if (question.length < 2) return;
  setStatus("query-status", "Retrieving vector hits and expanding the graph…");
  $("ask-btn").disabled = true;
  try {
    const result = await api("/query", {
      method: "POST",
      body: JSON.stringify({ question }),
    });
    renderQuery(result);
    await refreshGraph();
    setStatus("query-status", `Backend ${result.backend} · ${result.paths.length} paths`);
  } catch (err) {
    setStatus("query-status", err.message);
  } finally {
    $("ask-btn").disabled = false;
  }
}

async function ingestText() {
  const title = $("doc-title").value.trim() || "untitled";
  const text = $("doc-text").value.trim();
  if (text.length < 8) {
    setStatus("ingest-status", "Need a longer passage to extract relations.");
    return;
  }
  setStatus("ingest-status", "Extracting entities and relations…");
  try {
    const stats = await api("/ingest/text", {
      method: "POST",
      body: JSON.stringify({ title, text }),
    });
    $("doc-text").value = "";
    await refreshStats();
    await refreshGraph();
    setStatus("ingest-status", `Indexed ${stats.chunks} chunks · ${stats.relations} relations`);
  } catch (err) {
    setStatus("ingest-status", err.message);
  }
}

async function ingestSample() {
  setStatus("ingest-status", "Loading sample corpus…");
  try {
    const stats = await api("/ingest/sample", { method: "POST" });
    await refreshStats();
    await refreshGraph();
    setStatus("ingest-status", `Sample loaded · ${stats.nodes} entities`);
  } catch (err) {
    setStatus("ingest-status", err.message);
  }
}

function mountChips() {
  const wrap = $("chips");
  SUGGESTIONS.forEach((q) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip";
    btn.textContent = q;
    btn.addEventListener("click", () => {
      $("question").value = q;
      runQuery();
    });
    wrap.appendChild(btn);
  });
}

$("ask-btn").addEventListener("click", runQuery);
$("ingest-btn").addEventListener("click", ingestText);
$("sample-btn").addEventListener("click", ingestSample);

mountChips();
refreshStats()
  .then(refreshGraph)
  .then(runQuery)
  .catch((err) => setStatus("query-status", err.message));
