// gfem Mechanical – main controller: ribbon, outline tree, details pane, solve monitoring, results.
import { Viewer, b64f32 } from "./viewer.js";
import { Chart, fmt, fmtR } from "./charts.js";
import { openSketcher } from "./sketch.js";

// ------------------------------------------------------------------ API
const api = {
  async get(url) { const r = await fetch(url); const j = await r.json(); if (!r.ok || j.error) throw new Error(j.error || r.statusText); return j; },
  async post(url, body = {}) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json(); if (!r.ok || (j.error && !j.project)) throw new Error(j.error || r.statusText); return j;
  },
  async upload(file) { const r = await fetch("/api/upload?name=" + encodeURIComponent(file.name), { method: "POST", body: file }); return r.json(); },
};

const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const uid = (p, list) => { let n = 1; const ids = new Set(list.map(x => x.id)); while (ids.has(p + n)) n++; return p + n; };

const FEATURE_LABEL = { surface: "Surface", line: "Line", box: "Box", cylinder: "Cylinder", sphere: "Sphere", cone: "Cone", torus: "Torus", wedge: "Wedge", extrude: "Extrude",
  revolve: "Revolve", dxf_extrude: "DXF Extrude", boolean: "Boolean", fillet: "Fillet", chamfer: "Chamfer", transform: "Transform", pattern: "Pattern", import: "Import" };
const ICON = {
  model: "▣", geometry: "◈", feature: "▸", body: "■", materials: "◉", material: "●", connections: "⇄", contact: "⇄", named: "☰", ns: "▤",
  mesh: "▦", sizing: "◫", analysis: "⚙", settings: "⚙", support: "▼", load: "➜", thermal: "♨", solution: "✔", solinfo: "ℹ", result: "◐",
};

class App {
  constructor() {
    this.api = api;
    this.state = null; this.project = null; this.geom = null; this.meshDisp = null;
    this.treeSel = { kind: "model" };
    this.ui = { mode: "geometry", frame: 0, rtype: "total_deformation", deform: "auto", bands: 9, smooth: false, playing: false, probe: false, showMesh: false, undeformed: false };
    this.iters = []; this.incs = []; this.frames = []; this.seq = 0;
    this.status = { geometry: "none", mesh: "none", solve: "none" };
    this.viewer = new Viewer($("#viewport"));
    this.viewer.onSelect = sel => this.onSelect(sel);
    this.viewer.onProbe = (i, p) => this.onProbe(i, p);
    this.viewer.onHover = h => this.onHover(h);
    this.chartRes = new Chart($("#chart-res"), { logy: true, title: "Force convergence", xlabel: "cumulative iteration" });
    this.chartTime = new Chart($("#chart-time"), { title: "Time / load factor", xlabel: "increment" });
    this.chartGraph = new Chart($("#chart-graph"), { title: "Result vs time", xlabel: "time" });
    this._bindStatic();
    this.init();
  }

  // ---------------------------------------------------------------- bootstrap
  async init() {
    this.state = await api.get("/api/state");
    this.project = this.state.project;
    this.seq = this.state.seq;
    $("#sys-info").textContent = `${this.state.system.backend.toUpperCase()} · ${this.state.system.devices.join(", ")} · gfem ${this.state.system.version}`;
    const ex = await api.get("/api/examples");
    $("#examples-menu").innerHTML = ex.map(e => `<a data-ex="${e.key}" title="${esc(e.text)}">${esc(e.title)}</a>`).join("");
    $$("#examples-menu a").forEach(a => a.onclick = () => this.loadExample(a.dataset.ex));
    this.renderRibbon("home");
    if (this.project.geometry.features.length) await this.regenerate();
    this.render();
    this.listen();
    this.log("Welcome to gfem Mechanical. Start from Geometry (create or import CAD), or open an Example.");
  }

  listen() {
    const es = new EventSource("/api/events?since=" + this.seq);
    es.onmessage = ev => this.onEvent(JSON.parse(ev.data));
    es.onerror = () => { es.close(); setTimeout(() => this.listen(), 2000); };
  }

  onEvent(e) {
    this.seq = e.seq + 1;
    if (e.type === "log") this.log(e.msg, e.msg.startsWith("ERROR") ? "err" : e.msg.includes("warning") ? "warn" : "");
    else if (e.type === "iter") { this.iters.push(e); this.throttle("conv", () => this.drawConvergence(), 250); }
    else if (e.type === "inc") { this.incs.push(e); this.throttle("conv", () => this.drawConvergence(), 250); $("#st-job").textContent = `Solving… inc ${e.inc} t=${fmt(e.t)}`; }
    else if (e.type === "status") {
      $("#st-job").textContent = e.status === "running" ? "Solving…" : `Job ${e.status}`;
      $("#st-job").className = e.status;
      if (e.status !== "running") { this.status.solve = e.status === "done" ? "ok" : "failed"; this.loadResults(); }
      this.renderTree();
    } else if (e.type === "mesh") { this.loadMesh(false); }
  }

  throttle(k, fn, ms) { this._th = this._th || {}; if (this._th[k]) return; this._th[k] = setTimeout(() => { this._th[k] = null; fn(); }, ms); }

  log(msg, cls = "") {
    const d = document.createElement("div"); d.className = "logline " + cls;
    d.textContent = msg; const L = $("#log"); L.appendChild(d);
    while (L.children.length > 3000) L.removeChild(L.firstChild);
    L.scrollTop = L.scrollHeight;
    $("#st-msg").textContent = msg.length > 120 ? msg.slice(0, 120) + "…" : msg;
  }

  // ---------------------------------------------------------------- static bindings
  _bindStatic() {
    $$("#ribbon-tabs button").forEach(b => b.onclick = () => { $$("#ribbon-tabs button").forEach(x => x.classList.toggle("on", x === b)); this.renderRibbon(b.dataset.tab); });
    $$("#sel-filter button").forEach(b => b.onclick = () => this.setFilter(b.dataset.f));
    $$("#views button").forEach(b => b.onclick = () => b.dataset.v === "fit" ? this.viewer.fit() : this.viewer.view(b.dataset.v));
    $$("#modes button").forEach(b => b.onclick = () => this.setMode(b.dataset.m));
    $("#btn-selall").onclick = () => this.viewer.selectAll();
    $("#btn-selclear").onclick = () => this.viewer.clearSelection();
    $$(".btabs button").forEach(b => b.onclick = () => this.showBottom(b.dataset.b));
    $("#btn-new").onclick = () => this.newProject();
    $("#btn-save").onclick = () => this.saveProject();
    $("#btn-open").onclick = () => $("#file-proj").click();
    $("#file-proj").onchange = e => this.openProject(e.target.files[0]);
    $("#file-cad").onchange = e => this.importFile(e.target.files[0], "cad");
    $("#file-deck").onchange = e => this.importFile(e.target.files[0], "deck");
    $$("[data-export]").forEach(a => a.onclick = () => this.exportFile(a.dataset.export));
    $("#btn-shot").onclick = () => { const a = document.createElement("a"); a.href = this.viewer.screenshot(); a.download = (this.project.name || "gfem") + ".png"; a.click(); };
    $("#proj-name").onchange = e => { this.project.name = e.target.value; this.push(); };
    $("#frame-slider").oninput = e => { this.ui.frame = +e.target.value; this.showResult(); };
    $("#anim-play").onclick = () => this.toggleAnim();
    window.addEventListener("keydown", e => {
      if (e.target.matches("input,select,textarea")) return;
      if (e.key === "Escape") this.viewer.clearSelection();
      if ((e.ctrlKey || e.metaKey) && e.key === "a") { e.preventDefault(); this.viewer.selectAll(); }
      if (e.key === "Delete") this.deleteTreeSel();
      if (e.key === "F5") { e.preventDefault(); this.solve(); }
    });
    this._splitters();
    const drop = $("#viewport");
    drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("drop"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("drop"));
    drop.addEventListener("drop", e => { e.preventDefault(); drop.classList.remove("drop"); const f = e.dataTransfer.files[0]; if (!f) return;
      const ext = f.name.split(".").pop().toLowerCase();
      this.importFile(f, ["step", "stp", "iges", "igs", "brep", "stl", "dxf"].includes(ext) ? "cad" : "deck"); });
  }

  _splitters() {
    const drag = (el, fn) => el.addEventListener("pointerdown", e => { e.preventDefault(); const mv = ev => fn(ev); const up = () => { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); this.viewer.resize(); this.redrawCharts(); };
      window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up); });
    drag($("#split-main"), e => { $("#left").style.width = Math.max(200, Math.min(700, e.clientX)) + "px"; });
    drag($("#split-left"), e => { const r = $("#left").getBoundingClientRect(); $("#outline-panel").style.height = Math.max(80, e.clientY - r.top) + "px"; });
    drag($("#split-bottom"), e => { const r = $("#center").getBoundingClientRect(); $("#bottom").style.height = Math.max(60, r.bottom - e.clientY) + "px"; });
  }

  showBottom(name) { $$(".btabs button").forEach(b => b.classList.toggle("on", b.dataset.b === name)); $$(".bpane").forEach(p => p.classList.toggle("on", p.id === "b-" + name)); this.redrawCharts(); }
  redrawCharts() { this.chartRes.draw(); this.chartTime.draw(); this.chartGraph.draw(); }

  setFilter(f) { this.viewer.filter = f; $$("#sel-filter button").forEach(b => b.classList.toggle("on", b.dataset.f === f)); }
  setMode(m) {
    if (m === "mesh" && !this.meshDisp) { this.log("no mesh yet – Generate Mesh first"); return; }
    if (m === "result" && !this.frames.length) { this.log("no results yet – Solve first"); return; }
    this.ui.mode = m; this.viewer.setMode(m);
    $$("#modes button").forEach(b => b.classList.toggle("on", b.dataset.m === m));
    $("#legend").style.display = m === "result" ? "block" : "none";
    $("#anim-bar").style.display = m === "result" ? "flex" : "none";
    if (m !== "result" && this.ui.playing) this.toggleAnim();
  }

  // ---------------------------------------------------------------- server sync
  async push() { await api.post("/api/project", { project: this.project }); }

  async regenerate() {
    this.vpMsg("Generating geometry…");
    try {
      const r = await api.post("/api/regenerate", { project: this.project });
      if (r.error) { this.log("Geometry error: " + r.error, "err"); this.status.geometry = "failed"; this.vpMsg(""); this.renderTree(); return false; }
      this.project = r.project; this.geom = r; this.status.geometry = r.info ? "ok" : "none";
      this.applyBodyColors();
      this.viewer.setGeometry(r, !this._keepView);
      this.meshDisp = null; this.status.mesh = "none"; this.status.solve = this.frames.length ? "stale" : "none";
      if (r.info) this.log(`Geometry: ${r.info.bodies.length} bodies, ${r.info.faces.length} faces, ${r.info.edges.length} edges` + (r.info.messages?.length ? " – " + r.info.messages.join("; ") : ""));
      this.setMode("geometry");
    } catch (e) { this.log("Geometry error: " + e.message, "err"); }
    this.vpMsg(""); this.render();
    return true;
  }

  applyBodyColors() {
    const mats = Object.fromEntries(this.project.materials.map(m => [m.id, m.color || "#8aa9c9"]));
    const map = {};
    for (const [bid, b] of Object.entries(this.project.bodies || {})) map[bid] = b.color || mats[b.material] || "#b8c4d4";
    this.viewer.bodyColors = map;
  }

  vpMsg(t) { const m = $("#vp-msg"); m.textContent = t; m.style.display = t ? "" : "none"; }

  async meshNow() {
    this.vpMsg("Meshing…");
    try {
      const r = await api.post("/api/mesh", { project: this.project });
      this.setMeshDisplay(r);
    } catch (e) { this.log("Mesh error: " + e.message, "err"); this.status.mesh = "failed"; }
    this.vpMsg(""); this.render();
  }
  async loadMesh(switchMode = false) { try { const r = await api.get("/api/mesh"); if (r.xyz) { this.setMeshDisplay(r, switchMode); } } catch (e) { /* ignore */ } }
  setMeshDisplay(r, switchMode = true) {
    this.meshDisp = r; this.meshStats = r.stats; this.status.mesh = r.xyz ? "ok" : "none";
    this.viewer.setMesh(r);
    if (switchMode) this.setMode("mesh");
    this.renderTree(); if (this.treeSel.kind === "mesh") this.renderDetails();
  }

  async solve() {
    if (!this.project.conditions.length && !["modal", "deck"].includes(this.project.analysis.type)) { this.log("no boundary conditions / loads defined", "warn"); }
    this.iters = []; this.incs = []; this.drawConvergence();
    this.status.solve = "running"; this.renderTree();
    this.showBottom("solinfo");
    try { await api.post("/api/solve", { project: this.project }); }
    catch (e) { this.log("Solve error: " + e.message, "err"); this.status.solve = "failed"; this.renderTree(); }
  }
  async stop() { await api.post("/api/stop"); }

  async loadResults() {
    try {
      const fr = await api.get("/api/frames");
      this.frames = fr.frames; this.resultMeta = fr.meta;
      if (!this.frames.length) return;
      if (!this.meshDisp) await this.loadMesh(false);
      $("#frame-slider").max = this.frames.length - 1;
      this.ui.frame = this.frames.length - 1;
      const proc = this.frames[0].procedure;
      if (proc === "frequency" || proc === "buckle") this.ui.frame = 0;
      $("#frame-slider").value = this.ui.frame;
      const r = this.project.results[0];
      if (r && !this.treeSel.kind.startsWith("result")) this.treeSel = { kind: "result", id: r.id };
      this.renderTabular(); this.drawGraph();
      if (proc === "frequency" || proc === "buckle") this.showBottom("tabular");
      await this.showResult();
      this.render();
    } catch (e) { this.log("results: " + e.message, "err"); }
  }

  currentResultObject() {
    if (this.treeSel.kind === "result") return this.project.results.find(r => r.id === this.treeSel.id);
    return this.project.results[0];
  }

  async showResult() {
    if (!this.frames.length) return;
    const ro = this.currentResultObject();
    const rtype = ro ? ro.type : "total_deformation";
    const i = this.ui.frame;
    let r;
    try { r = await api.get(`/api/result?frame=${i}&type=${rtype}`); } catch (e) { this.log(e.message, "err"); return; }
    r.values = r.values ? b64f32(r.values) : null; r.U = r.U ? b64f32(r.U) : null;
    this.cur = r;
    // deformation scale (ANSYS auto: max displacement ~ 5% of model size)
    let scale = 1;
    const diag = this.geom?.info?.diag || this.state?.info?.diag || 1;
    const d = this.ui.deform;
    if (d === "auto") scale = r.umax > 0 ? 0.05 * diag / r.umax : 1; else scale = +d;
    if (rtype === "density" || r.procedure === "topopt") scale = 0;
    this.viewer.deform = scale;
    this.viewer.showMinMax = true;
    this.viewer.setResult(r, { showMesh: this.ui.showMesh, undeformed: this.ui.undeformed && this.viewer.mode !== "geometry" });
    this.setMode("result");
    this.renderLegend(r, ro, scale);
    const f = this.frames[i];
    let lab = `${i + 1}/${this.frames.length}  t=${fmt(f.time)}`;
    if (f.freq !== undefined) lab = `Mode ${f.mode}: ${fmt(f.freq)} Hz`;
    if (f.eigenvalue !== undefined) lab = `Mode ${f.mode}: load multiplier ${fmt(f.eigenvalue)}`;
    if (f.compliance !== undefined) lab = `Iteration ${i + 1}: compliance ${fmt(f.compliance)}`;
    $("#frame-label").textContent = lab;
    if (ro) { ro._min = r.min; ro._max = r.max; ro._time = f.time; }
    if (this.treeSel.kind === "result") this.renderDetails();
  }

  unitOf(rtype) {
    const u = this.state.units;
    if (rtype.includes("deformation")) return u.length;
    if (rtype.includes("stress") || rtype === "von_mises" || rtype.startsWith("principal") || rtype === "tresca" || rtype.startsWith("shear")) return u.stress;
    if (rtype === "temperature") return u.temp;
    if (rtype === "peeq" || rtype === "strain_eq") return u.length + "/" + u.length;
    return "";
  }

  renderLegend(r, ro, scale) {
    const L = $("#legend");
    const name = ro?.name || this.state.result_types[r.type] || r.type;
    const unit = this.unitOf(r.type);
    const f = this.frames[this.ui.frame];
    let sub = `Type: ${this.state.result_types[r.type] || r.type}<br>Unit: ${esc(unit)}<br>Time: ${fmt(f.time)}`;
    if (f.freq !== undefined) sub = `Mode ${f.mode} · Frequency: ${fmt(f.freq)} Hz<br>(normalised mode shape)`;
    if (f.eigenvalue !== undefined) sub = `Mode ${f.mode} · Load multiplier: ${fmt(f.eigenvalue)}<br>(normalised mode shape)`;
    if (f.label) sub = `Type: ${this.state.result_types[r.type] || r.type}<br>Unit: ${esc(unit)}<br>${esc(f.label)}`;
    if (!r.values) { L.innerHTML = `<div class="lg-title">${esc(name)}</div><div class="lg-sub">not available for this analysis</div>`; return; }
    const n = this.viewer.bands, cols = this.viewer.bandColors || [];
    let rows = "";
    for (let k = n - 1; k >= 0; k--) {
      const v1 = r.min + (r.max - r.min) * (k + 1) / n;
      const c = cols[k] ? "#" + cols[k].getHexString() : "#888";
      rows += `<div class="lg-row"><span class="sw" style="background:${c}"></span><span class="lv">${fmtR(v1, r.min, r.max)}${k === n - 1 ? " Max" : ""}</span></div>`;
    }
    rows += `<div class="lg-row"><span class="sw empty"></span><span class="lv">${fmtR(r.min, r.min, r.max)} Min</span></div>`;
    L.innerHTML = `<div class="lg-title">${esc(name)}</div><div class="lg-sub">${sub}</div><div class="lg-bands">${rows}</div>
      <div class="lg-foot">Deformation scale: ${fmt(scale)}${this.ui.deform === "auto" ? " (auto)" : ""}</div>`;
  }

  toggleAnim() {
    this.ui.playing = !this.ui.playing;
    $("#anim-play").textContent = this.ui.playing ? "❚❚" : "▶";
    const proc = this.frames[0]?.procedure;
    if (this.ui.playing && (proc === "frequency" || proc === "buckle")) { this.viewer.play("mode"); return; }
    this.viewer.play(null);
    clearInterval(this._animT);
    if (this.ui.playing) {
      if (this.frames.length <= 1) { this.viewer.play("ramp"); return; }
      this._animT = setInterval(async () => {
        if (this._busy) return; this._busy = true;
        this.ui.frame = (this.ui.frame + 1) % this.frames.length; $("#frame-slider").value = this.ui.frame;
        await this.showResult(); this._busy = false;
      }, 350);
    }
  }

  onProbe(i, p) {
    if (!this.cur || !this.cur.values) return;
    const v = this.cur.values[i];
    const label = `${fmt(v)} ${this.unitOf(this.cur.type)}`;
    this.viewer.probes = [...(this.viewer.probes || []), { i, label }];
    this.log(`Probe at (${fmt(p.x)}, ${fmt(p.y)}, ${fmt(p.z)}): ${label}`);
  }

  onHover(h) {
    const el = $("#hover");
    if (!h || h.face == null || this.ui.mode === "result") { el.style.display = "none"; return; }
    const f = this.geom?.info?.faces?.find(x => x.id === h.face);
    el.style.display = ""; el.textContent = f ? `Face ${f.id} · ${f.type}${f.area ? " · A=" + fmt(f.area) : ""}` : `Face ${h.face}`;
  }

  // ---------------------------------------------------------------- selection
  onSelect(sel) {
    const n = sel.ids.size;
    const kind = { 3: "Body", 2: "Face", 1: "Edge", 0: "Vertex" }[sel.dim];
    let txt = n ? `${n} ${kind}${n > 1 ? (kind === "Body" ? "ies" : "s").replace("ys", "ies") : ""} selected` : "";
    const info = this.geom?.info;
    if (n && info) {
      const key = { 3: "bodies", 2: "faces", 1: "edges", 0: "vertices" }[sel.dim];
      const list = info[key].filter(x => sel.ids.has(x.id));
      if (sel.dim === 2) txt += ` · area ${fmt(list.reduce((s, x) => s + (x.area || 0), 0))} ${this.state.units.length}²`;
      if (sel.dim === 1) txt += ` · length ${fmt(list.reduce((s, x) => s + (x.length || 0), 0))} ${this.state.units.length}`;
      if (sel.dim === 3) txt += ` · volume ${fmt(list.reduce((s, x) => s + (x.volume || 0), 0))} ${this.state.units.length}³`;
      if (sel.dim === 0 && list.length === 1) txt += ` · (${list[0].xyz.map(fmt).join(", ")})`;
      if (sel.dim === 2 && list.length === 1) txt += ` · ${list[0].type}`;
    }
    $("#st-sel").textContent = txt;
  }
  selectionScope() {
    const s = this.viewer.selection; if (!s.ids.size) return null;
    const info = this.geom?.info; const key = { 3: "bodies", 2: "faces", 1: "edges", 0: "vertices" }[s.dim];
    const ids = [...s.ids];
    const sigs = ids.map(i => info?.[key]?.find(x => x.id === i)?.sig || null);
    return { dim: s.dim, ids, sigs };
  }
  scopeText(sc) {
    if (!sc) return "<i>No Selection</i>";
    if (sc.named) { const ns = this.project.named_selections.find(n => n.id === sc.named); return "Named: " + esc(ns?.name || sc.named); }
    if (sc.set) return "Set: " + esc([].concat(sc.set).join(", "));
    const kind = { 3: "Body", 2: "Face", 1: "Edge", 0: "Vertex" }[sc.dim];
    return `${sc.ids.length} ${kind}${sc.ids.length !== 1 ? (kind === "Body" ? "ies" : "s").replace("ys", "ies") : ""}`;
  }

  // ---------------------------------------------------------------- ribbon
  renderRibbon(tab) {
    this.tab = tab;
    const B = (id, label, icon, title = "") => `<button class="rb" data-act="${id}" title="${esc(title || label)}"><span class="ic">${icon}</span><span>${label}</span></button>`;
    const G = (title, inner) => `<div class="rgroup"><div class="rbtns">${inner}</div><div class="rtitle">${title}</div></div>`;
    const t = {
      home: G("Project", B("new", "New", "🗋") + B("open", "Open", "📂") + B("save", "Save", "💾")) +
        G("Import", B("import-cad", "CAD", "⬢", "Import STEP / IGES / BREP / STL / DXF") + B("import-deck", "Solver deck", "🗎", "Import Abaqus .inp, Nastran .bdf, ANSYS .cdb, gmsh .msh")) +
        G("Workflow", B("geom", "Geometry", "◈") + B("mesh", "Mesh", "▦") + B("solve", "Solve", "⚡", "Solve (F5)") + B("results", "Results", "◐")) +
        G("Units", `<select id="rb-units" class="rbsel"><option value="mm">mm · N · s · MPa · t</option><option value="m">m · N · s · Pa · kg (SI)</option></select>`),
      geometry: G("Primitives", B("f-box", "Box", "⬛") + B("f-cylinder", "Cylinder", "⬤") + B("f-sphere", "Sphere", "●") + B("f-cone", "Cone", "▲") + B("f-torus", "Torus", "◯")) +
        G("Sketch", B("sketch", "Sketch…", "✎", "2D sketch → extrude / revolve / surface") + B("import-dxf", "DXF", "⌗", "Extrude a DXF drawing")) +
        G("Thin structures", B("f-surface", "Surface", "▱", "Planar surface body (shell, from a sketch)") + B("f-line", "Line", "╱", "Line body (beam) from points")) +
        G("Operations", B("f-union", "Union", "∪", "Fuse selected bodies") + B("f-subtract", "Subtract", "∖", "First selected body minus the others") + B("f-intersect", "Intersect", "∩") +
          B("f-fillet", "Fillet", "◜", "Fillet selected edges (edge filter)") + B("f-chamfer", "Chamfer", "◸") ) +
        G("Transform", B("f-move", "Move", "✥") + B("f-rotate", "Rotate", "⟳") + B("f-mirror", "Mirror", "⇋") + B("f-pattern", "Pattern", "⁘")) +
        G("Import / Export", B("import-cad", "Import CAD", "⬢") + B("exp-step", "STEP", "⇩")) +
        G("Options", `<label class="chk"><input type="checkbox" id="rb-share" ${this.project?.geometry.share_topology ? "checked" : ""}> Share topology</label>`),
      mesh: G("Mesh", B("mesh", "Generate", "▦") + B("mesh-clear", "Clear", "✕")) +
        G("Controls", B("m-sizing", "Sizing", "◫", "Local element size on selected faces / edges / bodies")) +
        G("Defaults", `<label>Size <input type="number" id="rb-msize" step="any" placeholder="auto" value="${this.project?.mesh.size ?? ""}"></label>
          <label>Order <select id="rb-morder"><option value="2">Quadratic</option><option value="1">Linear</option></select></label>
          <label>Method <select id="rb-mkind"><option value="tet">Tetrahedrons</option><option value="hex">Hex (sweepable)</option><option value="hexdom">Hex dominant</option></select></label>`),
      setup: G("Supports", B("c-fixed", "Fixed", "▼") + B("c-displacement", "Displacement", "↧") + B("c-frictionless", "Frictionless", "⊥") + B("c-simply_supported", "Simple sup.", "△") + B("c-remote_displacement", "Remote disp.", "⊙") + B("c-rigid_wall", "Rigid wall", "▬")) +
        G("Connectors", B("c-remote_force", "Remote force", "⊕") + B("c-bolt_pretension", "Bolt preten.", "⚲") + B("c-point_mass", "Point mass", "●") + B("c-elastic_support", "Elastic sup.", "≈")) +
        G("Loads", B("c-force", "Force", "➜") + B("c-pressure", "Pressure", "⇶") + B("c-moment", "Moment", "↻") + B("c-gravity", "Gravity", "⇓") +
          B("c-acceleration", "Accel.", "⇟") + B("c-rotational_velocity", "Rotation", "⟲") + B("c-body_temperature", "Thermal", "🌡")) +
        G("Thermal", B("c-temperature", "Temperature", "T") + B("c-heat_flux", "Heat flux", "q") + B("c-convection", "Convection", "h") + B("c-heat_generation", "Heat gen.", "Q") + B("c-radiation", "Radiation", "☼")) +
        G("Connections", B("k-bonded", "Bonded", "⇄") + B("k-frictionless", "Frictionless", "⇆") + B("k-frictional", "Frictional", "≋") + B("k-detect", "Detect", "⌖", "Detect contacting faces between bodies")) +
        G("Selection", B("ns-create", "Named sel.", "☰", "Create a named selection from the current selection")),
      solve: G("Analysis", `<select id="rb-atype" class="rbsel">${Object.entries(this.state?.analyses || {}).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select>`) +
        G("Run", B("solve", "Solve", "⚡", "Solve (F5)") + B("stop", "Stop", "■")) +
        G("Settings", `<label class="chk"><input type="checkbox" id="rb-nlgeom"> Large deflection</label>
           <label>Substeps <input type="number" id="rb-sub" min="1" value="1"></label>
           <label>Solver <select id="rb-lsolver"><option value="auto">Program controlled</option><option value="direct">Direct (sparse LU)</option><option value="amg">Iterative AMG-PCG (GPU)</option><option value="jacobi">Iterative Jacobi-PCG</option></select></label>`),
      results: G("Deformation", B("r-total_deformation", "Total", "⇲") + B("r-deformation_x", "Dir. X", "x") + B("r-deformation_y", "Dir. Y", "y") + B("r-deformation_z", "Dir. Z", "z")) +
        G("Stress", B("r-von_mises", "Equivalent", "σ") + B("r-principal_max", "Max princ.", "σ₁") + B("r-principal_min", "Min princ.", "σ₃") + B("r-stress_xx", "Normal X", "σx") + B("r-tresca", "Intensity", "τ")) +
        G("Other", B("r-peeq", "Plastic strain", "εp") + B("r-strain_eq", "Total strain", "ε") + B("r-temperature", "Temperature", "T") + B("r-safety_factor", "Safety factor", "SF") + B("r-fatigue_life", "Fatigue life", "N") + B("r-fatigue_sf", "Fatigue SF", "SFf") + B("r-density", "Topology", "◧")) +
        G("Display", `<label>Scale <select id="rb-scale"><option value="auto">Auto</option><option value="0">Undeformed</option><option value="1">True scale</option><option value="10">10×</option><option value="100">100×</option><option value="1000">1000×</option></select></label>
          <label>Bands <select id="rb-bands"><option>5</option><option>7</option><option selected>9</option><option>12</option><option>16</option><option value="smooth">Smooth</option></select></label>
          <label class="chk"><input type="checkbox" id="rb-mesh"> Show mesh</label><label class="chk"><input type="checkbox" id="rb-undef"> Undeformed edges</label>`) +
        G("Tools", B("probe", "Probe", "⌖", "Click on the model to read values") + B("probe-clear", "Clear probes", "✕") + B("anim", "Animate", "▶") + B("exp-vtu", "ParaView", "⇩")),
      display: G("View", B("v-iso", "Iso", "◇") + B("v-front", "Front", "▢") + B("v-top", "Top", "▭") + B("v-right", "Right", "▯") + B("v-fit", "Fit", "⤢")) +
        G("Style", `<label class="chk"><input type="checkbox" id="rb-edges" checked> Edges</label><label class="chk"><input type="checkbox" id="rb-wire"> Wireframe</label>
          <label class="chk"><input type="checkbox" id="rb-mlines" checked> Mesh lines</label>
          <label>Colormap <select id="rb-cmap"><option value="rainbow">Rainbow</option><option value="viridis">Viridis</option><option value="gray">Gray</option></select></label>`) +
        G("Section plane", `<label class="chk"><input type="checkbox" id="rb-sec"> On</label><select id="rb-secax"><option value="0">X</option><option value="1">Y</option><option value="2">Z</option></select>
          <input type="range" id="rb-secpos" min="0" max="1" step="0.005" value="0.5"><label class="chk"><input type="checkbox" id="rb-secflip"> Flip</label>`) +
        G("Capture", B("shot", "Image", "📷")),
    };
    $("#ribbon").innerHTML = t[tab] || "";
    $$("#ribbon [data-act]").forEach(b => b.onclick = () => this.action(b.dataset.act));
    const p = this.project; if (!p) return;
    const on = (id, ev, fn) => { const el = $(id); if (el) el[ev] = fn; };
    if ($("#rb-units")) { $("#rb-units").value = p.units; on("#rb-units", "onchange", e => this.changeUnits(e.target.value)); }
    on("#rb-share", "onchange", e => { p.geometry.share_topology = e.target.checked; this.regenerate(); });
    if ($("#rb-morder")) { $("#rb-morder").value = String(p.mesh.order); $("#rb-mkind").value = p.mesh.kind; }
    on("#rb-msize", "onchange", e => { p.mesh.size = e.target.value ? +e.target.value : null; this.meshChanged(); });
    on("#rb-morder", "onchange", e => { p.mesh.order = +e.target.value; this.meshChanged(); });
    on("#rb-mkind", "onchange", e => { p.mesh.kind = e.target.value; this.meshChanged(); });
    if ($("#rb-atype")) {
      $("#rb-atype").value = p.analysis.type; $("#rb-nlgeom").checked = !!p.analysis.nlgeom; $("#rb-sub").value = p.analysis.substeps; $("#rb-lsolver").value = p.analysis.linear_solver || "auto";
      on("#rb-atype", "onchange", e => { p.analysis.type = e.target.value; this.render(); this.push(); });
      on("#rb-nlgeom", "onchange", e => { p.analysis.nlgeom = e.target.checked; this.render(); });
      on("#rb-sub", "onchange", e => { p.analysis.substeps = +e.target.value; this.render(); });
      on("#rb-lsolver", "onchange", e => { p.analysis.linear_solver = e.target.value; this.render(); });
    }
    if ($("#rb-scale")) {
      $("#rb-scale").value = this.ui.deform; $("#rb-mesh").checked = this.ui.showMesh; $("#rb-undef").checked = this.ui.undeformed;
      on("#rb-scale", "onchange", e => { this.ui.deform = e.target.value; this.showResult(); });
      on("#rb-bands", "onchange", e => { const v = e.target.value; this.viewer.bands = v === "smooth" ? 9 : +v; this.viewer.smooth = v === "smooth"; this.showResult(); });
      on("#rb-mesh", "onchange", e => { this.ui.showMesh = e.target.checked; this.showResult(); });
      on("#rb-undef", "onchange", e => { this.ui.undeformed = e.target.checked; this.showResult(); });
    }
    on("#rb-edges", "onchange", e => this.viewer.setEdges(e.target.checked));
    on("#rb-wire", "onchange", e => this.viewer.setWire(e.target.checked));
    on("#rb-mlines", "onchange", e => this.viewer.setMeshLines(e.target.checked));
    on("#rb-cmap", "onchange", e => { this.viewer.cmapName = e.target.value; this.showResult(); });
    const sec = () => this.viewer.setSection($("#rb-sec").checked, +$("#rb-secax").value, +$("#rb-secpos").value, $("#rb-secflip").checked);
    ["#rb-sec", "#rb-secax", "#rb-secflip"].forEach(id => on(id, "onchange", sec)); on("#rb-secpos", "oninput", sec);
  }

  meshChanged() { this.meshDisp = null; this.status.mesh = "stale"; this.renderTree(); this.push(); }

  // ---------------------------------------------------------------- actions
  async action(a) {
    const p = this.project;
    const diag = this.geom?.info?.diag || 100;
    const s = v => +(v).toPrecision(2);
    if (a === "new") return this.newProject();
    if (a === "open") return $("#file-proj").click();
    if (a === "save") return this.saveProject();
    if (a === "import-cad") return $("#file-cad").click();
    if (a === "import-dxf") return $("#file-cad").click();
    if (a === "import-deck") return $("#file-deck").click();
    if (a === "geom") { this.setMode("geometry"); return this.select({ kind: "geometry" }); }
    if (a === "mesh") return this.meshNow();
    if (a === "mesh-clear") { this.meshDisp = null; this.status.mesh = "none"; this.setMode("geometry"); return this.renderTree(); }
    if (a === "solve") return this.solve();
    if (a === "stop") return this.stop();
    if (a === "results") return this.setMode("result");
    if (a === "sketch") return openSketcher(this, null);
    if (a === "exp-step") return this.exportFile("step");
    if (a === "exp-vtu") return this.exportFile("vtu");
    if (a === "shot") return $("#btn-shot").click();
    if (a === "probe") { this.viewer.probeMode = !this.viewer.probeMode; this.log("Probe mode " + (this.viewer.probeMode ? "on – click the model" : "off")); return; }
    if (a === "probe-clear") { this.viewer.probes = []; return; }
    if (a === "anim") return this.toggleAnim();
    if (a.startsWith("v-")) return a === "v-fit" ? this.viewer.fit() : this.viewer.view(a.slice(2));
    if (a.startsWith("f-")) return this.addFeature(a.slice(2), diag, s);
    if (a.startsWith("c-")) return this.addCondition(a.slice(2));
    if (a.startsWith("r-")) return this.addResult(a.slice(2));
    if (a === "m-sizing") {
      const sc = this.selectionScope();
      const it = { id: uid("S", p.mesh.local), name: "Sizing", scope: sc, size: s((p.mesh.size || diag / 20) / 3) };
      p.mesh.local.push(it); this.meshChanged(); return this.select({ kind: "sizing", id: it.id });
    }
    if (a === "k-bonded" || a === "k-frictionless" || a === "k-frictional") {
      const sc = this.selectionScope();
      const c = { id: uid("C", p.connections), type: a.slice(2), name: { "k-bonded": "Bonded", "k-frictionless": "Frictionless", "k-frictional": "Frictional" }[a], master: sc, slave: null, fkn: 1.0, mu: 0.2 };
      p.connections.push(c); this.push(); this.log("Contact created: select the target faces with 'Apply' in Details if needed.");
      return this.select({ kind: "contact", id: c.id });
    }
    if (a === "k-detect") {
      const r = await api.post("/api/detect_contacts", {});
      const finfo = Object.fromEntries((this.geom?.info?.faces || []).map(f => [f.id, f]));
      for (const [fa, fb] of r.pairs) {
        p.connections.push({ id: uid("C", p.connections), type: "bonded", name: `Bonded – F${fa}/F${fb}`,
          master: { dim: 2, ids: [fa], sigs: [finfo[fa]?.sig] }, slave: { dim: 2, ids: [fb], sigs: [finfo[fb]?.sig] } });
      }
      this.log(r.pairs.length ? `${r.pairs.length} contact pairs detected (bonded)` : "no contacting faces between different bodies found" + (p.geometry.share_topology ? " (share topology is on: touching bodies are already conformally joined)" : ""));
      this.push(); return this.render();
    }
    if (a === "ns-create") {
      const sc = this.selectionScope(); if (!sc) return this.log("select entities first", "warn");
      const n = { id: uid("NS", p.named_selections), name: `Selection ${p.named_selections.length + 1}`, scope: sc };
      p.named_selections.push(n); this.push(); return this.select({ kind: "ns", id: n.id });
    }
  }

  liveFeatures() {
    // features whose bodies are still alive (not consumed by a later boolean)
    const consumed = new Set();
    for (const f of this.project.geometry.features) {
      if (f.suppressed) continue;
      if (f.type === "boolean") { [].concat(f.target, f.keep_tool ? [] : f.tool).forEach(x => consumed.add(x)); }
      if (["fillet", "chamfer"].includes(f.type)) [].concat(f.target).forEach(x => consumed.add(x));
      if (f.type === "transform" && !f.copy) [].concat(f.target).forEach(x => consumed.add(x));
      if (f.type === "pattern" && f.fuse) [].concat(f.target).forEach(x => consumed.add(x));
    }
    return this.project.geometry.features.filter(f => !f.suppressed && !consumed.has(f.id) && !["boolean_tool"].includes(f.type));
  }

  featureOfBodies(ids) {
    // the CAD kernel reports which feature owns every final body (through fragments)
    const bodies = this.geom?.info?.bodies || [];
    const out = [];
    for (const id of ids) { const f = bodies.find(b => b.id === id)?.feature; if (f && !out.includes(f)) out.push(f); }
    return out;
  }
  bodyOfEdge(eid) {
    const info = this.geom?.info; const e = info?.edges.find(x => x.id === eid); if (!e) return null;
    for (const fid of e.faces || []) { const f = info.faces.find(x => x.id === fid); if (f?.bodies?.length) return f.bodies[0]; }
    return null;
  }

  async addFeature(kind, diag, s) {
    const p = this.project, F = p.geometry.features;
    const id = uid("F", F);
    const bb = this.geom?.info?.bbox || [0, 0, 0, 100, 100, 100];
    const L = s(diag ? diag / 3 : 50);
    let f = null;
    const ox = this.geom?.info ? s(bb[3]) : 0;
    if (kind === "box") f = { type: "box", x: ox, y: 0, z: 0, dx: L, dy: L, dz: L };
    if (kind === "cylinder") f = { type: "cylinder", x: ox, y: 0, z: 0, dx: 0, dy: 0, dz: L, r: s(L / 3) };
    if (kind === "sphere") f = { type: "sphere", x: ox, y: 0, z: 0, r: s(L / 2) };
    if (kind === "cone") f = { type: "cone", x: ox, y: 0, z: 0, dx: 0, dy: 0, dz: L, r1: s(L / 3), r2: 0 };
    if (kind === "torus") f = { type: "torus", x: ox, y: 0, z: 0, r1: s(L / 2), r2: s(L / 8) };
    if (kind === "surface") f = { type: "surface", plane: "XY", offset: 0, loops: [{ rect: [ox, 0, ox + L, L] }] };
    if (kind === "line") f = { type: "line", points: [[ox, 0, 0], [ox + L, 0, 0]] };
    if (["union", "subtract", "intersect"].includes(kind)) {
      const sel = this.viewer.selection;
      let ids = sel.dim === 3 ? [...sel.ids] : [];
      const feats = ids.length >= 2 ? this.featureOfBodies(ids) : this.liveFeatures().slice(-2).map(f => f.id);
      if (ids.length >= 2 && feats.length < 2) return this.log("the selected bodies belong to the same feature (split by Share topology) – turn Share topology off to combine overlapping bodies", "warn");
      if (feats.length < 2) return this.log("select at least two bodies (body filter) or create two bodies first", "warn");
      f = { type: "boolean", op: { union: "fuse", subtract: "cut", intersect: "intersect" }[kind], target: [feats[0]], tool: feats.slice(1) };
    }
    if (kind === "fillet" || kind === "chamfer") {
      const sel = this.viewer.selection;
      if (sel.dim !== 1 || !sel.ids.size) { this.setFilter("edge"); return this.log("select edges first (edge filter enabled), then press Fillet/Chamfer again", "warn"); }
      const einfo = this.geom.info.edges;
      const edges = [...sel.ids].map(i => ({ id: i, sig: einfo.find(e => e.id === i)?.sig }));
      const bid = this.bodyOfEdge(edges[0].id);
      const tgt = this.featureOfBodies([bid])[0] || this.liveFeatures().slice(-1)[0]?.id;
      f = { type: kind, target: [tgt], edges: edges.map(e => e.sig), r: s(diag / 60), d: s(diag / 60) };
    }
    if (["move", "rotate", "mirror", "pattern"].includes(kind)) {
      const sel = this.viewer.selection;
      const tgt = sel.dim === 3 && sel.ids.size ? this.featureOfBodies([...sel.ids]) : [this.liveFeatures().slice(-1)[0]?.id];
      if (!tgt[0]) return this.log("no body to transform", "warn");
      if (kind === "move") f = { type: "transform", op: "translate", target: tgt, vector: [L, 0, 0], copy: false };
      if (kind === "rotate") f = { type: "transform", op: "rotate", target: tgt, point: [0, 0, 0], axis: [0, 0, 1], angle: 90, copy: false };
      if (kind === "mirror") f = { type: "transform", op: "mirror", target: tgt, point: [0, 0, 0], normal: [1, 0, 0], copy: true };
      if (kind === "pattern") f = { type: "pattern", kind: "linear", target: tgt, direction: [1, 0, 0], count: 3, spacing: L, fuse: false };
    }
    if (!f) return;
    f.id = id; f.name = `${FEATURE_LABEL[f.type]}${f.op ? " (" + f.op + ")" : ""} ${id.slice(1)}`;
    F.push(f);
    this._keepView = this.geom?.info != null;
    const ok = await this.regenerate();
    this._keepView = false;
    if (ok) this.select({ kind: "feature", id });
  }

  async addSketchFeature(f, existing, boolOp, target) {
    const p = this.project, F = p.geometry.features;
    if (existing) { Object.assign(existing, f); await this.regenerate(); return this.select({ kind: "feature", id: existing.id }); }
    f.id = uid("F", F); f.name = `${{ revolve: "Revolve", surface: "Surface", extrude: "Extrude" }[f.type]} ${f.id.slice(1)}`; F.push(f);
    if (boolOp !== "new" && target) {
      const b = { id: uid("F", F), type: "boolean", op: boolOp, target: [target], tool: [f.id] }; b.name = `Boolean (${boolOp}) ${b.id.slice(1)}`; F.push(b);
    }
    await this.regenerate(); this.select({ kind: "feature", id: f.id });
  }

  addCondition(type) {
    const p = this.project, meta = this.state.conditions[type];
    let sc = this.selectionScope();
    if (meta.dims.length && sc && !meta.dims.includes(sc.dim)) { this.log(`${meta.label} accepts ${meta.dims.map(d => ["vertices", "edges", "faces", "bodies"][d]).join("/")} – selection ignored`, "warn"); sc = null; }
    const id = uid(meta.group === "support" ? "B" : meta.group === "thermal" ? "T" : "L", p.conditions);
    const c = { id, type, name: meta.label, scope: meta.dims.length ? sc : null };
    if (type === "force") c.vector = [0, -1000, 0];
    if (type === "pressure") c.value = 1.0;
    if (type === "moment") c.vector = [0, 0, 1e4];
    if (type === "displacement") c.components = [0, 0, 0];
    if (type === "gravity") c.vector = [0, -1, 0];
    if (type === "acceleration") c.vector = [0, -9806.65, 0];
    if (type === "rotational_velocity") { c.value = 100; c.axis = [0, 0, 1]; c.point = [0, 0, 0]; }
    if (type === "body_temperature") c.value = 100;
    if (type === "temperature") c.value = 100;
    if (type === "heat_flux") c.value = 0.01;
    if (type === "convection") { c.h = 2e-5; c.ambient = 22; }
    if (type === "heat_generation") c.value = 1e-4;
    if (type === "remote_displacement") { c.point = null; c.components = [0, 0, 0]; c.rotations = [0, 0, 0]; }
    if (type === "remote_force") { c.point = null; c.vector = [0, -1000, 0]; c.moment = [0, 0, 0]; }
    if (type === "bolt_pretension") { c.value = 10000; c.axis = null; }
    if (type === "point_mass") c.value = 0.001;
    if (type === "elastic_support") c.value = 10;
    if (type === "radiation") { c.emissivity = 0.8; c.ambient = 22; }
    if (type === "rigid_wall") { const bb = this.geom?.info?.bbox || [0, 0, 0, 1, 1, 1]; c.point = [0, bb[4] * 1.001, 0]; c.normal = [0, -1, 0]; c.motion = [0, -0.01 * (this.geom?.info?.diag || 1), 0]; }
    if (["temperature", "heat_flux", "convection", "heat_generation"].includes(type) && !p.analysis.type.startsWith("thermal")) {
      this.log("thermal boundary condition added – switch the analysis to Steady-State Thermal or Thermal-Stress to use it", "warn");
    }
    p.conditions.push(c); this.push();
    this.select({ kind: "condition", id });
  }

  addResult(type) {
    const p = this.project;
    const r = { id: uid("R", p.results), type, name: this.state.result_types[type] };
    p.results.push(r); this.push();
    this.select({ kind: "result", id: r.id });
  }

  async changeUnits(u) {
    if (u === this.project.units) return;
    if (!confirm("Changing the unit system converts material properties; geometry dimensions are kept as numbers. Continue?")) { $("#rb-units").value = this.project.units; return; }
    const f = u === "m" ? { E: 1e6, density: 1e12, cp: 1e-6, yield: 1e6, tangent: 1e6 } : { E: 1e-6, density: 1e-12, cp: 1e6, yield: 1e-6, tangent: 1e-6 };
    for (const m of this.project.materials) for (const k in f) if (m[k] !== undefined) m[k] *= f[k];
    this.project.units = u; await this.push(); this.state = await api.get("/api/state"); this.render();
  }

  // ---------------------------------------------------------------- files
  async newProject() { const st = await api.post("/api/new", { units: this.project?.units || "mm" }); this.state = st; this.project = st.project; this.geom = null; this.meshDisp = null; this.frames = [];
    this.viewer.setGeometry({ info: null, faces: [], edges: [] }); this.viewer.clear("geom"); this.viewer.clear("mesh"); this.viewer.clear("result"); this.setMode("geometry"); this.select({ kind: "model" }); this.render(); this.log("New project"); }
  async saveProject() {
    const blob = new Blob([JSON.stringify(this.project, null, 1)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = (this.project.name || "project").replace(/\s+/g, "_") + ".gfem.json"; a.click();
    await api.post("/api/save", {}); this.log("Project saved (download + server workdir)");
  }
  async openProject(file) {
    if (!file) return; const text = await file.text(); const project = JSON.parse(text);
    const r = await api.post("/api/open", { project }); this.applyOpen(r); this.log("Opened " + file.name);
  }
  async loadExample(key) {
    this.vpMsg("Loading example…");
    try { const r = await api.post("/api/example", { key }); this.applyOpen(r); this.log("Example loaded: " + r.state.project.name + " – press Solve (F5)"); }
    catch (e) { this.log(e.message, "err"); }
    this.vpMsg("");
  }
  applyOpen(r) {
    this.state = r.state; this.project = r.state.project; this.frames = []; this.meshDisp = null;
    this.geom = r; this.applyBodyColors(); this.viewer.setGeometry(r, true); this.status = { geometry: r.info ? "ok" : "none", mesh: "none", solve: "none" };
    $("#proj-name").value = this.project.name; this.setMode("geometry"); this.select({ kind: "model" }); this.renderRibbon(this.tab); this.render();
  }
  async importFile(file, kind) {
    if (!file) return;
    this.vpMsg(`Uploading ${file.name}…`);
    try {
      const up = await api.upload(file);
      this.vpMsg(`Importing ${file.name}…`);
      const r = await api.post("/api/import", { path: up.path });
      if (r.error) { this.log("Import failed: " + r.error, "err"); this.vpMsg(""); return; }
      this.project = r.project; this.geom = r; this.applyBodyColors(); this.viewer.setGeometry(r, true);
      this.status.geometry = "ok";
      if (r.mesh) { this.setMeshDisplay(r.mesh, false); this.log(`Solver deck imported: ${r.mesh.stats.nodes} nodes, ${r.mesh.stats.elements} elements ${JSON.stringify(r.mesh.stats.types)}`); }
      else this.log(`CAD imported: ${file.name} (${r.info.bodies.length} bodies, ${r.info.faces.length} faces)`);
      this.setMode("geometry");
    } catch (e) { this.log("Import failed: " + e.message, "err"); }
    this.vpMsg(""); this.render();
    $("#file-cad").value = ""; $("#file-deck").value = "";
  }
  exportFile(fmt) { const a = document.createElement("a"); a.href = "/api/export?fmt=" + fmt; a.click(); this.log("Export " + fmt + " requested"); }

  // ---------------------------------------------------------------- tree
  render() { this.renderTree(); this.renderDetails(); $("#st-units").textContent = this.state ? `Units: ${Object.values(this.state.units).slice(0, 4).join(", ")}` : ""; $("#proj-name").value = this.project?.name || ""; }

  treeModel() {
    const p = this.project, info = this.geom?.info;
    const stIcon = s => ({ ok: "ok", none: "todo", stale: "stale", failed: "fail", running: "run" }[s] || "");
    const condState = c => (this.state.conditions[c.type]?.dims.length && !(c.scope && (c.scope.ids?.length || c.scope.named || c.scope.set))) ? "fail" : "ok";
    const feats = p.geometry.features.map(f => ({ kind: "feature", id: f.id, label: f.name || f.id, icon: "feature", st: f.suppressed ? "sup" : "ok" }));
    const bodies = (info?.bodies || []).map(b => { const pb = p.bodies[String(b.id)] || {}; return { kind: "body", id: String(b.id), label: pb.name || `Body ${b.id}`, icon: "body", st: pb.suppressed ? "sup" : "ok" }; })
      .concat((info?.sheets || []).map(b => { const pb = p.bodies["S" + b.id] || {}; return { kind: "body", id: "S" + b.id, label: (pb.name || `Surface body ${b.id}`) + ` (t=${pb.thickness ?? "?"})`, icon: "body", st: pb.suppressed ? "sup" : "ok" }; }))
      .concat((info?.lines || []).map(b => { const pb = p.bodies["L" + b.id] || {}; return { kind: "body", id: "L" + b.id, label: (pb.name || `Line body ${b.id}`) + ` (${pb.profile || "RECT"})`, icon: "body", st: pb.suppressed ? "sup" : "ok" }; }));
    const anal = this.state.analyses[p.analysis.type];
    const conds = p.conditions.filter(c => this.relevant(c)).map(c => ({ kind: "condition", id: c.id, label: c.name, icon: this.state.conditions[c.type]?.group || "load", st: c.suppressed ? "sup" : condState(c) }));
    return { kind: "model", label: p.name || "Model", icon: "model", open: true, children: [
      { kind: "geometry", label: "Geometry", icon: "geometry", st: stIcon(this.status.geometry), open: true, children: [
        { kind: "features", label: "Features", icon: "feature", open: true, children: feats }, { kind: "bodies", label: `Bodies (${bodies.length})`, icon: "body", open: true, children: bodies }] },
      { kind: "materials", label: "Materials", icon: "materials", open: true, children: p.materials.map(m => ({ kind: "material", id: m.id, label: m.name, icon: "material", color: m.color })) },
      { kind: "connections", label: "Connections", icon: "connections", children: p.connections.map(c => ({ kind: "contact", id: c.id, label: c.name, icon: "contact", st: c.master && c.slave ? "ok" : "fail" })) },
      { kind: "named", label: "Named Selections", icon: "named", children: p.named_selections.map(n => ({ kind: "ns", id: n.id, label: n.name, icon: "ns" })) },
      { kind: "mesh", label: "Mesh", icon: "mesh", st: stIcon(this.status.mesh), children: p.mesh.local.map(l => ({ kind: "sizing", id: l.id, label: l.name, icon: "sizing" })) },
      { kind: "analysis", label: anal, icon: "analysis", open: true, children: [
        { kind: "settings", label: "Analysis Settings", icon: "settings" }, ...conds,
        { kind: "solution", label: "Solution", icon: "solution", st: stIcon(this.status.solve), open: true, children: [
          { kind: "solinfo", label: "Solution Information", icon: "solinfo" },
          ...p.results.map(r => ({ kind: "result", id: r.id, label: r.name, icon: "result", st: this.frames.length ? "ok" : "todo" }))] }] }] };
  }
  relevant(c) {
    const t = this.project.analysis.type, g = this.state.conditions[c.type]?.group;
    if (t === "thermal" || t === "thermal_transient") return g === "thermal";
    if (t === "thermal_stress") return true;
    return g !== "thermal";
  }

  renderTree() {
    if (!this.project) return;
    const root = this.treeModel();
    this._open = this._open || {};
    const sel = this.treeSel;
    const html = (n, depth) => {
      const key = n.kind + ":" + (n.id ?? "");
      const isOpen = this._open[key] ?? n.open ?? false;
      const kids = n.children || [];
      const active = sel.kind === n.kind && (sel.id ?? null) === (n.id ?? null);
      const st = n.st ? `<span class="tst ${n.st}"></span>` : "";
      const sw = n.color ? `<span class="tsw" style="background:${n.color}"></span>` : "";
      let s = `<div class="tn ${active ? "sel" : ""}" data-k="${n.kind}" data-id="${esc(n.id ?? "")}" style="padding-left:${6 + depth * 14}px">
        <span class="tw">${kids.length ? (isOpen ? "▾" : "▸") : ""}</span>${st}<span class="ti">${ICON[n.icon] || "•"}</span>${sw}<span class="tl">${esc(n.label)}</span></div>`;
      if (kids.length && isOpen) s += kids.map(k => html(k, depth + 1)).join("");
      return s;
    };
    $("#tree").innerHTML = html(root, 0);
    $$("#tree .tn").forEach(el => {
      el.onclick = e => {
        const key = el.dataset.k + ":" + el.dataset.id;
        if (e.target.classList.contains("tw")) { this._open[key] = !(this._open[key] ?? (el.querySelector(".tw").textContent === "▾")); return this.renderTree(); }
        this.select({ kind: el.dataset.k, id: el.dataset.id || undefined });
      };
      el.ondblclick = () => { const k = el.dataset.k; if (k === "solution" || k === "analysis") this.solve(); if (k === "mesh") this.meshNow(); if (k === "geometry") this.regenerate(); };
      el.oncontextmenu = e => { e.preventDefault(); this.select({ kind: el.dataset.k, id: el.dataset.id || undefined }); this.contextMenu(e, el.dataset.k, el.dataset.id); };
    });
  }

  contextMenu(e, kind, id) {
    document.querySelector(".ctx")?.remove();
    const items = [];
    const deletable = ["feature", "material", "contact", "ns", "sizing", "condition", "result"];
    if (deletable.includes(kind)) items.push(["Delete", () => this.deleteTreeSel()]);
    if (["feature", "condition", "contact", "body", "sizing"].includes(kind)) items.push(["Suppress / Unsuppress", () => this.toggleSuppress()]);
    if (kind === "feature") items.push(["Edit sketch…", () => { const f = this.project.geometry.features.find(x => x.id === id); if (f && (f.type === "extrude" || f.type === "revolve")) openSketcher(this, f); }]);
    if (kind === "materials") items.push(["Add material", () => this.addMaterial()]);
    if (kind === "mesh") items.push(["Generate Mesh", () => this.meshNow()]);
    if (kind === "solution" || kind === "analysis") items.push(["Solve", () => this.solve()]);
    if (kind === "result") items.push(["Evaluate / show", () => this.showResult()], ["Export ParaView (.vtu)", () => this.exportFile("vtu")]);
    if (!items.length) return;
    const m = document.createElement("div"); m.className = "ctx"; m.style.left = e.clientX + "px"; m.style.top = e.clientY + "px";
    m.innerHTML = items.map((it, i) => `<a data-i="${i}">${it[0]}</a>`).join("");
    document.body.appendChild(m);
    m.querySelectorAll("a").forEach(a => a.onclick = () => { m.remove(); items[+a.dataset.i][1](); });
    setTimeout(() => window.addEventListener("click", () => m.remove(), { once: true }), 0);
  }

  findItem(kind, id) {
    const p = this.project;
    return { feature: () => p.geometry.features.find(f => f.id === id), material: () => p.materials.find(m => m.id === id), contact: () => p.connections.find(c => c.id === id),
      ns: () => p.named_selections.find(n => n.id === id), sizing: () => p.mesh.local.find(l => l.id === id), condition: () => p.conditions.find(c => c.id === id),
      result: () => p.results.find(r => r.id === id), body: () => p.bodies[id] }[kind]?.();
  }
  async deleteTreeSel() {
    const { kind, id } = this.treeSel; const p = this.project;
    const lists = { feature: p.geometry.features, material: p.materials, contact: p.connections, ns: p.named_selections, sizing: p.mesh.local, condition: p.conditions, result: p.results };
    const L = lists[kind]; if (!L) return;
    if (kind === "material" && p.materials.length <= 1) return this.log("at least one material is required", "warn");
    const i = L.findIndex(x => x.id === id); if (i < 0) return;
    L.splice(i, 1);
    this.select({ kind: { feature: "geometry", material: "materials", contact: "connections", ns: "named", sizing: "mesh", condition: "analysis", result: "solution" }[kind] });
    if (kind === "feature") await this.regenerate(); else { if (kind === "sizing") this.meshChanged(); this.push(); this.render(); }
  }
  async toggleSuppress() {
    const it = this.findItem(this.treeSel.kind, this.treeSel.id); if (!it) return;
    it.suppressed = !it.suppressed;
    if (this.treeSel.kind === "feature") await this.regenerate(); else { this.push(); this.render(); }
  }

  select(s) {
    this.treeSel = s;
    this.renderTree(); this.renderDetails();
    // highlight scoped entities
    const it = this.findItem(s.kind, s.id);
    const hl = [];
    const add = (sc, color) => { if (sc && sc.ids) hl.push({ dim: sc.dim, ids: sc.ids, color }); if (sc && sc.named) { const n = this.project.named_selections.find(x => x.id === sc.named); if (n) add(n.scope, color); } };
    if (s.kind === "condition" && it) add(it.scope, this.state.conditions[it.type]?.group === "support" ? "#3b6fd8" : this.state.conditions[it.type]?.group === "thermal" ? "#e8890c" : "#d8312f");
    if (s.kind === "contact" && it) { add(it.master, "#d8312f"); add(it.slave, "#3b6fd8"); }
    if ((s.kind === "ns" || s.kind === "sizing") && it) add(it.scope, "#8e44ad");
    if (s.kind === "body") hl.push({ dim: 3, ids: [+s.id], color: "#f1c40f" });
    if (s.kind === "settings" || s.kind === "analysis") for (const c of this.project.conditions.filter(c => this.relevant(c) && !c.suppressed)) add(c.scope, this.state.conditions[c.type]?.group === "support" ? "#3b6fd8" : "#d8312f");
    this.viewer.setHighlights(hl);
    if (s.kind === "result" && this.frames.length) this.showResult();
    if (s.kind === "solinfo") { this.showBottom("solinfo"); }
    if (["geometry", "feature", "body", "materials", "material", "condition", "contact", "ns", "settings", "analysis"].includes(s.kind) && this.ui.mode === "result") this.setMode("geometry");
    if ((s.kind === "mesh" || s.kind === "sizing") && this.meshDisp && this.ui.mode !== "mesh") this.setMode("mesh");
  }

  // ---------------------------------------------------------------- details
  renderDetails() {
    if (!this.project) return;
    const { kind, id } = this.treeSel; const p = this.project; const u = this.state.units;
    const it = this.findItem(kind, id);
    const rows = [];          // [section, label, html]
    const sec = t => rows.push({ sec: t });
    const num = (label, obj, key, unit = "", opts = {}) => rows.push({ label, unit, html: `<input type="number" step="any" data-k="${key}" value="${obj[key] ?? ""}" ${opts.ro ? "readonly" : ""} placeholder="${opts.ph || ""}">`, obj, key, t: "num" });
    const txt = (label, obj, key) => rows.push({ label, html: `<input type="text" data-k="${key}" value="${esc(obj[key] ?? "")}">`, obj, key, t: "txt" });
    const sel = (label, obj, key, opts) => rows.push({ label, html: `<select data-k="${key}">${opts.map(([v, l]) => `<option value="${esc(v)}" ${String(obj[key]) === String(v) ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>`, obj, key, t: "sel" });
    const chk = (label, obj, key) => rows.push({ label, html: `<input type="checkbox" data-k="${key}" ${obj[key] ? "checked" : ""}>`, obj, key, t: "chk" });
    const vec = (label, obj, key, unit = "") => rows.push({ label, unit, html: `<div class="vec">${[0, 1, 2].map(i => `<input type="number" step="any" data-k="${key}" data-i="${i}" value="${(obj[key] || [0, 0, 0])[i] ?? ""}" placeholder="free">`).join("")}</div>`, obj, key, t: "vec" });
    const ro = (label, v) => rows.push({ label, html: `<span class="ro">${v}</span>` });
    const scope = (label, obj, key) => rows.push({ label, html: `<span class="scope">${this.scopeText(obj[key])}</span><button data-apply="${key}">Apply</button><button data-clear="${key}">✕</button>${p.named_selections.length ? `<select data-named="${key}"><option value="">Named…</option>${p.named_selections.map(n => `<option value="${n.id}">${esc(n.name)}</option>`).join("")}</select>` : ""}`, obj, key, t: "scope" });
    let title = "Details";
    const info = this.geom?.info;
    if (kind === "model") {
      title = "Details of \"Model\""; sec("Project"); txt("Name", p, "name");
      ro("Unit system", Object.values(u).slice(0, 5).join(", ")); ro("Compute device", `${this.state.system.backend} – ${this.state.system.devices.join(", ")}`);
      ro("Workdir", esc(this.state.workdir)); sec("Statistics"); ro("Bodies", info?.bodies.length ?? 0); ro("Faces", info?.faces.length ?? 0);
      ro("Nodes", this.meshStats?.nodes ?? "-"); ro("Elements", this.meshStats?.elements ?? "-");
    } else if (kind === "geometry" || kind === "features" || kind === "bodies") {
      title = "Details of \"Geometry\""; sec("Definition"); chk("Share topology (conformal multi-body mesh)", p.geometry, "share_topology");
      if (info) { sec("Bounding box"); ro("Length X", fmt(info.bbox[3] - info.bbox[0]) + " " + u.length); ro("Length Y", fmt(info.bbox[4] - info.bbox[1]) + " " + u.length); ro("Length Z", fmt(info.bbox[5] - info.bbox[2]) + " " + u.length);
        sec("Statistics"); ro("Bodies", info.bodies.length); ro("Faces", info.faces.length); ro("Edges", info.edges.length); ro("Vertices", info.vertices.length);
        const vol = info.bodies.reduce((s, b) => s + (b.volume || 0), 0); ro("Volume", fmt(vol) + " " + u.length + "³");
        const mass = info.bodies.reduce((s, b) => { const m = p.materials.find(m => m.id === p.bodies[String(b.id)]?.material); return s + (b.volume || 0) * (m?.density || 0); }, 0);
        ro("Mass", fmt(mass) + " " + u.mass); }
    } else if (kind === "feature" && it) {
      title = `Details of "${it.name}"`; sec("Feature"); txt("Name", it, "name"); ro("Type", FEATURE_LABEL[it.type]); chk("Suppressed", it, "suppressed");
      sec("Parameters");
      const L = u.length;
      const fields = { box: [["x", "X"], ["y", "Y"], ["z", "Z"], ["dx", "Length X"], ["dy", "Length Y"], ["dz", "Length Z"]], cylinder: [["x", "Base X"], ["y", "Base Y"], ["z", "Base Z"], ["dx", "Axis X"], ["dy", "Axis Y"], ["dz", "Axis Z"], ["r", "Radius"], ["angle", "Angle (°)"]],
        sphere: [["x", "Center X"], ["y", "Center Y"], ["z", "Center Z"], ["r", "Radius"]], cone: [["x", "Base X"], ["y", "Base Y"], ["z", "Base Z"], ["dx", "Axis X"], ["dy", "Axis Y"], ["dz", "Axis Z"], ["r1", "Base radius"], ["r2", "Top radius"]],
        torus: [["x", "Center X"], ["y", "Center Y"], ["z", "Center Z"], ["r1", "Major radius"], ["r2", "Minor radius"]], surface: [["offset", "Plane offset"]], extrude: [["offset", "Plane offset"], ["distance", "Depth"]], revolve: [["offset", "Plane offset"], ["angle", "Angle (°)"]],
        dxf_extrude: [["distance", "Depth"], ["scale", "Scale"]], fillet: [["r", "Radius"]], chamfer: [["d", "Distance"]], import: [["scale", "Scale"]] }[it.type] || [];
      for (const [k, l] of fields) num(l, it, k, ["angle", "scale"].includes(k) ? "" : L);
      if (it.type === "extrude" || it.type === "revolve") { ro("Profiles", it.loops.length); rows.push({ label: "Sketch", html: `<button data-sketch="1">Edit sketch…</button>` }); if (it.type === "extrude") chk("Symmetric", it, "symmetric"); }
      if (it.type === "boolean") { sel("Operation", it, "op", [["fuse", "Union"], ["cut", "Subtract"], ["intersect", "Intersect"]]); ro("Target", esc(it.target.join(", "))); ro("Tools", esc(it.tool.join(", "))); chk("Keep tool bodies", it, "keep_tool"); }
      if (it.type === "fillet" || it.type === "chamfer") { ro("Edges", it.edges === "all" ? "all" : it.edges.length); rows.push({ label: "Edges", html: `<button data-edges="1">Use selected edges</button><button data-edges="all">All edges</button>` }); }
      if (it.type === "transform") { sel("Operation", it, "op", [["translate", "Translate"], ["rotate", "Rotate"], ["mirror", "Mirror"], ["scale", "Scale"]]); chk("Copy", it, "copy");
        if (it.op === "translate") vec("Vector", it, "vector", L); if (it.op === "rotate") { vec("Point", it, "point", L); vec("Axis", it, "axis"); num("Angle (°)", it, "angle"); }
        if (it.op === "mirror") { vec("Point", it, "point", L); vec("Normal", it, "normal"); } if (it.op === "scale") { vec("Point", it, "point", L); num("Factor", it, "factor"); } }
      if (it.type === "pattern") { sel("Type", it, "kind", [["linear", "Linear"], ["circular", "Circular"]]); num("Count", it, "count"); chk("Fuse with original", it, "fuse");
        if (it.kind === "circular") { vec("Axis point", it, "point", L); vec("Axis", it, "axis"); num("Total angle (°)", it, "angle"); } else { vec("Direction", it, "direction"); num("Spacing", it, "spacing", L); } }
      if (it.type === "line") rows.push({ label: "Points (x y z per line)", html: `<textarea data-points="1" rows="4">${it.points.map(p => p.join(" ")).join("\n")}</textarea>` });
      if (it.type === "surface") { ro("Profiles", it.loops.length); rows.push({ label: "Sketch", html: `<button data-sketch="1">Edit sketch…</button>` }); }
      if (it.type === "import") { ro("File", esc(it.file.split(/[\\/]/).pop())); chk("Heal geometry", it, "heal"); }
    } else if (kind === "body") {
      const pb = p.bodies[id] || (p.bodies[id] = {}); const bi = info?.bodies.find(b => String(b.id) === id);
      title = `Details of "${pb.name || "Body " + id}"`; sec("Definition"); txt("Name", pb, "name"); chk("Suppressed", pb, "suppressed");
      sel("Material", pb, "material", p.materials.map(m => [m.id, m.name]));
      if (id.startsWith("S")) { sec("Shell section (S4 / S3, MITC4)"); num("Thickness", pb, "thickness", u.length);
        const sh = info?.sheets?.find(b => "S" + b.id === id); if (sh) ro("Area", fmt(sh.area) + " " + u.length + "²"); }
      if (id.startsWith("L")) { sec("Beam section (B31, Timoshenko)"); sel("Profile", pb, "profile", [["RECT", "Rectangle (b, h)"], ["CIRC", "Circle (r)"], ["PIPE", "Pipe (r, t)"], ["BOX", "Box (b, h, t)"], ["I", "I-beam (h, b, tf, tw)"], ["GENERAL", "General (A, I2, I3, J)"]]);
        rows.push({ label: "Dimensions", unit: u.length, html: `<input type="text" data-dims="1" value="${esc((pb.dims || []).join(", "))}">` });
        vec("Orientation n1 (local 2-axis)", pb, "n1");
        const ln = info?.lines?.find(b => "L" + b.id === id); if (ln) ro("Length", fmt(ln.length) + " " + u.length); }
      if (bi) { sec("Properties"); ro("Volume", fmt(bi.volume) + " " + u.length + "³"); const m = p.materials.find(m => m.id === pb.material); ro("Mass", fmt((bi.volume || 0) * (m?.density || 0)) + " " + u.mass);
        ro("Centroid", bi.com.map(fmt).join(", ")); }
    } else if (kind === "materials") {
      title = "Engineering Data"; sec("Library"); rows.push({ label: "Add from library", html: `<select data-lib="1"><option value="">choose…</option>${this.state.library.map(m => `<option>${esc(m.name)}</option>`).join("")}</select>` });
      rows.push({ label: "Custom", html: `<button data-addmat="1">Add custom material</button>` });
    } else if (kind === "material" && it) {
      title = `Details of "${it.name}"`; sec("Definition"); txt("Name", it, "name");
      sel("Model", it, "model", [["linear", "Isotropic elasticity"], ["plastic", "Bilinear / multilinear J2 plasticity"], ["neohooke", "Neo-Hookean hyperelastic"], ["mooney", "Mooney-Rivlin hyperelastic"], ["arruda", "Arruda-Boyce hyperelastic"], ["gent", "Gent hyperelastic"]]);
      if (it.model === "arruda" || it.model === "gent") { num("Shear modulus mu", it, "mu", u.stress); if (it.model === "arruda") num("Locking stretch", it, "lam_m"); else num("Jm", it, "Jm"); }
      rows.push({ label: "Color", html: `<input type="color" data-k="color" value="${it.color || "#8aa9c9"}">`, obj: it, key: "color", t: "txt" });
      sec("Elasticity"); num("Young's modulus", it, "E", u.stress); num("Poisson's ratio", it, "nu"); num("Density", it, "density", u.density);
      if (it.model === "plastic") { sec("Plasticity"); num("Yield strength", it, "yield", u.stress); num("Tangent modulus", it, "tangent", u.stress);
        rows.push({ label: "Hardening table (σy, εp per line)", html: `<textarea data-table="1" rows="3">${(it.table || []).map(r => r.join(", ")).join("\n")}</textarea>` }); }
      if (it.model === "neohooke") { sec("Hyperelastic (blank = from E, ν)"); num("C10", it, "C10", u.stress); num("D1", it, "D1", "1/" + u.stress); }
      if (it.model === "mooney") { sec("Hyperelastic"); num("C10", it, "C10", u.stress); num("C01", it, "C01", u.stress); num("D1", it, "D1", "1/" + u.stress); }
      sec("Fatigue (S-N, Basquin)"); num("Ultimate strength", it, "ultimate", u.stress); num("Fatigue strength coeff. Sf'", it, "Sf", u.stress);
      num("Fatigue strength exponent b", it, "b"); num("Endurance limit Se", it, "Se", u.stress);
      sec("Thermal"); num("Thermal expansion", it, "alpha", "1/°C"); num("Conductivity", it, "k", u.conductivity); num("Specific heat", it, "cp", u.cp);
      if (it.model !== "plastic") { sec("Strength"); num("Yield strength (safety factor)", it, "yield", u.stress); }
    } else if (kind === "mesh") {
      title = "Details of \"Mesh\""; sec("Defaults"); num("Element size", p.mesh, "size", u.length, { ph: "auto" });
      sel("Element order", p.mesh, "order", [["2", "Quadratic (C3D10 / C3D20)"], ["1", "Linear (C3D4 / C3D8)"]]);
      sel("Method", p.mesh, "kind", [["tet", "Tetrahedrons"], ["hex", "Hex (transfinite where possible)"], ["hexdom", "Hex dominant (subdivision)"]]);
      num("Curvature (elements per 2π)", p.mesh, "curvature");
      const st = this.meshStats;
      if (st) { sec("Statistics"); ro("Nodes", st.nodes); ro("Elements", st.elements); ro("Types", esc(Object.entries(st.types || {}).map(([k, v]) => `${k}: ${v}`).join(", ")));
        if (st.q_min !== undefined) { ro("Quality (min SICN)", fmt(st.q_min)); ro("Quality (avg)", fmt(st.q_avg)); rows.push({ label: "Quality histogram", html: `<canvas class="hist" width="220" height="60"></canvas>` }); }
        if (st.time) ro("Mesh time", fmt(st.time) + " s"); }
      rows.push({ label: "", html: `<button data-meshnow="1" class="primary">Generate Mesh</button>` });
    } else if (kind === "sizing" && it) {
      title = `Details of "${it.name}"`; sec("Scope"); scope("Geometry", it, "scope"); sec("Definition"); txt("Name", it, "name"); num("Element size", it, "size", u.length); chk("Suppressed", it, "suppressed");
    } else if (kind === "contact" && it) {
      title = `Details of "${it.name}"`; sec("Scope"); scope("Contact (master)", it, "master"); scope("Target (slave)", it, "slave");
      sec("Definition"); txt("Name", it, "name"); sel("Type", it, "type", [["bonded", "Bonded (MPC tie, dual mortar)"], ["frictionless", "Frictionless (augmented Lagrange)"], ["frictional", "Frictional (Coulomb)"]]);
      if (it.type === "frictional") num("Friction coefficient", it, "mu");
      if (it.type !== "bonded") num("Normal stiffness factor", it, "fkn"); chk("Suppressed", it, "suppressed");
    } else if (kind === "ns" && it) {
      title = `Details of "${it.name}"`; sec("Scope"); scope("Geometry", it, "scope"); sec("Definition"); txt("Name", it, "name");
    } else if (kind === "settings" || kind === "analysis") {
      const a = p.analysis; title = `Details of "Analysis Settings"`;
      sec("Analysis"); sel("Type", a, "type", Object.entries(this.state.analyses));
      if (a.type === "deck") { ro("Source", esc((p.mesh_source?.file || "").split(/[\\/]/).pop())); ro("Steps", "as defined in the imported deck"); }
      if (["static", "transient", "explicit", "thermal_stress", "thermal_transient"].includes(a.type)) { sec("Step Controls"); num("Step end time", a, "time", u.time);
        if (a.type !== "explicit") { num("Initial substeps", a, "substeps"); num("Maximum substeps", a, "max_substeps"); } num("Output frames", a, "frames"); }
      if (a.type === "transient" || a.type === "thermal_transient") num("Time step", a, "dt", u.time, { ph: "auto" });
      if (["static", "transient", "explicit", "thermal_stress"].includes(a.type)) { sec("Solver Controls"); chk("Large deflection (NLGEOM)", a, "nlgeom");
        sel("Solver type", a, "linear_solver", [["auto", "Program controlled"], ["direct", "Direct (sparse LU, host)"], ["gpu_direct", "Direct on GPU (cuSOLVER)"], ["amg", "Iterative AMG-PCG (GPU, FP64)"], ["mixed", "Iterative AMG-PCG mixed FP32/FP64 (GeForce/RTX)"], ["jacobi", "Iterative Jacobi-PCG"]]);
        sel("Tangent", a, "tangent", [["ebe", "Element-by-element (stored)"], ["jvp", "Matrix-free (AD JVP, minimum memory)"]]); }
      if (a.type === "transient" || a.type === "explicit") num("Mass damping", a, "damping", "1/s");
      if (["modal", "buckling", "nonlinear_buckling", "harmonic"].includes(a.type)) { sec("Eigen solver"); num("Number of modes", a, "nmodes");
        sel("Eigen solver", a, "eigensolver", [["auto", "Program controlled"], ["lanczos", "Shift-invert Lanczos (host)"], ["lobpcg", "Block LOBPCG + AMG (GPU, large models)"]]); }
      if (a.type === "nonlinear_buckling") { sec("Imperfection / Riks"); num("Imperfection amplitude", a, "imperfection_scale", u.length, { ph: "auto" }); }
      if (a.type === "response_spectrum" || a.type === "random_vibration") {
        const key = a.type === "response_spectrum" ? "spectrum" : "psd";
        sec(a.type === "response_spectrum" ? "Response spectrum" : "Base acceleration PSD");
        rows.push({ label: a.type === "response_spectrum" ? "f [Hz], Sa per line" : `f [Hz], PSD [(${u.accel})²/Hz] per line`,
          html: `<textarea data-tab2="${key}" rows="4">${(a[key] || (key === "spectrum" ? [[0.1, 9806.65], [100, 9806.65]] : [[20, 1e4], [2000, 1e4]])).map(r => r.join(", ")).join("\n")}</textarea>` });
        vec("Excitation direction", a, "direction"); num("Damping ratio", a, "damping_ratio");
        if (a.type === "response_spectrum") sel("Mode combination", a, "combination", [["SRSS", "SRSS"], ["CQC", "CQC"]]);
      }
      if (a.type === "harmonic") { sec("Frequency range"); num("Min frequency", a, "fmin", "Hz"); num("Max frequency", a, "fmax", "Hz"); num("Solution intervals", a, "nfreq"); num("Damping ratio", a, "damping_ratio"); }
      if (a.type === "topopt") { const t = a.topopt; sec("Topology Optimization"); num("Retained volume fraction", t, "volfrac"); num("Penalty (SIMP p)", t, "penal"); num("Filter radius", t, "rmin", u.length, { ph: "auto" });
        num("Max iterations", t, "iters"); chk("Exclude loaded / supported elements", t, "keep_bc"); }
      if (a.type.startsWith("thermal") || p.conditions.some(c => c.type === "body_temperature")) { sec("Thermal"); num("Initial / reference temperature", a, "initial_temperature", u.temp); }
      sec("Output"); chk("Write ParaView files", a, "write_vtu");
    } else if (kind === "condition" && it) {
      const meta = this.state.conditions[it.type]; title = `Details of "${it.name}"`;
      if (meta.dims.length) { sec("Scope"); scope("Geometry", it, "scope"); }
      sec("Definition"); txt("Name", it, "name"); ro("Type", meta.label); chk("Suppressed", it, "suppressed");
      const t = it.type;
      if (t === "force") vec("Force (X, Y, Z)", it, "vector", u.force);
      if (t === "moment") vec("Moment (X, Y, Z)", it, "vector", u.force + "·" + u.length);
      if (t === "pressure") num("Magnitude (+ into surface)", it, "value", u.stress);
      if (t === "displacement") vec("Components (blank = free)", it, "components", u.length);
      if (t === "gravity") vec("Direction", it, "vector");
      if (t === "acceleration") vec("Acceleration", it, "vector", u.accel);
      if (t === "rotational_velocity") { num("Angular velocity", it, "value", "rad/s"); vec("Axis", it, "axis"); vec("Axis point", it, "point", u.length); }
      if (t === "body_temperature" || t === "temperature") num("Temperature", it, "value", u.temp);
      if (t === "heat_flux") num("Heat flux (into body)", it, "value", u.flux);
      if (t === "convection") { num("Film coefficient", it, "h", u.film); num("Ambient temperature", it, "ambient", u.temp); }
      if (t === "heat_generation") num("Internal heat generation", it, "value", u.gen);
      if (t === "remote_displacement") { vec("Remote point (blank = centroid)", it, "point", u.length); vec("Translations (blank = free)", it, "components", u.length); vec("Rotations (blank = free)", it, "rotations", "rad"); }
      if (t === "remote_force") { vec("Remote point (blank = centroid)", it, "point", u.length); vec("Force", it, "vector", u.force); vec("Moment", it, "moment", u.force + "·" + u.length); }
      if (t === "point_mass") num("Mass", it, "value", u.mass);
      if (t === "elastic_support") num("Foundation stiffness", it, "value", u.force + "/" + u.length + "³");
      if (t === "radiation") { num("Emissivity", it, "emissivity"); num("Ambient temperature", it, "ambient", u.temp); }
      if (t === "bolt_pretension") { num("Pretension force", it, "value", u.force); vec("Bolt axis (blank = longest)", it, "axis"); }
      if (t === "displacement") vec("Rotations (shells/beams, blank = free)", it, "rotations", "rad");
      if (t === "rigid_wall") { vec("Wall point", it, "point", u.length); vec("Wall normal (into body)", it, "normal"); vec("Wall motion", it, "motion", u.length); num("Penalty factor", it, "fkn"); }
    } else if (kind === "solution" || kind === "solinfo") {
      title = "Details of \"Solution\""; sec("Status"); ro("Status", this.status.solve); ro("Frames", this.frames.length);
      if (this.resultMeta?.info) for (const [k, v] of Object.entries(this.resultMeta.info)) {
        if (v?.freq) { sec(`Frequencies (${k})`); v.freq.forEach((f, i) => ro(`Mode ${i + 1}`, fmt(f) + " Hz")); }
        if (v?.load_factors) { sec(`Load multipliers (${k})`); v.load_factors.forEach((f, i) => ro(`Mode ${i + 1}`, fmt(f))); }
        if (v?.lpf !== undefined) ro("Final LPF", fmt(v.lpf));
      }
      rows.push({ label: "", html: `<button data-solve="1" class="primary">Solve</button> <button data-stop="1">Stop</button>` });
    } else if (kind === "result" && it) {
      title = `Details of "${it.name}"`; sec("Definition"); txt("Name", it, "name"); sel("Type", it, "type", Object.entries(this.state.result_types));
      sec("Results"); ro("Minimum", it._min !== undefined ? fmtR(it._min, it._min, it._max) + " " + this.unitOf(it.type) : "-"); ro("Maximum", it._max !== undefined ? fmtR(it._max, it._min, it._max) + " " + this.unitOf(it.type) : "-");
      ro("Time", it._time !== undefined ? fmt(it._time) : "-");
      if (this.frames.length) rows.push({ label: "Display frame", html: `<select data-frame="1">${this.frames.map((f, i) => `<option value="${i}" ${i === this.ui.frame ? "selected" : ""}>${f.freq !== undefined ? `Mode ${f.mode}: ${fmt(f.freq)} Hz` : f.eigenvalue !== undefined ? `Mode ${f.mode}: ${fmt(f.eigenvalue)}` : `t = ${fmt(f.time)} (${esc(f.step)})`}</option>`).join("")}</select>` });
    }
    $("#details-title").textContent = title;
    let h = "<table class=\"pg\">";
    for (const r of rows) {
      if (r.sec) { h += `<tr class="sec"><td colspan="2">${esc(r.sec)}</td></tr>`; continue; }
      h += `<tr><td class="lbl">${esc(r.label)}${r.unit ? ` <span class="unit">[${esc(r.unit)}]</span>` : ""}</td><td class="val">${r.html}</td></tr>`;
    }
    $("#details").innerHTML = h + "</table>";
    // bind
    const D = $("#details");
    const rowByEl = el => rows.find(r => r.obj && r.key === el.dataset.k && (r.t !== "vec" || true));
    D.querySelectorAll("[data-k]").forEach(el => {
      el.onchange = async () => {
        const r = rows.find(r => r.obj && r.key === el.dataset.k); if (!r) return;
        const o = r.obj, k = r.key;
        if (r.t === "num") o[k] = el.value === "" ? null : +el.value;
        else if (r.t === "chk") o[k] = el.checked;
        else if (r.t === "vec") { const v = [...(o[k] || [0, 0, 0])]; v[+el.dataset.i] = el.value === "" ? null : +el.value; o[k] = v; }
        else if (r.t === "sel") o[k] = ["order"].includes(k) ? +el.value : el.value;
        else o[k] = el.value;
        await this.afterEdit(kind, it, k);
      };
    });
    D.querySelectorAll("[data-apply]").forEach(b => b.onclick = () => {
      const k = b.dataset.apply; const obj = it || this.findItem(kind, id); const sc = this.selectionScope();
      if (!sc) return this.log("select geometry in the viewport first", "warn");
      const meta = kind === "condition" ? this.state.conditions[obj.type] : null;
      if (meta && meta.dims.length && !meta.dims.includes(sc.dim)) return this.log(`${meta.label} cannot be applied to this entity type`, "warn");
      obj[k] = sc; this.viewer.clearSelection(); this.afterEdit(kind, obj, k); this.select(this.treeSel);
    });
    D.querySelectorAll("[data-clear]").forEach(b => b.onclick = () => { const obj = it; obj[b.dataset.clear] = null; this.afterEdit(kind, obj, b.dataset.clear); this.select(this.treeSel); });
    D.querySelectorAll("[data-named]").forEach(s => s.onchange = () => { if (!s.value) return; it[s.dataset.named] = { named: s.value }; this.afterEdit(kind, it, s.dataset.named); this.select(this.treeSel); });
    D.querySelector("[data-lib]")?.addEventListener("change", e => { if (!e.target.value) return; const lib = this.state.library.find(m => m.name === e.target.value);
      const m = { ...lib, id: uid("M", p.materials) }; p.materials.push(m); this.push(); this.select({ kind: "material", id: m.id }); });
    D.querySelector("[data-addmat]")?.addEventListener("click", () => this.addMaterial());
    D.querySelector("[data-meshnow]")?.addEventListener("click", () => this.meshNow());
    D.querySelector("[data-solve]")?.addEventListener("click", () => this.solve());
    D.querySelector("[data-stop]")?.addEventListener("click", () => this.stop());
    D.querySelector("[data-sketch]")?.addEventListener("click", () => openSketcher(this, it));
    D.querySelectorAll("[data-edges]").forEach(b => b.onclick = async () => {
      if (b.dataset.edges === "all") it.edges = "all";
      else { const s = this.viewer.selection; if (s.dim !== 1 || !s.ids.size) return this.log("select edges first", "warn");
        it.edges = [...s.ids].map(i => this.geom.info.edges.find(e => e.id === i)?.sig).filter(Boolean); }
      await this.regenerate(); this.select({ kind: "feature", id: it.id }); });
    D.querySelector("[data-tab2]")?.addEventListener("change", e => {
      const rows2 = e.target.value.trim().split(/\n+/).map(l => l.split(/[,\s]+/).map(Number)).filter(r => r.length >= 2 && r.every(isFinite));
      this.project.analysis[e.target.dataset.tab2] = rows2; this.push(); });
    D.querySelector("[data-dims]")?.addEventListener("change", e => { const pb = this.project.bodies[id];
      pb.dims = e.target.value.split(/[,\s]+/).map(Number).filter(isFinite); this.push(); this.renderTree(); });
    D.querySelector("[data-points]")?.addEventListener("change", async e => {
      it.points = e.target.value.trim().split(/\n+/).map(l => l.trim().split(/[,\s]+/).map(Number)).filter(r => r.length === 3 && r.every(isFinite));
      await this.regenerate(); this.select({ kind: "feature", id: it.id }); });
    D.querySelector("[data-table]")?.addEventListener("change", e => { const rowsT = e.target.value.trim().split(/\n+/).map(l => l.split(/[,\s]+/).map(Number)).filter(r => r.length >= 2 && r.every(isFinite));
      it.table = rowsT.length ? rowsT : null; this.push(); });
    D.querySelector("[data-frame]")?.addEventListener("change", e => { this.ui.frame = +e.target.value; $("#frame-slider").value = this.ui.frame; this.showResult(); });
    const hc = D.querySelector("canvas.hist");
    if (hc && this.meshStats?.q_hist) { const c = new Chart(hc, { bars: true }); c.set([{ data: this.meshStats.q_hist.map((v, i) => [(i + 0.5) / 10, v]), color: "#4a7bd0" }]); }
  }

  async afterEdit(kind, it, key) {
    const geomKinds = ["feature"];
    if (geomKinds.includes(kind) || (kind === "geometry" && key === "share_topology")) { this._keepView = true; await this.regenerate(); this._keepView = false; return this.select(this.treeSel); }
    if (kind === "body" || kind === "material") { this.applyBodyColors(); if (this.geom) this.viewer.setGeometry(this.geom, false); }
    if (kind === "mesh" || kind === "sizing") this.meshChanged();
    if (kind === "settings" || kind === "analysis") { this.renderRibbon(this.tab); }
    if (kind === "result" && key === "type") { it.name = this.state.result_types[it.type]; if (this.frames.length) await this.showResult(); }
    if (kind === "model" && key === "name") $("#proj-name").value = this.project.name;
    if (this.frames.length && !["result"].includes(kind)) this.status.solve = "stale";
    await this.push();
    this.renderTree(); this.renderDetails();
  }

  addMaterial() {
    const p = this.project; const base = this.state.library[0];
    const m = { ...base, id: uid("M", p.materials), name: "New material", color: "#" + Math.floor(Math.random() * 0xffffff).toString(16).padStart(6, "0") };
    p.materials.push(m); this.push(); this.select({ kind: "material", id: m.id });
  }

  // ---------------------------------------------------------------- charts / tables
  drawConvergence() {
    const it = this.iters.filter(x => x.residual !== null && x.residual !== undefined);
    this.chartRes.set([{ data: it.map(x => [x.n, x.residual]), color: "#c0392b", label: "force residual", markers: it.length < 300 },
      { data: it.length ? [[it[0].n, 5e-3], [it[it.length - 1].n, 5e-3]] : [], color: "#2471a3", dash: [5, 4], label: "criterion" }]);
    const inc = this.incs.filter(x => x.ok !== false);
    const series = [{ data: inc.map((x, i) => [i + 1, x.time_total ?? x.t]), color: "#1e8449", label: "time / LPF", markers: inc.length < 200 }];
    if (inc.some(x => x.compliance !== undefined)) series.push({ data: inc.map((x, i) => [i + 1, x.compliance]), color: "#8e44ad", label: "compliance" });
    this.chartTime.opts.title = inc.some(x => x.compliance !== undefined) ? "Compliance" : "Time / load factor";
    this.chartTime.set(inc.some(x => x.compliance !== undefined) ? series.slice(1) : series);
  }
  renderTabular() {
    const F = this.frames; if (!F.length) { $("#tabular").innerHTML = ""; return; }
    const cols = [["#", (f, i) => i + 1], ["Step", f => esc(f.step)], ["Time", f => fmt(f.time)]];
    if (F.some(f => f.freq !== undefined)) cols.push(["Frequency [Hz]", f => fmt(f.freq)]);
    if (F.some(f => f.eigenvalue !== undefined)) cols.push(["Load multiplier", f => fmt(f.eigenvalue)]);
    if (F.some(f => f.compliance !== undefined)) cols.push(["Compliance", f => fmt(f.compliance)]);
    if (F.some(f => f.umax !== undefined)) cols.push([`Max |U| [${this.state.units.length}]`, f => fmt(f.umax)]);
    if (F.some(f => f.smax !== undefined)) cols.push([`Max σ_eq [${this.state.units.stress}]`, f => fmt(f.smax)]);
    if (F.some(f => f.tmax !== undefined)) cols.push(["T min", f => fmt(f.tmin)], ["T max", f => fmt(f.tmax)]);
    $("#tabular").innerHTML = `<table class="tab"><tr>${cols.map(c => `<th>${c[0]}</th>`).join("")}</tr>${F.map((f, i) => `<tr data-i="${i}">${cols.map(c => `<td>${c[1](f, i)}</td>`).join("")}</tr>`).join("")}</table>`;
    $$("#tabular tr[data-i]").forEach(tr => tr.onclick = () => { this.ui.frame = +tr.dataset.i; $("#frame-slider").value = this.ui.frame; this.showResult(); });
  }
  drawGraph() {
    const F = this.frames;
    const s = [];
    if (F.some(f => f.umax !== undefined)) s.push({ data: F.map(f => [f.time, f.umax]), color: "#2471a3", label: "max |U|", markers: true });
    if (F.some(f => f.smax !== undefined)) { const m = Math.max(...F.map(f => f.smax || 0)), mu = Math.max(...F.map(f => f.umax || 0)) || 1;
      s.push({ data: F.map(f => [f.time, (f.smax || 0) / m * mu]), color: "#c0392b", label: "max σ_eq (scaled)", dash: [4, 3] }); }
    if (F.some(f => f.freq !== undefined)) { this.chartGraph.opts.bars = true; this.chartGraph.opts.title = "Natural frequencies [Hz]"; this.chartGraph.opts.xlabel = "mode"; s.length = 0; s.push({ data: F.map(f => [f.mode, f.freq]), color: "#2471a3" }); }
    else { this.chartGraph.opts.bars = false; this.chartGraph.opts.title = "Result vs time"; this.chartGraph.opts.xlabel = "time"; }
    this.chartGraph.set(s);
  }
}

window.app = new App();
