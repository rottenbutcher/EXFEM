// AI CAD dialog: describe a part (and/or attach a drawing); an LLM builds it as a feature tree.
// The server runs the agent loop (gfem/cad/ai.py) and streams "ai" events; API keys stay on the server.

const EXAMPLES = [
  ["Flange", "Flange: outer diameter 120 mm, thickness 15 mm, central hub d60 x 40 mm high with a d30 bore through everything, 6 bolt holes d11 equally spaced on a d95 bolt circle, 3 mm fillet where the hub meets the flange."],
  ["L-bracket", "L-shaped steel bracket, 5 mm plate, horizontal leg 80 x 60 mm, vertical leg 60 mm high, one triangular gusset 40 x 40 mm in the middle, two d9 holes in each leg, 3 mm inner bend fillet."],
  ["Shaft", "Stepped shaft along X: d40 x 60 long, then d30 x 80, then d25 x 40, 1 mm chamfers on the ends, a 8 x 4 mm keyway 40 mm long on the d30 section."],
  ["Heat sink", "Aluminium heat sink: base 100 x 80 x 6 mm, 11 fins 2 mm thick and 35 mm high along the 80 mm direction, evenly spaced, 4 M4 clearance holes (d4.5) in the corners of the base 5 mm from the edges."],
  ["Pipe tee", "Equal pipe tee: main pipe OD 60 ID 50 length 160 along X, branch pipe OD 60 ID 50 going up +Z 90 mm from the main axis, hollow inside and open at all three ends."],
  ["Lattice beam", "Planar truss for beam elements: bottom chord 2000 mm along X, top chord at height 300 mm, 8 panels with diagonal and vertical members, all as line bodies."],
];

const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export class AIDialog {
  constructor(app) {
    this.app = app; this.api = app.api;
    this.dlg = $("#ai-dlg"); this.atts = []; this.busy = false; this.rid = null; this.history = [];
    this.settings = null;
    $("#ai-chips").innerHTML = EXAMPLES.map(([k], i) => `<button data-i="${i}" title="${esc(EXAMPLES[i][1])}">${esc(k)}</button>`).join("");
    $("#ai-chips").onclick = e => { const b = e.target.closest("button"); if (b) { $("#ai-prompt").value = EXAMPLES[+b.dataset.i][1]; this.setMode("new"); } };
    $("#ai-close").onclick = $("#ai-ok").onclick = () => this.close();
    $("#ai-run").onclick = () => this.run();
    $("#ai-stop").onclick = () => this.api.post("/api/ai/stop");
    $("#ai-undo").onclick = () => this.undo();
    $("#ai-file").onchange = e => this.attach([...e.target.files]);
    $("#ai-provider").onchange = () => { this.fillModels(true); this.saveSettings(); };
    for (const id of ["#ai-model", "#ai-effort", "#ai-base", "#ai-turns", "#ai-visual"]) $(id).onchange = () => this.saveSettings();
    $("#ai-key-save").onclick = () => this.saveKey();
    $("#ai-prompt").addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); this.run(); } });
    this.dlg.addEventListener("keydown", e => { if (e.key === "Escape" && !this.busy) this.close(); });
    const pv = $(".ai-preview");
    pv.addEventListener("dragover", e => e.preventDefault());
    pv.addEventListener("drop", e => { e.preventDefault(); this.attach([...e.dataTransfer.files]); });
    $("#ai-prompt").addEventListener("paste", e => {
      const imgs = [...(e.clipboardData?.files || [])].filter(f => f.type.startsWith("image/"));
      if (imgs.length) { e.preventDefault(); this.attach(imgs.map((f, i) => new File([f], f.name || `pasted_${Date.now()}_${i}.png`, { type: f.type }))); }
    });
  }

  async open() {
    this.dlg.classList.add("open");
    const hasGeom = (this.app.project?.geometry.features || []).length > 0;
    if (!this.busy) this.setMode(hasGeom && this.history.length ? "modify" : "new");
    document.querySelector('input[name="ai-mode"][value="modify"]').disabled = !hasGeom;
    setTimeout(() => $("#ai-prompt").focus(), 50);
    await this.loadSettings();
  }
  loadSettings() {
    if (!this._loading) this._loading = this.api.get("/api/ai/settings")
      .then(r => { this.settings = r; this.fillSettings(); })
      .catch(e => this.line("Cannot load AI settings: " + e.message, "e"))
      .finally(() => { this._loading = null; });
    return this._loading;
  }
  close() { this.dlg.classList.remove("open"); }
  mode() { return document.querySelector('input[name="ai-mode"]:checked').value; }
  setMode(m) { const r = document.querySelector(`input[name="ai-mode"][value="${m}"]`); if (r && !r.disabled) r.checked = true; }

  // ---------------------------------------------------------------- settings
  fillSettings() {
    const s = this.settings;
    $("#ai-provider").innerHTML = Object.entries(s.providers).map(([k, p]) => `<option value="${k}">${esc(p.label)}</option>`).join("");
    $("#ai-provider").value = s.provider;
    $("#ai-effort").innerHTML = s.efforts.map(e => `<option>${e}</option>`).join("");
    $("#ai-effort").value = s.effort;
    $("#ai-base").value = s.base_url || "";
    $("#ai-turns").value = s.max_turns;
    $("#ai-visual").checked = !!s.visual;
    this.fillModels(false);
    $("#ai-model").value = s.model || s.providers[s.provider].default_model;
  }

  fillModels(reset) {
    const p = this.settings.providers[$("#ai-provider").value];
    $("#ai-models").innerHTML = p.models.map(m => `<option value="${esc(m)}">`).join("");
    if (reset) $("#ai-model").value = p.default_model;
    $("#ai-model").placeholder = p.default_model || "model name (e.g. qwen2.5-coder:32b)";
    const compat = $("#ai-provider").value === "compatible";
    document.querySelectorAll(".ai-burl").forEach(el => el.style.display = compat ? "" : "none");
    const info = $("#ai-keyinfo");
    if (!p.sdk) { info.className = "ai-keyinfo missing"; info.textContent = `Python package missing on the server: pip install ${$("#ai-provider").value === "anthropic" ? "anthropic" : "openai"}`; }
    else if (p.has_key) { info.className = "ai-keyinfo"; info.textContent = p.key_source === "env" ? `Using the key from the ${p.env} environment variable.` : "A key is saved on this computer (~/.gfem/ai.json). Enter a new one to replace it, or save empty to remove it."; }
    else if (compat) { info.className = "ai-keyinfo"; info.textContent = "Local servers usually need no key. Set the base URL of the OpenAI-compatible endpoint."; }
    else { info.className = "ai-keyinfo missing"; info.textContent = `No API key yet: paste one above and press Save (stored only on this computer), or set ${p.env} before starting gfem.`; }
    $("#ai-key").placeholder = p.has_key ? "•••••••• (saved)" : "paste API key";
  }

  async saveSettings() {
    try {
      this.settings = await this.api.post("/api/ai/settings", {
        provider: $("#ai-provider").value, model: $("#ai-model").value.trim(), effort: $("#ai-effort").value,
        base_url: $("#ai-base").value.trim(), max_turns: Math.max(1, Math.min(30, +$("#ai-turns").value || 10)), visual: $("#ai-visual").checked });
      this.fillModels(false);
    } catch (e) { this.line("Settings: " + e.message, "e"); }
  }

  async saveKey() {
    const key = $("#ai-key").value.trim();
    try {
      this.settings = await this.api.post("/api/ai/settings", { provider: $("#ai-provider").value, key });
      $("#ai-key").value = ""; this.fillModels(false);
      this.line(key ? "API key saved on this computer." : "API key removed.", "s");
    } catch (e) { this.line("Saving key failed: " + e.message, "e"); }
  }

  // ---------------------------------------------------------------- attachments
  async attach(files) {
    for (const f of files) {
      if (f.size > 25e6) { this.line(`${f.name}: larger than 25 MB, skipped`, "e"); continue; }
      try { const r = await this.api.upload(f); this.atts.push({ name: f.name, path: r.path }); }
      catch (e) { this.line(`${f.name}: upload failed (${e.message})`, "e"); }
    }
    this.renderAtts();
  }
  renderAtts() {
    $("#ai-att-list").innerHTML = this.atts.map((a, i) => `<span>${esc(a.name)}<b data-i="${i}" title="Remove">✕</b></span>`).join("");
    $("#ai-att-list").querySelectorAll("b").forEach(b => b.onclick = () => { this.atts.splice(+b.dataset.i, 1); this.renderAtts(); });
  }

  // ---------------------------------------------------------------- run
  async run() {
    if (this.busy) return;
    const prompt = $("#ai-prompt").value.trim();
    if (!prompt) { $("#ai-prompt").focus(); return; }
    if (!this.settings) await this.loadSettings();             // never send defaults of an unfilled form
    if (!this.settings) return;
    const p = this.settings?.providers?.[$("#ai-provider").value];
    if (p && !p.has_key && $("#ai-provider").value !== "compatible") {
      $("#ai-settings").open = true; $("#ai-key").focus();
      this.line("Enter an API key first (Model & API key).", "e"); return;
    }
    const mode = this.mode();
    const body = {
      prompt, mode, attachments: this.atts.map(a => a.path),
      base_features: mode === "modify" ? this.app.project.geometry.features : null,
      provider: $("#ai-provider").value, model: $("#ai-model").value.trim() || null, effort: $("#ai-effort").value,
      base_url: $("#ai-base").value.trim() || null, max_turns: +$("#ai-turns").value || null, visual: $("#ai-visual").checked,
    };
    this.setBusy(true, "Starting…");
    this.line((mode === "modify" ? "» (modify) " : "» ") + prompt + (this.atts.length ? `   [${this.atts.map(a => a.name).join(", ")}]` : ""), "u");
    try { const r = await this.api.post("/api/ai/generate", body); this.rid = r.rid; }
    catch (e) { this.setBusy(false, ""); this.line(e.message, "e"); }
  }

  setBusy(b, msg) {
    this.busy = b;
    $("#ai-run").disabled = b; $("#ai-stop").disabled = !b;
    $("#ai-status").textContent = msg || ""; $("#ai-status").className = b ? "busy" : "";
  }

  line(msg, cls = "a") {
    const d = document.createElement("div"); d.className = cls; d.textContent = msg;
    const L = $("#ai-log"); L.appendChild(d); L.scrollTop = L.scrollHeight;
  }

  showPng(b64) {
    if (!b64) return;
    const img = $("#ai-img"); img.src = "data:image/png;base64," + b64; img.className = "on"; $("#ai-img-msg").style.display = "none";
  }

  onEvent(e) {
    if (this.rid !== null && e.rid !== this.rid && e.step !== "start") return;
    if (e.step === "start") { this.rid = e.rid; return; }
    if (e.step === "log") {
      const m = e.msg.replace(/^AI: /, "");
      if (/^turn \d+/.test(m)) { $("#ai-status").textContent = m; return; }
      const cls = /^build \d+/.test(m) ? (/error\(s\)/.test(m) ? "e" : "b")
        : /^(tool input rejected|reached the limit|stopped by the user)/.test(m) ? "e" : "a";
      this.line(m, cls);
      return;
    }
    if (e.step === "build") {
      if (e.ok) this.showPng(e.png);
      $("#ai-status").textContent = e.ok ? `build ok: ${e.bodies} body(ies), ${(e.size || []).map(v => +v.toPrecision(4)).join(" × ")}` : "build failed – the model is fixing it…";
      return;
    }
    if (e.step === "done") {
      this.setBusy(false, "");
      const tok = e.usage && (e.usage.input || e.usage.output) ? ` · tokens ${e.usage.input + (e.usage.cache_read || 0)} in / ${e.usage.output} out` : "";
      if (e.builds !== undefined) $("#ai-usage").textContent = `${e.builds} build(s) · ${e.seconds}s${tok}`;
      if (!e.ok) { this.line("✖ " + (e.error || "no valid geometry"), "e"); return; }
      this.showPng(e.png);
      this.apply(e);
    }
  }

  async apply(e) {
    const app = this.app, g = app.project.geometry;
    this.history.push({ features: JSON.parse(JSON.stringify(g.features)), name: app.project.name, units: app.project.units });
    $("#ai-undo").disabled = false;
    g.features = e.features;
    if (this.mode() === "new" && e.name) {
      app.project.name = e.name; $("#proj-name").value = e.name;
    }
    const ok = await app.regenerate();
    this.line(ok ? `✔ applied to the model (${e.features.length} features)` : "✖ the result could not be regenerated in the GUI", ok ? "b" : "e");
    app.log(`AI CAD: ${e.features.length} features generated` + (e.name ? ` – ${e.name}` : ""));
    if (ok) { document.querySelector('input[name="ai-mode"][value="modify"]').disabled = false; this.setMode("modify"); $("#ai-prompt").value = ""; $("#ai-prompt").placeholder = "Describe a change (e.g. 'make the bolt holes M12 and add a 2 mm chamfer on the outer edge'), or switch to New part."; }
  }

  async undo() {
    const h = this.history.pop();
    if (!h) return;
    this.app.project.geometry.features = h.features;
    this.app.project.name = h.name; $("#proj-name").value = h.name;
    await this.app.regenerate();
    this.line("↶ restored the previous geometry", "s");
    $("#ai-undo").disabled = !this.history.length;
  }
}
