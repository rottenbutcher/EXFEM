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

const SIM_EXAMPLES = [
  ["Cantilever SF", "Aluminium cantilever bar 200 mm long, 20 x 10 mm section, fixed at x=0, 100 N downward at the free end. Safety factor at least 3; compare the deflection with beam theory."],
  ["Bracket sizing", "Steel L-bracket: 6 mm plate, base leg 100 x 60 mm on z=0, vertical leg 80 mm high at x=0, two d10 bolt holes in the base. Bolt holes fixed, 1500 N downward on the top face of the vertical leg. Safety factor >= 2; if not met, make the plate thicker and solve again."],
  ["Frequencies", "First three natural frequencies of a 300 x 30 x 3 mm steel cantilever strip fixed at one end; the first one must stay above 50 Hz – change the thickness if needed."],
  ["Plate with hole", "Steel plate 200 x 100 x 5 mm with a central d20 hole, fixed on the left end, 10 MPa tension pressure (pulling) on the right end. Report the stress concentration factor."],
  ["Buckling", "Steel column 1000 mm long, 20 x 20 mm square section, fixed at the base, axial compression on the top face. Find the critical buckling load and compare with Euler."],
];
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export class AIDialog {
  constructor(app) {
    this.app = app; this.api = app.api;
    this.dlg = $("#ai-dlg"); this.atts = []; this.busy = false; this.rid = null; this.history = [];
    this.settings = null;
    this.renderChips();
    $("#ai-chips").onclick = e => {
      const b = e.target.closest("button"); if (!b) return;
      const sim = this.mode() === "simulate";
      $("#ai-prompt").value = (sim ? SIM_EXAMPLES : EXAMPLES)[+b.dataset.i][1];
      if (!sim) this.setMode("new");
    };
    document.querySelectorAll('input[name="ai-mode"]').forEach(r => r.addEventListener("change", () => this.renderChips()));
    this.builds = [];
    $("#ai-strip").onclick = e => { const im = e.target.closest("img"); if (im) this.pickBuild(+im.dataset.i); };
    $("#ai-apply-build").onclick = () => this.applyBuild();
    $("#ai-report").onclick = () => this.report();
    $("#ai-close").onclick = $("#ai-ok").onclick = () => this.close();
    $("#ai-run").onclick = () => this.run();
    $("#ai-stop").onclick = () => this.api.post("/api/ai/stop");
    $("#ai-undo").onclick = () => this.undo();
    $("#ai-newchat").onclick = () => { this.session = null; this.line("— new conversation —", "s"); this.setMode("new"); $("#ai-prompt").focus(); };
    $("#ai-file").onchange = e => this.attach([...e.target.files]);
    $("#ai-provider").onchange = () => { this.fillModels(true); this.saveSettings(); };
    document.querySelectorAll("#ai-acc-seg button").forEach(b => b.onclick = () => this.pickAccount(b.dataset.p));
    $("#ai-signin").onclick = () => this.signIn();
    $("#ai-signout").onclick = () => this.signOut();
    $("#ai-recheck").onclick = () => this.refreshStatus(true);
    $("#ai-code-send").onclick = () => this.sendCode();
    $("#ai-code").addEventListener("keydown", e => { if (e.key === "Enter") this.sendCode(); });
    $("#ai-login-cancel").onclick = () => { this.api.post("/api/ai/login_cancel"); this.loginPanel(false); };
    this.cliStatus = {};
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

  prefillFix(errorText) {
    this._fix = "The last solve of this project failed with:\n" + (errorText || "(no message)") +
      "\nFind the cause (geometry, supports, loads, materials, mesh), fix the setup and solve again.";
  }

  async open() {
    this.dlg.classList.add("open");
    if (this._fix && !this.busy && !$("#ai-prompt").value.trim()) {
      $("#ai-prompt").value = this._fix; this._fix = null;
      setTimeout(() => this.setMode("simulate"), 0);
      setTimeout(() => this.renderChips(), 0);
    }
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
  report() {
    const p = this.app.project, h = p.ai_history || [];
    if (!h.length) { this.line("Nothing to report yet.", "s"); return; }
    const L = [`# ${p.name || "gfem"} - AI design record`, "", `Generated by gfem ${new Date().toISOString().slice(0, 16).replace("T", " ")}`, ""];
    h.forEach((r, i) => {
      L.push(`## ${i + 1}. ${r.mode === "simulate" ? "Simulation" : r.mode === "modify" ? "Change" : "New part"} (${r.provider}, ${r.model})`, "",
        "**Request**", "", "> " + String(r.prompt || "").split("\n").join("\n> "), "",
        `${r.builds ?? "?"} build(s), ${r.seconds ?? "?"} s` + (r.usage && r.usage.cost_usd ? `, about $${r.usage.cost_usd} API-equivalent` : ""), "",
        "**Result**", "", r.summary || "(no summary)", "");
    });
    L.push("## Model", "", `${p.geometry.features.length} features, ${p.conditions.length} conditions, analysis: ${p.analysis.type}`, "");
    const img = $("#ai-img");
    if (img.classList.contains("on")) L.push("![latest view](" + img.src + ")", "");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([L.join("\n")], { type: "text/markdown" }));
    a.download = (p.name || "gfem").split(/\s+/).join("_") + "_ai_report.md";
    a.click();
  }

  renderChips() {
    const list = this.mode() === "simulate" ? SIM_EXAMPLES : EXAMPLES;
    $("#ai-chips").innerHTML = list.map(([k], i) => `<button data-i="${i}" title="${esc(list[i][1])}">${esc(k)}</button>`).join("");
  }

  addThumb(png, features, kind) {
    if (!png) return;
    this.builds.push({ png, features, kind });
    const i = this.builds.length - 1;
    const im = document.createElement("img");
    im.src = "data:image/png;base64," + png; im.dataset.i = i; im.className = kind === "result" ? "res" : "";
    im.title = kind === "result" ? "result contour" : `build ${i + 1}` + (features ? "" : "");
    $("#ai-strip").appendChild(im); $("#ai-strip").scrollLeft = 1e9;
  }

  pickBuild(i) {
    const b = this.builds[i]; if (!b) return;
    this.showPng(b.png);
    document.querySelectorAll("#ai-strip img").forEach(x => x.classList.toggle("sel", +x.dataset.i === i));
    this._picked = i;
    $("#ai-apply-build").classList.toggle("on", !!b.features && !this.busy);
  }

  async applyBuild() {
    const b = this.builds[this._picked]; if (!b || !b.features) return;
    await this.apply({ features: b.features, name: null, builds: 1, seconds: 0 });
    $("#ai-apply-build").classList.remove("on");
  }

  isCli(p) { return p === "claude-cli" || p === "codex-cli"; }
  provider() { return this.account === "api" ? $("#ai-provider").value : this.account; }

  fillSettings() {
    const s = this.settings;
    $("#ai-provider").innerHTML = Object.entries(s.providers).map(([k, p]) => `<option value="${k}">${esc(p.label)}</option>`).join("");
    $("#ai-provider").value = this.isCli(s.provider) ? "anthropic" : s.provider;
    $("#ai-effort").innerHTML = s.efforts.map(e => `<option>${e}</option>`).join("");
    $("#ai-effort").value = s.effort;
    $("#ai-base").value = s.base_url || "";
    $("#ai-turns").value = s.max_turns;
    $("#ai-visual").checked = !!s.visual;
    this.account = this.isCli(s.provider) ? s.provider : "api";
    this.fillModels(false);
    $("#ai-model").value = s.model || "";
  }

  async pickAccount(acc) {
    if (this.busy) return;
    if (!this.settings) await this.loadSettings();
    if (!this.settings) return;
    this.account = acc;
    this.loginPanel(false);
    this.fillModels(true);
    this.saveSettings();
  }

  fillModels(reset) {
    if (!this.settings) return;
    const prov = this.provider(), cli = this.isCli(prov);
    document.querySelectorAll("#ai-acc-seg button").forEach(b => b.classList.toggle("on", b.dataset.p === this.account));
    document.querySelectorAll(".ai-apirow, .ai-keyrow").forEach(el => el.style.display = cli ? "none" : "");
    $(".ai-acc-btns").style.display = cli ? "" : "none";
    $("#ai-device-lbl").style.display = prov === "codex-cli" ? "" : "none";
    const compat = prov === "compatible";
    document.querySelectorAll(".ai-burl").forEach(el => el.style.display = compat ? "" : "none");
    const info = $("#ai-keyinfo");
    if (cli) {
      const c = this.settings.cli[prov];
      $("#ai-models").innerHTML = c.models.filter(Boolean).map(m => `<option value="${esc(m)}">`).join("");
      if (reset) $("#ai-model").value = "";
      $("#ai-model").placeholder = "default model of your account";
      info.className = "ai-keyinfo"; info.textContent = `Uses your ${c.account} plan through the official ${prov === "claude-cli" ? "Claude Code" : "Codex"} sign-in; no API key needed.`;
      $("#ai-device-lbl").style.display = prov === "codex-cli" ? "" : "none";
      this.renderStatus(); this.refreshStatus(false);
      return;
    }
    const p = this.settings.providers[prov];
    $("#ai-models").innerHTML = p.models.map(m => `<option value="${esc(m)}">`).join("");
    if (reset) $("#ai-model").value = p.default_model;
    $("#ai-model").placeholder = p.default_model || "model name (e.g. qwen2.5-coder:32b)";
    const st = $("#ai-acc-status");
    st.className = "ai-acc-status " + (p.has_key || compat ? "ok" : "bad");
    st.textContent = compat ? "local server (set the base URL below)" : p.has_key ? `API key ready (${p.key_source === "env" ? p.env : "saved"})` : "no API key yet – open Model & options";
    if (!p.sdk) { info.className = "ai-keyinfo missing"; info.textContent = `Python package missing on the server: pip install ${prov === "anthropic" ? "anthropic" : "openai"}`; }
    else if (p.has_key) { info.className = "ai-keyinfo"; info.textContent = p.key_source === "env" ? `Using the key from the ${p.env} environment variable.` : "A key is saved on this computer (~/.gfem/ai.json). Enter a new one to replace it, or save empty to remove it."; }
    else if (compat) { info.className = "ai-keyinfo"; info.textContent = "Local servers usually need no key. Set the base URL of the OpenAI-compatible endpoint."; }
    else { info.className = "ai-keyinfo missing"; info.textContent = `No API key yet: paste one above and press Save (stored only on this computer), or set ${p.env} before starting gfem.`; }
    $("#ai-key").placeholder = p.has_key ? "•••••••• (saved)" : "paste API key";
  }

  // ---------------------------------------------------------------- accounts (signed-in CLIs)
  async refreshStatus(force) {
    const prov = this.provider();
    if (!this.isCli(prov)) return;
    if (!force && this.cliStatus[prov] && Date.now() - this.cliStatus[prov]._t < 60000) return this.renderStatus();
    $("#ai-acc-status").className = "ai-acc-status"; $("#ai-acc-status").textContent = "checking…";
    try {
      const r = await this.api.get("/api/ai/cli_status?provider=" + prov);
      this.cliStatus[prov] = Object.assign(r[prov], { _t: Date.now() });
    } catch (e) { this.cliStatus[prov] = { installed: false, detail: e.message, _t: Date.now() }; }
    if (this.provider() === prov) this.renderStatus();
  }

  renderStatus() {
    const prov = this.provider(); if (!this.isCli(prov)) return;
    const st = this.cliStatus[prov], el = $("#ai-acc-status");
    const name = prov === "claude-cli" ? "Claude" : "ChatGPT";
    $("#ai-signout").style.display = st?.logged_in ? "" : "none";
    $("#ai-signin").style.display = st && st.installed && !st.logged_in ? "" : "none";
    if (!st) { el.className = "ai-acc-status"; el.textContent = "checking…"; return; }
    if (!st.installed) { el.className = "ai-acc-status bad"; el.innerHTML = `${prov === "claude-cli" ? "Claude Code" : "Codex"} CLI not installed – run <code>${esc(st.install)}</code> and press ⟳`; return; }
    el.className = "ai-acc-status " + (st.logged_in ? "ok" : "bad");
    el.textContent = st.logged_in ? `✔ signed in to ${name}` + (st.detail && st.detail.includes("@") ? ` (${st.detail})` : "") : `not signed in to ${name}`;
  }

  loginPanel(on) { $("#ai-login").classList.toggle("on", !!on); if (!on) { $("#ai-login-msg").textContent = ""; $("#ai-login-link").removeAttribute("href"); } }

  async signIn() {
    const prov = this.provider();
    const remote = !["localhost", "127.0.0.1", "::1", "[::1]"].includes(location.hostname);
    this.loginPanel(true);
    $("#ai-login-msg").textContent = remote ? "Starting sign-in… (gfem runs on another computer: use the link below)" : "Starting sign-in… a browser window opens; sign in there.";
    $("#ai-code-row").style.display = "";
    this._popup = remote ? window.open("", "gfem-signin") : null;     // opened in the click: not blocked
    try { await this.api.post("/api/ai/login", { provider: prov, device: prov === "codex-cli" && ($("#ai-device").checked || remote) }); }
    catch (e) { $("#ai-login-msg").textContent = "Sign-in failed to start: " + e.message; if (this._popup) this._popup.close(); }
  }

  async sendCode() {
    const code = $("#ai-code").value.trim(); if (!code) return;
    try { await this.api.post("/api/ai/login_code", { code }); $("#ai-code").value = ""; $("#ai-login-msg").textContent += "\ncode sent…"; }
    catch (e) { $("#ai-login-msg").textContent += "\n" + e.message; }
  }

  async signOut() {
    const prov = this.provider();
    if (!confirm(`Sign out of ${prov === "claude-cli" ? "Claude (Claude Code)" : "ChatGPT (Codex)"} on this computer?`)) return;
    try { const r = await this.api.post("/api/ai/logout", { provider: prov }); this.cliStatus[prov] = Object.assign(r, { _t: Date.now() }); this.renderStatus(); }
    catch (e) { this.line("Sign out failed: " + e.message, "e"); }
  }

  async saveSettings() {
    try {
      this.settings = await this.api.post("/api/ai/settings", {
        provider: this.provider(), model: $("#ai-model").value.trim(), effort: $("#ai-effort").value,
        base_url: $("#ai-base").value.trim(), max_turns: Math.max(1, Math.min(30, +$("#ai-turns").value || 10)), visual: $("#ai-visual").checked });
      this.fillModels(false);
    } catch (e) { this.line("Settings: " + e.message, "e"); }
  }

  async saveKey() {
    const key = $("#ai-key").value.trim();
    try {
      this.settings = await this.api.post("/api/ai/settings", { provider: $("#ai-provider").value, key });
      this.account = "api";
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
    const prov = this.provider();
    if (this.isCli(prov)) {
      if (!this.cliStatus[prov]) await this.refreshStatus(true);
      const st = this.cliStatus[prov];
      if (!st?.installed) { this.line(`Install the ${prov === "claude-cli" ? "Claude Code" : "Codex"} CLI first: ${st?.install || ""}`, "e"); return; }
      if (!st.logged_in) { this.line("Sign in first (button at the top).", "e"); $("#ai-signin").focus(); return; }
    } else {
      const p = this.settings?.providers?.[prov];
      if (p && !p.has_key && prov !== "compatible") {
        $("#ai-settings").open = true; $("#ai-key").focus();
        this.line("Enter an API key first (Model & options).", "e"); return;
      }
    }
    const mode = this.mode();
    const body = {
      prompt, mode, attachments: this.atts.map(a => a.path),
      base_features: mode === "modify" ? this.app.project.geometry.features : null,
      project: mode === "simulate" ? this.app.project : null,
      session: mode !== "new" && this.session && this.session.provider === prov ? this.session.id : null,
      provider: prov, model: $("#ai-model").value.trim() || null, effort: $("#ai-effort").value,
      base_url: $("#ai-base").value.trim() || null, max_turns: +$("#ai-turns").value || null, visual: $("#ai-visual").checked,
    };
    this._lastPrompt = prompt; this._lastMode = mode;
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
    const d = document.createElement("div"); d.className = cls;
    if (cls === "a") d.innerHTML = esc(msg).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>").replace(/^(\s*)[-*] /gm, "$1• ");
    else d.textContent = msg;
    const L = $("#ai-log"); L.appendChild(d); L.scrollTop = L.scrollHeight;
  }

  showPng(b64) {
    if (!b64) return;
    const img = $("#ai-img"); img.src = "data:image/png;base64," + b64; img.className = "on"; $("#ai-img-msg").style.display = "none";
    $("#ai-apply-build").classList.remove("on");
  }

  onEvent(e) {
    if (e.step === "login") {
      this.loginPanel(true);
      const m = $("#ai-login-msg"); m.textContent += (m.textContent ? "\n" : "") + e.line; m.scrollTop = m.scrollHeight;
      if (e.url) { $("#ai-login-link").href = e.url; if (this._popup && !this._popup.closed) { this._popup.location = e.url; this._popup = null; } }
      return;
    }
    if (e.step === "login_done") {
      this.cliStatus[e.provider] = Object.assign(e.status || {}, { _t: Date.now() });
      if (e.ok) { this.loginPanel(false); this.line(`Signed in to ${e.provider === "claude-cli" ? "Claude" : "ChatGPT"}.`, "b"); }
      else $("#ai-login-msg").textContent += "\nSign-in did not complete.";
      this.renderStatus(); return;
    }
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
    if (e.step === "sim") {
      if (e.png) { this.showPng(e.png); this.addThumb(e.png, null, "result"); }
      if (e.tool === "solve" && !e.error) $("#ai-status").textContent = "solved – the model is checking the results…";
      return;
    }
    if (e.step === "build") {
      if (e.ok) { this.showPng(e.png); this.addThumb(e.png, e.features, "build"); }
      $("#ai-status").textContent = e.ok ? `build ok: ${e.bodies} body(ies), ${(e.size || []).map(v => +v.toPrecision(4)).join(" × ")}` : "build failed – the model is fixing it…";
      return;
    }
    if (e.step === "done") {
      this.setBusy(false, "");
      const tok = (e.usage && (e.usage.input || e.usage.output) ? ` · tokens ${e.usage.input + (e.usage.cache_read || 0)} in / ${e.usage.output} out` : "")
        + (e.usage?.cost_usd ? ` · ≈ $${e.usage.cost_usd} (API-equivalent)` : "");
      if (e.builds !== undefined) $("#ai-usage").textContent = `${e.builds} build(s) · ${e.seconds}s${tok}`;
      if (!e.ok) { this.line("✖ " + (e.error || "no valid geometry"), "e"); return; }
      this.showPng(e.png);
      if (e.session) this.session = { id: e.session, provider: this.provider() };
      this.apply(e);
    }
  }

  async apply(e) {
    const app = this.app;
    const sim = !!e.project;
    this.history.push(sim ? { project: JSON.parse(JSON.stringify(app.project)) }
      : { features: JSON.parse(JSON.stringify(app.project.geometry.features)), name: app.project.name });
    $("#ai-undo").disabled = false;
    if (sim) {
      app.project = e.project; $("#proj-name").value = e.project.name || "";
      app.renderRibbon(app.tab || "home");
    } else {
      app.project.geometry.features = e.features;
      if (this.mode() === "new" && e.name) { app.project.name = e.name; $("#proj-name").value = e.name; }
    }
    (app.project.ai_history = app.project.ai_history || []).push({
      time: new Date().toISOString(), prompt: this._lastPrompt, mode: this._lastMode, provider: this.provider(),
      model: $("#ai-model").value.trim() || "default", builds: e.builds, seconds: e.seconds, usage: e.usage, summary: (e.summary || "").slice(0, 4000) });
    const ok = await app.regenerate();
    this.line(ok ? `✔ applied to the model (${e.features.length} features` + (sim ? `, ${app.project.conditions.length} conditions, ${app.project.analysis.type})` : ")")
      : "✖ the result could not be regenerated in the GUI", ok ? "b" : "e");
    app.log(`AI ${sim ? "simulation" : "CAD"}: ${e.features.length} features` + (e.name ? ` – ${e.name}` : ""));
    if (!ok) return;
    const modR = document.querySelector('input[name="ai-mode"][value="modify"]'); modR.disabled = false;
    if (sim) {
      if (e.solve_summary) { this.line("solving in the GUI to show the results…", "s"); app.solve(); }
      $("#ai-prompt").value = ""; $("#ai-prompt").placeholder = "Ask for a change (e.g. 'the safety factor must be at least 4 – thicken the bar') or a new analysis.";
    } else {
      this.setMode("modify"); $("#ai-prompt").value = "";
      $("#ai-prompt").placeholder = "Describe a change (e.g. 'make the bolt holes M12 and add a 2 mm chamfer on the outer edge'), or switch to New part.";
    }
  }

  async undo() {
    const h = this.history.pop();
    if (!h) return;
    if (h.project) { this.app.project = h.project; $("#proj-name").value = h.project.name || ""; }
    else { this.app.project.geometry.features = h.features; this.app.project.name = h.name; $("#proj-name").value = h.name; }
    await this.app.regenerate();
    this.line("↶ restored the previous geometry", "s");
    $("#ai-undo").disabled = !this.history.length;
  }
}
