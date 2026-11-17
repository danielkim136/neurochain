/* NeuroChain results page: interactive charts drawn as plain SVG.
   Every value comes from the JSON block #nc-data, written by
   neurochain.showcase from results/*.json. Nothing here computes a result;
   the code only selects, scales and draws. */
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("nc-data").textContent);
  const NS = "http://www.w3.org/2000/svg";
  const MINUS = "−";
  const C = { low: "var(--accent)", conv: "var(--ink-3)", ink: "var(--ink)", ink3: "var(--ink-3)", ink4: "var(--ink-4)" };

  // ---------- small helpers ----------
  function el(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    for (const key in attrs || {}) node.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(node);
    return node;
  }
  function text(parent, x, y, str, attrs) {
    const node = el("text", Object.assign({ x: x, y: y }, attrs || {}), parent);
    node.textContent = str;
    return node;
  }
  function fmt(v, d) {
    return (d === undefined ? v.toFixed(3) : v.toFixed(d)).replace("-", MINUS);
  }
  function signed(v, d) {
    const s = (v >= 0 ? "+" : "") + v.toFixed(d === undefined ? 3 : d);
    return s.replace("-", MINUS);
  }
  function linear(d0, d1, r0, r1) {
    const f = function (v) { return r0 + ((v - d0) / (d1 - d0)) * (r1 - r0); };
    f.domain = [d0, d1];
    f.invert = function (p) { return d0 + ((p - r0) / (r1 - r0)) * (d1 - d0); };
    return f;
  }
  function logScale(d0, d1, r0, r1) {
    const l0 = Math.log10(d0), l1 = Math.log10(d1);
    const f = function (v) { return r0 + ((Math.log10(v) - l0) / (l1 - l0)) * (r1 - r0); };
    f.domain = [d0, d1];
    return f;
  }
  function niceTicks(lo, hi, count) {
    const span = hi - lo;
    const raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(function (m) { return m * mag; }).find(function (s) { return span / s <= count; });
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function decimalsFor(ticks) {
    const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1;
    return Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  }
  function frame(host, height, margin) {
    host.innerHTML = "";
    const width = Math.max(300, host.clientWidth || 640);
    const svg = el("svg", { viewBox: "0 0 " + width + " " + height, width: width, height: height, role: "img" }, host);
    return { svg: svg, width: width, height: height, m: margin, x0: margin.l, x1: width - margin.r, y0: height - margin.b, y1: margin.t };
  }
  function yAxis(f, scale, ticks, label, digits) {
    const d = digits === undefined ? decimalsFor(ticks) : digits;
    ticks.forEach(function (t) {
      const y = scale(t);
      el("line", { x1: f.x0, x2: f.x1, y1: y, y2: y, class: "gridline" }, f.svg);
      text(f.svg, f.x0 - 8, y + 4, fmt(t, d), { "text-anchor": "end", class: "tick-label" });
    });
    if (label) text(f.svg, f.x0 - f.m.l + 4, f.y1 - 10, label, { class: "axis-title" });
  }
  function xAxis(f, scale, ticks, label, fmtTick) {
    el("line", { x1: f.x0, x2: f.x1, y1: f.y0, y2: f.y0, class: "baseline" }, f.svg);
    ticks.forEach(function (t) {
      const x = scale(t);
      el("line", { x1: x, x2: x, y1: f.y0, y2: f.y0 + 4, class: "baseline" }, f.svg);
      text(f.svg, x, f.y0 + 18, fmtTick ? fmtTick(t) : String(t), { "text-anchor": "middle", class: "tick-label" });
    });
    if (label) text(f.svg, f.x1, f.y0 + 36, label, { "text-anchor": "end", class: "axis-title" });
  }
  function path(points) {
    let d = "", pen = false;
    points.forEach(function (p) {
      if (p === null) { pen = false; return; }
      d += (pen ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1);
      pen = true;
    });
    return d;
  }
  function legend(host, items) {
    const ul = document.createElement("ul");
    ul.className = "legend";
    items.forEach(function (it) {
      const li = document.createElement("li");
      const sw = document.createElement("i");
      if (it.kind === "line") sw.className = "line";
      if (it.kind === "dash") sw.className = "dash";
      if (it.kind === "hollow" || it.kind === "ring") { sw.style.border = "2px solid " + it.color; sw.style.background = "transparent"; }
      else if (it.kind !== "dash") sw.style.background = it.color;
      if (it.kind === "square") sw.style.borderRadius = "2px";
      if (it.kind === "diamond") { sw.style.borderRadius = "1px"; sw.style.transform = "rotate(45deg) scale(0.85)"; }
      li.appendChild(sw);
      li.appendChild(document.createTextNode(it.label));
      ul.appendChild(li);
    });
    host.insertBefore(ul, host.firstChild);
  }

  // Tooltip shared by every chart.
  const tip = document.getElementById("tip");
  function showTip(html, evt) {
    tip.innerHTML = html;
    tip.hidden = false;
    const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
    tip.style.left = Math.max(8, x) + "px";
    tip.style.top = Math.max(8, y) + "px";
  }
  function hideTip() { tip.hidden = true; }

  // Crosshair for line charts: snaps to the nearest x in xs.
  function crosshair(f, xs, scaleX, html) {
    const line = el("line", { y1: f.y1, y2: f.y0, stroke: "var(--ink-3)", "stroke-width": 1, "stroke-dasharray": "3 3", visibility: "hidden" }, f.svg);
    const hit = el("rect", { x: f.x0, y: f.y1, width: f.x1 - f.x0, height: f.y0 - f.y1, class: "hit" }, f.svg);
    function move(evt) {
      const box = f.svg.getBoundingClientRect();
      const px = ((evt.clientX - box.left) / box.width) * f.width;
      let best = 0, dist = Infinity;
      xs.forEach(function (v, i) { const d = Math.abs(scaleX(v) - px); if (d < dist) { dist = d; best = i; } });
      const x = scaleX(xs[best]);
      line.setAttribute("x1", x); line.setAttribute("x2", x); line.setAttribute("visibility", "visible");
      showTip(html(best), evt);
    }
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerdown", move);
    hit.addEventListener("pointerleave", function () { line.setAttribute("visibility", "hidden"); hideTip(); });
  }

  function segmented(host, options, initial, onPick) {
    host.innerHTML = "";
    const buttons = options.map(function (opt) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = opt.label;
      b.setAttribute("aria-pressed", String(opt.value === initial));
      b.addEventListener("click", function () {
        buttons.forEach(function (o) { o.setAttribute("aria-pressed", "false"); });
        b.setAttribute("aria-pressed", "true");
        onPick(opt.value);
      });
      host.appendChild(b);
      return b;
    });
  }
  function fillSelect(select, options, initial) {
    options.forEach(function (o) {
      const opt = document.createElement("option");
      opt.value = o.value; opt.textContent = o.label;
      if (o.value === initial) opt.selected = true;
      select.appendChild(opt);
    });
  }

  const redraws = [];
  function register(host, draw) {
    redraws.push({ host: host, draw: draw });
    draw();
  }
  let lastWidth = window.innerWidth;
  window.addEventListener("resize", function () {
    if (Math.abs(window.innerWidth - lastWidth) < 8) return;
    lastWidth = window.innerWidth;
    redraws.forEach(function (r) { r.draw(); });
  });

  // ---------- Pareto (Figure 1, the hero) ----------
  (function () {
    const P = DATA.pareto;
    const host = document.getElementById("pareto-chart");
    const select = document.getElementById("pareto-dataset");
    fillSelect(select, Object.keys(P.datasets).map(function (k) { return { value: k, label: P.datasets[k].name }; }), "nason");
    let ds = "nason";
    // Feature is told apart by shape; colour is kept for the selected chain alone.
    const FEAT = {
      sbp: { label: "Spiking-band power", shape: "circle" },
      crossings: { label: "Threshold crossings", shape: "ring" },
      both: { label: "Both features", shape: "diamond" },
      raw: { label: "Every sample", shape: "square" },
    };
    const PARTS = ["Amplifier", "ADC", "Feature logic", "Radio"];

    function mark(parent, shape, x, y, r, color) {
      let m;
      if (shape === "circle" || shape === "ring") m = el("circle", { cx: x, cy: y, r: r }, parent);
      else if (shape === "square") m = el("rect", { x: x - r, y: y - r, width: 2 * r, height: 2 * r, rx: 1 }, parent);
      else m = el("path", { d: "M" + x + " " + (y - r - 1) + "L" + (x + r + 1) + " " + y + "L" + x + " " + (y + r + 1) + "L" + (x - r - 1) + " " + y + "Z" }, parent);
      m.setAttribute("style", shape === "ring" ? "fill:var(--sheet);stroke:" + color + ";stroke-width:1.5" : "fill:" + color + ";stroke:var(--sheet);stroke-width:1.5");
      return m;
    }

    function draw() {
      const info = P.datasets[ds];
      const idx = P.configs.map(function (_, i) { return i; }).filter(function (i) { return P.configs[i].r2[ds] !== undefined; });
      const f = frame(host, Math.min(380, Math.max(300, host.clientWidth * 0.66)), { l: 44, r: 12, t: 20, b: 44 });
      const uws = idx.map(function (i) { return P.configs[i].uw; });
      const r2s = idx.map(function (i) { return P.configs[i].r2[ds]; });
      const xs = logScale(Math.min.apply(null, uws) / 1.4, Math.max.apply(null, uws) * 1.4, f.x0, f.x1);
      const lo = Math.min.apply(null, r2s), hi = Math.max.apply(null, r2s), pad = (hi - lo) * 0.08;
      const ticks = niceTicks(lo - pad, hi + pad, 5);
      const ys = linear(ticks[0], ticks[ticks.length - 1], f.y0, f.y1);
      yAxis(f, ys, ticks, "R², held out");
      const xt = [0.1, 0.3, 1, 3, 10, 30].filter(function (t) { return t >= xs.domain[0] && t <= xs.domain[1]; });
      xAxis(f, xs, xt, "µW per channel", function (t) { return String(t); });

      const base = P.configs[info.baseline], target = (1 - P.tolerance) * base.r2[ds];
      el("line", { x1: f.x0, x2: f.x1, y1: ys(target), y2: ys(target), stroke: "var(--ink-4)", "stroke-dasharray": "4 4" }, f.svg);
      text(f.svg, f.x1, ys(target) + 16, "within " + Math.round(P.tolerance * 100) + "% of the conventional chain", { class: "muted-label", "text-anchor": "end" });

      const front = info.front.slice().sort(function (a, b) { return P.configs[a].uw - P.configs[b].uw; });
      el("path", { d: path(front.map(function (i) { return [xs(P.configs[i].uw), ys(P.configs[i].r2[ds])]; })), fill: "none", stroke: "var(--ink-3)", "stroke-width": 1.5 }, f.svg);

      const marks = el("g", {}, f.svg);
      idx.forEach(function (i) {
        const c = P.configs[i], x = xs(c.uw), y = ys(c.r2[ds]);
        if (i !== info.winner) mark(marks, FEAT[c.feature].shape, x, y, 4, "var(--ink-4)");
      });
      const w = P.configs[info.winner], wx = xs(w.uw), wy = ys(w.r2[ds]);
      mark(f.svg, FEAT[w.feature].shape, wx, wy, 6, "var(--accent)");
      const bx = xs(base.uw), by = ys(base.r2[ds]);
      el("circle", { cx: bx, cy: by, r: 8, fill: "none", stroke: "var(--ink-2)", "stroke-width": 1.5 }, f.svg);
      const wRight = wx < (f.x0 + f.x1) / 2;
      text(f.svg, wx + (wRight ? 12 : -12), wy - 12, "Selected, " + fmt(w.uw, 2) + " µW", { class: "direct-accent", "text-anchor": wRight ? "start" : "end" });
      text(f.svg, bx - 12, by - 14, "Conventional, " + fmt(base.uw, 2) + " µW", { class: "direct", "text-anchor": "end" });

      idx.forEach(function (i) {
        const c = P.configs[i];
        const hit = el("circle", { cx: xs(c.uw), cy: ys(c.r2[ds]), r: 10, class: "hit" }, f.svg);
        function show(evt) {
          let rows = "";
          c.parts.forEach(function (v, k) { rows += "<br><span class='muted'>" + PARTS[k] + "</span> <span class='mono'>" + (v < 0.01 ? v.toPrecision(2) : fmt(v, 3)) + "</span>"; });
          showTip("<b>" + c.label + "</b><br><span class='mono'>R² " + fmt(c.r2[ds]) + " · " + fmt(c.uw, 2) + " µW</span>" + rows, evt);
        }
        hit.addEventListener("pointerenter", show);
        hit.addEventListener("pointerdown", show);
        hit.addEventListener("pointerleave", hideTip);
      });
      legend(host, Object.keys(FEAT).filter(function (k) { return idx.some(function (i) { return P.configs[i].feature === k; }); })
        .map(function (k) { return { label: FEAT[k].label, color: "var(--ink-4)", kind: FEAT[k].shape }; })
        .concat([{ label: "Pareto front", color: "var(--ink-3)", kind: "line" }]));
    }
    select.addEventListener("change", function () { ds = select.value; draw(); });
    register(host, draw);
  })();

  // ---------- Decoders ----------
  (function () {
    const D = DATA.decoders;
    const host = document.getElementById("dec-chart");
    const select = document.getElementById("dec-dataset");
    const keys = Object.keys(D.datasets);
    fillSelect(select, keys.map(function (k) { return { value: k, label: D.datasets[k].name }; }), "nason");
    let ds = "nason";

    function draw() {
      const entry = D.datasets[ds];
      const names = Object.keys(D.names);
      const rowH = 54;
      const f = frame(host, names.length * rowH + 60, { l: 150, r: 20, t: 10, b: 46 });
      let lo = Infinity, hi = -Infinity;
      names.forEach(function (n) { ["conventional", "winner"].forEach(function (r) { lo = Math.min(lo, entry.rows[n][r][1]); hi = Math.max(hi, entry.rows[n][r][2]); }); });
      const ticks = niceTicks(lo - 0.03, hi + 0.03, 6);
      const xs = linear(ticks[0], ticks[ticks.length - 1], f.x0, f.x1);
      ticks.forEach(function (t) { el("line", { x1: xs(t), x2: xs(t), y1: f.y1, y2: f.y0, class: "gridline" }, f.svg); });
      xAxis(f, xs, ticks, "R², held out (95 % interval)", function (t) { return fmt(t, decimalsFor(ticks)); });
      names.forEach(function (n, k) {
        const yc = f.y1 + k * rowH + rowH / 2;
        text(f.svg, 0, yc - 2, D.names[n], { class: "direct" });
        const diff = entry.rows[n].diff;
        text(f.svg, 0, yc + 15, "Δ " + signed(diff[0]) + " [" + signed(diff[1]) + ", " + signed(diff[2]) + "]", { class: "tick-label" });
        [["conventional", C.conv, -7], ["winner", C.low, 7]].forEach(function (spec) {
          const v = entry.rows[n][spec[0]], y = yc + spec[2];
          el("line", { x1: xs(v[1]), x2: xs(v[2]), y1: y, y2: y, style: "stroke:" + spec[1], "stroke-width": 2, "stroke-linecap": "round" }, f.svg);
          el("circle", { cx: xs(v[0]), cy: y, r: 5, style: "fill:" + spec[1] + ";stroke:var(--sheet);stroke-width:2" }, f.svg);
          const hit = el("rect", { x: xs(v[1]) - 6, y: y - 8, width: xs(v[2]) - xs(v[1]) + 12, height: 16, class: "hit" }, f.svg);
          hit.addEventListener("pointerenter", function (evt) {
            showTip("<b>" + D.names[n] + "</b> · " + entry[spec[0]].label + "<br><span class='mono'>R² " + fmt(v[0]) + " [" + fmt(v[1]) + ", " + fmt(v[2]) + "] · " + fmt(entry[spec[0]].uw, 2) + " µW</span>", evt);
          });
          hit.addEventListener("pointerleave", hideTip);
        });
      });
      legend(host, [
        { label: "Conventional: " + entry.conventional.label + " (" + fmt(entry.conventional.uw, 2) + " µW)", color: C.conv },
        { label: "Selected for this dataset: " + entry.winner.label + " (" + fmt(entry.winner.uw, 2) + " µW)", color: C.low },
      ]);
    }
    select.addEventListener("change", function () { ds = select.value; draw(); });
    register(host, draw);

    // Example trace.
    const ex = D.example;
    const thost = document.getElementById("trace-chart");
    let pick = "gru";
    document.getElementById("trace-caption").textContent =
      "Held-out finger velocity on Nason, " + (ex.truth.length * ex.bin_s).toFixed(0) + " s in " + Math.round(ex.bin_s * 1000) + " ms bins, decoded from the selected chain (" + ex.label + "). Velocity in full range per second.";
    function drawTrace() {
      const f = frame(thost, 260, { l: 46, r: 16, t: 24, b: 44 });
      const n = ex.truth.length, t = ex.truth.map(function (_, i) { return i * ex.bin_s; });
      const all = ex.truth.concat(ex[pick]);
      const ticks = niceTicks(Math.min.apply(null, all), Math.max.apply(null, all), 5);
      const ys = linear(ticks[0], ticks[ticks.length - 1], f.y0, f.y1);
      const xs = linear(0, t[n - 1], f.x0, f.x1);
      yAxis(f, ys, ticks, "velocity");
      xAxis(f, xs, niceTicks(0, t[n - 1], 6), "seconds", function (v) { return String(v); });
      el("path", { d: path(ex.truth.map(function (v, i) { return [xs(t[i]), ys(v)]; })), fill: "none", stroke: "var(--ink)", "stroke-width": 2, opacity: 0.85 }, f.svg);
      el("path", { d: path(ex[pick].map(function (v, i) { return [xs(t[i]), ys(v)]; })), fill: "none", style: "stroke:" + C.low, "stroke-width": 2 }, f.svg);
      crosshair(f, t, xs, function (i) {
        return "<span class='mono'>" + t[i].toFixed(2) + " s</span><br>True <span class='mono'>" + fmt(ex.truth[i], 2) + "</span><br>" + D.names[pick] + " <span class='mono'>" + fmt(ex[pick][i], 2) + "</span>";
      });
      legend(thost, [{ label: "True velocity", color: "var(--ink)", kind: "line" }, { label: D.names[pick] + ", decoded", color: C.low, kind: "line" }]);
    }
    segmented(document.getElementById("trace-pick"), Object.keys(D.names).map(function (k) { return { value: k, label: D.names[k] }; }), pick, function (v) { pick = v; drawTrace(); });
    register(thost, drawTrace);
  })();

  // ---------- Stimulation ----------
  (function () {
    const S = DATA.stim, tr = S.trace;
    const host = document.getElementById("stim-chart");
    let pick = "template";
    const NAMES = { none: "No treatment", blank: "Blanking", template: "Template canceller" };

    function draw() {
      host.innerHTML = "";
      const width = Math.max(300, host.clientWidth || 640);
      const h1 = 170, h2 = 210, gap = 26;
      const svg = el("svg", { viewBox: "0 0 " + width + " " + (h1 + h2 + gap), width: width, height: h1 + h2 + gap, role: "img" }, host);
      const m = { l: 54, r: 16 };
      const t = tr.time_ms, xs = linear(t[0], t[t.length - 1], m.l, width - m.r);
      const step = t[1] - t[0];

      function panel(top, height, domain, label) {
        const f = { svg: svg, width: width, height: height, m: { l: m.l, r: m.r, t: 18, b: 30 }, x0: m.l, x1: width - m.r, y1: top + 18, y0: top + height - 30 };
        const ticks = niceTicks(domain[0], domain[1], 4);
        const ys = linear(ticks[0], ticks[ticks.length - 1], f.y0, f.y1);
        yAxis(f, ys, ticks, label, 0);
        tr.onsets_ms.forEach(function (o) {
          el("rect", { x: xs(o), y: f.y1, width: Math.max(2, xs(o + 0.7) - xs(o)), height: f.y0 - f.y1, fill: "var(--neutral-soft)" }, svg);
        });
        return { f: f, ys: ys };
      }
      const top = panel(0, h1, [-1100, 1100], "µV, what the ADC sees");
      [-1000, 1000].forEach(function (v) {
        el("line", { x1: m.l, x2: width - m.r, y1: top.ys(v), y2: top.ys(v), stroke: "var(--ink-3)", "stroke-dasharray": "4 4" }, svg);
      });
      text(svg, width - m.r, top.ys(1000) - 5, "ADC range ±1 mV", { "text-anchor": "end", class: "muted-label" });
      el("path", { d: path(tr.raw_uv.map(function (v, i) { return [xs(t[i]), top.ys(v)]; })), fill: "none", style: "stroke:" + C.ink4, "stroke-width": 1.5 }, svg);

      const series = pick === "none" ? tr.raw_uv : tr[pick + "_uv"];
      const valid = pick === "none" ? null : tr[pick + "_valid"];
      const lim = Math.max(30, Math.max.apply(null, tr.clean_uv.map(Math.abs)) * 1.6);
      const bot = panel(h1 + gap, h2, [-lim, lim], "µV, clean and treated");
      const clipId = "stimclip";
      const clip = el("clipPath", { id: clipId }, svg);
      el("rect", { x: m.l, y: bot.f.y1, width: width - m.l - m.r, height: bot.f.y0 - bot.f.y1 }, clip);
      el("path", { d: path(tr.clean_uv.map(function (v, i) { return [xs(t[i]), bot.ys(v)]; })), fill: "none", stroke: "var(--ink-2)", "stroke-width": 1.2, "clip-path": "url(#" + clipId + ")" }, svg);
      el("path", { d: path(series.map(function (v, i) { return valid && !valid[i] ? null : [xs(t[i]), bot.ys(v)]; })), fill: "none", style: "stroke:" + C.low, "stroke-width": 2, "clip-path": "url(#" + clipId + ")" }, svg);
      if (valid) {
        series.forEach(function (_, i) {
          if (!valid[i]) el("rect", { x: xs(t[i]) - 1, y: bot.f.y0 - 4, width: Math.max(2, xs(t[i] + step) - xs(t[i]) - 1), height: 4, fill: "var(--ink-3)" }, svg);
        });
      }
      xAxis(bot.f, xs, niceTicks(t[0], t[t.length - 1], 6), "ms", function (v) { return String(v); });
      crosshair(Object.assign({}, bot.f, { y1: top.f.y1 }), t, xs, function (i) {
        return "<span class='mono'>" + t[i].toFixed(1) + " ms</span><br>ADC input <span class='mono'>" + fmt(tr.raw_uv[i], 1) + " µV</span><br>Clean <span class='mono'>" + fmt(tr.clean_uv[i], 1) + " µV</span><br>" + NAMES[pick] + " " +
          (valid && !valid[i] ? "<span class='muted'>marked invalid</span>" : "<span class='mono'>" + fmt(series[i], 1) + " µV</span>");
      });
      legend(host, [
        { label: "ADC input with artifact", color: C.ink4, kind: "line" },
        { label: "Clean recording", color: "var(--ink-2)", kind: "line" },
        { label: NAMES[pick] + " output", color: C.low, kind: "line" },
        { label: "Pulse", color: "var(--neutral-soft)", kind: "square" },
      ]);
    }
    segmented(document.getElementById("stim-pick"), Object.keys(NAMES).map(function (k) { return { value: k, label: NAMES[k] }; }), pick, function (v) { pick = v; draw(); });
    register(host, draw);

    // R² by treatment.
    const rhost = document.getElementById("stim-r2");
    const CHAINS = { sbp: ["Selected chain", C.low], crossings: ["Conventional chain", C.conv] };
    function drawR2() {
      const rows = [];
      ["sbp", "crossings"].forEach(function (ch) { ["none", "blank", "template"].forEach(function (tm) { rows.push([ch, tm]); }); });
      const rowH = 34;
      const f = frame(rhost, rows.length * rowH + 70, { l: 200, r: 20, t: 10, b: 46 });
      let lo = Infinity, hi = -Infinity;
      S.rows.forEach(function (r) { lo = Math.min(lo, r.r2[1]); hi = Math.max(hi, r.r2[2]); });
      const ticks = niceTicks(lo - 0.05, Math.max(hi, S.clean.sbp, S.clean.crossings) + 0.05, 6);
      const xs = linear(ticks[0], ticks[ticks.length - 1], f.x0, f.x1);
      ticks.forEach(function (tk) { el("line", { x1: xs(tk), x2: xs(tk), y1: f.y1, y2: f.y0, class: "gridline" }, f.svg); });
      xAxis(f, xs, ticks, "R² on stimulated bins, GRU", function (v) { return fmt(v, 1); });
      rows.forEach(function (rw, k) {
        const y = f.y1 + k * rowH + rowH / 2 + (rw[0] === "crossings" ? 12 : 0);
        if (k === 0 || k === 3) text(f.svg, 0, y - 22, CHAINS[rw[0]][0], { class: "direct" });
        text(f.svg, 12, y + 4, NAMES[rw[1]], {});
        const clean = S.clean[rw[0]];
        el("line", { x1: xs(clean), x2: xs(clean), y1: y - 11, y2: y + 11, stroke: "var(--ink-3)", "stroke-dasharray": "3 2", "stroke-width": 1.5 }, f.svg);
        [["without stimulation", -5, true], ["with stimulation", 5, false]].forEach(function (spec) {
          const r = S.rows.find(function (q) { return q.chain === rw[0] && q.treatment === rw[1] && q.fitted === spec[0]; });
          if (!r) return;
          const yy = y + spec[1], col = CHAINS[rw[0]][1];
          el("line", { x1: xs(r.r2[1]), x2: xs(r.r2[2]), y1: yy, y2: yy, style: "stroke:" + col, "stroke-width": 2, "stroke-linecap": "round" }, f.svg);
          el("circle", { cx: xs(r.r2[0]), cy: yy, r: 4.5, style: spec[2] ? "fill:var(--sheet);stroke:" + col + ";stroke-width:2" : "fill:" + col + ";stroke:var(--sheet);stroke-width:1.5" }, f.svg);
          const hit = el("rect", { x: Math.min(xs(r.r2[1]), xs(r.r2[0])) - 8, y: yy - 6, width: Math.abs(xs(r.r2[2]) - xs(r.r2[1])) + 16, height: 12, class: "hit" }, f.svg);
          hit.addEventListener("pointerenter", function (evt) {
            showTip("<b>" + CHAINS[rw[0]][0] + ", " + NAMES[rw[1]].toLowerCase() + "</b><br>Decoder fitted " + spec[0] + "<br><span class='mono'>R² " + fmt(r.r2[0]) + " [" + fmt(r.r2[1]) + ", " + fmt(r.r2[2]) + "]<br>change " + signed(r.loss[0]) + " [" + signed(r.loss[1]) + ", " + signed(r.loss[2]) + "]</span>", evt);
          });
          hit.addEventListener("pointerleave", hideTip);
        });
      });
      legend(rhost, [
        { label: "Decoder fitted without stimulation", color: "var(--ink-2)", kind: "hollow" },
        { label: "Refitted with stimulation on", color: "var(--ink-2)" },
        { label: "No stimulation", kind: "dash" },
      ]);
    }
    register(rhost, drawR2);
  })();

  // ---------- Aging ----------
  (function () {
    const A = DATA.aging;
    const host = document.getElementById("age-chart");
    const cSel = document.getElementById("age-condition"), dSel = document.getElementById("age-decoder");
    fillSelect(cSel, Object.keys(A.conditions).map(function (k) {
      const c = A.conditions[k];
      return { value: k, label: (c.chain === "sbp" ? "Selected chain (SBP)" : "Conventional chain (crossings)") + ", " + c.name.split(",")[0] + " · " + c.uw.toFixed(2) + " µW" };
    }), "nason_sbp");
    fillSelect(dSel, Object.keys(DATA.decoders.names).map(function (k) { return { value: k, label: DATA.decoders.names[k] }; }), "gru");
    const MODES = { fixed: ["Left alone", "var(--ink-4)", "var(--neutral-soft)"], zscore: ["Running z-score", C.low, "var(--accent-soft)"], refit: ["Refitted", "var(--ink-2)", "var(--neutral-soft)"] };

    function draw() {
      const cond = cSel.value, dec = dSel.value;
      const rows = A.rows.filter(function (r) { return r[0] === cond && r[1] === dec; });
      const weeks = Array.from(new Set(rows.map(function (r) { return r[2]; }))).sort(function (a, b) { return a - b; });
      const f = frame(host, 340, { l: 46, r: 120, t: 24, b: 46 });
      const lo = Math.min.apply(null, rows.map(function (r) { return r[5]; })), hi = Math.max.apply(null, rows.map(function (r) { return r[6]; }));
      const ticks = niceTicks(Math.min(0, lo - 0.03), hi + 0.03, 5);
      const ys = linear(ticks[0], ticks[ticks.length - 1], f.y0, f.y1);
      const xs = linear(0, weeks[weeks.length - 1], f.x0, f.x1);
      yAxis(f, ys, ticks, "R², held out");
      xAxis(f, xs, weeks.filter(function (w) { return w !== 2 && w !== 8; }), "simulated weeks since implant", function (v) { return String(v); });
      const kx = xs(A.key_week);
      el("line", { x1: kx, x2: kx, y1: f.y1, y2: f.y0, stroke: "var(--ink-3)", "stroke-dasharray": "2 3" }, f.svg);
      text(f.svg, kx + 6, f.y1 + 4, "real NoMAD gap", { class: "muted-label" });
      const series = {};
      Object.keys(MODES).forEach(function (mode) {
        const pts = weeks.map(function (w) { return rows.find(function (r) { return r[2] === w && r[3] === mode; }); });
        if (pts.some(function (p) { return !p; })) {
          // week 0 is shared: the stabilised and refitted decoders start where the fixed one does
          const base = rows.find(function (r) { return r[2] === 0 && r[3] === "fixed"; });
          for (let i = 0; i < pts.length; i++) if (!pts[i] && weeks[i] === 0) pts[i] = base;
        }
        series[mode] = pts;
        if (pts.some(function (p) { return !p; })) return;
        const band = pts.map(function (p) { return [xs(p[2]), ys(p[6])]; }).concat(pts.slice().reverse().map(function (p) { return [xs(p[2]), ys(p[5])]; }));
        el("path", { d: path(band) + "Z", style: "fill:" + MODES[mode][2] }, f.svg);
      });
      const ends = [];
      Object.keys(MODES).forEach(function (mode) {
        const pts = series[mode];
        if (pts.some(function (p) { return !p; })) return;
        el("path", { d: path(pts.map(function (p) { return [xs(p[2]), ys(p[4])]; })), fill: "none", style: "stroke:" + MODES[mode][1], "stroke-width": 2 }, f.svg);
        pts.forEach(function (p) { el("circle", { cx: xs(p[2]), cy: ys(p[4]), r: 3, style: "fill:" + MODES[mode][1] + ";stroke:var(--sheet);stroke-width:1.5" }, f.svg); });
        ends.push({ y: ys(pts[pts.length - 1][4]), label: MODES[mode][0], cls: mode === "zscore" ? "direct-accent" : "direct" });
      });
      ends.sort(function (a, b) { return a.y - b.y; });
      for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 15) ends[i].y = ends[i - 1].y + 15;
      ends.forEach(function (e) { text(f.svg, f.x1 + 8, e.y + 4, e.label, { class: e.cls }); });
      crosshair(f, weeks, xs, function (i) {
        let s = "<b>Week " + weeks[i] + "</b>";
        Object.keys(MODES).forEach(function (mode) {
          const p = series[mode][i];
          if (p) s += "<br>" + MODES[mode][0] + " <span class='mono'>" + fmt(p[4]) + " [" + fmt(p[5]) + ", " + fmt(p[6]) + "]</span>";
        });
        return s;
      });
      legend(host, Object.keys(MODES).map(function (k) { return { label: MODES[k][0], color: MODES[k][1], kind: "line" }; }));
    }
    cSel.addEventListener("change", draw);
    dSel.addEventListener("change", draw);
    register(host, draw);
  })();

  // ---------- RTL ----------
  (function () {
    const R = DATA.rtl;
    const host = document.getElementById("rtl-chart");
    const n = R.codes.length, t = R.codes.map(function (_, i) { return (i / R.fs_hz) * 1000; });
    document.getElementById("rtl-caption").textContent =
      "Channel " + R.channel + ", " + (n / R.fs_hz).toFixed(1) + " s at " + R.fs_hz / 1000 + " kS/s, in 8-bit ADC codes (LSB; the Neuralink data has no published voltage scale). Stimulation trains rail the ADC; the canceller learns the artifact over successive pulses. " +
      (R.identical ? "The Python specification produces the same sequence, sample for sample, for every output in this window." : "The Python specification differs from the RTL in this window.");
    function draw() {
      const f = frame(host, 280, { l: 46, r: 16, t: 24, b: 44 });
      const canc = R.cancelled.map(function (v) { return v / R.scale; });
      const ticks = niceTicks(-130, 130, 4);
      const ys = linear(ticks[0], ticks[ticks.length - 1], f.y0, f.y1);
      const xs = linear(0, t[n - 1], f.x0, f.x1);
      yAxis(f, ys, ticks, "ADC code", 0);
      xAxis(f, xs, niceTicks(0, t[n - 1], 6), "ms", function (v) { return String(v); });
      el("path", { d: path(R.codes.map(function (v, i) { return [xs(t[i]), ys(v)]; })), fill: "none", style: "stroke:" + C.ink4, "stroke-width": 1.2 }, f.svg);
      el("path", { d: path(canc.map(function (v, i) { return R.valid[i] ? [xs(t[i]), ys(v)] : null; })), fill: "none", style: "stroke:" + C.low, "stroke-width": 1.5 }, f.svg);
      crosshair(f, t, xs, function (i) {
        return "<span class='mono'>" + t[i].toFixed(1) + " ms</span><br>ADC code <span class='mono'>" + R.codes[i] + "</span><br>RTL canceller " + (R.valid[i] ? "<span class='mono'>" + fmt(canc[i], 2) + "</span>" : "<span class='muted'>blanked</span>");
      });
      legend(host, [{ label: "ADC code with artifacts", color: C.ink4, kind: "line" }, { label: "Canceller output, RTL", color: C.low, kind: "line" }]);
    }
    register(host, draw);
  })();
})();
