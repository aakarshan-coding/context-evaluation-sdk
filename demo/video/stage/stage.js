"use strict";
// The renderer calls init() once, then step() once per frame. Every motion is a tween on a fixed
// 30 fps clock, so the same captures always render the same video.

const FPS = 30, W = 1760, H = 864, DASH_SCALE = 1.25;
const $ = (id) => document.getElementById(id);
const ease = (p) => (p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2);
const lerp = (a, b, p) => a + (b - a) * p;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

const timeline = [];
let T = 0;
function add(dur, start, update, advance) {
  timeline.push({ t0: T, dur, start, update, begun: false, done: false });
  if (advance) T += dur;
}
const seq = (dur, start, update) => add(dur, start, update, true);
const par = (dur, start, update) => add(dur, start, update, false);
const wait = (s) => { T += s; };
const fire = (fn) => add(0, fn, null, true);

const marks = {};
let seed = 7;
function rand() {
  seed = (seed + 0x6d2b79f5) | 0;
  let x = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
  return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
}

const view = { s: 1, fx: W / 2, fy: H / 2 };
function contentToScreen(p) {
  const c = $("content").getBoundingClientRect();
  return { x: c.left + W / 2 + (p.x - view.fx) * view.s, y: c.top + H / 2 + (p.y - view.fy) * view.s };
}
function screenToContent(x, y) {
  const c = $("content").getBoundingClientRect();
  return { x: (x - c.left - W / 2) / view.s + view.fx, y: (y - c.top - H / 2) / view.s + view.fy };
}
function toContentRect(r, frame) {
  let { left, top, right, bottom } = r;
  if (frame) {
    const f = frame.getBoundingClientRect(), k = f.width / frame.offsetWidth;
    [left, right, top, bottom] = [f.left + left * k, f.left + right * k, f.top + top * k, f.top + bottom * k];
  }
  const a = screenToContent(left, top), b = screenToContent(right, bottom);
  return { x: a.x, y: a.y, w: b.x - a.x, h: b.y - a.y };
}
function union(rects) {
  const x = Math.min(...rects.map((r) => r.x)), y = Math.min(...rects.map((r) => r.y));
  const x2 = Math.max(...rects.map((r) => r.x + r.w)), y2 = Math.max(...rects.map((r) => r.y + r.h));
  return { x, y, w: x2 - x, h: y2 - y };
}
function pad(r, px) { return { x: r.x - px, y: r.y - px, w: r.w + 2 * px, h: r.h + 2 * px }; }

function zoomTo(getRect, dur = 0.9, max = 2.0) {
  let from, to;
  seq(dur, () => {
    from = { ...view };
    const r = getRect();
    const s = clamp(Math.min((0.9 * W) / r.w, (0.8 * H) / r.h), 1, max);
    to = { s, fx: clamp(r.x + r.w / 2, W / (2 * s), W - W / (2 * s)), fy: clamp(r.y + r.h / 2, H / (2 * s), H - H / (2 * s)) };
  }, (p) => {
    const e = ease(p);
    view.s = lerp(from.s, to.s, e); view.fx = lerp(from.fx, to.fx, e); view.fy = lerp(from.fy, to.fy, e);
  });
}
const zoomOut = (dur = 0.8) => zoomTo(() => ({ x: 0, y: 0, w: W, h: H }), dur);

let captionOpacity = 0;
function caption(text, dur = 0.6) {
  let from, swapped;
  par(dur, () => { from = captionOpacity; swapped = false; }, (p) => {
    if (p < 0.4 && from > 0) { captionOpacity = from * (1 - p / 0.4); }
    else {
      if (!swapped) { $("caption").textContent = text; swapped = true; }
      captionOpacity = text ? clamp((p - 0.4) / 0.6, 0, 1) : 0;
    }
    $("caption").style.opacity = captionOpacity;
  });
}
function fadeWindow(show, dur = 0.8) {
  par(dur, null, (p) => {
    const e = ease(show ? p : 1 - p);
    $("window").style.opacity = e;
    $("window").style.transform = `scale(${lerp(0.96, 1, e)})`;
    $("card").style.opacity = 1 - e;
  });
}
function setCard(title, sub, extra) {
  fire(() => { $("card-title").textContent = title; $("card-sub").textContent = sub; $("card-extra").innerHTML = extra || ""; });
}
function showLayer(name, title, dur = 0.6) {
  fire(() => { $("wtitle").textContent = title; $("window").classList.toggle("light", name === "dash"); });
  par(dur, null, (p) => { $("dash").style.opacity = name === "dash" ? p : 1 - p; });
}
function speedBadge(text, show, dur = 0.4) {
  if (show) fire(() => { $("speed").textContent = text; });
  par(dur, null, (p) => { $("speed").style.opacity = show ? p : 1 - p; });
}

const term = { scroll: 0, caret: null, path: "~/demo", typing: false };
function termClear(path) {
  fire(() => { $("term-lines").replaceChildren(); term.scroll = 0; term.path = path; });
}
function command(cmd, { paste = false, cps = 17 } = {}) {
  let text;
  fire(() => {
    const line = document.createElement("div");
    line.className = "line cmd";
    const path = Object.assign(document.createElement("span"), { className: "path", textContent: term.path });
    const prompt = Object.assign(document.createElement("span"), { className: "prompt", textContent: " $ " });
    text = document.createElement("span");
    term.caret = Object.assign(document.createElement("span"), { className: "caret" });
    line.append(path, prompt, text, term.caret);
    $("term-lines").append(line);
  });
  wait(0.5);
  if (paste) {
    fire(() => { text.textContent = cmd; });
    wait(0.45);
  } else {
    const times = [];
    let total = 0;
    for (const ch of cmd) { total += (ch === " " ? 1.6 : 1) * (0.6 + 0.8 * rand()) / cps; times.push(total); }
    seq(total, () => { term.typing = true; }, (p) => {
      const n = times.filter((x) => x <= p * total + 1e-9).length;
      text.textContent = cmd.slice(0, n);
    });
    fire(() => { term.typing = false; });
    wait(0.35);
  }
  fire(() => { term.caret.remove(); term.caret = null; });
}
// Lines appear quickly; the real wait for the command is cut.
function output(text, { perLine = 0.03, max = 1.0, after = 0.25 } = {}) {
  const lines = text.split("\n"), block = { nodes: [] };
  wait(after);
  seq(Math.min(lines.length * perLine, max), null, (p) => {
    while (block.nodes.length < Math.ceil(p * lines.length)) {
      const text = lines[block.nodes.length];
      const d = Object.assign(document.createElement("div"), { className: "line", textContent: text });
      // A wrapped line continues under its own indent, as the reader expects from aligned columns.
      const indent = text.length - text.trimStart().length;
      Object.assign(d.style, { paddingLeft: `calc(8px + ${indent}ch)`, textIndent: `-${indent}ch` });
      $("term-lines").append(d);
      block.nodes.push(d);
    }
  });
  return block;
}
function textRect(node, chars) {
  const range = document.createRange();
  const textNode = node.firstChild;
  range.setStart(textNode, 0);
  range.setEnd(textNode, Math.min(chars ?? textNode.length, textNode.length));
  return toContentRect(range.getBoundingClientRect());
}
function linesMatching(block, re) { return block.nodes.filter((n) => re.test(n.textContent)); }
function highlight(block, re) { fire(() => linesMatching(block, re).forEach((n) => n.classList.add("hl"))); }
function unhighlight() { fire(() => document.querySelectorAll(".line.hl").forEach((n) => n.classList.remove("hl"))); }
function zoomLines(block, re, chars, max) {
  zoomTo(() => pad(union(linesMatching(block, re).map((n) => textRect(n, chars))), 14), 0.9, max);
}
function updateTerm() {
  const lines = $("term-lines");
  const target = Math.max(0, lines.offsetHeight - H + 8);
  const next = Math.abs(target - term.scroll) < 0.5 ? target : term.scroll + (target - term.scroll) * 0.3;
  const changed = next !== term.scroll;
  term.scroll = next;
  lines.style.transform = `translateY(${-term.scroll}px)`;
  return changed;
}

const ui = $("ui");
const udoc = () => ui.contentDocument;
const uiRect = (el) => toContentRect(el.getBoundingClientRect(), ui);
const cur = { x: W * 0.62, y: H * 0.7, press: 1, shown: 0 };
function showCursor(show, dur = 0.3) { par(dur, null, (p) => { cur.shown = show ? p : 1 - p; }); }
function moveTo(getPoint, dur = 1.0) {
  let from, to, ctrl;
  seq(dur, () => {
    from = { x: cur.x, y: cur.y };
    to = getPoint();
    const dx = to.x - from.x, dy = to.y - from.y;
    ctrl = { x: (from.x + to.x) / 2 - dy * 0.22, y: (from.y + to.y) / 2 + dx * 0.22 };
  }, (p) => {
    const e = ease(p), u = 1 - e;
    cur.x = u * u * from.x + 2 * u * e * ctrl.x + e * e * to.x;
    cur.y = u * u * from.y + 2 * u * e * ctrl.y + e * e * to.y;
  });
}
function pointIn(getEl, fx = 0.3, fy = 0.5) {
  return () => { const r = uiRect(getEl()); return { x: r.x + r.w * fx, y: r.y + r.h * fy }; };
}
function click(getEl) {
  wait(0.35);
  seq(0.12, null, (p) => { cur.press = lerp(1, 0.82, p); });
  fire(() => { getEl().click(); });
  par(0.6, null, (p) => {
    const s = contentToScreen(cur), size = lerp(10, 90, ease(p));
    Object.assign($("ripple").style, { opacity: 1 - p, width: size + "px", height: size + "px", transform: `translate(${s.x - size / 2}px, ${s.y - size / 2}px)` });
  });
  seq(0.15, null, (p) => { cur.press = lerp(0.82, 1, p); });
}
function scrollTo(getY, dur = 1.1) {
  let from, to;
  seq(dur, () => {
    const win = ui.contentWindow;
    from = win.scrollY;
    to = clamp(getY(), 0, udoc().documentElement.scrollHeight - win.innerHeight);
  }, (p) => { ui.contentWindow.scrollTo(0, lerp(from, to, ease(p))); });
}
const q = (sel) => udoc().querySelector(sel);
function h3(text) { return [...udoc().querySelectorAll("#detail h3")].find((h) => h.textContent.startsWith(text)); }

let frame = 0, inflight = 0, caretOn = true;
window.step = async () => {
  const t = frame / FPS;
  frame += 1;
  let dirty = false, fired = false;
  for (const tw of timeline) {
    if (tw.done || t < tw.t0) continue;
    if (!tw.begun) { tw.begun = true; if (tw.start) tw.start(); fired = true; }
    const p = tw.dur > 0 ? clamp((t - tw.t0) / tw.dur, 0, 1) : 1;
    if (tw.update) tw.update(p);
    dirty = true;
    if (p >= 1) tw.done = true;
  }
  if (fired) await settle();
  dirty = updateTerm() || dirty;
  const blink = term.typing || Math.floor(t / 0.55) % 2 === 0;
  if (term.caret && blink !== caretOn) dirty = true;
  caretOn = blink;
  if (term.caret) term.caret.style.opacity = blink ? 1 : 0;
  $("zoom").style.transform = `translate(${W / 2 - view.fx * view.s}px, ${H / 2 - view.fy * view.s}px) scale(${view.s})`;
  const s = contentToScreen(cur);
  Object.assign($("cursor").style, { opacity: cur.shown, transform: `translate(${s.x - 3}px, ${s.y - 2}px) scale(${cur.press})` });
  return dirty || fired;
};
async function settle() {
  do { await new Promise((r) => setTimeout(r, 60)); } while (inflight > 0);
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
}

async function loadDashboard() {
  ui.style.width = W / DASH_SCALE + "px";
  ui.style.height = H / DASH_SCALE + "px";
  ui.style.transform = `scale(${DASH_SCALE})`;
  await new Promise((r) => { ui.onload = r; ui.src = "/"; });
  for (let i = 0; i < 100 && !(q("#failures button") && q("#experiments table")); i++) await new Promise((r) => setTimeout(r, 100));
  const style = udoc().createElement("style");
  style.textContent = "html { scrollbar-width: none; } ::-webkit-scrollbar { display: none; }";
  udoc().head.append(style);
  const win = ui.contentWindow, original = win.fetch.bind(win);
  win.fetch = (...args) => { inflight += 1; return original(...args).finally(() => { inflight -= 1; }); };
}

window.init = async () => {
  const [demo, real] = await Promise.all(["captured/demo.json", "captured/realbot.json"].map((p) => fetch(p).then((r) => r.json())));
  await loadDashboard();
  $("window").style.opacity = 0;
  build(demo.steps, real.steps);
  return { frames: Math.ceil((T + 0.2) * FPS), marks };
};

function build(demo, real) {
  const out = (step, opts) => output(step.out, opts);

  // Title card.
  setCard("contexteval", "Find out why your RAG bot is wrong, and the setting that fixes it.");
  wait(2.8);

  // The fake-provider demo in a terminal.
  showLayer("term", "Terminal: ~/demo", 0);
  fadeWindow(true); wait(0.6);
  caption("Install from PyPI");
  command(demo[0].cmd, { paste: true }); out(demo[0]); wait(1.0);
  caption("Create a project and import the planted test cases");
  command(demo[1].cmd); out(demo[1]);
  command(demo[2].cmd, { cps: 24 }); out(demo[2]);
  command(demo[3].cmd, { cps: 24 }); out(demo[3]); wait(0.6);

  marks.gifStart = T;
  caption("Run every case through the demo bot");
  command(demo[4].cmd, { cps: 24 });
  const run = out(demo[4]);
  highlight(run, /cases, \d+ passed/); zoomLines(run, /cases, \d+ passed/, undefined, 1.9); wait(1.8); zoomOut(); unhighlight();

  caption("Diagnose: every failure gets a label and its evidence");
  command(demo[5].cmd, { cps: 24 });
  const labels = /^ {2}(missing|irrelevant|ignored|not_context|flaky) /;
  const diag = out(demo[5], { max: 0.8 });
  highlight(diag, labels); zoomLines(diag, labels, undefined, 1.8); wait(2.4); zoomOut(); unhighlight();

  caption("Sweep: try the settings the labels call for");
  command(demo[6].cmd, { cps: 24 });
  const sweep = out(demo[6], { max: 1.2 });
  wait(0.8);
  highlight(sweep, /^Set /); zoomLines(sweep, /^Set /, 50, 2.0); wait(2.4); zoomOut(); unhighlight();

  marks.gifEnd = T;
  // The dashboard over the same project.
  caption("contexteval ui: the same results in the dashboard");
  showLayer("dash", "contexteval ui  ·  127.0.0.1:8765", 0.7); wait(0.9);
  const ignored = () => [...udoc().querySelectorAll("#failures button")].find((b) => b.querySelector(".b-ignored"));
  showCursor(true);
  caption("Open the failure labeled ignored");
  moveTo(pointIn(ignored, 0.35, 0.3), 1.1);
  click(ignored); wait(0.5);
  caption("The answer missed the expected fact");
  zoomTo(() => pad(union([uiRect(h3("Answer vs expected")), uiRect(q("#detail .pair"))]), 12), 1.0, 2.0); wait(2.0);
  caption("Moving the gold chunk to the top fixed it");
  zoomTo(() => pad(union([uiRect(h3("Diagnosis")), uiRect(h3("Diagnosis").nextElementSibling)]), 12), 0.9, 2.0); wait(2.0);
  zoomOut(0.7);
  caption("The gold chunk sat third in the prompt");
  scrollTo(() => udoc().querySelector("#detail li.gold").getBoundingClientRect().top + ui.contentWindow.scrollY - 220, 1.1);
  zoomTo(() => pad(uiRect(q("#detail li.gold")), 14), 0.9, 1.9); wait(2.0); zoomOut(0.7);
  caption("The sweep's runs, with the recommended row in bold");
  scrollTo(() => udoc().getElementById("exp-title").getBoundingClientRect().top + ui.contentWindow.scrollY - 40, 1.3);
  moveTo(pointIn(() => q("#experiments tr.best td"), 0.85, 0.5), 0.9);
  zoomTo(() => pad(union([uiRect(q("#experiments tr.best")), uiRect(q("#experiments .recommendation"))]), 14), 1.0, 1.8); wait(2.4);
  zoomOut(0.7); showCursor(false);

  // The real bot.
  caption("On a real support bot (gpt-4.1-mini)");
  termClear("~/ridgeline");
  showLayer("term", "Terminal: ~/ridgeline", 0.7); wait(0.8);
  for (const step of real.slice(0, -1)) {
    command(step.cmd, { cps: 32, paste: step === real[0] });
    out(step, { max: 0.6 });
    wait(0.25);
  }
  const final = real[real.length - 1];
  command(final.cmd, { cps: 30 });
  const minutes = Math.round(final.seconds / 60);
  speedBadge(`▶▶ sped up:${minutes} min of sweep in 3 s`, true);
  const realSweep = out(final, { perLine: 0.08, max: 3.0 });
  speedBadge("", false);
  wait(0.6);
  caption("The sweep's recommendation, checked on holdout cases");
  highlight(realSweep, /^Set /); zoomLines(realSweep, /^Set /, 57, 2.0); wait(3.0); zoomOut(); unhighlight();

  // End card.
  caption("");
  setCard("contexteval", "", '<div class="pill">pip install contexteval</div><div class="url">github.com/aakarshan-coding/context-evaluation-sdk</div>');
  fadeWindow(false, 0.9); wait(0.9);
  wait(3.6);
}
