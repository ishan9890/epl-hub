import { kitColours } from "../teams.js";

const KEYS = ["home", "draw", "away"];
const CLOSE_CALL = 0.05;   // top two outcomes within 5 points: too close to call
let firstRender = true;    // the entrance animation plays once per page load

const pct = (x) => Math.round(x * 100);
const clarity = (m) => Math.max(...Object.values(m.probs));
const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function utcDate(iso, options) {
  return new Date(iso).toLocaleDateString("en-GB", { ...options, timeZone: "UTC" });
}

function dayLong(iso) {
  return utcDate(iso, { weekday: "long", day: "numeric", month: "long" });
}

function dayShort(iso) {
  return utcDate(iso, { weekday: "short", day: "numeric", month: "short" });
}

function rangeLabel(first, last) {
  if (first === last) return dayLong(first);
  const sameMonth = first.slice(0, 7) === last.slice(0, 7);
  const left = utcDate(first, sameMonth
    ? { weekday: "long", day: "numeric" }
    : { weekday: "long", day: "numeric", month: "long" });
  return `${left} to ${dayLong(last)}`;
}

function verdict(m) {
  const ranked = KEYS.map((k) => [k, m.probs[k]]).sort((a, b) => b[1] - a[1]);
  if (ranked[0][1] - ranked[1][1] < CLOSE_CALL) {
    return { key: null, text: "Too close to call" };
  }
  const key = ranked[0][0];
  const text = key === "draw" ? "Leans towards a draw" : `Leans ${key === "home" ? m.home : m.away}`;
  return { key, text };
}

function figure(value) {
  const box = el("div", "fig");
  box.append(el("span", "num", String(pct(value))), el("span", "pc", "%"));
  return box;
}

function fixture(m, index) {
  const kit = kitColours(m.home, m.away);
  const v = verdict(m);

  const row = el("article", "fixture");
  row.style.setProperty("--row", index);
  row.style.setProperty("--home", kit.home);
  row.style.setProperty("--away", kit.away);
  row.style.setProperty("--hp", m.probs.home * 100);
  row.style.setProperty("--dp", m.probs.draw * 100);

  const home = el("div", "side home");
  home.append(el("div", "team", m.home), figure(m.probs.home));

  const mid = el("div", "mid");
  mid.append(figure(m.probs.draw), el("small", "", "draw"));

  const away = el("div", "side away");
  away.append(el("div", "team", m.away), figure(m.probs.away));

  const sides = el("div", "sides");
  sides.append(home, mid, away);

  const tape = el("div", "tape");
  tape.setAttribute("role", "img");
  tape.setAttribute(
    "aria-label",
    `${m.home} ${pct(m.probs.home)} percent, draw ${pct(m.probs.draw)} percent, ` +
    `${m.away} ${pct(m.probs.away)} percent. ${v.text}.`
  );
  for (const key of KEYS) {
    const seg = el("i", key === v.key ? "lean" : "");
    seg.dataset.key = key;
    seg.style.setProperty("--p", m.probs[key] * 100);
    tape.append(seg);
  }

  const meta = el("div", "meta");
  meta.append(el("span", "", v.text), el("span", "when", dayShort(m.date)));

  row.append(sides, tape, meta);
  return row;
}

function groupByDay(entries) {
  const groups = new Map();
  for (const entry of entries) {
    if (!groups.has(entry.m.date)) groups.set(entry.m.date, []);
    groups.get(entry.m.date).push(entry);
  }
  return [...groups.entries()].sort((a, b) => a[0].localeCompare(b[0]));
}

// Moves the existing rows into the chosen layout (it never rebuilds them).
function drawList(list, entries, mode) {
  list.replaceChildren();

  if (mode === "day") {
    for (const [date, group] of groupByDay(entries)) {
      const section = el("section", "day");
      const head = el("div", "day-head");
      head.append(
        el("h2", "", dayLong(date)),
        el("span", "", group.length === 1 ? "1 match" : `${group.length} matches`),
      );
      section.append(head, ...group.map((entry) => entry.row));
      list.append(section);
    }
    return;
  }

  const flat = el("section", "fixtures-flat");
  const ordered = [...entries].sort((a, b) => clarity(b.m) - clarity(a.m));
  flat.append(...ordered.map((entry) => entry.row));
  list.append(flat);
}

// Gives every row a temporary name so the browser can glide each one to its new place.
function withSortTransition(rows, change) {
  if (!document.startViewTransition || prefersReducedMotion()) {
    change();
    return;
  }
  rows.forEach((row, i) => { row.style.viewTransitionName = `fx-${i}`; });
  const clear = () => rows.forEach((row) => { row.style.viewTransitionName = ""; });
  document.startViewTransition(change).finished.then(clear, clear);
}

function trustNote(data) {
  const model = data.model;
  const stamp = new Date(data.generated_at).toLocaleString("en-GB", {
    day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
  });
  let text = `Updated ${stamp}.`;
  if (model?.test_matches) {
    text += ` On the latest ${model.test_matches} matches the model picked ${pct(model.accuracy)}% ` +
      `of results correctly, against ${pct(model.baseline_accuracy)}% for always choosing the most ` +
      `common result. Probability score (log loss, lower is better): ${model.log_loss} against ` +
      `${model.baseline_log_loss}.`;
  }
  return el("p", "trust", text);
}

export function render(container, data) {
  const matches = data.matches ?? [];
  const dates = matches.map((m) => m.date).sort();

  const head = el("div", "view-head");
  const titles = el("div");
  titles.append(
    el("h1", "view-title", "Next matchweek"),
    el("p", "view-sub", dates.length ? rangeLabel(dates[0], dates[dates.length - 1]) : "Waiting for fixtures"),
  );
  head.append(titles);
  container.append(head);

  if (matches.length === 0) {
    container.append(
      el("div", "notice", data.message || "No fixtures yet. Predictions appear here once the next round is published."),
      trustNote(data),
    );
    return;
  }

  const entries = matches.map((m, i) => ({ m, row: fixture(m, i) }));
  const rows = entries.map((entry) => entry.row);

  const list = el("div", "fixture-list");
  if (firstRender) list.classList.add("enter");

  let mode = "day";
  const options = [["day", "By day"], ["clarity", "Clearest first"]];

  const buttons = options.map(([value, label]) => {
    const button = el("button", "", label);
    button.type = "button";
    button.addEventListener("click", () => {
      if (mode === value) return;
      mode = value;
      syncPressed();
      list.classList.remove("enter");           // moving a row would replay its sweep
      withSortTransition(rows, () => drawList(list, entries, mode));
    });
    return button;
  });

  const syncPressed = () => {
    buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(options[i][0] === mode)));
  };
  syncPressed();

  const control = el("div", "seg-control");
  control.setAttribute("role", "group");
  control.setAttribute("aria-label", "Order of matches");
  control.append(...buttons);
  head.append(control);

  drawList(list, entries, mode);
  container.append(list, trustNote(data));
  firstRender = false;
}