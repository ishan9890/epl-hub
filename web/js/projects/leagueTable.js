import { teamInfo } from "../teams.js";

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const INK = "#12281D";
const WORDS = { W: "win", D: "draw", L: "loss" };

// Columns after the team name. "opt" columns are hidden on narrow screens.
const COLUMNS = [
  { key: "played", label: "P" },
  { key: "won", label: "W" },
  { key: "drawn", label: "D" },
  { key: "lost", label: "L" },
  { key: "gf", label: "GF", opt: true },
  { key: "ga", label: "GA", opt: true },
  { key: "gd", label: "GD" },
  { key: "points", label: "Pts", className: "pts" },
];

// How bright a colour looks (0 = black, 1 = white), using the standard WCAG formula.
function luminance(hex) {
  const n = parseInt(hex.slice(1), 16);
  const channel = (v) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel((n >> 16) & 255) + 0.7152 * channel((n >> 8) & 255) + 0.0722 * channel(n & 255);
}

// Choose white or dark text, whichever is easier to read on this background.
function textOn(hex) {
  const L = luminance(hex);
  const againstWhite = 1.05 / (L + 0.05);
  const againstInk = (L + 0.05) / (luminance(INK) + 0.05);
  return againstWhite >= againstInk ? "#FFFFFF" : INK;
}

function formCell(form, team) {
  const td = el("td", "form-cell");
  if (!Array.isArray(form) || form.length === 0) {
    td.textContent = "–";
    return td;
  }
  const wrap = el("span", "form");
  wrap.setAttribute("role", "img");
  wrap.setAttribute(
    "aria-label",
    `${team}, last ${form.length} results, oldest first: ${form.map((r) => WORDS[r] ?? r).join(", ")}`,
  );
  for (const r of form) {
    const mark = el("span", "mark", r);
    mark.dataset.r = r;
    wrap.append(mark);
  }
  td.append(wrap);
  return td;
}

export function render(container, data) {
  const head = el("div", "view-head");
  const titles = el("div");
  titles.append(
    el("h1", "view-title", "League table"),
    el("p", "view-sub", `Updated ${new Date(data.generated_at).toLocaleString()}`),
  );
  head.append(titles);

  const headRow = el("tr");
  headRow.append(el("th", "", "#"), el("th", "", "Team"));
  for (const col of COLUMNS) {
    headRow.append(el("th", col.opt ? "opt" : col.className ?? "", col.label));
  }
  headRow.append(el("th", "form-cell", "Form"));
  const thead = el("thead");
  thead.append(headRow);

  const tbody = el("tbody");
  for (const row of data.rows) {
    const info = teamInfo(row.team);
    const tr = el("tr");

    tr.append(el("td", "pos", String(row.pos)));

    const club = el("td", "club");
    const code = el("span", "code", info.code);
    code.style.background = info.color;
    code.style.color = textOn(info.color);
    club.append(code, el("span", "club-name", row.team));
    tr.append(club);

    for (const col of COLUMNS) {
      const value = col.key === "gd" && row.gd > 0 ? `+${row.gd}` : String(row[col.key]);
      tr.append(el("td", col.opt ? "opt" : col.className ?? "", value));
    }
    tr.append(formCell(row.form, row.team));
    tbody.append(tr);
  }

  const table = el("table", "league");
  table.append(thead, tbody);
  const wrap = el("div", "table-wrap");
  wrap.append(table);

  container.append(
    head,
    wrap,
    el("p", "trust", "Form shows each team's last five league results, oldest on the left. W is a win, D a draw and L a loss."),
  );
}