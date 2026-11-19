const KEYS = ["home", "draw", "away"];
const pct = (x) => `${Math.round(x * 100)}%`;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatDate(iso) {
  return new Date(iso).toLocaleDateString("en-GB", {
    weekday: "short", day: "numeric", month: "short", timeZone: "UTC",
  });
}

function pickLabel(m) {
  if (m.pick === "home") return `${m.home} win`;
  if (m.pick === "away") return `${m.away} win`;
  return "Draw";
}

const confidence = (m) => Math.max(...Object.values(m.probs));

const sorters = {
  date: (a, b) => a.date.localeCompare(b.date),
  confidence: (a, b) => confidence(b) - confidence(a),
};

function matchCard(m) {
  const card = el("article", "card");

  const bar = el("div", "bar");
  for (const key of KEYS) {
    const seg = el("div", `seg ${key}`, pct(m.probs[key]));
    seg.style.width = `${m.probs[key] * 100}%`;
    bar.append(seg);
  }

  card.append(
    el("h3", "", `${m.home} vs ${m.away}`),
    el("p", "muted", formatDate(m.date)),
    bar,
    el("p", "pick", `Model leans: ${pickLabel(m)}`),
  );
  return card;
}

function legend() {
  const box = el("div", "legend");
  box.append(
    el("span", "home", "Home win"),
    el("span", "draw", "Draw"),
    el("span", "away", "Away win"),
  );
  return box;
}

function modelNote(m) {
  if (!m) return document.createTextNode("");
  return el(
    "p", "muted",
    `Back-test on ${m.test_matches} matches (${m.test_period}): ` +
    `${pct(m.accuracy)} of results picked correctly, versus ${pct(m.baseline_accuracy)} ` +
    `for always picking the most common result. ` +
    `Log loss ${m.log_loss} (baseline ${m.baseline_log_loss}; lower is better).`
  );
}

export function render(container, data) {
  container.append(
    el("h2", "", "Next matchweek"),
    el("p", "muted", `Updated ${new Date(data.generated_at).toLocaleString()}`),
  );

  const matches = data.matches ?? [];
  if (matches.length === 0) {
    container.append(
      el("div", "notice", data.message || "No predictions yet."),
      modelNote(data.model),
    );
    return;
  }

  const select = el("select");
  select.setAttribute("aria-label", "Sort matches");
  select.append(
    new Option("Sort by kickoff date", "date"),
    new Option("Sort by model confidence", "confidence"),
  );
  const controls = el("div", "controls");
  controls.append(select);

  const grid = el("section", "grid");
  const draw = () => {
    const sorted = [...matches].sort(sorters[select.value]);
    grid.replaceChildren(...sorted.map(matchCard));
  };
  select.addEventListener("change", draw);
  draw();

  container.append(controls, legend(), grid, modelNote(data.model));
}