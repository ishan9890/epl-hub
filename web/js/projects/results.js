function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const pct = (x) => `${Math.round(x * 100)}%`;

function formatDate(iso) {
  return new Date(iso).toLocaleDateString("en-GB", {
    day: "numeric", month: "short", timeZone: "UTC",
  });
}

function formatLongDate(iso) {
  return new Date(iso).toLocaleDateString("en-GB", {
    day: "numeric", month: "long", year: "numeric", timeZone: "UTC",
  });
}

function pickText(m) {
  const label = m.pick === "draw" ? "Draw" : `${m.pick === "home" ? m.home : m.away} win`;
  return `${label} (${pct(m.probs[m.pick])})`;
}

function table(headers, rows) {
  const thead = el("thead");
  const headRow = el("tr");
  for (const h of headers) headRow.append(el("th", "", h));
  thead.append(headRow);

  const tbody = el("tbody");
  for (const cells of rows) {
    const tr = el("tr");
    for (const c of cells) tr.append(el("td", "", c));
    tbody.append(tr);
  }

  const t = el("table", "results-table");
  t.append(thead, tbody);
  const wrap = el("div", "table-wrap");
  wrap.append(t);
  return wrap;
}

function scoreboard(summary) {
  const box = el("div", "notice");
  if (!summary || summary.matches_played === 0) {
    const waiting = summary?.matches_pending ?? 0;
    box.textContent = waiting > 0
      ? `Waiting for the first predicted round to finish (${waiting} matches pending).`
      : "No predicted matches have been played yet.";
    return box;
  }

  const n = summary.matches_played;
  box.append(
    el("p", "", `Scored so far: ${n} matches in ${summary.rounds} round(s)` +
      (summary.matches_pending ? `, plus ${summary.matches_pending} pending.` : ".")),
    el("p", "", `Accuracy: ${pct(summary.accuracy)} (typical error about ±${pct(summary.accuracy_se)}) ` +
      `versus ${pct(summary.baseline_accuracy)} for always picking the most common result.`),
    el("p", "", `Log loss: ${summary.log_loss} versus ${summary.baseline_log_loss} for the baseline ` +
      `(lower is better).`),
  );
  if (n < 100) {
    box.append(el("p", "muted",
      "Small sample: with this few matches the numbers are mostly luck. " +
      "Wait for several more rounds before judging the model."));
  }
  return box;
}

function roundBlock(round, open) {
  const details = el("details", "round");
  details.open = open;

  const head = el("summary");
  head.append(el("strong", "", round.label));

  if (round.has_predictions) {
    head.append(el("span", "chip chip-pred", "Predictions made"));
    head.append(el("span", "chip", round.model_name));
    const stats = round.played > 0
      ? `${round.correct}/${round.played} correct · log loss ${round.log_loss} ` +
        `(baseline ${round.baseline_log_loss})` +
        (round.pending ? ` · ${round.pending} pending` : "")
      : `Waiting for results (${round.pending} matches)`;
    head.append(el("span", "stats", stats));
  } else {
    head.append(el("span", "chip", "No predictions made"));
  }
  details.append(head);

  if (round.has_predictions) {
    details.append(table(
      ["Date", "Match", "Prediction", "Result", "Outcome"],
      round.matches.map((m) => [
        formatDate(m.date),
        `${m.home} v ${m.away}`,
        pickText(m),
        m.result ? m.score : "Pending",
        m.result ? (m.correct ? "✓ Correct" : "✗ Wrong") : "",
      ]),
    ));
  } else {
    details.append(table(
      ["Date", "Match", "Result"],
      round.matches.map((m) => [formatDate(m.date), `${m.home} v ${m.away}`, m.score]),
    ));
  }
  return details;
}

export function render(container, data) {
  container.append(
    el("h2", "", "Results"),
    el("p", "muted", `Updated ${new Date(data.generated_at).toLocaleString()}`),
  );

  if (data.tracking_started) {
    container.append(el("p", "muted",
      `Predictions have been saved before kickoff since the round starting ` +
      `${formatLongDate(data.tracking_started)}. Earlier rounds show results only, ` +
      `because no predictions were made for them.`));
  }

  container.append(scoreboard(data.summary));

  const rounds = data.rounds ?? [];
  if (rounds.length === 0) {
    container.append(el("div", "notice", data.message || "No results yet."));
    return;
  }
  rounds.forEach((round, i) => container.append(roundBlock(round, i < 2)));
}