function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const COLUMNS = [
  ["pos", "#"], ["team", "Team"], ["played", "P"], ["won", "W"], ["drawn", "D"],
  ["lost", "L"], ["gf", "GF"], ["ga", "GA"], ["gd", "GD"], ["points", "Pts"],
];

export function render(container, data) {
  container.append(
    el("h2", "", "League table"),
    el("p", "muted", `Updated ${new Date(data.generated_at).toLocaleString()}`),
  );

  const thead = el("thead");
  const headRow = el("tr");
  for (const [, label] of COLUMNS) headRow.append(el("th", "", label));
  thead.append(headRow);

  const tbody = el("tbody");
  for (const row of data.rows) {
    const tr = el("tr");
    for (const [key] of COLUMNS) tr.append(el("td", "", String(row[key])));
    tbody.append(tr);
  }

  const table = el("table");
  table.append(thead, tbody);

  const wrap = el("div", "table-wrap");
  wrap.append(table);
  container.append(wrap);
}