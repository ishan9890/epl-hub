import { loadJSON } from "./api.js";

const nav = document.querySelector("#nav");
const app = document.querySelector("#app");
const FADE_MS = 160;

const indicator = document.createElement("span");
indicator.className = "nav-indicator";

let latest = 0;          // identifies the newest request, so a slow older one can't overwrite it
let shown = false;       // has a real view been drawn yet?
let currentId = null;

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function placeIndicator() {
  const active = nav.querySelector("a.active");
  if (!active) return;
  indicator.style.setProperty("--x", `${active.offsetLeft}px`);
  indicator.style.setProperty("--w", `${active.offsetWidth}px`);
}

async function showProject(project) {
  const ticket = ++latest;
  const wasShown = shown;

  if (wasShown) app.classList.add("leaving");
  else app.textContent = "Loading…";

  try {
    const [module, data] = await Promise.all([
      import(project.module),
      loadJSON(project.data),
      wasShown ? wait(FADE_MS) : Promise.resolve(),
    ]);
    if (ticket !== latest) return;

    if (typeof module.render !== "function") {
      throw new Error(`${project.module} must export a render(container, data) function`);
    }

    app.replaceChildren();
    module.render(app, data);
    shown = true;
    if (wasShown) window.scrollTo(0, 0);
  } catch (err) {
    if (ticket !== latest) return;
    console.error(err);
    app.textContent = `Something went wrong: ${err.message}`;
  } finally {
    if (ticket === latest) app.classList.remove("leaving");
  }
}

function route(projects) {
  const id = location.hash.slice(1);
  const project = projects.find((p) => p.id === id) ?? projects[0];

  for (const link of nav.querySelectorAll("a")) {
    link.classList.toggle("active", link.dataset.id === project.id);
  }
  placeIndicator();

  if (project.id === currentId && shown) return;
  currentId = project.id;
  showProject(project);
}

async function start() {
  try {
    const projects = await loadJSON("data/projects.json");
    if (projects.length === 0) {
      app.textContent = "No projects yet.";
      return;
    }

    for (const p of projects) {
      const link = document.createElement("a");
      link.href = `#${p.id}`;
      link.textContent = p.title;
      link.dataset.id = p.id;
      nav.append(link);
    }
    nav.append(indicator);

    window.addEventListener("hashchange", () => route(projects));
    window.addEventListener("resize", placeIndicator);
    route(projects);

    // the font changes link widths once it loads, so measure again; then allow sliding
    document.fonts?.ready.then(placeIndicator);
    requestAnimationFrame(() => requestAnimationFrame(() => indicator.classList.add("ready")));
  } catch (err) {
    console.error(err);
    app.textContent = `Could not start the site: ${err.message}`;
  }
}

start();