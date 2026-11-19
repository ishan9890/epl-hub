import { loadJSON } from "./api.js";

const nav = document.querySelector("#nav");
const app = document.querySelector("#app");

async function showProject(project) {
  app.textContent = "Loading…";
  try {
    const [module, data] = await Promise.all([
      import(project.module),
      loadJSON(project.data),
    ]);

    if (typeof module.render !== "function") {
      throw new Error(`${project.module} must export a render(container, data) function`);
    }

    app.replaceChildren();
    module.render(app, data);
  } catch (err) {
    console.error(err);
    app.textContent = `Something went wrong: ${err.message}`;
  }
}

function route(projects) {
  const id = location.hash.slice(1);
  const project = projects.find((p) => p.id === id) ?? projects[0];

  for (const link of nav.querySelectorAll("a")) {
    link.classList.toggle("active", link.dataset.id === project.id);
  }
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

    window.addEventListener("hashchange", () => route(projects));
    route(projects);
  } catch (err) {
    console.error(err);
    app.textContent = `Could not start the site: ${err.message}`;
  }
}

start();