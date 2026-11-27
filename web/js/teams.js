// Best-effort club colours: a primary kit colour and an alternate used on a clash.
// Keys must match the team names in your data (football-data spelling).
const TEAMS = {
  "Arsenal":       { code: "ARS", color: "#DB0007", alt: "#063672" },
  "Aston Villa":   { code: "AVL", color: "#7A1F3D", alt: "#95BFE5" },
  "Bournemouth":   { code: "BOU", color: "#D71920", alt: "#1E1E1E" },
  "Brentford":     { code: "BRE", color: "#E30613", alt: "#F2B400" },
  "Brighton":      { code: "BHA", color: "#0057B8", alt: "#F2D000" },
  "Chelsea":       { code: "CHE", color: "#034694", alt: "#D9A400" },
  "Coventry":      { code: "COV", color: "#59CBE8", alt: "#1B2A49" },
  "Crystal Palace":{ code: "CRY", color: "#1B458F", alt: "#C4122E" },
  "Everton":       { code: "EVE", color: "#003399", alt: "#E8C100" },
  "Fulham":        { code: "FUL", color: "#222222", alt: "#C8102E" },
  "Hull":          { code: "HUL", color: "#E58E0A", alt: "#1E1E1E" },
  "Ipswich":       { code: "IPS", color: "#2B5AA6", alt: "#F2A900" },
  "Leeds":         { code: "LEE", color: "#1D428A", alt: "#FFCD00" },
  "Liverpool":     { code: "LIV", color: "#C8102E", alt: "#00A398" },
  "Man City":      { code: "MCI", color: "#6CABDD", alt: "#1C2C5B" },
  "Man United":    { code: "MUN", color: "#DA291C", alt: "#14213D" },
  "Newcastle":     { code: "NEW", color: "#241F20", alt: "#41B6E6" },
  "Nott'm Forest": { code: "NFO", color: "#DD0000", alt: "#1F2A44" },
  "Sunderland":    { code: "SUN", color: "#EB172B", alt: "#1E1E1E" },
  "Tottenham":     { code: "TOT", color: "#132257", alt: "#E6D600" },
};

const FALLBACK = { color: "#5B6E61", alt: "#9AA89E" };
const CLASH = 100;   // colours closer than this count as a clash (tweak by eye)
const warned = new Set();

export function teamInfo(name) {
  const known = TEAMS[name];
  if (known) return known;
  if (!warned.has(name)) {
    warned.add(name);
    console.warn(`No colours for "${name}" in teams.js; using a neutral grey.`);
  }
  return { code: name.slice(0, 3).toUpperCase(), ...FALLBACK };
}

function rgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

// "Redmean" colour difference: a cheap formula that tracks how different two colours look.
function distance(hexA, hexB) {
  const a = rgb(hexA);
  const b = rgb(hexB);
  const rmean = (a.r + b.r) / 2;
  const dr = a.r - b.r;
  const dg = a.g - b.g;
  const db = a.b - b.b;
  return Math.sqrt(
    (2 + rmean / 256) * dr * dr + 4 * dg * dg + (2 + (255 - rmean) / 256) * db * db
  );
}

// Home keeps its colour. Away switches to its alternate only if that avoids a clash.
export function kitColours(homeName, awayName) {
  const home = teamInfo(homeName);
  const away = teamInfo(awayName);

  const primary = distance(home.color, away.color);
  if (primary >= CLASH) return { home: home.color, away: away.color, clash: false };

  const alternate = distance(home.color, away.alt);
  return alternate > primary
    ? { home: home.color, away: away.alt, clash: true }
    : { home: home.color, away: away.color, clash: true };
}