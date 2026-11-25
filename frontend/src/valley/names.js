// Spanish labels and number/date formats for the valley dashboard (FR-15).

export const CLASS_NAMES = {
  alfalfa: "Alfalfa",
  avocado: "Palta",
  grape: "Uva",
  "other annual crops": "Otros cultivos anuales",
  "fallow/bare": "Sin cultivo (descanso)",
  "planted, not yet identifiable": "Sembrado, aún no identificable",
  "no clear images": "Sin imágenes claras",
};

export const PLANTED = "planted, not yet identifiable";
export const NO_CLEAR_IMAGES = "no clear images";

export function className(name) {
  return CLASS_NAMES[name] ?? name;
}

const number = new Intl.NumberFormat("es-PE", { maximumFractionDigits: 0 });
const signed = new Intl.NumberFormat("es-PE", { maximumFractionDigits: 0, signDisplay: "always" });
const signedPercent = new Intl.NumberFormat("es-PE", {
  style: "percent",
  maximumFractionDigits: 0,
  signDisplay: "always",
});

export function hectares(value) {
  return `${number.format(value)} ha`;
}

export function change(value) {
  return `${signed.format(value)} ha`;
}

export function percentChange(value) {
  return signedPercent.format(value);
}

// "2026-11-22" (a date) or "2026-11-24T17:20:17" (a local time) -> "22 nov. 2026"
export function date(iso) {
  if (!iso) return "—";
  const when = iso.length === 10 ? new Date(`${iso}T12:00:00`) : new Date(iso);
  return when.toLocaleDateString("es-PE", { day: "numeric", month: "short", year: "numeric" });
}

// "2026-11-22" -> "22 nov."
export function dayMonth(iso) {
  return new Date(`${iso}T12:00:00`).toLocaleDateString("es-PE", {
    day: "numeric",
    month: "short",
  });
}
