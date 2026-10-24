// Number and date formatting shared by the components. Missing values (null) show as "—".

export const dash = "—";

export function fixed(value, digits = 2) {
  return value == null ? dash : value.toFixed(digits);
}

export function percent(value) {
  return value == null ? dash : `${Math.round(value * 100)}%`;
}

export function shortDate(iso) {
  return iso ? new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-US", {
    year: "numeric", month: "short", day: "numeric", timeZone: "UTC",
  }) : dash;
}

// "2023/24" -> [Aug 1 2023, Jul 31 2024] as UTC timestamps (ms).
export function campaignRange(label) {
  const start = Number(label.slice(0, 4));
  return [Date.UTC(start, 7, 1), Date.UTC(start + 1, 6, 31)];
}
