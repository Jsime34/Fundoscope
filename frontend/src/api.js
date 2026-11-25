// Calls to the FastAPI backend (proxied by Vite in development).

export async function fetchHistory(file) {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/api/history", { method: "POST", body: form });
  if (!response.ok) {
    let detail = `The server answered ${response.status}.`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // not JSON: keep the generic message
    }
    throw new Error(detail);
  }
  return response.json();
}

// The latest crop count of a valley (FR-15). Error messages are in Spanish: the page is.
export async function fetchValley(slug) {
  const response = await fetch(`/api/valleys/${encodeURIComponent(slug)}`);
  if (response.status === 404) throw new Error("este valle todavía no tiene un conteo.");
  if (!response.ok) throw new Error(`el servidor respondió ${response.status}.`);
  return response.json();
}
