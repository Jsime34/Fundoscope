// Calls to the FastAPI backend (proxied by Vite in development).

const POLL_MS = 2000; // how often a running analysis is checked

async function request(url, options) {
  const response = await fetch(url, options);
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

// Store a parcel file (FR-02); nothing is downloaded yet. Returns the parcel with its id.
export function uploadParcel(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/api/parcels", { method: "POST", body: form });
}

// Queue an analysis of a stored parcel (FR-13). Returns the job at once.
export function startAnalysis(parcelId) {
  return request(`/api/parcels/${encodeURIComponent(parcelId)}/analyses`, { method: "POST" });
}

// Check the job every 2 s until it ends; `onUpdate` gets each state. Returns the done job.
export async function waitForJob(job, onUpdate) {
  let current = job;
  for (;;) {
    onUpdate(current);
    if (current.status === "done") return current;
    if (current.status === "failed") throw new Error(current.error ?? "The analysis failed.");
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
    current = await request(`/api/jobs/${encodeURIComponent(current.id)}`);
  }
}

// The latest crop count of a valley (FR-15). Error messages are in Spanish: the page is.
export async function fetchValley(slug) {
  const response = await fetch(`/api/valleys/${encodeURIComponent(slug)}`);
  if (response.status === 404) throw new Error("este valle todavía no tiene un conteo.");
  if (!response.ok) throw new Error(`el servidor respondió ${response.status}.`);
  return response.json();
}

// Parcels stored in the database, newest first (FR-02).
export function fetchStoredParcels() {
  return request("/api/parcels");
}

// A parcel's report, rebuilt from the database with no satellite request.
export function fetchStoredHistory(id) {
  return request(`/api/parcels/${encodeURIComponent(id)}/history`);
}
