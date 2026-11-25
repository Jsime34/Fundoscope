import "./styles.css";

import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";

// Two pages, each in its own bundle: the parcel report (/) and the valley crop monitor
// (/valle/<slug>), so a phone opening the valley doesn't download the report's charts.
const App = lazy(() => import("./App"));
const ValleyPage = lazy(() => import("./valley/ValleyPage"));

const valley = window.location.pathname.match(/^\/valle\/([a-z0-9-]+)\/?$/);

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <Suspense fallback={null}>{valley ? <ValleyPage slug={valley[1]} /> : <App />}</Suspense>
  </StrictMode>,
);
