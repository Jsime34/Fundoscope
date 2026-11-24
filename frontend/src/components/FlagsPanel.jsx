import { Fragment } from "react";

import { percent } from "../format";

// Readable names for the values behind each flag (see backend analysis/flags.py).
const NAMES = {
  active_area: "Active area",
  median_before: "Usual active area (median of earlier campaigns)",
  threshold: "Threshold (half of the usual)",
  clear_dates: "Clear satellite dates",
  minimum: "Minimum for reliable values",
  previous_campaign: "Previous campaign",
  previous_crop: "Previous crop",
  previous_confidence: "Confidence then",
  crop: "Crop",
  confidence: "Confidence now",
  integrated_ndvi: "Integrated NDVI",
  median_others: "Usual (median of the other campaigns)",
  normal_low: "Normal range from",
  normal_high: "Normal range to",
  robust_z: "Robust z-score",
};
const SHARES = new Set(["active_area", "median_before", "threshold", "previous_confidence", "confidence"]);

function value(key, raw) {
  if (SHARES.has(key)) return percent(raw);
  if (typeof raw === "number" && !Number.isInteger(raw)) return raw.toFixed(1);
  return String(raw);
}

// Flags on the parcel's own history (abandonment, crop change, unusual campaign, low data),
// each with a "Why?" that shows the values behind it.
export default function FlagsPanel({ flags }) {
  if (!flags.length) {
    return <p className="hint">No flags: nothing unusual in this parcel's own history.</p>;
  }
  return (
    <ul className="flags">
      {flags.map((flag) => (
        <li key={`${flag.campaign}-${flag.kind}`} className={flag.kind === "low data" ? "flag minor" : "flag"}>
          <span className="flag-campaign">{flag.campaign}</span>
          <div>
            <p>{flag.message}</p>
            <details>
              <summary>Why?</summary>
              <dl className="facts">
                {Object.entries(flag.evidence).map(([key, raw]) => (
                  <Fragment key={key}>
                    <dt>{NAMES[key] ?? key}</dt>
                    <dd>{value(key, raw)}</dd>
                  </Fragment>
                ))}
              </dl>
            </details>
          </div>
        </li>
      ))}
    </ul>
  );
}
