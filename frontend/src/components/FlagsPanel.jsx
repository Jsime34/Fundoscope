import { Fragment } from "react";

import { campaignRange, percent, shortDate } from "../format";

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
  parcel_ndvi: "Parcel mean NDVI",
  neighbors_p10: "Neighbors' 10th percentile",
  neighbors_median: "Neighbors' median",
  neighbor_pixels: "Neighbor pixels (10 m)",
  months_compared: "Months compared",
};
const SHARES = new Set(["active_area", "median_before", "threshold", "previous_confidence", "confidence"]);
const NDVI = new Set(["parcel_ndvi", "neighbors_p10", "neighbors_median"]);

function value(key, raw) {
  if (SHARES.has(key)) return percent(raw);
  if (NDVI.has(key)) return raw.toFixed(2);
  if (key === "neighbor_pixels") return raw.toLocaleString("en-US");
  if (typeof raw === "number" && !Number.isInteger(raw)) return raw.toFixed(1);
  return String(raw);
}

// The campaign's clear satellite dates: how many, and the first and last (FR-12: every flag's
// evidence shows its dates and number of clear dates).
function clearDates(label, campaigns, observations) {
  const campaign = campaigns.find((c) => c.campaign === label);
  if (!campaign) return null;
  const [start, end] = campaignRange(label);
  const days = observations
    .map((o) => o.date)
    .filter((day) => {
      const time = Date.parse(`${day}T00:00:00Z`);
      return time >= start && time <= end;
    });
  if (!days.length) return String(campaign.n_obs);
  return `${campaign.n_obs}, from ${shortDate(days[0])} to ${shortDate(days.at(-1))}`;
}

// Flags on the parcel's own history (abandonment, crop change, unusual campaign, low data)
// and against its neighbors (underperforming), each with a "Why?" that shows the values.
export default function FlagsPanel({ flags, campaigns = [], observations = [] }) {
  if (!flags.length) {
    return <p className="hint">No flags: nothing unusual in this parcel's own history.</p>;
  }
  return (
    <ul className="flags">
      {flags.map((flag) => {
        const dates = clearDates(flag.campaign, campaigns, observations) ?? flag.evidence.clear_dates;
        const facts = Object.entries(flag.evidence).filter(([key]) => key !== "clear_dates");
        if (dates != null) facts.push(["clear_dates", dates]); // last, with the first and last date
        return (
          <li key={`${flag.campaign}-${flag.kind}`} className={flag.kind === "low data" ? "flag minor" : "flag"}>
            <span className="flag-campaign">{flag.campaign}</span>
            <div>
              <p>{flag.message}</p>
              <details>
                <summary>Why?</summary>
                <dl className="facts">
                  {facts.map(([key, raw]) => (
                    <Fragment key={key}>
                      <dt>{NAMES[key] ?? key}</dt>
                      <dd>{value(key, raw)}</dd>
                    </Fragment>
                  ))}
                </dl>
              </details>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
