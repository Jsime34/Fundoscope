import { shortDate } from "../format";

// `run` is the analysis job that just updated the report (null when it was reopened).
export default function SummaryCard({ result, run }) {
  const { parcel, summary, campaigns } = result;
  const complete = campaigns.filter((c) => c.complete).length;
  const downloaded = run && (run.processing_units > 0 || run.credits > 0);
  const firstRun =
    downloaded && run.processing_units === summary.processing_units && run.credits === summary.credits;
  return (
    <div className="card">
      <div className="parcel-name">{parcel.name || "Unnamed parcel"}</div>
      <div className="parcel-sub">
        {parcel.area_ha.toFixed(2)} ha · {parcel.zone}
      </div>
      <dl className="facts">
        <dt>Satellite dates</dt>
        <dd>
          {summary.satellite_dates} ({shortDate(summary.first_date)} – {shortDate(summary.last_date)})
        </dd>
        <dt>Removed</dt>
        <dd>
          {summary.dropped_cloudy} cloudy · {summary.dropped_dips} cloud dips
        </dd>
        <dt>Campaigns</dt>
        <dd>
          {campaigns.length} ({complete} complete)
        </dd>
        <dt>Cost</dt>
        <dd>
          {summary.processing_units.toFixed(1)} processing units · {summary.credits.toFixed(0)} credits
          so far{" "}
          {!downloaded && <span className="badge cached">stored: nothing new downloaded</span>}
          {downloaded && !firstRun && (
            <span className="cost-update">
              (this update: {run.processing_units.toFixed(1)} units · {run.credits.toFixed(0)} credits)
            </span>
          )}
        </dd>
      </dl>
      {parcel.repaired && (
        <div className="note">The outline was invalid and was repaired automatically. Check the drawing.</div>
      )}
      {summary.active_area_failures > 0 && (
        <div className="note">
          Active area is missing for {summary.active_area_failures} campaign(s): the satellite service
          failed. Try again later.
        </div>
      )}
    </div>
  );
}
