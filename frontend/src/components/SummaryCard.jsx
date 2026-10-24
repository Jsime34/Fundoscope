import { shortDate } from "../format";

export default function SummaryCard({ result }) {
  const { parcel, summary, campaigns } = result;
  const complete = campaigns.filter((c) => c.complete).length;
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
          {summary.processing_units.toFixed(1)} processing units · {summary.credits.toFixed(0)} credits{" "}
          {summary.cached && <span className="badge cached">from cache</span>}
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
