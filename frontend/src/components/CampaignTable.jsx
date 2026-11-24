import { dash, fixed, percent } from "../format";

// The predicted crop and how sure the model is (mean probability over the parcel's pixels).
function Crop({ name, confidence }) {
  if (!name) return dash;
  return (
    <span className="crop">
      {name} <span className="confidence">{percent(confidence)}</span>
    </span>
  );
}

function ActiveArea({ value }) {
  if (value == null) return "—";
  return (
    <span className="meter">
      <span className="track">
        <span className="fill" style={{ width: `${Math.round(value * 100)}%` }} />
      </span>
      {percent(value)}
    </span>
  );
}

// One row per agricultural campaign (August-July).
export default function CampaignTable({ campaigns }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Campaign</th>
            <th title="Predicted by the crop model; the percentage is its confidence. In-progress campaigns have no crop yet.">
              Crop
            </th>
            <th>Peak NDVI</th>
            <th>Mean NDVI</th>
            <th>Mean NDWI</th>
            <th title="Area under the NDVI curve above 0.1, in NDVI x days">Integrated</th>
            <th title="Share of the parcel whose peak NDVI passed 0.4">Active area</th>
            <th>Clear dates</th>
            <th>Coverage</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {campaigns.map((c) => (
            <tr key={c.campaign}>
              <td>{c.campaign}</td>
              <td>
                <Crop name={c.crop} confidence={c.crop_confidence} />
              </td>
              <td>{fixed(c.peak_ndvi)}</td>
              <td>{fixed(c.mean_ndvi)}</td>
              <td>{fixed(c.mean_ndwi)}</td>
              <td>{fixed(c.integrated_ndvi, 0)}</td>
              <td>
                <ActiveArea value={c.active_area} />
              </td>
              <td>{c.n_obs}</td>
              <td>{percent(c.coverage)}</td>
              <td>{c.complete ? "Complete" : <span className="badge partial">In progress</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
