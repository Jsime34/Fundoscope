// FR-08: each recent campaign of the parcel against the farmland 50 m to 2 km around it.
// One row per campaign: the neighbors' p10-p90 (light) and p25-p75 (darker) bands, their
// median (tick), and the parcel (dot). All rows share one NDVI scale.

const BAND_TEXT = {
  "below p10": "Below 90 % of its neighbors",
  "p10-p25": "In the lower quarter of its neighbors",
  "p25-p50": "A little below the median neighbor",
  "p50-p75": "A little above the median neighbor",
  "p75-p90": "In the upper quarter of its neighbors",
  "above p90": "Greener than 90 % of its neighbors",
};

function scale(bands) {
  const low = Math.min(...bands.map((b) => Math.min(b.p10, b.parcel_ndvi)));
  const high = Math.max(...bands.map((b) => Math.max(b.p90, b.parcel_ndvi)));
  const pad = (high - low) * 0.08 || 0.05;
  return (value) => `${((value - (low - pad)) / (high - low + 2 * pad)) * 100}%`;
}

function label(band) {
  return band.months === 12 ? band.campaign : `${band.campaign} (${band.months} months so far)`;
}

export default function NeighborsPanel({ neighbors }) {
  if (!neighbors) {
    return <p className="hint">Not compared.</p>;
  }
  if (!neighbors.bands.length) {
    return <p className="hint">{neighbors.note}</p>;
  }
  const at = scale(neighbors.bands);
  return (
    <>
      <p className="hint">
        Mean NDVI of the parcel against the farmland 50 m to 2 km around it, from the valley's
        monthly composites. Light bar: the middle 80 % of the neighbors; darker bar: the middle
        50 %; tick: the median neighbor; dot: this parcel.
      </p>
      <ul className="neighbor-rows">
        {neighbors.bands.map((band) => (
          <li key={band.campaign} className="neighbor-row">
            <span className="neighbor-campaign">{label(band)}</span>
            <span className="neighbor-text">
              {BAND_TEXT[band.band] ?? band.band}
              {band.underperforming && <span className="badge partial">Underperforming</span>}
            </span>
            <span
              className="neighbor-chart"
              title={`Parcel ${band.parcel_ndvi.toFixed(2)} · neighbors p10 ${band.p10.toFixed(2)}, median ${band.p50.toFixed(2)}, p90 ${band.p90.toFixed(2)} · ${band.neighbor_pixels.toLocaleString("en-US")} pixels`}
            >
              <span className="outer" style={{ left: at(band.p10), width: `calc(${at(band.p90)} - ${at(band.p10)})` }} />
              <span className="inner" style={{ left: at(band.p25), width: `calc(${at(band.p75)} - ${at(band.p25)})` }} />
              <span className="median" style={{ left: at(band.p50) }} />
              <span className="parcel" style={{ left: at(band.parcel_ndvi) }} />
            </span>
            <span className="neighbor-values">
              {band.parcel_ndvi.toFixed(2)} <span className="muted">vs. median {band.p50.toFixed(2)}</span>
            </span>
          </li>
        ))}
      </ul>
    </>
  );
}
