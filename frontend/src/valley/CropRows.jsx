import { change, className, hectares, percentChange, PLANTED } from "./names";

// One row's chart: this campaign (dot), the previous one at the same date (dark tick), and
// the corrected 95 % range (thin bar below). Each row has its own scale, from 0 to its
// largest value, because "other annual crops" is ten times the other classes.
function RowChart({ item, previous, color }) {
  const top = Math.max(item.hectares, previous ?? 0, item.high ?? 0) * 1.08 || 1;
  const at = (value) => `${(value / top) * 100}%`;
  const tip = [
    `Esta campaña: ${hectares(item.hectares)}`,
    previous != null ? `Campaña anterior a la misma fecha: ${hectares(previous)}` : null,
    item.low != null ? `Rango corregido (95 %): ${hectares(item.low)} a ${hectares(item.high)}` : null,
  ]
    .filter(Boolean)
    .join("\n");
  return (
    <span className="row-chart" title={tip}>
      <span className="track" />
      {item.low != null && (
        <span
          className="range"
          style={{ left: at(item.low), width: at(item.high - item.low), background: color }}
        />
      )}
      {previous != null && <span className="previous" style={{ left: at(previous) }} />}
      <span className="dot" style={{ left: at(item.hectares), background: color }} />
    </span>
  );
}

function Swatch({ name, color }) {
  const style =
    name === PLANTED
      ? { background: `repeating-linear-gradient(45deg, ${color} 0 3px, #99cd99 3px 6px)` }
      : { background: color };
  return <span className="swatch" style={style} aria-hidden="true" />;
}

// "Hectáreas por cultivo": one row per class, this campaign against the previous one at the
// same date. Tapping a row shows that class on the map.
export default function CropRows({ current, previous, selected, onSelect }) {
  const before = Object.fromEntries((previous?.classes ?? []).map((c) => [c.name, c.hectares]));
  const rows = current.classes.filter((c) => c.pixels > 0 || before[c.name] > 0);
  return (
    <ul className="crop-rows">
      {rows.map((item) => {
        const old = before[item.name];
        const delta = old != null ? item.hectares - old : null;
        return (
          <li key={item.name}>
            <button
              type="button"
              className={`crop-row${selected === item.name ? " selected" : ""}`}
              aria-pressed={selected === item.name}
              onClick={() => onSelect(item.name)}
            >
              <span className="crop-head">
                <Swatch name={item.name} color={item.color} />
                <span className="crop-name">{className(item.name)}</span>
                <span className="crop-ha">{hectares(item.hectares)}</span>
              </span>
              <span className="crop-change">
                {delta != null ? (
                  <>
                    <span aria-hidden="true">{delta > 0 ? "▲" : delta < 0 ? "▼" : "="}</span>{" "}
                    {change(delta)}
                    {old > 0 ? ` (${percentChange(delta / old)})` : ""} · antes {hectares(old)}
                  </>
                ) : (
                  "sin dato de la campaña anterior"
                )}
              </span>
              <RowChart item={item} previous={old} color={item.color} />
              <span className="crop-range">
                {item.low != null
                  ? `Rango corregido (95 %): ${hectares(item.low)} a ${hectares(item.high)}`
                  : "Sin rango: se cuenta por una regla, no por el modelo"}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

export { Swatch };
