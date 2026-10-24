import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceArea,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { campaignRange, fixed, shortDate } from "../format";

// Chart palette (light mode): one hue per measure, text in neutral ink.
const COLORS = {
  ndvi: "#2a78d6",
  ndviDots: "#86b6ef",
  ndwi: "#eb6834",
  band: "#f0efec",
  grid: "#e1e0d9",
  axis: "#c3c2b7",
  muted: "#898781",
};
const DAY = 86_400_000;

function toTime(iso) {
  return Date.parse(`${iso}T00:00:00Z`);
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const values = Object.fromEntries(payload.map((p) => [p.dataKey, p.value]));
  return (
    <div className="chart-tooltip">
      <div className="date">{shortDate(new Date(label).toISOString().slice(0, 10))}</div>
      <div className="row">
        <span className="swatch" style={{ background: COLORS.ndvi }} /> NDVI {fixed(values.ndvi)}
      </div>
      <div className="row">
        <span className="swatch" style={{ background: COLORS.ndwi }} /> NDWI {fixed(values.ndwi)}
      </div>
    </div>
  );
}

// NDVI and NDWI over time: clear satellite dates as dots, the cleaned curves as lines,
// and the agricultural campaigns (Aug-Jul) as alternating bands.
export default function NdviChart({ series, observations, campaigns }) {
  const line = series.map((p) => ({ t: toTime(p.date), ndvi: p.ndvi, ndwi: p.ndwi }));
  const dots = observations.map((p) => ({ t: toTime(p.date), clear: p.ndvi }));
  if (line.length === 0) return <p className="hint">No clear satellite images to plot.</p>;

  const first = line[0].t;
  const last = line[line.length - 1].t;
  const yearTicks = [];
  for (let y = new Date(first).getUTCFullYear() + 1; Date.UTC(y, 0, 1) <= last; y += 1) {
    yearTicks.push(Date.UTC(y, 0, 1));
  }
  const bands = campaigns.map((c, i) => {
    const [start, end] = campaignRange(c.campaign);
    const x1 = Math.max(start, first);
    const x2 = Math.min(end, last);
    return { label: c.campaign, x1, x2, shaded: i % 2 === 0, wide: x2 - x1 > 150 * DAY };
  });
  // Ticks every 0.2; go below 0 only if NDWI does (bare desert can be slightly negative).
  const lowest = Math.min(0, ...line.map((p) => p.ndwi ?? 0));
  const yMin = Math.floor(lowest * 5) / 5;
  const yTicks = [];
  for (let v = yMin; v <= 1.0001; v += 0.2) yTicks.push(Math.round(v * 10) / 10);

  return (
    <ResponsiveContainer width="100%" height={340}>
      <ComposedChart data={line} margin={{ top: 26, right: 14, bottom: 0, left: -10 }}>
        {bands.map((b) => (
          <ReferenceArea
            key={b.label}
            x1={b.x1}
            x2={b.x2}
            fill={b.shaded ? COLORS.band : "#ffffff"}
            fillOpacity={b.shaded ? 1 : 0}
            ifOverflow="hidden"
            label={
              b.wide
                ? { value: b.label, position: "insideTop", dy: -22, fill: COLORS.muted, fontSize: 11 }
                : undefined
            }
          />
        ))}
        <CartesianGrid vertical={false} stroke={COLORS.grid} />
        <XAxis
          dataKey="t"
          type="number"
          domain={[first, last]}
          ticks={yearTicks}
          tickFormatter={(t) => new Date(t).getUTCFullYear()}
          stroke={COLORS.axis}
          tick={{ fill: COLORS.muted, fontSize: 12 }}
        />
        <YAxis
          type="number"
          domain={[yMin, 1]}
          ticks={yTicks}
          interval={0}
          stroke={COLORS.axis}
          tick={{ fill: COLORS.muted, fontSize: 12 }}
          width={44}
        />
        <Tooltip content={<ChartTooltip />} cursor={{ stroke: COLORS.axis }} />
        <Legend verticalAlign="bottom" height={30} iconSize={10} wrapperStyle={{ fontSize: 12.5 }} />
        <Scatter
          name="Clear satellite date (NDVI)"
          data={dots}
          dataKey="clear"
          fill={COLORS.ndviDots}
          shape={(props) => <circle cx={props.cx} cy={props.cy} r={2.2} fill={COLORS.ndviDots} />}
          isAnimationActive={false}
        />
        <Line
          name="NDVI, cleaned (green leaves)"
          dataKey="ndvi"
          stroke={COLORS.ndvi}
          strokeWidth={2}
          dot={false}
          connectNulls={false}
          isAnimationActive={false}
        />
        <Line
          name="NDWI, cleaned (canopy water)"
          dataKey="ndwi"
          stroke={COLORS.ndwi}
          strokeWidth={2}
          dot={false}
          connectNulls={false}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
