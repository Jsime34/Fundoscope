import { percent } from "../format";

// FR-06 and FR-12: the predicted crop of each campaign and how sure the model is, one tile per
// campaign (Figure 6). The swatch is the crop's color on the valley map; the crop's name is
// always written, so color is never the only cue. A tile with a flag names it and links to the
// flags.

const UNSURE = 0.6; // below this, a crop change isn't flagged either (backend analysis/flags.py)
// The flag a tile names when a campaign has several (the others are counted), most serious first.
const FLAG_ORDER = [
  "possible abandonment",
  "perennial removed",
  "perennial replaced",
  "perennial established",
  "crop change",
  "anomalous campaign",
  "underperforming",
  "low data",
];
const MINOR_FLAGS = new Set(["low data"]);

function rank(flag) {
  const index = FLAG_ORDER.indexOf(flag.kind);
  return index < 0 ? FLAG_ORDER.length : index;
}

function capitalize(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

// "fallow/bare" -> "Fallow / bare", "other annual crops" -> "Other annual crops"
export function cropName(name) {
  return capitalize(name.replace("/", " / "));
}

function Crop({ campaign }) {
  if (!campaign.crop) {
    return (
      <span className="tile-none">
        {campaign.complete ? "Not predicted" : "Too early to tell: predicted when the campaign ends"}
      </span>
    );
  }
  const unsure = campaign.crop_confidence < UNSURE;
  return (
    <>
      <span className="tile-crop">
        <span className="swatch" style={{ background: campaign.crop_color ?? "var(--muted)" }} />
        {cropName(campaign.crop)}
      </span>
      <span className="tile-confidence">
        confidence {percent(campaign.crop_confidence)}
        {unsure && " · uncertain"}
      </span>
    </>
  );
}

export default function CropTimeline({ campaigns, flags }) {
  const predicted = campaigns.some((c) => c.crop);
  const complete = campaigns.some((c) => c.complete);
  return (
    <>
      <p className="hint">
        The crop model's prediction for each campaign (August to July). The confidence is the
        crop's average probability over the parcel's pixels; below {percent(UNSURE)} it is marked
        uncertain. Colors match the valley crop map.
      </p>
      {complete && !predicted && (
        <p className="note">The server has no crop model loaded, so no crop was predicted.</p>
      )}
      <ol className="timeline">
        {campaigns.map((campaign) => {
          const own = flags.filter((f) => f.campaign === campaign.campaign);
          const flagged = own.some((f) => !MINOR_FLAGS.has(f.kind));
          const classes = ["tile", campaign.complete ? "" : "partial", flagged ? "flagged" : ""];
          return (
            <li key={campaign.campaign} className={classes.join(" ").trim()}>
              <span className="tile-campaign">
                {campaign.campaign}
                {!campaign.complete && " · in progress"}
              </span>
              <Crop campaign={campaign} />
              {own.length > 0 && (
                <a
                  className="tile-flags"
                  href="#flags"
                  title={own.map((f) => capitalize(f.kind)).join(", ")}
                >
                  {capitalize(own.toSorted((a, b) => rank(a) - rank(b))[0].kind)}
                  {own.length > 1 && ` +${own.length - 1} more`}
                </a>
              )}
            </li>
          );
        })}
      </ol>
    </>
  );
}
