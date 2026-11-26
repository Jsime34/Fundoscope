import { useEffect, useState } from "react";

import { fetchStoredHistory, startAnalysis, uploadParcel, waitForJob } from "./api";
import CampaignTable from "./components/CampaignTable";
import CropTimeline from "./components/CropTimeline";
import FlagsPanel from "./components/FlagsPanel";
import NeighborsPanel from "./components/NeighborsPanel";
import NdviChart from "./components/NdviChart";
import ParcelMap from "./components/ParcelMap";
import StoredParcels from "./components/StoredParcels";
import SummaryCard from "./components/SummaryCard";
import UploadPanel from "./components/UploadPanel";

function useElapsedSeconds(running) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!running) return undefined;
    setSeconds(0);
    const timer = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, [running]);
  return seconds;
}

function Status({ state, fileName, seconds, error, step, job }) {
  if (state === "loading") {
    const progress = job?.progress ?? 0;
    return (
      <div className="status loading" role="status">
        <div className="status-line">
          <span className="spinner" />
          {fileName}: {job?.step ?? step} ({seconds} s)
        </div>
        {job && (
          <>
            <progress className="job-progress" value={progress} max={100} aria-label="Analysis progress">
              {progress} %
            </progress>
            <div className="status-hint">
              The first analysis downloads 8 years of Sentinel-2 data (about 2–3 minutes); later ones
              only download what is new. It runs on the server: if you close this page, the parcel
              will be under "Analyzed before" once it is done.
            </div>
          </>
        )}
      </div>
    );
  }
  if (state === "error") {
    return (
      <div className="status error" role="alert">
        {error}
      </div>
    );
  }
  return null;
}

export default function App() {
  const [state, setState] = useState("idle"); // idle | loading | done | error
  const [result, setResult] = useState(null); // the report on screen
  const [run, setRun] = useState(null); // the analysis job that produced it (null: reopened)
  const [job, setJob] = useState(null); // the analysis in progress (FR-13)
  const [step, setStep] = useState(""); // what is happening before a job exists
  const [error, setError] = useState("");
  const [fileName, setFileName] = useState("");
  const seconds = useElapsedSeconds(state === "loading");

  async function load(name, firstStep, request) {
    setFileName(name);
    setStep(firstStep);
    setJob(null);
    setState("loading");
    setError("");
    try {
      const loaded = await request();
      setResult(loaded.report);
      setRun(loaded.run);
      setState("done");
    } catch (err) {
      setError(err.message);
      setState("error");
    }
  }

  // Queue the analysis, follow its progress, then read the stored report.
  async function runAnalysis(parcelId) {
    const done = await waitForJob(await startAnalysis(parcelId), setJob);
    return { report: await fetchStoredHistory(parcelId), run: done };
  }

  const analyze = (file) =>
    load(file.name, "uploading the parcel", async () => runAnalysis((await uploadParcel(file)).id));
  const reopen = (parcel) =>
    parcel.observed_through
      ? load(parcel.name ?? "the parcel", "reading the stored report", async () => ({
          report: await fetchStoredHistory(parcel.id),
          run: null,
        }))
      : load(parcel.name ?? "the parcel", "starting the analysis", () => runAnalysis(parcel.id));

  const busy = state === "loading";
  const status = (
    <Status state={state} fileName={fileName} seconds={seconds} error={error} step={step} job={job} />
  );

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">Fundoscope</span>
        <span className="tagline">Satellite history of a farm parcel · coastal Peru</span>
      </header>

      {!result ? (
        <section className="hero">
          <h1>How has this parcel really been farmed?</h1>
          <p>
            Upload a parcel boundary and get its campaign-by-campaign history since 2018 from free
            Copernicus Sentinel-2 imagery: how green and well-watered it was, and how much of it was
            actually farmed.
          </p>
          <UploadPanel onFile={analyze} busy={busy} large />
          {status}
          <StoredParcels onOpen={reopen} busy={busy} />
        </section>
      ) : (
        <main className="grid">
          <div className="grid top">
            <div className="stack">
              <SummaryCard result={result} run={run} />
              <div className="card">
                <UploadPanel onFile={analyze} busy={busy} />
                {status}
              </div>
            </div>
            <div className="card map-card">
              <ParcelMap
                key={JSON.stringify(result.parcel.bounds)}
                geometry={result.parcel.geometry}
                bounds={result.parcel.bounds}
                ring={result.neighbors?.ring ?? null}
              />
            </div>
          </div>

          <div className="card">
            <h2>Crop by campaign</h2>
            <CropTimeline campaigns={result.campaigns} flags={result.flags ?? []} />
          </div>

          <div className="card">
            <div className="chart-head">
              <h2>Vegetation (NDVI) and canopy water (NDWI) since 2018</h2>
              <span className="hint">Shaded bands: agricultural campaigns, August to July</span>
            </div>
            <NdviChart
              series={result.series}
              observations={result.observations}
              campaigns={result.campaigns}
            />
          </div>

          <div className="card">
            <h2>Campaign by campaign</h2>
            <CampaignTable campaigns={result.campaigns} />
          </div>

          <div className="grid pair">
            <div className="card" id="flags">
              <h2>Flags</h2>
              <FlagsPanel
                flags={result.flags ?? []}
                campaigns={result.campaigns}
                observations={result.observations}
              />
            </div>

            <div className="card">
              <h2>Compared with the farmland around it</h2>
              <NeighborsPanel neighbors={result.neighbors ?? null} />
            </div>
          </div>
        </main>
      )}

      <footer className="footer">
        Contains modified Copernicus Sentinel data, processed through the Copernicus Data Space
        Ecosystem. Basemap imagery © Esri, Maxar, Earthstar Geographics. Fundoscope is a screening
        tool, not a legal or agronomic appraisal.
      </footer>
    </div>
  );
}
