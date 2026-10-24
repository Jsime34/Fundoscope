import { useEffect, useState } from "react";

import { fetchHistory } from "./api";
import CampaignTable from "./components/CampaignTable";
import NdviChart from "./components/NdviChart";
import ParcelMap from "./components/ParcelMap";
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

function Status({ state, fileName, seconds, error }) {
  if (state === "loading") {
    return (
      <div className="status loading" role="status">
        <span className="spinner" />
        Analyzing {fileName}: downloading 8 years of Sentinel-2 data ({seconds} s). The first run
        takes about 2–3 minutes; a repeat is instant.
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
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [fileName, setFileName] = useState("");
  const seconds = useElapsedSeconds(state === "loading");

  async function analyze(file) {
    setFileName(file.name);
    setState("loading");
    setError("");
    try {
      setResult(await fetchHistory(file));
      setState("done");
    } catch (err) {
      setError(err.message);
      setState("error");
    }
  }

  const busy = state === "loading";
  const status = <Status state={state} fileName={fileName} seconds={seconds} error={error} />;

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
        </section>
      ) : (
        <main className="grid">
          <div className="grid top">
            <div className="stack">
              <SummaryCard result={result} />
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
              />
            </div>
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
