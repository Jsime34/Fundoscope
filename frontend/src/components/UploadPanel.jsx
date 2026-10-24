import { useRef, useState } from "react";

const ACCEPT = ".kml,.kmz,.geojson,.json";

// Drop zone + file picker. Calls onFile(file) once per chosen file.
export default function UploadPanel({ onFile, busy, large = false }) {
  const input = useRef(null);
  const [over, setOver] = useState(false);

  function pick(files) {
    if (!busy && files && files.length > 0) onFile(files[0]);
  }

  return (
    <div
      className={`dropzone${over ? " over" : ""}${busy ? " busy" : ""}`}
      role="button"
      tabIndex={0}
      aria-label="Upload a parcel boundary file"
      onClick={() => !busy && input.current?.click()}
      onKeyDown={(e) => {
        if (!busy && (e.key === "Enter" || e.key === " ")) input.current?.click();
      }}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        pick(e.dataTransfer.files);
      }}
    >
      <strong>{large ? "Drop a parcel boundary here" : "Analyze another parcel"}</strong>
      <span>KML, KMZ or GeoJSON · or click to choose a file</span>
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        onChange={(e) => {
          pick(e.target.files);
          e.target.value = "";
        }}
      />
    </div>
  );
}
