import "maplibre-gl/dist/maplibre-gl.css";

import { Map as MapLibreMap, NavigationControl, ScaleControl, setWorkerUrl } from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
import { useEffect, useRef } from "react";

// MapLibre 6 loads its worker from a separate file; Vite's dependency pre-bundling would
// lose it, so point MapLibre at the file explicitly (works in dev and in the build).
setWorkerUrl(workerUrl);

const STYLE = {
  version: 8,
  sources: {
    imagery: {
      type: "raster",
      tiles: [
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      ],
      tileSize: 256,
      maxzoom: 19,
      attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
    },
  },
  layers: [{ id: "imagery", type: "raster", source: "imagery" }],
};

// Satellite basemap with the parcel outline. Re-created for each parcel (the parent sets `key`).
export default function ParcelMap({ geometry, bounds }) {
  const container = useRef(null);

  useEffect(() => {
    const map = new MapLibreMap({
      container: container.current,
      style: STYLE,
      bounds,
      fitBoundsOptions: { padding: 60, maxZoom: 17 },
    });
    map.addControl(new NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new ScaleControl({ unit: "metric" }), "bottom-left");
    map.on("load", () => {
      map.addSource("parcel", { type: "geojson", data: { type: "Feature", properties: {}, geometry } });
      map.addLayer({
        id: "parcel-fill",
        type: "fill",
        source: "parcel",
        paint: { "fill-color": "#ffffff", "fill-opacity": 0.12 },
      });
      map.addLayer({
        id: "parcel-line",
        type: "line",
        source: "parcel",
        paint: { "line-color": "#ffd23f", "line-width": 2.5 },
      });
    });
    return () => map.remove();
  }, [geometry, bounds]);

  return <div ref={container} className="map" aria-label="Map of the parcel on satellite imagery" />;
}
