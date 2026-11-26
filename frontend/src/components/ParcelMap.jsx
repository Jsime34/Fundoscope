import "maplibre-gl/dist/maplibre-gl.css";

import { Map as MapLibreMap, NavigationControl, ScaleControl, setWorkerUrl } from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
import { useEffect, useRef, useState } from "react";

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

const RING_LAYERS = ["ring-fill", "ring-line"];

function boundsOf(polygon) {
  const points = polygon.coordinates.flat(1);
  const lons = points.map((p) => p[0]);
  const lats = points.map((p) => p[1]);
  return [Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats)];
}

// Satellite basemap with the parcel outline and, inside the pilot area, the 50 m-2 km neighbor
// ring (FR-11), which can be shown and hidden. Re-created for each parcel (the parent sets `key`).
export default function ParcelMap({ geometry, bounds, ring }) {
  const container = useRef(null);
  const mapRef = useRef(null);
  const [showRing, setShowRing] = useState(false);

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
      if (ring) {
        map.addSource("ring", { type: "geojson", data: { type: "Feature", properties: {}, geometry: ring } });
        map.addLayer({
          id: "ring-fill",
          type: "fill",
          source: "ring",
          layout: { visibility: "none" },
          paint: { "fill-color": "#ffffff", "fill-opacity": 0.1 },
        });
        map.addLayer({
          id: "ring-line",
          type: "line",
          source: "ring",
          layout: { visibility: "none" },
          paint: { "line-color": "#ffffff", "line-width": 1.5, "line-dasharray": [3, 2] },
        });
      }
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
    mapRef.current = map;
    return () => map.remove();
  }, [geometry, bounds, ring]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ring) return;
    const apply = () => {
      for (const id of RING_LAYERS) {
        map.setLayoutProperty(id, "visibility", showRing ? "visible" : "none");
      }
      if (showRing) map.fitBounds(boundsOf(ring), { padding: 30 });
      else map.fitBounds(bounds, { padding: 60, maxZoom: 17 });
    };
    if (map.getLayer(RING_LAYERS[0])) apply();
    else map.once("idle", apply);
  }, [showRing, ring, bounds]);

  return (
    <div className="map-wrap">
      <div ref={container} className="map" aria-label="Map of the parcel on satellite imagery" />
      {ring && (
        <label className="map-toggle">
          <input type="checkbox" checked={showRing} onChange={(e) => setShowRing(e.target.checked)} />
          Neighbor ring (50 m–2 km)
        </label>
      )}
    </div>
  );
}
