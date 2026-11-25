import "maplibre-gl/dist/maplibre-gl.css";

import { Map as MapLibreMap, NavigationControl, ScaleControl, setWorkerUrl } from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
import { useEffect, useRef } from "react";

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
      attribution: "Imágenes © Esri, Maxar, Earthstar Geographics",
    },
  },
  layers: [{ id: "imagery", type: "raster", source: "imagery" }],
};

function boundsOf(corners) {
  const lons = corners.map((c) => c[0]);
  const lats = corners.map((c) => c[1]);
  return [Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats)];
}

// The satellite basemap with the farmland in a light wash and one crop in its color.
// The images are transparent PNGs on the valley's grid; `corners` places them.
export default function ValleyMap({ corners, farmlandImage, classImage, visible, label }) {
  const container = useRef(null);
  const map = useRef(null);
  const ready = useRef(false);

  useEffect(() => {
    const instance = new MapLibreMap({
      container: container.current,
      style: STYLE,
      bounds: boundsOf(corners),
      fitBoundsOptions: { padding: 12 },
      attributionControl: { compact: true },
    });
    instance.addControl(new NavigationControl({ showCompass: false }), "top-right");
    instance.addControl(new ScaleControl({ unit: "metric" }), "bottom-left");
    instance.on("load", () => {
      instance.addSource("farmland", { type: "image", url: farmlandImage, coordinates: corners });
      instance.addLayer({ id: "farmland", type: "raster", source: "farmland" });
      instance.addSource("crop", { type: "image", url: classImage, coordinates: corners });
      instance.addLayer({
        id: "crop",
        type: "raster",
        source: "crop",
        paint: { "raster-resampling": "nearest" },
      });
      // The compact attribution starts open and would cover the map on a phone: close it.
      container.current
        ?.querySelector(".maplibregl-ctrl-attrib")
        ?.classList.remove("maplibregl-compact-show");
      ready.current = true;
      instance.fire("fundoscope:ready");
    });
    map.current = instance;
    return () => {
      ready.current = false;
      instance.remove();
    };
    // The map is built once; the effects below update its images and visibility.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const instance = map.current;
    const update = () => {
      instance.getSource("farmland").updateImage({ url: farmlandImage, coordinates: corners });
      instance.getSource("crop").updateImage({ url: classImage, coordinates: corners });
      for (const id of ["farmland", "crop"]) {
        instance.setLayoutProperty(id, "visibility", visible ? "visible" : "none");
      }
    };
    if (ready.current) update();
    else instance.once("fundoscope:ready", update);
  }, [corners, farmlandImage, classImage, visible]);

  return <div ref={container} className="valley-map" role="img" aria-label={label} />;
}
