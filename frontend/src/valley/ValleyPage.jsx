import "./valley.css";

import { lazy, Suspense, useEffect, useState } from "react";

import { fetchValley } from "../api";
import CropRows, { Swatch } from "./CropRows";
import { className, date, dayMonth, hectares, NO_CLEAR_IMAGES } from "./names";

const ValleyMap = lazy(() => import("./ValleyMap"));

const VALLEY_NAMES = { majes: "Pampa de Majes" };

function firstSelection(count) {
  const crops = count.classes.filter((c) => c.pixels > 0 && c.name !== NO_CLEAR_IMAGES);
  return crops.toSorted((a, b) => b.hectares - a.hectares)[0]?.name;
}

function CompleteCampaigns({ counts }) {
  const whole = counts.filter((count) => !count.in_progress);
  if (!whole.length) return null;
  const names = whole[0].classes.filter((c) => whole.some((w) => w.classes.find((x) => x.name === c.name)?.pixels));
  return (
    <section className="card">
      <h2>Campañas completas</h2>
      <div className="table-wrap">
        <table className="complete">
          <thead>
            <tr>
              <th scope="col">Cultivo</th>
              {whole.map((count) => (
                <th scope="col" key={count.key}>{count.campaign}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {names.map((item) => (
              <tr key={item.name}>
                <td>{className(item.name)}</td>
                {whole.map((count) => (
                  <td key={count.key}>{hectares(count.classes.find((c) => c.name === item.name)?.hectares ?? 0)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// The valley crop monitor (FR-15): /valle/<slug>, in Spanish, for a phone first.
export default function ValleyPage({ slug }) {
  const [valley, setValley] = useState(null);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(null);
  const [showMap, setShowMap] = useState(true);
  const title = VALLEY_NAMES[slug] ?? slug;

  useEffect(() => {
    document.documentElement.lang = "es";
    document.title = `${title} · Fundoscope`;
    fetchValley(slug)
      .then((body) => {
        setValley(body);
        setSelected(firstSelection(body.counts[0]));
      })
      .catch((err) => setError(err.message));
  }, [slug, title]);

  if (error) {
    return (
      <div className="valley">
        <h1>{title}</h1>
        <p className="status error" role="alert">No se pudo cargar el conteo: {error}</p>
      </div>
    );
  }
  if (!valley) {
    return (
      <div className="valley">
        <h1>{title}</h1>
        <p className="status loading" role="status">Cargando…</p>
      </div>
    );
  }

  const [current, previous] = valley.counts;
  const comparable = current.in_progress && previous && previous.campaign !== current.campaign;
  const chosen = current.classes.find((c) => c.name === selected) ?? current.classes[0];

  return (
    <div className="valley">
      <header className="valley-head">
        <span className="brand">Fundoscope</span>
        <h1>{title}</h1>
        <p className="valley-sub">
          Campaña {current.campaign}
          {current.in_progress ? ` (agosto al ${dayMonth(current.through)})` : ""}
        </p>
        <p className="valley-dates">
          <span>Actualizado: {date(valley.updated_at)}</span>
          <span>Imagen más reciente: {date(valley.newest_image)}</span>
        </p>
      </header>

      <section className="card map-section">
        <div className="map-head">
          <h2>Mapa de cultivos</h2>
          <label className="switch">
            <input type="checkbox" checked={showMap} onChange={(e) => setShowMap(e.target.checked)} />
            Mostrar
          </label>
        </div>
        <Suspense fallback={<div className="valley-map" />}>
          <ValleyMap
            corners={valley.corners}
            farmlandImage={current.farmland_image}
            classImage={chosen.image}
            visible={showMap}
            label={`Mapa: ${className(chosen.name)} sobre la imagen satelital`}
          />
        </Suspense>
        <div className="legend" aria-label="Leyenda">
          <span className="legend-item">
            <Swatch name={chosen.name} color={chosen.color} /> {className(chosen.name)}
          </span>
          <span className="legend-item">
            <span className="swatch farmland" aria-hidden="true" /> Otras tierras de cultivo
          </span>
          <span className="legend-hint">Toque un cultivo de la lista para verlo en el mapa.</span>
        </div>
      </section>

      <section className="card">
        <h2>Hectáreas por cultivo</h2>
        <p className="hint">
          {comparable
            ? `Esta campaña hasta el ${dayMonth(current.through)}, comparada con ${previous.campaign} hasta la misma fecha. `
            : ""}
          Punto: esta campaña. Raya oscura: campaña anterior. Barra delgada: rango corregido. Cada
          fila tiene su propia escala.
        </p>
        <CropRows
          current={current}
          previous={comparable ? previous : null}
          selected={chosen.name}
          onSelect={setSelected}
        />
        <p className="farmland-total">Tierras de cultivo: {hectares(valley.farmland_ha)}</p>
      </section>

      <CompleteCampaigns counts={valley.counts} />

      <section className="card reading">
        <h2>Cómo leer estas cifras</h2>
        <p>
          Son estimaciones satelitales, no un censo. Cada píxel de 10 m de tierra de cultivo
          (verde en alguna de las tres últimas campañas, sin zonas urbanas ni agua) se clasifica
          con un modelo entrenado con parcelas etiquetadas; las hectáreas son esos píxeles. Para la
          campaña en curso se usa un modelo de los meses que van de la campaña, y la campaña anterior
          se cuenta hasta la misma fecha con el mismo modelo.
        </p>
        <p>
          El rango corregido (95 %) ajusta las hectáreas por los errores conocidos del modelo
          (Olofsson et al., 2014). Es aproximado, porque las parcelas de referencia no son una
          muestra al azar del valle. “Sembrado, aún no identificable” son campos recién sembrados
          cuyo cultivo todavía no se distingue.
        </p>
      </section>

      <footer className="footer">
        Contiene datos modificados de Copernicus Sentinel, procesados en Copernicus Data Space
        Ecosystem. Imágenes de fondo © Esri, Maxar, Earthstar Geographics. Fundoscope es una
        herramienta de consulta, no una tasación agronómica.
      </footer>
    </div>
  );
}
