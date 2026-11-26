import { useEffect, useState } from "react";

import { fetchStoredParcels } from "../api";
import { shortDate } from "../format";

// The parcels already analyzed (stored in the database): open one without downloading again.
export default function StoredParcels({ onOpen, busy }) {
  const [parcels, setParcels] = useState(null);

  useEffect(() => {
    fetchStoredParcels()
      .then(setParcels)
      .catch(() => setParcels([])); // no database: just no list
  }, []);

  if (!parcels?.length) return null;
  return (
    <div className="stored">
      <h2>Analyzed before</h2>
      <ul>
        {parcels.map((parcel) => (
          <li key={parcel.id}>
            <button type="button" disabled={busy} onClick={() => onOpen(parcel)}>
              <span className="stored-name">{parcel.name ?? "Unnamed parcel"}</span>
              <span className="stored-meta">
                {parcel.area_ha.toFixed(1)} ha · data to {shortDate(parcel.observed_through)}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
