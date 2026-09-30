"""Evalscripts: small JavaScript programs that Sentinel Hub runs on every pixel.

Each one declares its input bands and outputs in `setup()` and computes them per pixel in
`evaluatePixel()`. The `dataMask` output tells the Statistical API which pixels to count:
we set it to 0 for pixels that are not clear, so clouds never enter the statistics.
"""

# Scene Classification Layer (SCL) classes treated as not clear:
# 0 no data, 1 saturated/defective, 3 cloud shadow, 8 cloud (medium), 9 cloud (high), 10 cirrus.
NOT_CLEAR_SCL_CLASSES = (0, 1, 3, 8, 9, 10)

DAILY_INDICES = f"""
//VERSION=3
function setup() {{
  return {{
    input: [{{ bands: ["B04", "B08", "B8A", "B11", "SCL", "dataMask"] }}],
    output: [
      {{ id: "ndvi", bands: 1, sampleType: "FLOAT32" }},
      {{ id: "ndwi", bands: 1, sampleType: "FLOAT32" }},
      {{ id: "dataMask", bands: 1 }}
    ]
  }};
}}
const NOT_CLEAR = {list(NOT_CLEAR_SCL_CLASSES)};
function evaluatePixel(s) {{
  const clear = s.dataMask === 1 && NOT_CLEAR.indexOf(s.SCL) === -1;
  return {{
    ndvi: [(s.B08 - s.B04) / (s.B08 + s.B04)],
    ndwi: [(s.B8A - s.B11) / (s.B8A + s.B11)],
    dataMask: [clear ? 1 : 0]
  }};
}}
"""
"""NDVI (B08, B04) and NDWI (B8A, B11) per clear pixel, for the Statistical API."""
