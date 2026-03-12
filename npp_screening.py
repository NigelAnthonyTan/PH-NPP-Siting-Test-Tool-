"""
NPP Site Screening Tool — Philippines
======================================
Screens candidate areas for Nuclear Power Plant siting using IAEA safety
criteria for proximity to:
  • Active capable fault lines  → exclusion zone : 5 km   (IAEA NS-R-3)
  • Active volcanoes            → exclusion zone : 30 km  (IAEA NS-R-3)
  • Coastline                   → cooling access flag : within 2 km
                                  (UK NNB siting standard; IAEA non-safety
                                   siting factor per NS-R-3)

Data inputs (place all files in the same folder as this script):
  - fault.shp / .dbf / .prj / .shx / .cpg   — Philippine active fault lines
  - philippines_boundary.shp (+ companions)  — PH administrative boundary
  - volcanoes.xls                             — Smithsonian GVP v5.x (SpreadsheetML XML)

Outputs (saved to ./output/):
  - npp_screening_map.png        — static siting suitability map
  - npp_screening_map.html       — interactive Folium map
  - viable_grid_cells.csv        — candidate grid cell centroids with coastal flag
  - screening_summary.txt        — run summary and statistics
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from shapely.geometry import Point
from shapely.ops import unary_union
import folium
from folium.plugins import MiniMap
from lxml import etree

# ── Configuration ──────────────────────────────────────────────────────────────

FAULT_BUFFER_KM    = 50      # IAEA: exclusion within 5 km of capable faults
VOLCANO_BUFFER_KM  = 100     # IAEA: exclusion within 30 km of active volcanoes
GRID_SPACING_KM    = 5     # resolution of candidate grid (10 km x 10 km cells)
VOLCANO_CLIP_KM    = 250    # clip global dataset to this radius around PH
COASTAL_ACCESS_KM  = 2      # UK NNB standard: within 2 km of coast = viable
                             # for once-through seawater cooling access

# Geometry simplification tolerance (metres).
# 500 m is imperceptible at national screening scale and slashes memory ~10-20x.
SIMPLIFY_TOLERANCE_M = 500

WGS84    = "EPSG:4326"
PH_PROJ  = "EPSG:32651"    # UTM Zone 51N — metric CRS for the Philippines
OUTPUT_DIR = "output"

# ── Helpers ────────────────────────────────────────────────────────────────────

def km_to_m(km):
    return km * 1_000


def load_faults(path="fault.shp"):
    print(f"[1/6] Loading fault data from '{path}' ...")
    faults = gpd.read_file(path)
    if faults.crs is None:
        faults = faults.set_crs(WGS84)
    faults = faults.to_crs(PH_PROJ)

    # Drop null geometries
    before = len(faults)
    faults = faults[~faults.geometry.isna()].copy()

    # Drop geometries with non-finite (NaN/Inf) coordinates
    def is_finite_geom(geom):
        try:
            bounds = geom.bounds
            return all(np.isfinite(b) for b in bounds)
        except Exception:
            return False

    faults = faults[faults.geometry.apply(is_finite_geom)].copy()
    dropped = before - len(faults)
    if dropped > 0:
        print(f"      Dropped {dropped} invalid/corrupt geometry features.")

    # Simplify to reduce vertex count — 500 m tolerance is imperceptible
    # at national screening scale but cuts memory significantly
    faults["geometry"] = faults.geometry.simplify(
        SIMPLIFY_TOLERANCE_M, preserve_topology=True)
    print(f"      {len(faults)} fault features loaded (geometry simplified to "
          f"{SIMPLIFY_TOLERANCE_M} m tolerance).")
    return faults


def load_volcanoes(path="volcanoes.xls"):
    """
    Parse Smithsonian GVP SpreadsheetML XML (.xls) file.
      Row 1  — metadata title (skip)
      Row 2  — column headers
      Row 3+ — data
    """
    print(f"[2/6] Loading volcano data from '{path}' ...")

    SS = "urn:schemas-microsoft-com:office:spreadsheet"
    xml_parser = etree.XMLParser(recover=True, encoding="utf-8")
    with open(path, "rb") as f:
        content = f.read()
    root = etree.fromstring(content, parser=xml_parser)
    rows = root.findall(f".//{{{SS}}}Row")

    def get_row_values(row):
        cells = row.findall(f"{{{SS}}}Cell")
        return [
            (c.find(f"{{{SS}}}Data").text or "")
            if c.find(f"{{{SS}}}Data") is not None else ""
            for c in cells
        ]

    headers = get_row_values(rows[1])   # row index 1 = header row
    data    = [get_row_values(r) for r in rows[2:]]

    df = pd.DataFrame(data, columns=headers)
    df = df[df["Latitude"].str.strip() != ""]
    df["Latitude"]  = pd.to_numeric(df["Latitude"],  errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"])

    geometry  = [Point(lon, lat) for lon, lat in zip(df["Longitude"], df["Latitude"])]
    volcanoes = gpd.GeoDataFrame(df, geometry=geometry, crs=WGS84)
    volcanoes = volcanoes.to_crs(PH_PROJ)
    print(f"      {len(volcanoes)} Holocene volcanoes loaded from GVP catalogue.")
    return volcanoes


def load_boundary(path="philippines_boundary.shp"):
    print(f"[3/6] Loading Philippine boundary from '{path}' ...")
    boundary = gpd.read_file(path)
    if boundary.crs is None:
        boundary = boundary.set_crs(WGS84)
    boundary = boundary.to_crs(PH_PROJ)
    # Simplify boundary for containment checks — keeps grid step fast
    boundary["geometry"] = boundary.geometry.simplify(
        SIMPLIFY_TOLERANCE_M, preserve_topology=True)
    ph_union = boundary.geometry.unary_union
    print(f"      Boundary loaded (geometry simplified).")
    return boundary, ph_union


def extract_coastline(ph_union):
    """
    Derive the Philippine coastline by extracting the outer boundary of the
    land polygon. No additional shapefile needed — the coast is the edge of
    the boundary. Returns a single LineString/MultiLineString geometry and
    a buffered polygon representing the 2 km coastal access zone.
    """
    print(f"      Extracting coastline and building {COASTAL_ACCESS_KM} km "
          f"coastal access zone ...")
    coastline = ph_union.boundary
    coastal_zone = coastline.buffer(km_to_m(COASTAL_ACCESS_KM))
    return coastline, coastal_zone


def build_exclusion_zones(faults, volcanoes, ph_union):
    print(f"[4/6] Building exclusion zones ...")
    ph_clip       = ph_union.buffer(km_to_m(VOLCANO_CLIP_KM))
    volc_regional = volcanoes[volcanoes.geometry.within(ph_clip)].copy()

    ph_count = len(volcanoes[volcanoes["Country"].str.contains("Philippines", na=False)])
    print(f"      Philippine volcanoes in dataset : {ph_count}")
    print(f"      Volcanoes within {VOLCANO_CLIP_KM} km of PH : {len(volc_regional)}")

    # Buffer each feature individually, simplify immediately, then union.
    # This avoids holding all high-res buffered polygons in memory at once.
    print("      Buffering faults ...")
    fault_buffers = [
        geom.buffer(km_to_m(FAULT_BUFFER_KM)).simplify(SIMPLIFY_TOLERANCE_M)
        for geom in faults.geometry
    ]
    fault_excl_geom = unary_union(fault_buffers)
    del fault_buffers

    print("      Buffering volcanoes ...")
    volcano_buffers = [
        geom.buffer(km_to_m(VOLCANO_BUFFER_KM)).simplify(SIMPLIFY_TOLERANCE_M)
        for geom in volc_regional.geometry
    ]
    volcano_excl_geom = unary_union(volcano_buffers)
    del volcano_buffers

    combined_excl = fault_excl_geom.union(volcano_excl_geom)

    print(f"      Fault exclusion    : {FAULT_BUFFER_KM} km buffer applied.")
    print(f"      Volcano exclusion  : {VOLCANO_BUFFER_KM} km buffer applied.")
    return fault_excl_geom, volcano_excl_geom, combined_excl, volc_regional


def generate_candidate_grid(ph_union, combined_excl, coastal_zone):
    print(f"[5/6] Generating {GRID_SPACING_KM} km candidate grid ...")
    minx, miny, maxx, maxy = ph_union.bounds
    step = km_to_m(GRID_SPACING_KM)

    viable      = []
    coastal_flags = []
    total       = 0
    for x in np.arange(minx, maxx, step):
        for y in np.arange(miny, maxy, step):
            pt = Point(x + step / 2, y + step / 2)
            if ph_union.contains(pt):
                total += 1
                if not combined_excl.contains(pt):
                    viable.append(pt)
                    coastal_flags.append(coastal_zone.contains(pt))

    pct          = 100 * len(viable) / total if total else 0
    coastal_count = sum(coastal_flags)
    print(f"      Grid cells within PH            : {total}")
    print(f"      Viable cells post-exclusion     : {len(viable)}  ({pct:.1f}% of land area)")
    print(f"      Viable cells with coastal access: {coastal_count}  "
          f"(within {COASTAL_ACCESS_KM} km of coast)")

    viable_gdf = gpd.GeoDataFrame(geometry=viable, crs=PH_PROJ).to_crs(WGS84)
    viable_gdf["longitude"]      = viable_gdf.geometry.x
    viable_gdf["latitude"]       = viable_gdf.geometry.y
    viable_gdf["coastal_access"] = coastal_flags
    return viable_gdf, total


def make_static_map(boundary, faults, volc_regional,
                    fault_excl_geom, volcano_excl_geom,
                    coastline, coastal_zone, viable_gdf):
    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.set_facecolor("#d6eaf8")

    boundary.to_crs(PH_PROJ).plot(
        ax=ax, color="#f0ece2", edgecolor="#999", linewidth=0.4, zorder=2)

    # Coastal access zone (subtle blue band along coast)
    gpd.GeoDataFrame(geometry=[coastal_zone], crs=PH_PROJ).plot(
        ax=ax, color="#2980b9", alpha=0.18, zorder=3)

    gpd.GeoDataFrame(geometry=[fault_excl_geom], crs=PH_PROJ).plot(
        ax=ax, color="#e74c3c", alpha=0.35, zorder=4)
    gpd.GeoDataFrame(geometry=[volcano_excl_geom], crs=PH_PROJ).plot(
        ax=ax, color="#e67e22", alpha=0.30, zorder=5)

    faults.plot(ax=ax, color="#c0392b", linewidth=0.6, zorder=6)

    volc_proj = volc_regional.to_crs(PH_PROJ)
    volc_proj.plot(ax=ax, color="#d35400", marker="^", markersize=35, zorder=7)

    # Annotate volcano names
    for _, row in volc_proj.iterrows():
        ax.annotate(
            row.get("Volcano Name", ""),
            xy=(row.geometry.x, row.geometry.y),
            xytext=(4, 4), textcoords="offset points",
            fontsize=5.5, color="#7d3c00", zorder=9
        )

    # Split viable sites: coastal access vs. inland viable
    viable_proj   = viable_gdf.to_crs(PH_PROJ)
    coastal_sites = viable_proj[viable_proj["coastal_access"]]
    inland_sites  = viable_proj[~viable_proj["coastal_access"]]

    if len(inland_sites):
        inland_sites.plot(ax=ax, color="#95a5a6", markersize=3,
                          alpha=0.55, zorder=8)
    if len(coastal_sites):
        coastal_sites.plot(ax=ax, color="#27ae60", markersize=4,
                           alpha=0.75, zorder=9)

    legend_elements = [
        mpatches.Patch(color="#2980b9", alpha=0.35,
                       label=f"Coastal access zone ({COASTAL_ACCESS_KM} km)"),
        mpatches.Patch(color="#e74c3c", alpha=0.5,
                       label=f"Fault exclusion zone ({FAULT_BUFFER_KM} km)"),
        mpatches.Patch(color="#e67e22", alpha=0.5,
                       label=f"Volcano exclusion zone ({VOLCANO_BUFFER_KM} km)"),
        Line2D([0], [0], color="#c0392b", lw=1.5,
               label="Active capable faults"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#d35400",
               markersize=10, label="Active volcanoes (regional)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#27ae60",
               markersize=7, label=f"Viable + coastal access (<{COASTAL_ACCESS_KM} km)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#95a5a6",
               markersize=5, label="Viable (no coastal access)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9,
              framealpha=0.9, title="Legend", title_fontsize=9)
    ax.set_title(
        "Nuclear Power Plant Siting — High-Level Screening\n"
        f"Philippines  |  Fault excl.: {FAULT_BUFFER_KM} km  |  "
        f"Volcano excl.: {VOLCANO_BUFFER_KM} km  |  "
        f"Coastal access: {COASTAL_ACCESS_KM} km  |  Grid: {GRID_SPACING_KM} km",
        fontsize=11, fontweight="bold", pad=12
    )
    ax.set_xlabel("Easting (m, UTM Zone 51N)")
    ax.set_ylabel("Northing (m, UTM Zone 51N)")

    out_png = os.path.join(OUTPUT_DIR, "npp_screening_map.png")
    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"      Static map saved  -> {out_png}")


def make_interactive_map(boundary, faults, volc_regional, viable_gdf):
    m = folium.Map(location=[12.5, 122.5], zoom_start=6,
                   tiles="CartoDB positron")

    folium.GeoJson(
        boundary.to_crs(WGS84).__geo_interface__,
        name="Philippine Boundary",
        style_function=lambda _: {
            "fillColor": "#f0ece2", "color": "#888",
            "weight": 1, "fillOpacity": 0.4}
    ).add_to(m)

    folium.GeoJson(
        faults.to_crs(WGS84).__geo_interface__,
        name=f"Active Faults (+/- {FAULT_BUFFER_KM} km exclusion)",
        style_function=lambda _: {
            "color": "#c0392b", "weight": 1.5, "opacity": 0.8}
    ).add_to(m)

    volc_group = folium.FeatureGroup(
        name=f"Active Volcanoes (+/- {VOLCANO_BUFFER_KM} km exclusion)")
    for _, row in volc_regional.to_crs(WGS84).iterrows():
        popup_html = (
            f"<b>{row.get('Volcano Name','Volcano')}</b><br>"
            f"Country: {row.get('Country','')}<br>"
            f"Last eruption: {row.get('Last Known Eruption','Unknown')}<br>"
            f"Lat: {row.geometry.y:.4f}, Lon: {row.geometry.x:.4f}"
        )
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=folium.Popup(popup_html, max_width=220),
            icon=folium.Icon(color="orange", icon="fire", prefix="fa")
        ).add_to(volc_group)
    volc_group.add_to(m)

    # Coastal access sites (green) — primary candidates
    coastal_group = folium.FeatureGroup(
        name=f"Viable + Coastal Access (within {COASTAL_ACCESS_KM} km of coast)")
    coastal_sites = viable_gdf[viable_gdf["coastal_access"]].iloc[::2]
    for _, row in coastal_sites.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4, color="#27ae60", fill=True,
            fill_color="#27ae60", fill_opacity=0.75,
            popup=(f"<b>Viable — Coastal Access</b><br>"
                   f"Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f}")
        ).add_to(coastal_group)
    coastal_group.add_to(m)

    # Inland viable sites (grey) — pass hazard screen but lack coastal access
    inland_group = folium.FeatureGroup(
        name="Viable (no coastal access)")
    inland_sites = viable_gdf[~viable_gdf["coastal_access"]].iloc[::2]
    for _, row in inland_sites.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3, color="#7f8c8d", fill=True,
            fill_color="#7f8c8d", fill_opacity=0.55,
            popup=(f"<b>Viable — No Coastal Access</b><br>"
                   f"Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f}")
        ).add_to(inland_group)
    inland_group.add_to(m)

    MiniMap(toggle_display=True).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    out_html = os.path.join(OUTPUT_DIR, "npp_screening_map.html")
    m.save(out_html)
    print(f"      Interactive map   -> {out_html}")


def write_summary(viable_gdf, total_cells, volc_regional):
    pct          = 100 * len(viable_gdf) / total_cells if total_cells else 0
    coastal_count = viable_gdf["coastal_access"].sum()
    coastal_pct   = 100 * coastal_count / len(viable_gdf) if len(viable_gdf) else 0
    ph_volc = volc_regional[
        volc_regional["Country"].str.contains("Philippines", na=False)]
    other   = volc_regional[
        ~volc_regional["Country"].str.contains("Philippines", na=False)]

    lines = [
        "=" * 64,
        "  NPP SITE SCREENING SUMMARY — PHILIPPINES",
        "=" * 64,
        "",
        "Siting Criteria Applied:",
        f"  Fault exclusion radius   : {FAULT_BUFFER_KM} km  (IAEA NS-R-3)",
        f"  Volcano exclusion radius : {VOLCANO_BUFFER_KM} km  (IAEA NS-R-3)",
        f"  Coastal access threshold : {COASTAL_ACCESS_KM} km  (UK NNB standard)",
        f"  Candidate grid spacing   : {GRID_SPACING_KM} km",
        "",
        "Results:",
        f"  Grid cells within PH territory          : {total_cells}",
        f"  Viable cells post-exclusion             : {len(viable_gdf)}  ({pct:.1f}% of land area)",
        f"  Viable cells WITH coastal access        : {coastal_count}  ({coastal_pct:.1f}% of viable)",
        f"  Viable cells WITHOUT coastal access     : {len(viable_gdf) - coastal_count}",
        "",
        "Note: 'Coastal access' = within 2 km of coastline, indicating",
        "feasibility for once-through seawater cooling. Sites without",
        "coastal access are not excluded — they remain viable candidates",
        "subject to alternative cooling system design (e.g. cooling towers",
        "with freshwater source), to be assessed in the next screening stage.",
        "",
        f"Philippine volcanoes in GVP dataset ({len(ph_volc)}):",
    ]
    for _, v in ph_volc.sort_values("Volcano Name").iterrows():
        lines.append(
            f"  - {v['Volcano Name']:<32s}  Last eruption: "
            f"{v.get('Last Known Eruption','Unknown')}")

    if len(other):
        lines += ["", f"Other regional volcanoes within {VOLCANO_CLIP_KM} km ({len(other)}):"]
        for _, v in other.sort_values("Country").iterrows():
            lines.append(f"  - {v['Volcano Name']:<32s}  ({v.get('Country','')})")

    lines += [
        "",
        "-" * 64,
        "DISCLAIMER: High-level screen only. Full IAEA site evaluation",
        "required, including PSHA, tsunami/flood, population, hydrology,",
        "extreme external events, land use and grid connectivity studies.",
        "Coastal sites must also be assessed for tsunami inundation risk.",
        "=" * 64,
    ]

    out_txt = os.path.join(OUTPUT_DIR, "screening_summary.txt")
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    print(f"      Summary saved     -> {out_txt}")
    print()
    for ln in lines:
        print("  " + ln)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "=" * 64)
    print("  NPP Siting Screening Tool — Philippines")
    print("=" * 64 + "\n")

    faults             = load_faults("fault.shp")
    volcanoes          = load_volcanoes("volcanoes.xls")
    boundary, ph_union = load_boundary("philippines_boundary.shp")
    coastline, coastal_zone = extract_coastline(ph_union)

    fault_excl, volcano_excl, combined_excl, volc_regional = \
        build_exclusion_zones(faults, volcanoes, ph_union)

    viable_gdf, total_cells = generate_candidate_grid(
        ph_union, combined_excl, coastal_zone)

    csv_path = os.path.join(OUTPUT_DIR, "viable_grid_cells.csv")
    viable_gdf[["latitude", "longitude", "coastal_access"]].to_csv(
        csv_path, index=False)
    print(f"      Viable cells CSV  -> {csv_path}")

    print("[6/6] Generating maps ...")
    make_static_map(boundary, faults, volc_regional,
                    fault_excl, volcano_excl,
                    coastline, coastal_zone, viable_gdf)
    make_interactive_map(boundary, faults, volc_regional, viable_gdf)
    write_summary(viable_gdf, total_cells, volc_regional)

    print(f"\n  Done. All outputs saved to ./{OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
