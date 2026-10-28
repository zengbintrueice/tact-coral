# download_gcbd_global.py — global GCBD download + per-region availability counts
#
# Purpose: fetch the FULL global Global Coral Bleaching Database (van Woesik &
# Kratochwill 2022, Sci. Data 9:20; BCO-DMO DOI 10.26008/1912/bco-dmo.773466.2)
# as one flat per-Sample_ID CSV, then count AVAILABLE observations per sea region.
# The shipped `data/china_bco_dmo_bleaching.csv` is a regional subset of this
# database used by Steps 9/10; this script obtains the global parent dataset.
#
# Download mirror chain (first success wins):
#   1. BCO-DMO direct flat CSV (canonical export, exact flat schema):
#      https://datadocs.bco-dmo.org/dataset/773466/file/B11vA82u7y2Owp/global_bleaching_environmental.csv
#   2. GitHub author repo via codeload (works from CN networks where datadocs and
#      raw.githubusercontent.com time out):
#      https://codeload.github.com/rvanwoesik/Global-Coral-Bleaching-Database/zip/refs/heads/main
#      -> Global_Coral_Bleaching_Database_SQLite_11_23_21.zip -> flatten via SQL join.
# A cached copy of either artifact in CACHE_DIR short-circuits the download.
#
# Availability criteria (documented, applied uniformly to all regions):
#   usable_geo_year : valid lat/lon AND valid Date_Year
#   usable_severity : usable_geo_year AND severity informative
#                     (Severity_Code in {1,2,3} or Percent_Bleached valid)
#   severe          : usable_severity AND Severity_Code == 3 (>50% bleached)
#
# Outputs:
#   data/gcbd_global_bleaching_environmental.csv   (flat, one row per Sample_ID)
#   results/gcbd_global_region_counts/gcbd_region_counts.csv
#   results/gcbd_global_region_counts/gcbd_by_year.csv
#   results/gcbd_global_region_counts/gcbd_download_manifest.json
#
# Run:
#   & 'D:\...\piecf_env\Scripts\python.exe' data/source_builders/download_gcbd_global.py

import hashlib
import io
import json
import sqlite3
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

LTTS_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = LTTS_ROOT / "data"
CACHE_DIR = DATA_DIR / "raw_sources"
OUT_DIR = LTTS_ROOT / "results" / "gcbd_global_region_counts"

BCODMO_CSV_URL = ("https://datadocs.bco-dmo.org/dataset/773466/file/"
                  "B11vA82u7y2Owp/global_bleaching_environmental.csv")
GITHUB_ZIP_URL = ("https://codeload.github.com/rvanwoesik/"
                  "Global-Coral-Bleaching-Database/zip/refs/heads/main")
INNER_SQLITE_NAME = "Global_Coral_Bleaching_Database_11_23_21.db"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
TIMEOUT = 180

FLAT_OUT = DATA_DIR / "gcbd_global_bleaching_environmental.csv"
CACHE_ZIP = CACHE_DIR / "gcbd_author_repo.zip"
CACHE_SQLITE = CACHE_DIR / "Global_Coral_Bleaching_Database_11_23_21.db"
CACHE_FLAT = CACHE_DIR / "global_bleaching_environmental.csv"

# Coarse regions of interest (lon_min, lon_max, lat_min, lat_max) — diagnostic
# grouping only, coarse boxes may overlap; never used for any model/analysis split.
REGIONS_OF_INTEREST = {
    "South China Sea": (105, 122, 0, 25),
    "Great Barrier Reef": (142, 154, -28, -10),
    "Coral Triangle": (95, 140, -12, 20),
    "Caribbean": (-95, -55, 5, 30),
    "Red Sea": (32, 44, 12, 30),
    "Arabian Gulf": (47, 58, 22, 30),
    "West Indian Ocean": (30, 80, -30, 12),
    "Tropical Eastern Pacific": (-110, -75, -10, 25),
    "South Pacific (Polynesia/Melanesia)": (150, 210, -30, -5),
    "Hawaii": (-165, -150, 15, 30),
    "Japan/Kuroshio": (122, 145, 20, 35),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def http_download(url: str, dest: Path) -> bool:
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(part, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        if part.stat().st_size == 0:
            raise RuntimeError("empty download")
        part.replace(dest)
        return True
    except Exception as e:
        print(f"  [download] {url} failed: {type(e).__name__}: {e}")
        if part.exists():
            part.unlink()
        return False


def extract_or_flatten() -> pd.DataFrame:
    """Ensure the SQLite DB is extracted from the cached repo zip, then flatten."""
    if not CACHE_SQLITE.exists():
        z = zipfile.ZipFile(CACHE_ZIP)
        inner = [n for n in z.namelist() if n.endswith(INNER_SQLITE_NAME)]
        if not inner:
            sys.exit(f"{INNER_SQLITE_NAME} not found inside repo zip")
        CACHE_SQLITE.write_bytes(z.read(inner[0]))
    return flatten_sqlite(CACHE_SQLITE)


def flatten_sqlite(db_path: Path) -> pd.DataFrame:
    con = sqlite3.connect(str(db_path))
    q = """
    SELECT
      s.Site_ID, s.Data_Source, s.Latitude_Degrees, s.Longitude_Degrees,
      o.Ocean_Name, r.Realm_Name, ec.Ecoregion_Name, c.Country_Name,
      s.State_Island_Province_Name, s.City_Town_Name, s.Site_Name,
      s.Distance_to_Shore, s.Exposure, s.Turbidity, s.Cyclone_Frequency,
      e.Reef_ID, e.Date_Day, e.Date_Month, e.Date_Year, e.Depth_m,
      b.Bleaching_Level, b.Percent_Bleached, b.Severity_Code,
      b.Bleaching_Prevalence_Score,
      env.ClimSST, env.Temperature_Kelvin, env.Temperature_Maximum,
      env.Windspeed, env.SSTA, env.SSTA_DHWMax, env.TSA, env.TSA_DHW,
      e.Sample_ID
    FROM Sample_Event_tbl e
    JOIN Site_Info_tbl s ON e.Site_ID = s.Site_ID
    LEFT JOIN Ocean_Name_LUT o ON s.Ocean_Name = o.Ocean_ID
    LEFT JOIN Realm_Name_LUT r ON s.Realm_Name = r.Realm_ID
    LEFT JOIN Ecoregion_Name_LUT ec ON s.Ecoregion_Name = ec.Ecoregion_ID
    LEFT JOIN Country_Name_LUT c ON s.Country_Name = c.Country_ID
    LEFT JOIN (
      SELECT Sample_ID,
             MAX(Bleaching_Level) AS Bleaching_Level,
             MAX(Percent_Bleached) AS Percent_Bleached,
             MAX(Severity_Code) AS Severity_Code,
             MAX(Bleaching_Prevalence_Score) AS Bleaching_Prevalence_Score
      FROM Bleaching_tbl GROUP BY Sample_ID
    ) b ON e.Sample_ID = b.Sample_ID
    LEFT JOIN Environmental_tbl env ON e.Sample_ID = env.Sample_ID
    """
    df = pd.read_sql_query(q, con)
    con.close()
    return df


def load_flat(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # canonical BCO-DMO export lacks a per-row severity code but has severity fields
    if "Severity_Code" not in df.columns:
        df["Severity_Code"] = pd.NA
    if "Sample_ID" not in df.columns:
        df["Sample_ID"] = df["Site_ID"].astype(str) + "_" + df["Date"].astype(str)
    return df


def availability(df: pd.DataFrame) -> pd.DataFrame:
    lat = pd.to_numeric(df["Latitude_Degrees"], errors="coerce")
    lon = pd.to_numeric(df["Longitude_Degrees"], errors="coerce")
    yr = pd.to_numeric(df["Date_Year"], errors="coerce")
    geo_year = lat.between(-90, 90) & lon.between(-180, 180) & yr.between(1980, 2020)
    sev = pd.to_numeric(df.get("Severity_Code"), errors="coerce")
    pct = pd.to_numeric(df.get("Percent_Bleached", df.get("Percent_Bleaching")),
                        errors="coerce")
    sev_info = sev.isin([1, 2, 3]) | pct.notna()
    df = df.copy()
    df["_year"] = yr
    df["_usable_geo_year"] = geo_year
    df["_usable_severity"] = geo_year & sev_info
    df["_severe"] = geo_year & (sev == 3)
    return df


def assign_region(lat, lon):
    for name, (lo_x, hi_x, lo_y, hi_y) in REGIONS_OF_INTEREST.items():
        if lo_x <= lon <= hi_x and lo_y <= lat <= hi_y:
            return name
    return "Other"


def region_counts(df: pd.DataFrame, group_col: str, top=None) -> pd.DataFrame:
    sub = df[df["_usable_geo_year"]]
    rows = []
    for key, g in sub.groupby(group_col, dropna=False):
        key = str(key)
        rows.append({
            group_col: key,
            "n_usable": len(g),
            "n_severity": int(g["_usable_severity"].sum()),
            "n_severe": int(g["_severe"].sum()),
        })
    out = pd.DataFrame(rows).sort_values("n_usable", ascending=False)
    if top:
        out = out.head(top)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": {}, "definition": {
        "usable_geo_year": "valid lat/lon AND 1980<=Date_Year<=2020",
        "usable_severity": "usable_geo_year AND (Severity_Code in {1,2,3} or "
                           "Percent_Bleached present)",
        "severe": "usable_severity AND Severity_Code==3 (>50% bleached)",
    }}

    # ---- acquire global flat table (local cache first, then network) ----
    used_source = None
    if FLAT_OUT.exists():
        print(f"[cache] {FLAT_OUT.name} already present — reusing")
        used_source = "existing data/gcbd_global_bleaching_environmental.csv"
        df = load_flat(FLAT_OUT)
    elif CACHE_FLAT.exists():
        print(f"[cache] {CACHE_FLAT.name} — using cached BCO-DMO flat CSV")
        used_source = str(CACHE_FLAT)
        df = load_flat(CACHE_FLAT)
    elif CACHE_SQLITE.exists() or CACHE_ZIP.exists():
        print("[cache] SQLite / repo zip cached — no network needed")
        df = extract_or_flatten()
        used_source = "GitHub rvanwoesik/Global-Coral-Bleaching-Database (SQLite, cached)"
        manifest["sources"]["sqlite_sha256"] = (
            sha256_file(CACHE_SQLITE) if CACHE_SQLITE.exists() else None)
    else:
        # mirror 1: BCO-DMO direct flat CSV
        print("[mirror 1] BCO-DMO direct flat CSV ...")
        if http_download(BCODMO_CSV_URL, CACHE_FLAT):
            used_source = BCODMO_CSV_URL
            df = load_flat(CACHE_FLAT)
            manifest["sources"]["bcodmo_csv_sha256"] = sha256_file(CACHE_FLAT)
        else:
            # mirror 2: GitHub author repo via codeload -> SQLite
            print("[mirror 2] GitHub codeload repo zip -> SQLite ...")
            if not http_download(GITHUB_ZIP_URL, CACHE_ZIP):
                sys.exit("All download mirrors failed.")
            df = extract_or_flatten()
            used_source = "GitHub rvanwoesik/Global-Coral-Bleaching-Database (SQLite)"
            manifest["sources"]["sqlite_sha256"] = sha256_file(CACHE_SQLITE)

    manifest["sources"]["used"] = used_source
    manifest["sources"]["bcodmo_csv_url"] = BCODMO_CSV_URL
    manifest["sources"]["github_zip_url"] = GITHUB_ZIP_URL
    print(f"Global GCBD rows (per Sample_ID): {len(df)}")

    df = availability(df)
    FLAT_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FLAT_OUT, index=False, encoding="utf-8")
    manifest["outputs"] = {"flat_csv_sha256": sha256_file(FLAT_OUT)}
    print(f"[out] {FLAT_OUT}  ({FLAT_OUT.stat().st_size/1e6:.1f} MB)")

    lat = pd.to_numeric(df["Latitude_Degrees"], errors="coerce")
    lon = pd.to_numeric(df["Longitude_Degrees"], errors="coerce")
    df["_region"] = [
        assign_region(a, b) if (pd.notna(a) and pd.notna(b)) else "Other"
        for a, b in zip(lat, lon)
    ]

    counts = region_counts(df, "_region").rename(columns={"_region": "region"})
    counts["group"] = "region_of_interest"
    for col, top in [("Ocean_Name", None), ("Realm_Name", None),
                     ("Ecoregion_Name", 30), ("Country_Name", 30)]:
        c = region_counts(df, col)
        c = c.rename(columns={col: "region"})
        c.insert(0, "group", col)
        counts = pd.concat([counts, c], ignore_index=True)
    counts.to_csv(OUT_DIR / "gcbd_region_counts.csv", index=False)
    print("[out] gcbd_region_counts.csv")

    yearly = (df[df["_usable_geo_year"]].groupby("_year")
              .agg(n_usable=("_usable_geo_year", "sum"),
                   n_severity=("_usable_severity", "sum"),
                   n_severe=("_severe", "sum"))
              .reset_index().rename(columns={"_year": "year"}))
    yearly.to_csv(OUT_DIR / "gcbd_by_year.csv", index=False)

    manifest["totals"] = {
        "rows_total": int(len(df)),
        "usable_geo_year": int(df["_usable_geo_year"].sum()),
        "usable_severity": int(df["_usable_severity"].sum()),
        "severe_severity_code3": int(df["_severe"].sum()),
    }
    with open(OUT_DIR / "gcbd_download_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n== Totals ==")
    for k, v in manifest["totals"].items():
        print(f"  {k}: {v}")
    print("\n== Regions of interest ==")
    print(counts[counts["group"] == "region_of_interest"].to_string(index=False))
    print("\n== Ocean ==")
    print(counts[counts["group"] == "Ocean_Name"].to_string(index=False))
    print("\n== Realm ==")
    print(counts[counts["group"] == "Realm_Name"].to_string(index=False))
    print(f"\nDone. Manifest: {OUT_DIR / 'gcbd_download_manifest.json'}")


if __name__ == "__main__":
    main()
