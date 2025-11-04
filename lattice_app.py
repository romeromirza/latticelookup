
import re
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Lattice Info Lookup", layout="centered")
st.title("Lattice Volume Filling Fraction and Beam Radius")
st.caption("Simply choose a lattice type and desired permittivity. Volume filling fraction is independent of lattice type.")

@st.cache_data
def load_tables():
    base = Path(__file__).parent
    tables = {}
    for name, fname in {
        "octet": "octet_info.csv",
        "diamond": "diamond_info.csv",
        "fluorite": "fluorite_info.csv",
        "gyroid": "gyroid_info.csv",
    }.items():
        df = pd.read_csv(base / fname)
        # Extract size columns and sort by the numeric size
        size_cols = []
        for c in df.columns:
            m = re.match(r"\[(\d+\.?\d*)mm cell\] Beam radius \(mm\)", str(c))
            if m:
                size_cols.append((float(m.group(1)), c))
        size_cols.sort(key=lambda x: x[0])
        tables[name] = {
            "df": df.sort_values("Er").reset_index(drop=True),
            "sizes": [s for s, _ in size_cols],
            "size_cols": [c for _, c in size_cols],
            "er_min": float(df["Er"].min()),
            "er_max": float(df["Er"].max()),
        }
    return tables

def linear_interp(x, x1, y1, x2, y2):
    if x2 == x1:
        return float(y1)
    t = (x - x1) / (x2 - x1)
    return float((1 - t) * y1 + t * y2)

def neighbors(sorted_vals, x):
    # assumes sorted ascending
    if x <= sorted_vals[0]:
        return sorted_vals[0], sorted_vals[0]
    if x >= sorted_vals[-1]:
        return sorted_vals[-1], sorted_vals[-1]
    # find bracketing values
    for i in range(len(sorted_vals) - 1):
        if sorted_vals[i] <= x <= sorted_vals[i + 1]:
            return sorted_vals[i], sorted_vals[i + 1]
    return sorted_vals[-1], sorted_vals[-1]

def interpolate_vf(tbl, er):
    df = tbl["df"]
    ers = df["Er"].to_numpy()
    vfs = df["Vf"].to_numpy()
    e1, e2 = neighbors(list(ers), er)
    # get v1, v2
    v1 = float(df.loc[df["Er"] == e1, "Vf"].iloc[0])
    v2 = float(df.loc[df["Er"] == e2, "Vf"].iloc[0])
    return linear_interp(er, e1, v1, e2, v2)

def interpolate_beam_radius(tbl, er, cell_size):
    df = tbl["df"]
    sizes = tbl["sizes"]
    size_cols = tbl["size_cols"]
    ers = df["Er"].to_numpy()

    # bracket along Er and size
    e1, e2 = neighbors(list(ers), er)
    s1, s2 = neighbors(sizes, cell_size)

    # fetch corner values
    row_e1 = df.index[df["Er"] == e1][0]
    row_e2 = df.index[df["Er"] == e2][0]
    col_s1 = size_cols[sizes.index(s1)]
    col_s2 = size_cols[sizes.index(s2)]

    r11 = float(df.loc[row_e1, col_s1])
    r12 = float(df.loc[row_e1, col_s2])
    r21 = float(df.loc[row_e2, col_s1])
    r22 = float(df.loc[row_e2, col_s2])

    # handle degenerate edges
    if e1 == e2 and s1 == s2:
        return r11
    if e1 == e2:
        return linear_interp(cell_size, s1, r11, s2, r12)
    if s1 == s2:
        return linear_interp(er, e1, r11, e2, r21)

    # bilinear interpolation
    # first interpolate along size at e1 and e2
    r1 = linear_interp(cell_size, s1, r11, s2, r12)
    r2 = linear_interp(cell_size, s1, r21, s2, r22)
    # then interpolate between e1 and e2
    return linear_interp(er, e1, r1, e2, r2)

tables = load_tables()

with st.sidebar:
    st.subheader("Inputs")
    lattice = st.selectbox("Lattice type", list(tables.keys()), index=0, format_func=lambda x: x.capitalize())
    er_min = tables[lattice]["er_min"]
    er_max = tables[lattice]["er_max"]
    er = st.number_input("Permittivity (εr)", min_value=float(er_min), max_value=float(er_max),
                         value=float(np.clip(1.5, er_min, er_max)), step=0.01, format="%.2f")
    cell_size = st.slider("Unit cell size (mm)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)

tbl = tables[lattice]
vf = interpolate_vf(tbl, er)
br = interpolate_beam_radius(tbl, er, cell_size)

st.markdown(f"### Results for **{lattice.capitalize()}**")
c1, c2 = st.columns(2)
with c1:
    st.metric("Filling fraction (0-1)", f"{vf:.3f}")
with c2:
    st.metric("Beam radius (mm)", f"{br:.3f}")

st.divider()
with st.expander("Details and ranges"):
    st.write(f"εr range in data: {tbl['er_min']:.2f} to {tbl['er_max']:.2f}")
    st.write("Unit cell size range: 1.0 to 10.0 mm in 0.5 mm increments in the tables. Values are linearly interpolated.")

st.caption("Note: εr outside the table range is clamped to the nearest available values before interpolation.")
