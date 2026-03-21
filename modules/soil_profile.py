import math

import numpy as np
import pandas as pd


def get_unit_labels(unit_system: str) -> dict:
    if unit_system == "SI":
        return {
            "length": "m",
            "unit_weight": "kN/m³",
            "cohesion": "kPa",
            "pressure": "kPa",
            "gamma_w": 9.81,
        }
    return {
        "length": "ft",
        "unit_weight": "pcf",
        "cohesion": "psf",
        "pressure": "psf",
        "gamma_w": 62.4,
    }


def get_soil_column_names(unit_system: str) -> dict:
    units = get_unit_labels(unit_system)
    return {
        "layer_name": "Layer Name",
        "thickness": f"Thk. ({units['length']})",
        "gamma_moist": f"γ moist ({units['unit_weight']})",
        "gamma_sat": f"γ sat ({units['unit_weight']})",
        "phi": "Phi (deg)",
        "c": f"c ({units['cohesion']})",
    }


def default_soil_df(unit_system: str) -> pd.DataFrame:
    cols = get_soil_column_names(unit_system)

    if unit_system == "SI":
        return pd.DataFrame(
            [
                {
                    cols["layer_name"]: "Layer 1",
                    cols["thickness"]: 5.0,
                    cols["gamma_moist"]: 18.0,
                    cols["gamma_sat"]: 20.0,
                    cols["phi"]: 30.0,
                    cols["c"]: 0.0,
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                cols["layer_name"]: "Layer 1",
                cols["thickness"]: 10.0,
                cols["gamma_moist"]: 110.0,
                cols["gamma_sat"]: 125.0,
                cols["phi"]: 30.0,
                cols["c"]: 0.0,
            }
        ]
    )


def clean_soil_df(df: pd.DataFrame, unit_system: str) -> pd.DataFrame:
    df = df.copy()
    cols = get_soil_column_names(unit_system)

    expected_cols = [
        cols["layer_name"],
        cols["thickness"],
        cols["gamma_moist"],
        cols["gamma_sat"],
        cols["phi"],
        cols["c"],
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[expected_cols]
    df = df.dropna(how="all")

    df[cols["layer_name"]] = df[cols["layer_name"]].fillna("").astype(str).str.strip()

    numeric_cols = [
        cols["thickness"],
        cols["gamma_moist"],
        cols["gamma_sat"],
        cols["phi"],
        cols["c"],
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)
    df.loc[df[cols["layer_name"]] == "", cols["layer_name"]] = "Layer"

    df = df.rename(
        columns={
            cols["layer_name"]: "Layer Name",
            cols["thickness"]: "Thickness",
            cols["gamma_moist"]: "Gamma moist",
            cols["gamma_sat"]: "Gamma sat",
            cols["phi"]: "Phi (deg)",
            cols["c"]: "c",
        }
    )

    df = df.reset_index(drop=True)
    df.insert(0, "Layer No.", range(1, len(df) + 1))
    return df


def build_width_array(b_min: float, b_max: float, b_inc: float) -> np.ndarray:
    n_steps = int(round((b_max - b_min) / b_inc))
    values = b_min + np.arange(n_steps + 1) * b_inc
    values = values[values <= b_max + 1e-9]
    return np.round(values, 10)


def get_layer_segments_within_interval(
    soil_df: pd.DataFrame,
    z_top: float,
    z_bottom: float,
) -> list[dict]:
    segments = []
    current_top = 0.0

    for _, row in soil_df.iterrows():
        layer_top = current_top
        layer_bottom = current_top + float(row["Thickness"])

        overlap_top = max(z_top, layer_top)
        overlap_bottom = min(z_bottom, layer_bottom)
        overlap = overlap_bottom - overlap_top

        if overlap > 0:
            segments.append(
                {
                    "Layer No.": int(row["Layer No."]),
                    "Layer Name": row["Layer Name"],
                    "z_top": overlap_top,
                    "z_bottom": overlap_bottom,
                    "thickness": overlap,
                    "Gamma moist": float(row["Gamma moist"]),
                    "Gamma sat": float(row["Gamma sat"]),
                    "Phi (deg)": float(row["Phi (deg)"]),
                    "c": float(row["c"]),
                }
            )

        current_top = layer_bottom
        if current_top >= z_bottom:
            break

    return segments


def effective_unit_weight_for_segment(
    seg_top: float,
    seg_bottom: float,
    gamma_moist: float,
    gamma_sat: float,
    groundwater_depth: float,
    gamma_w: float,
) -> float:
    if groundwater_depth <= seg_top:
        return gamma_sat - gamma_w

    if groundwater_depth >= seg_bottom:
        return gamma_moist

    above = groundwater_depth - seg_top
    below = seg_bottom - groundwater_depth
    gamma_sub = gamma_sat - gamma_w
    return (gamma_moist * above + gamma_sub * below) / (seg_bottom - seg_top)


def compute_effective_surcharge_at_base(
    soil_df: pd.DataFrame,
    df_depth: float,
    groundwater_depth: float,
    gamma_w: float,
) -> float:
    segments = get_layer_segments_within_interval(soil_df, 0.0, df_depth)
    q_eff = 0.0

    for seg in segments:
        gamma_eff = effective_unit_weight_for_segment(
            seg_top=seg["z_top"],
            seg_bottom=seg["z_bottom"],
            gamma_moist=seg["Gamma moist"],
            gamma_sat=seg["Gamma sat"],
            groundwater_depth=groundwater_depth,
            gamma_w=gamma_w,
        )
        q_eff += gamma_eff * seg["thickness"]

    return q_eff


def compute_weighted_avg_below_base(
    soil_df: pd.DataFrame,
    df_depth: float,
    avg_depth: float,
    groundwater_depth: float,
    gamma_w: float,
) -> tuple[float, float, float]:
    z_top = df_depth
    z_bottom = df_depth + avg_depth

    segments = get_layer_segments_within_interval(soil_df, z_top, z_bottom)
    if len(segments) == 0:
        raise ValueError("No soil exists below the footing base within the averaging depth.")

    total_h = sum(seg["thickness"] for seg in segments)
    if total_h <= 0:
        raise ValueError("Averaging depth below footing base is zero or invalid.")

    c_sum = 0.0
    tan_phi_sum = 0.0
    gamma_eff_sum = 0.0

    for seg in segments:
        h_i = seg["thickness"]
        phi_i_rad = math.radians(seg["Phi (deg)"])
        gamma_eff_i = effective_unit_weight_for_segment(
            seg_top=seg["z_top"],
            seg_bottom=seg["z_bottom"],
            gamma_moist=seg["Gamma moist"],
            gamma_sat=seg["Gamma sat"],
            groundwater_depth=groundwater_depth,
            gamma_w=gamma_w,
        )

        c_sum += seg["c"] * h_i
        tan_phi_sum += math.tan(phi_i_rad) * h_i
        gamma_eff_sum += gamma_eff_i * h_i

    c_av = c_sum / total_h
    phi_av_rad = math.atan(tan_phi_sum / total_h)
    phi_av_deg = math.degrees(phi_av_rad)
    gamma_av = gamma_eff_sum / total_h

    return c_av, phi_av_deg, gamma_av


def iterate_averaged_parameters(
    soil_df: pd.DataFrame,
    B: float,
    df_depth: float,
    groundwater_depth: float,
    gamma_w: float,
    wedge_method: str,
    tol_deg: float = 0.01,
    max_iter: int = 50,
) -> tuple[float, float, float, float, int]:
    base_segments = get_layer_segments_within_interval(soil_df, df_depth, df_depth + 1e-9)
    if len(base_segments) == 0:
        raise ValueError("Footing base is below the defined soil profile.")

    founding_phi_deg = base_segments[0]["Phi (deg)"]
    phi_guess_deg = founding_phi_deg

    for iteration in range(1, max_iter + 1):
        if wedge_method == "Terzaghi":
            alpha_deg = phi_guess_deg
        else:
            alpha_deg = 45.0 + phi_guess_deg / 2.0

        alpha_rad = math.radians(alpha_deg)
        avg_depth = 0.5 * B * math.tan(alpha_rad)

        if avg_depth <= 0:
            c_av, phi_av_deg, gamma_av = compute_weighted_avg_below_base(
                soil_df=soil_df,
                df_depth=df_depth,
                avg_depth=1e-6,
                groundwater_depth=groundwater_depth,
                gamma_w=gamma_w,
            )
        else:
            c_av, phi_av_deg, gamma_av = compute_weighted_avg_below_base(
                soil_df=soil_df,
                df_depth=df_depth,
                avg_depth=avg_depth,
                groundwater_depth=groundwater_depth,
                gamma_w=gamma_w,
            )

        if abs(phi_av_deg - phi_guess_deg) < tol_deg:
            final_avg_depth = avg_depth
            return c_av, phi_av_deg, gamma_av, final_avg_depth, iteration

        phi_guess_deg = phi_av_deg

    final_alpha_deg = 45.0 + phi_guess_deg / 2.0 if wedge_method == "Meyerhof" else phi_guess_deg
    final_avg_depth = 0.5 * B * math.tan(math.radians(final_alpha_deg))
    return c_av, phi_av_deg, gamma_av, final_avg_depth, max_iter
