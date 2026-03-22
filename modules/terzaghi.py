import math

import numpy as np
import pandas as pd

from modules.soil_profile import (
    compute_effective_surcharge_at_base,
    get_unit_labels,
    iterate_averaged_parameters,
)


def terzaghi_factors(phi_deg: float) -> tuple[float, float, float]:
    phi_rad = math.radians(phi_deg)

    if abs(phi_deg) < 1e-10:
        Nq = 1.0
        Nc = 5.7
        Ngamma = 0.0
        return Nc, Nq, Ngamma

    a_phi = math.exp(math.pi * (0.75 - phi_deg / 360.0) * math.tan(phi_rad))
    angle_rad = math.radians(45.0 + phi_deg / 2.0)
    Nq = (a_phi ** 2) / (2.0 * (math.cos(angle_rad) ** 2))
    Nc = (Nq - 1.0) / math.tan(phi_rad)

    sin_4phi = math.sin(math.radians(4.0 * phi_deg))
    denominator = 1.0 + 0.4 * sin_4phi
    Ngamma = 2.0 * (Nq + 1.0) * math.tan(phi_rad) / denominator

    return Nc, Nq, Ngamma


def _get_common_terzaghi_values(
    soil_df: pd.DataFrame,
    B: float,
    df_depth: float,
    groundwater_depth: float,
    wedge_method: str,
    unit_system: str,
) -> dict:
    units = get_unit_labels(unit_system)
    gamma_w = units["gamma_w"]

    q_eff = compute_effective_surcharge_at_base(
        soil_df=soil_df,
        df_depth=df_depth,
        groundwater_depth=groundwater_depth,
        gamma_w=gamma_w,
    )

    c_av, phi_av_deg, gamma_av, avg_depth, n_iter = iterate_averaged_parameters(
        soil_df=soil_df,
        B=float(B),
        df_depth=df_depth,
        groundwater_depth=groundwater_depth,
        gamma_w=gamma_w,
        wedge_method=wedge_method,
    )

    Nc, Nq, Ngamma = terzaghi_factors(phi_av_deg)

    return {
        "q_eff": q_eff,
        "c_av": c_av,
        "phi_av_deg": phi_av_deg,
        "gamma_av": gamma_av,
        "avg_depth": avg_depth,
        "n_iter": n_iter,
        "Nc": Nc,
        "Nq": Nq,
        "Ngamma": Ngamma,
    }


def _get_design_value(
    q_net_ult: float,
    design_framework: str,
    fs_value: float | None,
    phi_r_value: float | None,
) -> float:
    if design_framework == "ASD":
        return q_net_ult / float(fs_value)
    return float(phi_r_value) * q_net_ult


def calculate_terzaghi_strip_results(
    soil_df: pd.DataFrame,
    widths: np.ndarray,
    df_depth: float,
    groundwater_depth: float,
    wedge_method: str,
    design_framework: str,
    fs_value: float | None,
    phi_r_value: float | None,
    unit_system: str,
) -> pd.DataFrame:
    units = get_unit_labels(unit_system)
    results = []

    for B in widths:
        vals = _get_common_terzaghi_values(
            soil_df=soil_df,
            B=float(B),
            df_depth=df_depth,
            groundwater_depth=groundwater_depth,
            wedge_method=wedge_method,
            unit_system=unit_system,
        )

        q_net_ult = (
            vals["c_av"] * vals["Nc"]
            + vals["q_eff"] * vals["Nq"]
            + 0.5 * vals["gamma_av"] * float(B) * vals["Ngamma"]
        )

        q_design = _get_design_value(
            q_net_ult=q_net_ult,
            design_framework=design_framework,
            fs_value=fs_value,
            phi_r_value=phi_r_value,
        )

        results.append(
            {
                "Method": "Terzaghi",
                "Footing Shape": "Strip",
                f"B ({units['length']})": float(B),
                f"q' ({units['pressure']})": vals["q_eff"],
                f"c_av ({units['cohesion']})": vals["c_av"],
                "phi_av (deg)": vals["phi_av_deg"],
                f"gamma'_av ({units['unit_weight']})": vals["gamma_av"],
                f"H ({units['length']})": vals["avg_depth"],
                "Iterations": vals["n_iter"],
                "Nc": vals["Nc"],
                "Nq": vals["Nq"],
                "Ngamma": vals["Ngamma"],
                f"q_net_ult ({units['pressure']})": q_net_ult,
                f"q_design ({units['pressure']})": q_design,
            }
        )

    return pd.DataFrame(results)


def calculate_terzaghi_square_results(
    soil_df: pd.DataFrame,
    widths: np.ndarray,
    df_depth: float,
    groundwater_depth: float,
    wedge_method: str,
    design_framework: str,
    fs_value: float | None,
    phi_r_value: float | None,
    unit_system: str,
) -> pd.DataFrame:
    units = get_unit_labels(unit_system)
    results = []

    for B in widths:
        vals = _get_common_terzaghi_values(
            soil_df=soil_df,
            B=float(B),
            df_depth=df_depth,
            groundwater_depth=groundwater_depth,
            wedge_method=wedge_method,
            unit_system=unit_system,
        )

        q_net_ult = (
            1.3 * vals["c_av"] * vals["Nc"]
            + vals["q_eff"] * vals["Nq"]
            + 0.4 * vals["gamma_av"] * float(B) * vals["Ngamma"]
        )

        q_design = _get_design_value(
            q_net_ult=q_net_ult,
            design_framework=design_framework,
            fs_value=fs_value,
            phi_r_value=phi_r_value,
        )

        results.append(
            {
                "Method": "Terzaghi",
                "Footing Shape": "Square",
                f"B ({units['length']})": float(B),
                f"L ({units['length']})": float(B),
                "L/B": 1.0,
                f"q' ({units['pressure']})": vals["q_eff"],
                f"c_av ({units['cohesion']})": vals["c_av"],
                "phi_av (deg)": vals["phi_av_deg"],
                f"gamma'_av ({units['unit_weight']})": vals["gamma_av"],
                f"H ({units['length']})": vals["avg_depth"],
                "Iterations": vals["n_iter"],
                "Nc": vals["Nc"],
                "Nq": vals["Nq"],
                "Ngamma": vals["Ngamma"],
                f"q_net_ult ({units['pressure']})": q_net_ult,
                f"q_design ({units['pressure']})": q_design,
            }
        )

    return pd.DataFrame(results)


def calculate_terzaghi_rectangular_results(
    soil_df: pd.DataFrame,
    widths: np.ndarray,
    length_to_width_ratio: float,
    df_depth: float,
    groundwater_depth: float,
    wedge_method: str,
    design_framework: str,
    fs_value: float | None,
    phi_r_value: float | None,
    unit_system: str,
) -> pd.DataFrame:
    units = get_unit_labels(unit_system)
    results = []

    for B in widths:
        L = float(length_to_width_ratio) * float(B)
        B_over_L = float(B) / L

        vals = _get_common_terzaghi_values(
            soil_df=soil_df,
            B=float(B),
            df_depth=df_depth,
            groundwater_depth=groundwater_depth,
            wedge_method=wedge_method,
            unit_system=unit_system,
        )

        if abs(vals["Nc"]) < 1e-12:
            Fcs = 1.0
        else:
            Fcs = 1.0 + B_over_L * (vals["Nq"] / vals["Nc"])

        Fqs = 1.0 + B_over_L * math.tan(math.radians(vals["phi_av_deg"]))
        Fgammas = 1.0 - 0.4 * B_over_L

        q_net_ult = (
            vals["c_av"] * vals["Nc"] * Fcs
            + vals["q_eff"] * vals["Nq"] * Fqs
            + 0.5 * vals["gamma_av"] * float(B) * vals["Ngamma"] * Fgammas
        )

        q_design = _get_design_value(
            q_net_ult=q_net_ult,
            design_framework=design_framework,
            fs_value=fs_value,
            phi_r_value=phi_r_value,
        )

        results.append(
            {
                "Method": "Terzaghi",
                "Footing Shape": "Rectangular",
                f"B ({units['length']})": float(B),
                f"L ({units['length']})": L,
                "L/B": float(length_to_width_ratio),
                "B/L": B_over_L,
                "Fcs": Fcs,
                "Fqs": Fqs,
                "Fγs": Fgammas,
                f"q' ({units['pressure']})": vals["q_eff"],
                f"c_av ({units['cohesion']})": vals["c_av"],
                "phi_av (deg)": vals["phi_av_deg"],
                f"gamma'_av ({units['unit_weight']})": vals["gamma_av"],
                f"H ({units['length']})": vals["avg_depth"],
                "Iterations": vals["n_iter"],
                "Nc": vals["Nc"],
                "Nq": vals["Nq"],
                "Ngamma": vals["Ngamma"],
                f"q_net_ult ({units['pressure']})": q_net_ult,
                f"q_design ({units['pressure']})": q_design,
            }
        )

    return pd.DataFrame(results)

def calculate_terzaghi_circular_results(
    soil_df: pd.DataFrame,
    radii: np.ndarray,
    df_depth: float,
    groundwater_depth: float,
    wedge_method: str,
    design_framework: str,
    fs_value: float | None,
    phi_r_value: float | None,
    unit_system: str,
) -> pd.DataFrame:
    units = get_unit_labels(unit_system)
    results = []

    for R in radii:
        B = 2.0 * float(R)

        vals = _get_common_terzaghi_values(
            soil_df=soil_df,
            B=B,
            df_depth=df_depth,
            groundwater_depth=groundwater_depth,
            wedge_method=wedge_method,
            unit_system=unit_system,
        )

        q_net_ult = (
            1.3 * vals["c_av"] * vals["Nc"]
            + vals["q_eff"] * vals["Nq"]
            + 0.3 * vals["gamma_av"] * B * vals["Ngamma"]
        )

        q_design = _get_design_value(
            q_net_ult=q_net_ult,
            design_framework=design_framework,
            fs_value=fs_value,
            phi_r_value=phi_r_value,
        )

        results.append(
            {
                "Method": "Terzaghi",
                "Footing Shape": "Circular",
                f"R ({units['length']})": float(R),
                f"B=2R ({units['length']})": B,
                f"q' ({units['pressure']})": vals["q_eff"],
                f"c_av ({units['cohesion']})": vals["c_av"],
                "phi_av (deg)": vals["phi_av_deg"],
                f"gamma'_av ({units['unit_weight']})": vals["gamma_av"],
                f"H ({units['length']})": vals["avg_depth"],
                "Iterations": vals["n_iter"],
                "Nc": vals["Nc"],
                "Nq": vals["Nq"],
                "Ngamma": vals["Ngamma"],
                f"q_net_ult ({units['pressure']})": q_net_ult,
                f"q_design ({units['pressure']})": q_design,
            }
        )

    return pd.DataFrame(results)
