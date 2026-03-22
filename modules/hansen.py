import math

import numpy as np
import pandas as pd

from modules.soil_profile import (
    compute_effective_surcharge_at_base,
    get_unit_labels,
    iterate_averaged_parameters,
)


def hansen_factors(phi_deg: float) -> tuple[float, float, float]:
    phi_rad = math.radians(phi_deg)

    if abs(phi_deg) < 1e-10:
        Nq = 1.0
        Nc = 5.14
        Ngamma = 0.0
        return Nc, Nq, Ngamma

    Nq = math.exp(math.pi * math.tan(phi_rad)) * (
        math.tan(math.radians(45.0 + phi_deg / 2.0)) ** 2
    )
    Nc = (Nq - 1.0) / math.tan(phi_rad)
    Ngamma = 1.5 * (Nq - 1.0) * math.tan(phi_rad)

    return Nc, Nq, Ngamma


def _get_common_hansen_values(
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

    Nc, Nq, Ngamma = hansen_factors(phi_av_deg)

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


def _hansen_shape_factors(phi_deg: float, Nc: float, Nq: float, B_over_L: float) -> tuple[float, float, float]:
    phi_rad = math.radians(phi_deg)

    if abs(Nc) < 1e-12:
        sc = 1.0
    else:
        sc = 1.0 + B_over_L * (Nq / Nc)

    sq = 1.0 + B_over_L * math.sin(phi_rad)
    sgamma = 1.0 - 0.4 * B_over_L

    return sc, sq, sgamma


def _hansen_depth_factors(phi_deg: float, Nc: float, Df: float, B: float) -> tuple[float, float, float]:
    Df_over_B = Df / B

    if abs(phi_deg) < 1e-10:
        if Df_over_B <= 1.0:
            dc = 1.0 + 0.4 * Df_over_B
        else:
            dc = 1.0 + 0.4 * math.atan(Df_over_B)  # radians
        dq = 1.0
        dgamma = 1.0
        return dc, dq, dgamma

    phi_rad = math.radians(phi_deg)
    sin_phi = math.sin(phi_rad)
    tan_phi = math.tan(phi_rad)

    if Df_over_B <= 1.0:
        dq = 1.0 + 2.0 * tan_phi * ((1.0 - sin_phi) ** 2) * Df_over_B
    else:
        dq = 1.0 + 2.0 * tan_phi * ((1.0 - sin_phi) ** 2) * math.atan(Df_over_B)

    dc = dq - (1.0 - dq) / (Nc * tan_phi)
    dgamma = 1.0

    return dc, dq, dgamma


def _hansen_ground_factors(phi_deg: float, ground_angle_deg: float) -> tuple[float, float, float]:
    beta_rad = math.radians(ground_angle_deg)

    if abs(phi_deg) < 1e-10:
        gc = 1.0 - ground_angle_deg / 147.0
    else:
        gc = 1.0 - ground_angle_deg / 147.0

    common = (1.0 - 0.5 * math.tan(beta_rad)) ** 5
    gq = common
    ggamma = common

    return gc, gq, ggamma


def _hansen_base_factors(phi_deg: float, base_angle_deg: float) -> tuple[float, float, float]:
    eta_rad = math.radians(base_angle_deg)

    if abs(phi_deg) < 1e-10:
        bc = 1.0 - base_angle_deg / 147.0
        bq = 1.0
        bgamma = 1.0
        return bc, bq, bgamma

    phi_rad = math.radians(phi_deg)

    bc = 1.0 - base_angle_deg / 147.0
    bq = math.exp(-2.0 * eta_rad * math.tan(phi_rad))
    bgamma = math.exp(-2.7 * eta_rad * math.tan(phi_rad))

    return bc, bq, bgamma


def _hansen_load_inclination_factors() -> tuple[float, float, float]:
    return 1.0, 1.0, 1.0


def _calculate_hansen_results_for_shape(
    soil_df: pd.DataFrame,
    widths: np.ndarray,
    footing_shape: str,
    df_depth: float,
    groundwater_depth: float,
    wedge_method: str,
    design_framework: str,
    fs_value: float | None,
    phi_r_value: float | None,
    unit_system: str,
    base_angle: float,
    ground_angle: float,
    length_to_width_ratio: float | None = None,
    radii_mode: bool = False,
) -> pd.DataFrame:
    units = get_unit_labels(unit_system)
    results = []

    for value in widths:
        if radii_mode:
            R = float(value)
            B = 2.0 * R
        else:
            B = float(value)
            R = None

        vals = _get_common_hansen_values(
            soil_df=soil_df,
            B=B,
            df_depth=df_depth,
            groundwater_depth=groundwater_depth,
            wedge_method=wedge_method,
            unit_system=unit_system,
        )

        if footing_shape == "Strip":
            B_over_L = 0.0
            L = None
            L_over_B = None
        elif footing_shape == "Square":
            B_over_L = 1.0
            L = B
            L_over_B = 1.0
        elif footing_shape == "Rectangular":
            L_over_B = float(length_to_width_ratio)
            L = L_over_B * B
            B_over_L = B / L
        elif footing_shape == "Circular":
            B_over_L = 1.0  # equivalent square
            L = B
            L_over_B = 1.0
        else:
            raise ValueError(f"Unsupported footing shape: {footing_shape}")

        sc, sq, sgamma = _hansen_shape_factors(
            phi_deg=vals["phi_av_deg"],
            Nc=vals["Nc"],
            Nq=vals["Nq"],
            B_over_L=B_over_L,
        )

        dc, dq, dgamma = _hansen_depth_factors(
            phi_deg=vals["phi_av_deg"],
            Nc=vals["Nc"],
            Df=df_depth,
            B=B,
        )

        gc, gq, ggamma = _hansen_ground_factors(
            phi_deg=vals["phi_av_deg"],
            ground_angle_deg=ground_angle,
        )

        bc, bq, bgamma = _hansen_base_factors(
            phi_deg=vals["phi_av_deg"],
            base_angle_deg=base_angle,
        )

        ic, iq, igamma = _hansen_load_inclination_factors()

        q_net_ult = (
            vals["c_av"] * vals["Nc"] * sc * dc * ic * gc * bc
            + vals["q_eff"] * vals["Nq"] * sq * dq * iq * gq * bq
            + 0.5 * vals["gamma_av"] * B * vals["Ngamma"] * sgamma * dgamma * igamma * ggamma * bgamma
        )

        q_design = _get_design_value(
            q_net_ult=q_net_ult,
            design_framework=design_framework,
            fs_value=fs_value,
            phi_r_value=phi_r_value,
        )

        row = {
            "Method": "Hansen",
            "Footing Shape": footing_shape,
            f"q' ({units['pressure']})": vals["q_eff"],
            f"c_av ({units['cohesion']})": vals["c_av"],
            "phi_av (deg)": vals["phi_av_deg"],
            f"gamma'_av ({units['unit_weight']})": vals["gamma_av"],
            f"H ({units['length']})": vals["avg_depth"],
            "Iterations": vals["n_iter"],
            "Nc": vals["Nc"],
            "Nq": vals["Nq"],
            "Ngamma": vals["Ngamma"],
            "sc": sc,
            "sq": sq,
            "sgamma": sgamma,
            "dc": dc,
            "dq": dq,
            "dgamma": dgamma,
            "ic": ic,
            "iq": iq,
            "igamma": igamma,
            "gc": gc,
            "gq": gq,
            "ggamma": ggamma,
            "bc": bc,
            "bq": bq,
            "bgamma": bgamma,
            f"q_net_ult ({units['pressure']})": q_net_ult,
            f"q_design ({units['pressure']})": q_design,
        }

        if radii_mode:
            row[f"R ({units['length']})"] = R
            row[f"B=2R ({units['length']})"] = B
            row[f"L ({units['length']})"] = L
            row["L/B"] = L_over_B
        else:
            row[f"B ({units['length']})"] = B
            if L is not None:
                row[f"L ({units['length']})"] = L
                row["L/B"] = L_over_B
                row["B/L"] = B_over_L

        results.append(row)

    return pd.DataFrame(results)


def calculate_hansen_strip_results(
    soil_df: pd.DataFrame,
    widths: np.ndarray,
    df_depth: float,
    groundwater_depth: float,
    wedge_method: str,
    design_framework: str,
    fs_value: float | None,
    phi_r_value: float | None,
    unit_system: str,
    base_angle: float,
    ground_angle: float,
) -> pd.DataFrame:
    return _calculate_hansen_results_for_shape(
        soil_df=soil_df,
        widths=widths,
        footing_shape="Strip",
        df_depth=df_depth,
        groundwater_depth=groundwater_depth,
        wedge_method=wedge_method,
        design_framework=design_framework,
        fs_value=fs_value,
        phi_r_value=phi_r_value,
        unit_system=unit_system,
        base_angle=base_angle,
        ground_angle=ground_angle,
    )


def calculate_hansen_square_results(
    soil_df: pd.DataFrame,
    widths: np.ndarray,
    df_depth: float,
    groundwater_depth: float,
    wedge_method: str,
    design_framework: str,
    fs_value: float | None,
    phi_r_value: float | None,
    unit_system: str,
    base_angle: float,
    ground_angle: float,
) -> pd.DataFrame:
    return _calculate_hansen_results_for_shape(
        soil_df=soil_df,
        widths=widths,
        footing_shape="Square",
        df_depth=df_depth,
        groundwater_depth=groundwater_depth,
        wedge_method=wedge_method,
        design_framework=design_framework,
        fs_value=fs_value,
        phi_r_value=phi_r_value,
        unit_system=unit_system,
        base_angle=base_angle,
        ground_angle=ground_angle,
    )


def calculate_hansen_rectangular_results(
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
    base_angle: float,
    ground_angle: float,
) -> pd.DataFrame:
    return _calculate_hansen_results_for_shape(
        soil_df=soil_df,
        widths=widths,
        footing_shape="Rectangular",
        df_depth=df_depth,
        groundwater_depth=groundwater_depth,
        wedge_method=wedge_method,
        design_framework=design_framework,
        fs_value=fs_value,
        phi_r_value=phi_r_value,
        unit_system=unit_system,
        base_angle=base_angle,
        ground_angle=ground_angle,
        length_to_width_ratio=length_to_width_ratio,
    )


def calculate_hansen_circular_results(
    soil_df: pd.DataFrame,
    radii: np.ndarray,
    df_depth: float,
    groundwater_depth: float,
    wedge_method: str,
    design_framework: str,
    fs_value: float | None,
    phi_r_value: float | None,
    unit_system: str,
    base_angle: float,
    ground_angle: float,
) -> pd.DataFrame:
    return _calculate_hansen_results_for_shape(
        soil_df=soil_df,
        widths=radii,
        footing_shape="Circular",
        df_depth=df_depth,
        groundwater_depth=groundwater_depth,
        wedge_method=wedge_method,
        design_framework=design_framework,
        fs_value=fs_value,
        phi_r_value=phi_r_value,
        unit_system=unit_system,
        base_angle=base_angle,
        ground_angle=ground_angle,
        radii_mode=True,
    )
