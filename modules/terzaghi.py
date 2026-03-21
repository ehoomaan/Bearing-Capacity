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
    gamma_w = units["gamma_w"]
    results = []

    q_eff = compute_effective_surcharge_at_base(
        soil_df=soil_df,
        df_depth=df_depth,
        groundwater_depth=groundwater_depth,
        gamma_w=gamma_w,
    )

    for B in widths:
        c_av, phi_av_deg, gamma_av, avg_depth, n_iter = iterate_averaged_parameters(
            soil_df=soil_df,
            B=float(B),
            df_depth=df_depth,
            groundwater_depth=groundwater_depth,
            gamma_w=gamma_w,
            wedge_method=wedge_method,
        )

        Nc, Nq, Ngamma = terzaghi_factors(phi_av_deg)

        q_net_ult = c_av * Nc + q_eff * Nq + 0.5 * gamma_av * float(B) * Ngamma

        if design_framework == "ASD":
            q_design = q_net_ult / float(fs_value)
        else:
            q_design = float(phi_r_value) * q_net_ult

        results.append(
            {
                f"B ({units['length']})": float(B),
                f"q' ({units['pressure']})": q_eff,
                f"c_av ({units['cohesion']})": c_av,
                "phi_av (deg)": phi_av_deg,
                f"gamma'_av ({units['unit_weight']})": gamma_av,
                f"H ({units['length']})": avg_depth,
                "Iterations": n_iter,
                "Nc": Nc,
                "Nq": Nq,
                "Ngamma": Ngamma,
                f"q_net_ult ({units['pressure']})": q_net_ult,
                f"q_design ({units['pressure']})": q_design,
            }
        )

    return pd.DataFrame(results)
