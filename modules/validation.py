import pandas as pd


def validate_inputs(
    soil_df: pd.DataFrame,
    footing_shape: str,
    df_depth: float,
    b_min: float | None,
    b_max: float | None,
    b_inc: float | None,
    length_l: float | None,
    r_min: float | None,
    r_max: float | None,
    r_inc: float | None,
    design_framework: str,
    fs_value: float | None,
    phi_r_value: float | None,
    selected_methods: list[str],
) -> list[str]:
    errors = []

    if soil_df.empty:
        errors.append("At least one valid soil layer is required.")

    if df_depth < 0:
        errors.append("Embedment depth Df must be greater than or equal to zero.")

    if len(selected_methods) == 0:
        errors.append("Select at least one bearing-capacity method.")

    if design_framework == "ASD":
        if fs_value is None or fs_value <= 0:
            errors.append("Factor of safety must be greater than zero for ASD.")
    else:
        if phi_r_value is None or phi_r_value <= 0:
            errors.append("Resistance factor must be greater than zero for LRFD.")

    if footing_shape in ["Strip", "Rectangular"]:
        if b_min is None or b_max is None or b_inc is None:
            errors.append("Width range inputs are required.")
        else:
            if b_min <= 0 or b_max <= 0 or b_inc <= 0:
                errors.append("B_min, B_max, and B_increment must be greater than zero.")
            if b_max < b_min:
                errors.append("B_max must be greater than or equal to B_min.")

    if footing_shape == "Rectangular":
        if length_l is None or length_l <= 0:
            errors.append("Length L must be greater than zero for rectangular footing.")
        elif b_max is not None and length_l < b_max:
            errors.append("For rectangular footing, L should be greater than or equal to the largest B value.")

    if footing_shape == "Circular":
        if r_min is None or r_max is None or r_inc is None:
            errors.append("Radius range inputs are required.")
        else:
            if r_min <= 0 or r_max <= 0 or r_inc <= 0:
                errors.append("R_min, R_max, and R_increment must be greater than zero.")
            if r_max < r_min:
                errors.append("R_max must be greater than or equal to R_min.")

    if not soil_df.empty:
        for _, row in soil_df.iterrows():
            if row["Thickness"] <= 0:
                errors.append(f"Layer {int(row['Layer No.'])}: Thickness must be greater than zero.")
            if row["Gamma moist"] <= 0:
                errors.append(f"Layer {int(row['Layer No.'])}: Unit weight must be greater than zero.")
            if row["Gamma sat"] <= 0:
                errors.append(f"Layer {int(row['Layer No.'])}: Saturated unit weight must be greater than zero.")
            if row["Phi (deg)"] < 0:
                errors.append(f"Layer {int(row['Layer No.'])}: Phi must be greater than or equal to zero.")
            if row["c"] < 0:
                errors.append(f"Layer {int(row['Layer No.'])}: Cohesion must be greater than or equal to zero.")

    return errors
