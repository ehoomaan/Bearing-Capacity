import math
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Bearing Capacity Calculator", layout="wide")


def get_unit_labels(unit_system: str) -> dict:
    if unit_system == "SI":
        return {
            "length": "m",
            "unit_weight": "kN/m³",
            "cohesion": "kPa",
        }
    return {
        "length": "ft",
        "unit_weight": "pcf",
        "cohesion": "psf",
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

    # Remove fully blank rows
    df = df.dropna(how="all")

    # Clean text column
    df[cols["layer_name"]] = df[cols["layer_name"]].fillna("").astype(str).str.strip()

    # Convert numeric columns
    numeric_cols = [
        cols["thickness"],
        cols["gamma_moist"],
        cols["gamma_sat"],
        cols["phi"],
        cols["c"],
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep rows only when all numeric fields are provided
    df = df.dropna(subset=numeric_cols)

    # Give a default name to unnamed but otherwise valid rows
    df.loc[df[cols["layer_name"]] == "", cols["layer_name"]] = "Layer"

    # Rename to stable internal names
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


st.title("Shallow Foundation Bearing Capacity Calculator")

with st.sidebar:
    st.header("Project Settings")
    unit_system = st.selectbox("Unit system", ["SI", "USCS"], index=0)
    design_framework = st.selectbox("Design framework", ["ASD", "LRFD"], index=0)

col1, col2 = st.columns([1.35, 0.95])

with col1:
    with st.expander("Project Information", expanded=False):
        project_name = st.text_input("Project name")
        project_location = st.text_input("Project location")
        prepared_by = st.text_input("Prepared by")
        project_date = st.date_input("Date", value=date.today())

    with st.expander("Soil Properties", expanded=True):
        groundwater_depth = st.number_input(
            "Groundwater depth below ground surface",
            min_value=0.0,
            value=10.0,
            step=0.5,
        )

        st.write("Soil layers")
        soil_editor_df = st.data_editor(
            default_soil_df(unit_system),
            num_rows="dynamic",
            use_container_width=True,
            key=f"soil_table_{unit_system}",
        )

        cleaned_soil_df = clean_soil_df(soil_editor_df, unit_system)

with col2:
    with st.expander("Footing Properties", expanded=True):
        foot_col1, foot_col2 = st.columns([1.35, 0.85])

        with foot_col1:
            footing_shape = st.selectbox("Footing shape", ["Strip", "Rectangular", "Circular"])

            df_depth = st.number_input("Embedment depth, Df", min_value=0.0, value=1.5, step=0.1)
            base_angle = st.number_input("Base inclination angle (deg)", min_value=0.0, value=0.0, step=0.5)
            ground_angle = st.number_input("Ground inclination angle (deg)", min_value=0.0, value=0.0, step=0.5)
            load_angle = st.number_input("Load inclination angle (deg)", min_value=0.0, value=0.0, step=0.5)

            b_min = b_max = b_inc = None
            length_l = None
            r_min = r_max = r_inc = None

            if footing_shape in ["Strip", "Rectangular"]:
                b_min = st.number_input("B_min", min_value=0.01, value=1.0, step=0.1)
                b_max = st.number_input("B_max", min_value=0.01, value=3.0, step=0.1)
                b_inc = st.number_input("B_increment", min_value=0.01, value=0.5, step=0.1)

            if footing_shape == "Rectangular":
                length_l = st.number_input("L", min_value=0.01, value=3.0, step=0.1)

            if footing_shape == "Circular":
                r_min = st.number_input("R_min", min_value=0.01, value=0.5, step=0.1)
                r_max = st.number_input("R_max", min_value=0.01, value=1.5, step=0.1)
                r_inc = st.number_input("R_increment", min_value=0.01, value=0.25, step=0.1)

        with foot_col2:
            st.markdown("**Geometry sketch**")
            st.info("Reserved area for footing geometry image.")

    with st.expander("Analysis Options", expanded=True):
        selected_methods = st.multiselect(
            "Bearing-capacity method(s)",
            ["Terzaghi", "Vesic", "Hansen", "Lyamin", "Loukidis and Salgado"],
            default=["Terzaghi"],
        )

        wedge_method = st.selectbox("Failure-wedge averaging method", ["Terzaghi", "Meyerhof"])

        fs_value = None
        phi_r_value = None

        if design_framework == "ASD":
            fs_value = st.number_input("Factor of safety", min_value=0.1, value=3.0, step=0.1)
        else:
            phi_r_value = st.number_input("Resistance factor", min_value=0.01, value=0.5, step=0.05)

run_analysis = st.button("Run Analysis", type="primary")

if run_analysis:
    errors = validate_inputs(
        soil_df=cleaned_soil_df,
        footing_shape=footing_shape,
        df_depth=df_depth,
        b_min=b_min,
        b_max=b_max,
        b_inc=b_inc,
        length_l=length_l,
        r_min=r_min,
        r_max=r_max,
        r_inc=r_inc,
        design_framework=design_framework,
        fs_value=fs_value,
        phi_r_value=phi_r_value,
        selected_methods=selected_methods,
    )

    if errors:
        st.error("Please fix the following issues before running the analysis:")
        for err in errors:
            st.write(f"- {err}")
    else:
        st.success("Input collection is working correctly.")
        st.subheader("Collected Input Data")

        input_summary = {
            "unit_system": unit_system,
            "design_framework": design_framework,
            "project_name": project_name,
            "project_location": project_location,
            "prepared_by": prepared_by,
            "project_date": str(project_date),
            "groundwater_depth": groundwater_depth,
            "footing_shape": footing_shape,
            "Df": df_depth,
            "base_angle": base_angle,
            "ground_angle": ground_angle,
            "load_angle": load_angle,
            "methods": selected_methods,
            "wedge_method": wedge_method,
            "FS": fs_value,
            "phi_r": phi_r_value,
            "B_min": b_min,
            "B_max": b_max,
            "B_increment": b_inc,
            "L": length_l,
            "R_min": r_min,
            "R_max": r_max,
            "R_increment": r_inc,
        }

        st.json(input_summary)
        st.dataframe(cleaned_soil_df, use_container_width=True)
