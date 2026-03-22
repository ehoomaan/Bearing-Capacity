from datetime import date
from pathlib import Path

import streamlit as st

from modules.soil_profile import (
    build_width_array,
    clean_soil_df,
    default_soil_df,
)
from modules.terzaghi import calculate_terzaghi_strip_results
from modules.validation import validate_inputs

def get_static_geometry_image() -> str | None:
    image_path = Path("assets/foundation_geometry.png")
    if image_path.exists():
        return str(image_path)
    return None
    
st.set_page_config(page_title="Bearing Capacity App", layout="wide")

st.title("Bearing Capacity App")

with st.sidebar:
    st.header("Global Settings")
    unit_system = st.selectbox("Unit system", ["SI", "USCS"], index=0)
    design_framework = st.selectbox("Design framework", ["ASD", "LRFD"], index=0)

main_col1, main_col2 = st.columns([1.45, 1.0])

with main_col1:
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
    
    st.markdown("**Geometry Sketch**")
    image_path = get_static_geometry_image()
    if image_path:
        st.image(image_path, use_container_width=True)
    else:
        st.info("Geometry image not found.")

with main_col2:
    right_col1, right_col2 = st.columns([1.0, 1.05])

    with right_col1:
        with st.expander("Footing Properties", expanded=True):
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

    with right_col2:
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

        if footing_shape == "Strip" and selected_methods == ["Terzaghi"]:
            widths = build_width_array(b_min, b_max, b_inc)
            results_df = calculate_terzaghi_strip_results(
                soil_df=cleaned_soil_df,
                widths=widths,
                df_depth=df_depth,
                groundwater_depth=groundwater_depth,
                wedge_method=wedge_method,
                design_framework=design_framework,
                fs_value=fs_value,
                phi_r_value=phi_r_value,
                unit_system=unit_system,
            )

            st.subheader("Terzaghi Strip Footing Results")
            st.dataframe(results_df, use_container_width=True)
            st.info("This step displays the Terzaghi strip-footing results table only. Plotting will be added next.")
        else:
            st.warning("This step supports only the Terzaghi method with strip footing.")
