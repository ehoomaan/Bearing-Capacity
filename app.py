from datetime import date
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit as st

from modules.soil_profile import (
    build_width_array,
    clean_soil_df,
    default_soil_df,
)

from modules.terzaghi import (
    calculate_terzaghi_rectangular_results,
    calculate_terzaghi_square_results,
    calculate_terzaghi_strip_results,
    calculate_terzaghi_circular_results,
)
from modules.validation import validate_inputs
def get_plot_columns(results_df, footing_shape: str, unit_system: str) -> tuple[str, str]:
    length_unit = "m" if unit_system == "SI" else "ft"
    pressure_unit = "kPa" if unit_system == "SI" else "psf"

    if footing_shape == "Circular":
        x_col = f"R ({length_unit})"
    else:
        x_col = f"B ({length_unit})"

    y_col = f"q_design ({pressure_unit})"
    return x_col, y_col


def get_axis_labels(footing_shape: str, design_framework: str, unit_system: str) -> tuple[str, str]:
    length_unit = "m" if unit_system == "SI" else "ft"
    pressure_unit = "kPa" if unit_system == "SI" else "psf"

    if footing_shape == "Circular":
        x_label = f"Footing Radius, R ({length_unit})"
    else:
        x_label = f"Footing Width, B ({length_unit})"

    if design_framework == "ASD":
        y_label = f"Net Allowable Bearing Pressure ({pressure_unit})"
    else:
        y_label = f"Net Factored Bearing Resistance ({pressure_unit})"

    return x_label, y_label


def plot_results(results_df, footing_shape: str, design_framework: str, unit_system: str):
    x_col, y_col = get_plot_columns(results_df, footing_shape, unit_system)
    x_label, y_label = get_axis_labels(footing_shape, design_framework, unit_system)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(results_df[x_col], results_df[y_col], marker="o")
    ax.set_xlabel(x_label,fontsize=8)
    ax.tick_params(axis='x', labelsize=10)
    ax.set_ylabel(y_label,fontsize=8)
    ax.set_title("Bearing Capacity vs Footing Size",fontsize=10,fontweight='bold')
    ax.grid(True)

    return fig

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
    right_col1, right_col2 = st.columns([1.15, 0.95])

    with right_col1:
        with st.expander("Footing Properties", expanded=True):
            footing_shape = st.selectbox(
                "Footing shape",
                ["Strip", "Square", "Rectangular", "Circular"],
            )

            df_depth = st.number_input("Embedment depth, Df", min_value=0.0, value=1.5, step=0.1)
            base_angle = st.number_input("Base inclination angle (deg)", min_value=0.0, value=0.0, step=0.5)
            ground_angle = st.number_input("Ground inclination angle (deg)", min_value=0.0, value=0.0, step=0.5)
            load_angle = st.number_input("Load inclination angle (deg)", min_value=0.0, value=0.0, step=0.5)

            b_min = b_max = b_inc = None
            length_to_width_ratio = None
            r_min = r_max = r_inc = None

            if footing_shape in ["Strip", "Square", "Rectangular"]:
                b_min = st.number_input("B_min", min_value=0.01, value=1.0, step=0.1)
                b_max = st.number_input("B_max", min_value=0.01, value=3.0, step=0.1)
                b_inc = st.number_input("B_increment", min_value=0.01, value=0.5, step=0.1)

            if footing_shape == "Rectangular":
                length_to_width_ratio = st.number_input(
                    "L/B ratio",
                    min_value=1.01,
                    value=2.0,
                    step=0.1,
                )

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
        length_to_width_ratio=length_to_width_ratio,
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

        if selected_methods == ["Terzaghi"]:
            if footing_shape == "Strip":
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
                fig = plot_results(
                    results_df=results_df,
                    footing_shape=footing_shape,
                    design_framework=design_framework,
                    unit_system=unit_system,
                )
                plot_col1, plot_col2, plot_col3 = st.columns([1, 2, 1])
                with plot_col2:
                    st.pyplot(fig)                

            elif footing_shape == "Square":
                widths = build_width_array(b_min, b_max, b_inc)
                results_df = calculate_terzaghi_square_results(
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

                st.subheader("Terzaghi Square Footing Results")
                st.dataframe(results_df, use_container_width=True)
                fig = plot_results(
                    results_df=results_df,
                    footing_shape=footing_shape,
                    design_framework=design_framework,
                    unit_system=unit_system,
                )
                plot_col1, plot_col2, plot_col3 = st.columns([1, 2, 1])
                with plot_col2:
                    st.pyplot(fig)

            elif footing_shape == "Rectangular":
                widths = build_width_array(b_min, b_max, b_inc)
                results_df = calculate_terzaghi_rectangular_results(
                    soil_df=cleaned_soil_df,
                    widths=widths,
                    length_to_width_ratio=length_to_width_ratio,
                    df_depth=df_depth,
                    groundwater_depth=groundwater_depth,
                    wedge_method=wedge_method,
                    design_framework=design_framework,
                    fs_value=fs_value,
                    phi_r_value=phi_r_value,
                    unit_system=unit_system,
                )

                st.subheader("Terzaghi Rectangular Footing Results")
                st.dataframe(results_df, use_container_width=True)
                fig = plot_results(
                    results_df=results_df,
                    footing_shape=footing_shape,
                    design_framework=design_framework,
                        unit_system=unit_system,
                )
                plot_col1, plot_col2, plot_col3 = st.columns([1, 2, 1])
                with plot_col2:
                    st.pyplot(fig)

            elif footing_shape == "Circular":
                radii = build_width_array(r_min, r_max, r_inc)
                results_df = calculate_terzaghi_circular_results(
                    soil_df=cleaned_soil_df,
                    radii=radii,
                    df_depth=df_depth,
                    groundwater_depth=groundwater_depth,
                    wedge_method=wedge_method,
                    design_framework=design_framework,
                    fs_value=fs_value,
                    phi_r_value=phi_r_value,
                    unit_system=unit_system,
                )

                st.subheader("Terzaghi Circular Footing Results")
                st.dataframe(results_df, use_container_width=True)
                fig = plot_results(
                    results_df=results_df,
                    footing_shape=footing_shape,
                    design_framework=design_framework,
                        unit_system=unit_system,
                )
                plot_col1, plot_col2, plot_col3 = st.columns([1, 2, 1])
                with plot_col2:
                    st.pyplot(fig)
            else:
                st.warning("This step currently supports only the Terzaghi method.")
