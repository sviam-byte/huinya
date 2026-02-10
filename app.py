#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Light Streamlit demo.
В облаке НЕ генерим полный Excel (слишком тяжело). Полный отчёт — через cli.py локально.
Также НЕ импортируем tool.py на уровне модуля (cold start в Streamlit Cloud).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from types import ModuleType
from typing import List, Mapping, Sequence

import streamlit as st


def _tool() -> ModuleType:
    """Ленивая загрузка heavy-модуля для Streamlit Cloud."""
    import tool
    return tool


def _resolve_selected_methods(selected: Sequence[str], mapping: Mapping[str, object]) -> List[str]:
    """Оставляет только методы, которые реально доступны в mapping."""
    return [m for m in selected if m in mapping]


def _is_cloud_env() -> bool:
    """Пытаемся определить облачную среду Streamlit, чтобы не грузить тяжелые экспорты."""
    return (
        os.getenv("STREAMLIT_CLOUD") == "true"
        or os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud"
        or os.getenv("STREAMLIT_SHARING") == "true"
    )


def main() -> None:
    """Запускает Streamlit UI."""
    st.set_page_config(page_title="Time Series Connectivity Demo", layout="wide")
    st.title("Time Series Connectivity Demo")
    st.caption("Демо: heatmap/connectome. Полный Excel — локально: `python cli.py <file>`")

    uploaded_file = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])
    col1, col2, col3 = st.columns(3)
    with col1:
        lag = st.number_input("Lag", min_value=1, max_value=50, value=1, step=1)
    with col2:
        threshold = st.number_input(
            "Weight threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
        )
    with col3:
        normalize = st.checkbox("normalize", value=True)


    # Для p-value (Granger full / directed) нужен отдельный порог
    alpha = st.number_input(
        "p-value alpha (for Granger p-values)",
        min_value=1e-6,
        max_value=0.5,
        value=0.05,
        format="%.6f",
    )

    st.subheader("Input parsing")
    header_mode = st.selectbox("Header", ["auto", "yes", "no"], index=0)
    time_col_mode = st.selectbox("Time column", ["auto", "none"], index=0)
    transpose_mode = st.selectbox("Transpose", ["auto", "yes", "no"], index=0)

    st.subheader("Preprocessing")
    preprocess_enabled = st.checkbox("Enable preprocessing", value=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        remove_outliers = st.checkbox("outliers", value=True)
    with col5:
        log_transform = st.checkbox("log", value=False)
    with col6:
        quiet_warnings = st.checkbox("quiet warnings", value=False)

    tool = _tool()
    tool.configure_warnings(quiet=quiet_warnings)
    if not tool.PYINFORM_AVAILABLE:
        st.info("TE-методы скрыты: установи pyinform (локально), если нужно.")

    method_options = tool.STABLE_METHODS + tool.EXPERIMENTAL_METHODS
    if not tool.PYINFORM_AVAILABLE:
        method_options = [m for m in method_options if not m.startswith("te_")]
    selected_methods = st.multiselect(
        "Methods",
        options=method_options,
        default=[m for m in tool.STABLE_METHODS if m in method_options],
    )

    if any(method in tool.EXPERIMENTAL_METHODS for method in selected_methods):
        st.warning("Часть выбранных методов помечена как experimental.")

    is_cloud = _is_cloud_env()
    generate_excel = False
    generate_html = True
    generate_site = False
    if is_cloud:
        st.info("В облаке полный Excel-отчёт отключён (слишком тяжело для Streamlit Cloud).")
    else:
        generate_excel = st.checkbox(
            "Generate full Excel report (slow)",
            value=False,
            help="Создаёт полный отчёт. Может занять время и много памяти.",
        )
    generate_html = st.checkbox("Generate HTML report (fast)", value=True)
    if not is_cloud:
        generate_site = st.checkbox("Generate mini-site report (zip)", value=False)

    if st.button("Run", type="primary"):
        if not uploaded_file:
            st.error("Сначала загрузите файл CSV/XLSX.")
            return

        suffix = os.path.splitext(uploaded_file.name)[1] or ".csv"
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, f"input{suffix}")
            output_path = os.path.join(tmp_dir, "AllMethods_Full.xlsx")

            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            enable_experimental = any(m in tool.EXPERIMENTAL_METHODS for m in selected_methods)
            engine = tool.BigMasterTool(enable_experimental=enable_experimental)
            engine.lag_ranges = {v: range(1, lag + 1) for v in tool.method_mapping}

            with st.spinner("Обработка данных..."):
                engine.load_data_excel(
                    input_path,
                    header=header_mode,
                    time_col=time_col_mode,
                    transpose=transpose_mode,
                    preprocess=preprocess_enabled,
                    log_transform=log_transform,
                    remove_outliers=remove_outliers,
                    normalize=normalize,
                    fill_missing=True,
                    check_stationarity=False,
                )
                engine.run_all_methods()
                html_path = os.path.join(tmp_dir, "report.html")
                site_dir = os.path.join(tmp_dir, "site_report")
                site_zip = os.path.join(tmp_dir, "site_report.zip")
                if generate_excel:
                    engine.export_big_excel(
                        output_path,
                        threshold=threshold,
                        p_value_alpha=alpha,
                        window_size=100,
                        overlap=50,
                        log_transform=log_transform,
                        remove_outliers=remove_outliers,
                        normalize=normalize,
                        fill_missing=True,
                        check_stationarity=False,
                    )
                if generate_html:
                    engine.export_html_report(html_path, graph_threshold=threshold, p_alpha=float(alpha))
                if generate_site:
                    engine.export_site_report(site_dir, graph_threshold=threshold, p_alpha=float(alpha), zip_path=site_zip)

            resolved_methods = _resolve_selected_methods(selected_methods, tool.method_mapping)
            if not resolved_methods:
                st.info("Методы не выбраны или недоступны.")
                return

            st.subheader("Heatmaps")
            for method in resolved_methods[:3]:
                matrix = tool.compute_connectivity_variant(engine.data_normalized, method, lag=lag)
                heatmap = tool.plot_heatmap(matrix, f"{method} Heatmap", legend_text=f"Lag={lag}")
                st.image(heatmap, caption=method)

            st.subheader("Connectome")
            primary_method = resolved_methods[0]
            matrix = tool.compute_connectivity_variant(engine.data_normalized, primary_method, lag=lag)
            directed = tool.is_directed_method(primary_method)
            invert = tool.is_pvalue_method(primary_method)
            thr = float(alpha) if invert else threshold
            connectome = tool.plot_connectome(
                matrix,
                f"{primary_method} Connectome",
                threshold=thr,
                directed=directed,
                invert_threshold=invert,
                legend_text=f"Lag={lag}",
            )
            st.image(connectome, caption=primary_method)

            if generate_excel:
                with open(output_path, "rb") as f:
                    st.download_button(
                        "Download Excel",
                        data=f.read(),
                        file_name="AllMethods_Full.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            if generate_html:
                with open(html_path, "rb") as f:
                    st.download_button(
                        "Download HTML report",
                        data=f.read(),
                        file_name="report.html",
                        mime="text/html",
                    )
            if generate_site and (not is_cloud):
                with open(site_zip, "rb") as f:
                    st.download_button(
                        "Download mini-site (zip)",
                        data=f.read(),
                        file_name="site_report.zip",
                        mime="application/zip",
                    )
            else:
                st.caption(
                    "Полный Excel-отчёт: `python cli.py <file> --lags N --graph-threshold T` (локально)"
                )


if __name__ == "__main__":
    main()
