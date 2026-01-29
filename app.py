#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Minimal Streamlit facade for BigMasterTool."""

from __future__ import annotations

import os
import tempfile
from typing import List

import streamlit as st

from tool import (
    BigMasterTool,
    EXPERIMENTAL_METHODS,
    STABLE_METHODS,
    compute_connectivity_variant,
    configure_warnings,
    method_mapping,
    plot_connectome,
    plot_heatmap,
)


def _resolve_selected_methods(selected: List[str]) -> List[str]:
    """Оставляет только методы, которые реально доступны в mapping."""
    return [m for m in selected if m in method_mapping]


def main() -> None:
    """Запускает Streamlit UI."""
    st.set_page_config(page_title="Time Series Connectivity Demo", layout="wide")
    st.title("Time Series Connectivity Demo")
    st.caption("Загрузите файл и получите базовый отчёт по связям.")

    configure_warnings(quiet=False)

    uploaded_file = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])
    col1, col2, col3 = st.columns(3)
    with col1:
        lag = st.number_input("Lag", min_value=1, max_value=50, value=1, step=1)
    with col2:
        threshold = st.number_input(
            "Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
        )
    with col3:
        normalize = st.checkbox("normalize", value=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        remove_outliers = st.checkbox("outliers", value=True)
    with col5:
        log_transform = st.checkbox("log", value=False)
    with col6:
        quiet_warnings = st.checkbox("quiet warnings", value=False)

    method_options = STABLE_METHODS + EXPERIMENTAL_METHODS
    selected_methods = st.multiselect(
        "Methods",
        options=method_options,
        default=STABLE_METHODS,
    )

    if any(method in EXPERIMENTAL_METHODS for method in selected_methods):
        st.warning("Часть выбранных методов помечена как experimental.")

    if st.button("Run", type="primary"):
        if not uploaded_file:
            st.error("Сначала загрузите файл CSV/XLSX.")
            return

        configure_warnings(quiet=quiet_warnings)
        suffix = os.path.splitext(uploaded_file.name)[1] or ".csv"
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, f"input{suffix}")
            output_path = os.path.join(tmp_dir, "AllMethods_Full.xlsx")

            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            tool = BigMasterTool(enable_experimental=False)
            tool.lag_ranges = {v: range(1, lag + 1) for v in method_mapping}

            with st.spinner("Обработка данных..."):
                tool.load_data_excel(
                    input_path,
                    log_transform=log_transform,
                    remove_outliers=remove_outliers,
                    normalize=normalize,
                    fill_missing=True,
                    check_stationarity=False,
                )
                tool.run_all_methods()
                tool.export_big_excel(
                    output_path,
                    threshold=threshold,
                    window_size=100,
                    overlap=50,
                    log_transform=log_transform,
                    remove_outliers=remove_outliers,
                    normalize=normalize,
                    fill_missing=True,
                    check_stationarity=False,
                )

            resolved_methods = _resolve_selected_methods(selected_methods)
            if not resolved_methods:
                st.info("Методы не выбраны или недоступны.")
                return

            st.subheader("Heatmaps")
            for method in resolved_methods[:2]:
                matrix = compute_connectivity_variant(
                    tool.data_normalized,
                    method,
                    lag=lag,
                )
                heatmap = plot_heatmap(
                    matrix,
                    f"{method} Heatmap",
                    legend_text=f"Lag={lag}",
                )
                st.image(heatmap, caption=method)

            st.subheader("Connectome")
            primary_method = resolved_methods[0]
            matrix = compute_connectivity_variant(
                tool.data_normalized,
                primary_method,
                lag=lag,
            )
            directed = "directed" in primary_method or "partial" in primary_method
            invert = "granger" in primary_method
            connectome = plot_connectome(
                matrix,
                f"{primary_method} Connectome",
                threshold=threshold,
                directed=directed,
                invert_threshold=invert,
                legend_text=f"Lag={lag}",
            )
            st.image(connectome, caption=primary_method)

            with open(output_path, "rb") as f:
                st.download_button(
                    "Download Excel",
                    data=f.read(),
                    file_name="AllMethods_Full.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )


if __name__ == "__main__":
    main()
