#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Light Streamlit demo.
В облаке НЕ генерим полный Excel (слишком тяжело). Полный отчёт — через cli.py локально.
Также НЕ импортируем tool.py на уровне модуля (cold start в Streamlit Cloud).
"""

from __future__ import annotations

import os
import tempfile
from typing import List

import streamlit as st

def _tool():
    """Ленивая загрузка heavy-модуля для Streamlit Cloud."""
    import tool
    return tool


def _resolve_selected_methods(selected: List[str], mapping) -> List[str]:
    """Оставляет только методы, которые реально доступны в mapping."""
    return [m for m in selected if m in mapping]


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

    tool = _tool()
    tool.configure_warnings(quiet=quiet_warnings)
    if not tool.PYINFORM_AVAILABLE:
        st.info("TE-методы скрыты: установи pyinform (локально), если нужно.")

    method_options = tool.STABLE_METHODS + tool.EXPERIMENTAL_METHODS
    selected_methods = st.multiselect(
        "Methods",
        options=method_options,
        default=tool.STABLE_METHODS,
    )

    if any(method in tool.EXPERIMENTAL_METHODS for method in selected_methods):
        st.warning("Часть выбранных методов помечена как experimental.")

    if st.button("Run", type="primary"):
        if not uploaded_file:
            st.error("Сначала загрузите файл CSV/XLSX.")
            return

        suffix = os.path.splitext(uploaded_file.name)[1] or ".csv"
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, f"input{suffix}")

            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            engine = tool.BigMasterTool(enable_experimental=False)
            engine.lag_ranges = {v: range(1, lag + 1) for v in tool.method_mapping}

            with st.spinner("Обработка данных..."):
                engine.load_data_excel(
                    input_path,
                    log_transform=log_transform,
                    remove_outliers=remove_outliers,
                    normalize=normalize,
                    fill_missing=True,
                    check_stationarity=False,
                )
                engine.run_all_methods()

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
            directed = "directed" in primary_method or "partial" in primary_method
            invert = "granger" in primary_method
            connectome = tool.plot_connectome(
                matrix,
                f"{primary_method} Connectome",
                threshold=threshold,
                directed=directed,
                invert_threshold=invert,
                legend_text=f"Lag={lag}",
            )
            st.image(connectome, caption=primary_method)

            st.caption("Полный Excel-отчёт: `python cli.py <file> --lags N --graph-threshold T` (локально)")


if __name__ == "__main__":
    main()
