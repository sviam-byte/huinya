#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
import os

from tool import BigMasterTool, configure_warnings, method_mapping, save_folder


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute connectivity measures for multivariate time series.")
    p.add_argument("input_file", help="Path to input CSV or Excel file with time series data")
    p.add_argument("--lags", type=int, default=5, help="Max lag/model order (for Granger, TE, etc.)")
    p.add_argument("--log", action="store_true", help="Apply logarithm transform to data (for positive-valued data)")
    p.add_argument("--no-outliers", action="store_true", help="Disable outlier removal")
    p.add_argument("--no-normalize", action="store_true", help="Disable normalization of data")
    p.add_argument("--no-stationarity-check", action="store_true", help="Disable stationarity check (ADF test)")
    p.add_argument("--graph-threshold", type=float, default=0.5, help="Threshold for graph edges")
    p.add_argument("--output", default=None, help="Output Excel file path")
    p.add_argument("--quiet-warnings", action="store_true", help="Suppress warnings")
    p.add_argument("--experimental", action="store_true", help="Enable experimental sliding-window analyses")
    return p


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_warnings(quiet=args.quiet_warnings)

    filepath = os.path.abspath(args.input_file)
    output_path = args.output or os.path.join(save_folder, "AllMethods_Full.xlsx")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tool = BigMasterTool(enable_experimental=args.experimental)
    tool.lag_ranges = {v: range(1, args.lags + 1) for v in method_mapping}

    tool.load_data_excel(
        filepath,
        log_transform=args.log,
        remove_outliers=not args.no_outliers,
        normalize=not args.no_normalize,
        fill_missing=True,
        check_stationarity=not args.no_stationarity_check,
    )
    tool.run_all_methods()
    tool.export_big_excel(
        output_path,
        threshold=args.graph_threshold,
        window_size=100,
        overlap=50,
        log_transform=args.log,
        remove_outliers=not args.no_outliers,
        normalize=not args.no_normalize,
        fill_missing=True,
        check_stationarity=not args.no_stationarity_check,
    )

    print("Готово. Excel сохранён в:", output_path)


if __name__ == "__main__":
    main()
