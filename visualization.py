"""
Visualization Module for GPW Smart Analyzer

This module provides Chart.js visualization generation for stock price analysis
including moving averages and trend indicators.

Author: GPW Smart Analyzer Team
License: MIT
"""

import json
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from config import CHART_COLORS, CHART_OUTPUT_DIR, ENABLE_CHART_GENERATION, CHART_MAX_STOCKS

import logging
logger = logging.getLogger(__name__)


def generate_price_ma_chart(ticker: str, company_name: str, df: pd.DataFrame,
                           ma5: Optional[float], ma10: Optional[float],
                           ma20: Optional[float], trend_label: str) -> Dict[str, Any]:
    """
    Generate a Chart.js 3.x line chart configuration for price and moving averages.

    Args:
        ticker: Stock ticker symbol
        company_name: Full company name
        df: DataFrame with historical price data
        ma5: 5-day moving average
        ma10: 10-day moving average
        ma20: 20-day moving average
        trend_label: Polish trend label

    Returns:
        Chart.js configuration dictionary
    """
    if df is None or df.empty:
        logger.warning(f"No data available for chart generation: {ticker}")
        return None

    try:
        # Prepare data for Chart.js
        dates = df.index.strftime('%Y-%m-%d').tolist()
        close_prices = df['Close'].tolist()

        # Calculate moving averages for the entire period
        ma5_data = df['Close'].rolling(window=5).mean().tolist()
        ma10_data = df['Close'].rolling(window=10).mean().tolist()
        ma20_data = df['Close'].rolling(window=20).mean().tolist()

        # Convert NaN to None for Chart.js
        def clean_data(data):
            return [None if pd.isna(x) else float(x) for x in data]

        close_data_clean = clean_data(close_prices)
        ma5_data_clean = clean_data(ma5_data)
        ma10_data_clean = clean_data(ma10_data)
        ma20_data_clean = clean_data(ma20_data)

        # Create Chart.js configuration
        chart_config = {
            "type": "line",
            "data": {
                "labels": dates,
                "datasets": [
                    {
                        "label": "Cena zamknięcia",
                        "data": close_data_clean,
                        "borderColor": CHART_COLORS['price'],
                        "backgroundColor": f"{CHART_COLORS['price']}20",
                        "borderWidth": 2,
                        "pointRadius": 1,
                        "pointHoverRadius": 5,
                        "tension": 0.1
                    },
                    {
                        "label": "MA5",
                        "data": ma5_data_clean,
                        "borderColor": CHART_COLORS['ma5'],
                        "backgroundColor": f"{CHART_COLORS['ma5']}20",
                        "borderWidth": 2,
                        "pointRadius": 0,
                        "pointHoverRadius": 4,
                        "tension": 0.1
                    },
                    {
                        "label": "MA10",
                        "data": ma10_data_clean,
                        "borderColor": CHART_COLORS['ma10'],
                        "backgroundColor": f"{CHART_COLORS['ma10']}20",
                        "borderWidth": 2,
                        "pointRadius": 0,
                        "pointHoverRadius": 4,
                        "tension": 0.1
                    },
                    {
                        "label": "MA20",
                        "data": ma20_data_clean,
                        "borderColor": CHART_COLORS['ma20'],
                        "backgroundColor": f"{CHART_COLORS['ma20']}20",
                        "borderWidth": 2,
                        "pointRadius": 0,
                        "pointHoverRadius": 4,
                        "tension": 0.1
                    }
                ]
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": f"{company_name} ({ticker}) - Trend: {trend_label}",
                        "font": {
                            "size": 16,
                            "weight": "bold"
                        }
                    },
                    "legend": {
                        "display": True,
                        "position": "top"
                    },
                    "tooltip": {
                        "mode": "index",
                        "intersect": False,
                        "callbacks": {
                            "label": {
                                "type": "function",
                                "body": "return context.dataset.label + ': ' + context.parsed.y.toFixed(2) + ' PLN';"
                            }
                        }
                    }
                },
                "scales": {
                    "x": {
                        "display": True,
                        "title": {
                            "display": True,
                            "text": "Data"
                        },
                        "ticks": {
                            "maxTicksLimit": 10
                        }
                    },
                    "y": {
                        "display": True,
                        "title": {
                            "display": True,
                            "text": "Cena (PLN)"
                        },
                        "ticks": {
                            "callback": {
                                "type": "function",
                                "body": "return value.toFixed(2) + ' PLN';"
                            }
                        }
                    }
                },
                "interaction": {
                    "mode": "nearest",
                    "axis": "x",
                    "intersect": False
                }
            }
        }

        return chart_config

    except Exception as e:
        logger.error(f"Error generating chart for {ticker}: {e}")
        return None


def save_chart_json(chart_config: Dict[str, Any], ticker: str,
                    output_dir: str = CHART_OUTPUT_DIR) -> Optional[str]:
    """
    Save chart configuration as JSON file.

    Args:
        chart_config: Chart.js configuration dictionary
        ticker: Stock ticker symbol
        output_dir: Output directory for chart files

    Returns:
        Path to saved file or None on failure
    """
    if chart_config is None:
        logger.warning(f"No chart configuration to save for {ticker}")
        return None

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{ticker}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        # Save JSON with proper formatting
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chart_config, f, indent=2, ensure_ascii=False)

        logger.info(f"Chart configuration saved to: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Error saving chart JSON for {ticker}: {e}")
        return None


def print_chart_json(chart_config: Dict[str, Any], ticker: str) -> None:
    """
    Print chart configuration to console.

    Args:
        chart_config: Chart.js configuration dictionary
        ticker: Stock ticker symbol
    """
    if chart_config is None:
        logger.warning(f"No chart configuration to print for {ticker}")
        return

    try:
        print(f"\n📊 Wykres Chart.js dla {ticker}:")
        print("=" * 60)

        # Print formatted JSON
        json_str = json.dumps(chart_config, indent=2, ensure_ascii=False)
        print(json_str)

        print("=" * 60)

    except Exception as e:
        logger.error(f"Error printing chart JSON for {ticker}: {e}")


def generate_charts_for_stocks(stocks: List[Dict[str, Any]], max_charts: int = CHART_MAX_STOCKS) -> List[str]:
    """
    Generate charts for multiple stocks.

    Args:
        stocks: List of stock dictionaries with technical data
        max_charts: Maximum number of charts to generate

    Returns:
        List of paths to generated chart files
    """
    if not ENABLE_CHART_GENERATION:
        logger.info("Chart generation is disabled")
        return []

    chart_files = []
    generated_count = 0

    for i, stock in enumerate(stocks):
        if generated_count >= max_charts:
            break

        ticker = stock.get('ticker', 'UNKNOWN')
        company_name = stock.get('name', ticker)

        # Check if historical prices are available
        historical_prices = stock.get('historical_prices')
        if historical_prices is None or historical_prices.empty:
            logger.warning(f"No historical data available for {ticker}")
            continue

        # Get technical indicators
        ma5 = stock.get('ma5')
        ma10 = stock.get('ma10')
        ma20 = stock.get('ma20')
        trend_label = stock.get('trend_label', 'Nieznany')

        # Generate chart configuration
        chart_config = generate_price_ma_chart(
            ticker, company_name, historical_prices,
            ma5, ma10, ma20, trend_label
        )

        if chart_config is not None:
            # Save chart file
            chart_file = save_chart_json(chart_config, ticker)
            if chart_file:
                chart_files.append(chart_file)
                generated_count += 1

    logger.info(f"Generated {generated_count} charts out of {len(stocks)} stocks")
    return chart_files


def create_summary_charts_summary(stocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a summary of generated charts for reporting.

    Args:
        stocks: List of stock dictionaries

    Returns:
        Summary dictionary with chart statistics
    """
    total_stocks = len(stocks)
    charts_generated = 0

    trend_counts = {
        'Wzrostowy': 0,
        'Spadkowy': 0,
        'Boczny': 0,
        'Nieznany': 0
    }

    for stock in stocks:
        if stock.get('historical_prices') is not None:
            charts_generated += 1

        trend_label = stock.get('trend_label', 'Nieznany')
        if trend_label in trend_counts:
            trend_counts[trend_label] += 1

    summary = {
        'total_stocks': total_stocks,
        'charts_generated': charts_generated,
        'charts_directory': CHART_OUTPUT_DIR,
        'trend_distribution': trend_counts,
        'generation_timestamp': datetime.now().isoformat()
    }

    return summary


def export_charts_summary_to_json(summary: Dict[str, Any], filename: str = "charts_summary.json") -> Optional[str]:
    """
    Export charts summary to JSON file.

    Args:
        summary: Charts summary dictionary
        filename: Output filename

    Returns:
        Path to saved file or None on failure
    """
    try:
        output_path = os.path.join(CHART_OUTPUT_DIR, filename)
        os.makedirs(CHART_OUTPUT_DIR, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Charts summary exported to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error exporting charts summary: {e}")
        return None


def print_charts_summary(summary: Dict[str, Any]) -> None:
    """
    Print charts summary to console.

    Args:
        summary: Charts summary dictionary
    """
    print(f"\n📊 Podsumowanie wykresów:")
    print(f"   • Łącznie spółek: {summary['total_stocks']}")
    print(f"   • Wygenerowanych wykresów: {summary['charts_generated']}")
    print(f"   • Katalog wyjściowy: {summary['charts_directory']}")
    print(f"\n📈 Rozkład trendów:")
    for trend, count in summary['trend_distribution'].items():
        percentage = (count / summary['total_stocks']) * 100 if summary['total_stocks'] > 0 else 0
        print(f"   • {trend}: {count} ({percentage:.1f}%)")
    print("=" * 50)