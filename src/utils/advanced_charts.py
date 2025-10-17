"""
Advanced Professional Charts Module for GPW Smart Analyzer

This module provides comprehensive professional Chart.js configurations including:
- Candlestick charts with OHLCV data
- Multiple technical indicators (RSI, MACD, ADX, Bollinger Bands)
- Volume analysis
- Support/Resistance levels
- Professional styling like TradingView

Author: GPW Smart Analyzer Team
License: MIT
"""

import json
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Professional color schemes for charts
PROFESSIONAL_COLORS = {
    'background': '#1e1e1e',
    'grid': '#2a2a2a',
    'text': '#d1d4dc',
    'green': '#00c853',
    'red': '#ff5252',
    'blue': '#2196f3',
    'orange': '#ff9800',
    'purple': '#9c27b0',
    'yellow': '#ffeb3b',
    'cyan': '#00bcd4'
}

def generate_candlestick_chart(ticker: str, company_name: str, df: pd.DataFrame,
                             ma5: Optional[float] = None, ma10: Optional[float] = None,
                             ma20: Optional[float] = None, trend_label: str = "Boczny") -> Dict[str, Any]:
    """
    Generate professional candlestick chart with moving averages and volume.

    Args:
        ticker: Stock ticker symbol
        company_name: Full company name
        df: DataFrame with OHLCV data
        ma5: 5-day moving average
        ma10: 10-day moving average
        ma20: 20-day moving average
        trend_label: Polish trend label

    Returns:
        Chart.js configuration dictionary
    """
    if df is None or df.empty:
        logger.warning(f"No data available for candlestick chart: {ticker}")
        return None

    try:
        # Prepare data
        dates = df.index.strftime('%Y-%m-%d').tolist()

        # Candlestick data
        opens = df['Open'].tolist()
        highs = df['High'].tolist()
        lows = df['Low'].tolist()
        closes = df['Close'].tolist()
        volumes = df['Volume'].tolist()

        # Calculate moving averages
        ma5_data = df['Close'].rolling(window=5).mean().tolist()
        ma10_data = df['Close'].rolling(window=10).mean().tolist()
        ma20_data = df['Close'].rolling(window=20).mean().tolist()

        # Calculate Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_upper = df['Close'].rolling(window=bb_period).mean() + (df['Close'].rolling(window=bb_period).std() * bb_std)
        bb_lower = df['Close'].rolling(window=bb_period).mean() - (df['Close'].rolling(window=bb_period).std() * bb_std)
        bb_middle = df['Close'].rolling(window=bb_period).mean()

        # Color coding for candlesticks
        colors = ['#00c853' if close >= open else '#ff5252' for open, close in zip(opens, closes)]

        # Chart configuration
        chart_config = {
            "type": "candlestick",
            "data": {
                "labels": dates,
                "datasets": [
                    {
                        "label": "Cena",
                        "data": [
                            {
                                "x": dates[i],
                                "o": opens[i],
                                "h": highs[i],
                                "l": lows[i],
                                "c": closes[i]
                            } for i in range(len(dates))
                        ],
                        "type": "candlestick",
                        "yAxisID": "y",
                        "borderColor": colors,
                        "wickColor": "#666",
                        "color": colors
                    },
                    {
                        "label": "MA5",
                        "data": ma5_data,
                        "type": "line",
                        "borderColor": PROFESSIONAL_COLORS['blue'],
                        "backgroundColor": "transparent",
                        "borderWidth": 2,
                        "pointRadius": 0,
                        "tension": 0.1,
                        "yAxisID": "y"
                    },
                    {
                        "label": "MA10",
                        "data": ma10_data,
                        "type": "line",
                        "borderColor": PROFESSIONAL_COLORS['orange'],
                        "backgroundColor": "transparent",
                        "borderWidth": 2,
                        "pointRadius": 0,
                        "tension": 0.1,
                        "yAxisID": "y"
                    },
                    {
                        "label": "MA20",
                        "data": ma20_data,
                        "type": "line",
                        "borderColor": PROFESSIONAL_COLORS['purple'],
                        "backgroundColor": "transparent",
                        "borderWidth": 2,
                        "pointRadius": 0,
                        "tension": 0.1,
                        "yAxisID": "y"
                    },
                    {
                        "label": "Bollinger Upper",
                        "data": bb_upper.tolist(),
                        "type": "line",
                        "borderColor": PROFESSIONAL_COLORS['yellow'],
                        "backgroundColor": "transparent",
                        "borderWidth": 1,
                        "pointRadius": 0,
                        "borderDash": [5, 5],
                        "yAxisID": "y"
                    },
                    {
                        "label": "Bollinger Lower",
                        "data": bb_lower.tolist(),
                        "type": "line",
                        "borderColor": PROFESSIONAL_COLORS['yellow'],
                        "backgroundColor": "transparent",
                        "borderWidth": 1,
                        "pointRadius": 0,
                        "borderDash": [5, 5],
                        "yAxisID": "y"
                    },
                    {
                        "label": "Wolumen",
                        "data": volumes,
                        "type": "bar",
                        "backgroundColor": [
                            PROFESSIONAL_COLORS['green'] if close >= open else PROFESSIONAL_COLORS['red']
                            for open, close in zip(opens, closes)
                        ],
                        "yAxisID": "y1",
                        "barPercentage": 0.6
                    }
                ]
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "backgroundColor": PROFESSIONAL_COLORS['background'],
                "plugins": {
                    "title": {
                        "display": True,
                        "text": f"{company_name} ({ticker}) - {trend_label}",
                        "color": PROFESSIONAL_COLORS['text'],
                        "font": {
                            "size": 18,
                            "weight": "bold"
                        }
                    },
                    "legend": {
                        "display": True,
                        "position": "top",
                        "labels": {
                            "color": PROFESSIONAL_COLORS['text'],
                            "usePointStyle": True,
                            "padding": 20
                        }
                    },
                    "tooltip": {
                        "mode": "index",
                        "intersect": False,
                        "backgroundColor": "rgba(0, 0, 0, 0.8)",
                        "titleColor": PROFESSIONAL_COLORS['text'],
                        "bodyColor": PROFESSIONAL_COLORS['text'],
                        "borderColor": PROFESSIONAL_COLORS['grid'],
                        "callbacks": {
                            "label": {
                                "type": "function",
                                "body": """
                                function(context) {
                                    const label = context.dataset.label || '';
                                    if (label === 'Cena') {
                                        const o = context.raw.o;
                                        const h = context.raw.h;
                                        const l = context.raw.l;
                                        const c = context.raw.c;
                                        return [
                                            'O: ' + o.toFixed(2) + ' PLN',
                                            'H: ' + h.toFixed(2) + ' PLN',
                                            'L: ' + l.toFixed(2) + ' PLN',
                                            'C: ' + c.toFixed(2) + ' PLN'
                                        ];
                                    }
                                    return label + ': ' + context.parsed.y.toFixed(2) + ' PLN';
                                }
                                """
                            }
                        }
                    }
                },
                "scales": {
                    "x": {
                        "grid": {
                            "color": PROFESSIONAL_COLORS['grid'],
                            "borderColor": PROFESSIONAL_COLORS['grid']
                        },
                        "ticks": {
                            "color": PROFESSIONAL_COLORS['text'],
                            "maxTicksLimit": 10
                        }
                    },
                    "y": {
                        "type": "linear",
                        "display": True,
                        "position": "left",
                        "grid": {
                            "color": PROFESSIONAL_COLORS['grid'],
                            "borderColor": PROFESSIONAL_COLORS['grid']
                        },
                        "ticks": {
                            "color": PROFESSIONAL_COLORS['text'],
                            "callback": {
                                "type": "function",
                                "body": "return value.toFixed(2) + ' PLN';"
                            }
                        },
                        "title": {
                            "display": True,
                            "text": "Cena (PLN)",
                            "color": PROFESSIONAL_COLORS['text']
                        }
                    },
                    "y1": {
                        "type": "linear",
                        "display": True,
                        "position": "right",
                        "grid": {
                            "drawOnChartArea": False
                        },
                        "ticks": {
                            "color": PROFESSIONAL_COLORS['text'],
                            "callback": {
                                "type": "function",
                                "body": """
                                function(value) {
                                    if (value >= 1000000) {
                                        return (value / 1000000).toFixed(1) + 'M';
                                    }
                                    if (value >= 1000) {
                                        return (value / 1000).toFixed(1) + 'K';
                                    }
                                    return value.toFixed(0);
                                }
                                """
                            }
                        },
                        "title": {
                            "display": True,
                            "text": "Wolumen",
                            "color": PROFESSIONAL_COLORS['text']
                        }
                    }
                },
                "layout": {
                    "padding": {
                        "left": 10,
                        "right": 10,
                        "top": 10,
                        "bottom": 10
                    }
                }
            }
        }

        return chart_config

    except Exception as e:
        logger.error(f"Error generating candlestick chart for {ticker}: {e}")
        return None


def generate_rsi_chart(ticker: str, df: pd.DataFrame, rsi_period: int = 14) -> Dict[str, Any]:
    """
    Generate RSI indicator chart.

    Args:
        ticker: Stock ticker symbol
        df: DataFrame with price data
        rsi_period: RSI calculation period

    Returns:
        Chart.js configuration dictionary
    """
    try:
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        dates = df.index.strftime('%Y-%m-%d').tolist()
        rsi_data = rsi.tolist()

        chart_config = {
            "type": "line",
            "data": {
                "labels": dates,
                "datasets": [
                    {
                        "label": "RSI",
                        "data": rsi_data,
                        "borderColor": PROFESSIONAL_COLORS['cyan'],
                        "backgroundColor": f"{PROFESSIONAL_COLORS['cyan']}20",
                        "borderWidth": 2,
                        "pointRadius": 1,
                        "tension": 0.1
                    }
                ]
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "backgroundColor": PROFESSIONAL_COLORS['background'],
                "plugins": {
                    "title": {
                        "display": True,
                        "text": f"{ticker} - RSI ({rsi_period})",
                        "color": PROFESSIONAL_COLORS['text'],
                        "font": {"size": 16, "weight": "bold"}
                    },
                    "legend": {
                        "display": False
                    }
                },
                "scales": {
                    "x": {
                        "grid": {"color": PROFESSIONAL_COLORS['grid']},
                        "ticks": {"color": PROFESSIONAL_COLORS['text']}
                    },
                    "y": {
                        "min": 0,
                        "max": 100,
                        "grid": {"color": PROFESSIONAL_COLORS['grid']},
                        "ticks": {"color": PROFESSIONAL_COLORS['text']},
                        "title": {
                            "display": True,
                            "text": "RSI",
                            "color": PROFESSIONAL_COLORS['text']
                        }
                    }
                },
                "plugins": {
                    "annotation": {
                        "annotations": {
                            "overbought": {
                                "type": "line",
                                "yMin": 70,
                                "yMax": 70,
                                "borderColor": PROFESSIONAL_COLORS['red'],
                                "borderWidth": 2,
                                "borderDash": [5, 5],
                                "label": {
                                    "content": "Overbought (70)",
                                    "enabled": True,
                                    "position": "end"
                                }
                            },
                            "oversold": {
                                "type": "line",
                                "yMin": 30,
                                "yMax": 30,
                                "borderColor": PROFESSIONAL_COLORS['green'],
                                "borderWidth": 2,
                                "borderDash": [5, 5],
                                "label": {
                                    "content": "Oversold (30)",
                                    "enabled": True,
                                    "position": "end"
                                }
                            }
                        }
                    }
                }
            }
        }

        return chart_config

    except Exception as e:
        logger.error(f"Error generating RSI chart for {ticker}: {e}")
        return None


def generate_macd_chart(ticker: str, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Any]:
    """
    Generate MACD indicator chart.

    Args:
        ticker: Stock ticker symbol
        df: DataFrame with price data
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period

    Returns:
        Chart.js configuration dictionary
    """
    try:
        # Calculate MACD
        ema_fast = df['Close'].ewm(span=fast).mean()
        ema_slow = df['Close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        dates = df.index.strftime('%Y-%m-%d').tolist()

        chart_config = {
            "type": "line",
            "data": {
                "labels": dates,
                "datasets": [
                    {
                        "label": "MACD",
                        "data": macd_line.tolist(),
                        "borderColor": PROFESSIONAL_COLORS['blue'],
                        "backgroundColor": "transparent",
                        "borderWidth": 2,
                        "pointRadius": 0,
                        "yAxisID": "y"
                    },
                    {
                        "label": "Signal",
                        "data": signal_line.tolist(),
                        "borderColor": PROFESSIONAL_COLORS['red'],
                        "backgroundColor": "transparent",
                        "borderWidth": 2,
                        "pointRadius": 0,
                        "yAxisID": "y"
                    },
                    {
                        "label": "Histogram",
                        "data": histogram.tolist(),
                        "type": "bar",
                        "backgroundColor": histogram.apply(lambda x: PROFESSIONAL_COLORS['green'] if x >= 0 else PROFESSIONAL_COLORS['red']).tolist(),
                        "yAxisID": "y"
                    }
                ]
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "backgroundColor": PROFESSIONAL_COLORS['background'],
                "plugins": {
                    "title": {
                        "display": True,
                        "text": f"{ticker} - MACD ({fast},{slow},{signal})",
                        "color": PROFESSIONAL_COLORS['text'],
                        "font": {"size": 16, "weight": "bold"}
                    },
                    "legend": {
                        "display": True,
                        "position": "top",
                        "labels": {"color": PROFESSIONAL_COLORS['text']}
                    }
                },
                "scales": {
                    "x": {
                        "grid": {"color": PROFESSIONAL_COLORS['grid']},
                        "ticks": {"color": PROFESSIONAL_COLORS['text']}
                    },
                    "y": {
                        "grid": {"color": PROFESSIONAL_COLORS['grid']},
                        "ticks": {"color": PROFESSIONAL_COLORS['text']},
                        "title": {
                            "display": True,
                            "text": "MACD",
                            "color": PROFESSIONAL_COLORS['text']
                        }
                    }
                }
            }
        }

        return chart_config

    except Exception as e:
        logger.error(f"Error generating MACD chart for {ticker}: {e}")
        return None


def generate_volume_chart(ticker: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate volume analysis chart with moving average.

    Args:
        ticker: Stock ticker symbol
        df: DataFrame with OHLCV data

    Returns:
        Chart.js configuration dictionary
    """
    try:
        dates = df.index.strftime('%Y-%m-%d').tolist()
        volumes = df['Volume'].tolist()
        volume_ma = df['Volume'].rolling(window=20).mean().tolist()

        # Color bars based on price movement
        closes = df['Close'].tolist()
        opens = df['Open'].tolist()
        colors = [PROFESSIONAL_COLORS['green'] if close >= open else PROFESSIONAL_COLORS['red']
                 for open, close in zip(opens, closes)]

        chart_config = {
            "type": "bar",
            "data": {
                "labels": dates,
                "datasets": [
                    {
                        "label": "Wolumen",
                        "data": volumes,
                        "backgroundColor": colors,
                        "borderColor": colors,
                        "borderWidth": 1
                    },
                    {
                        "label": "MA20",
                        "data": volume_ma,
                        "type": "line",
                        "borderColor": PROFESSIONAL_COLORS['orange'],
                        "backgroundColor": "transparent",
                        "borderWidth": 2,
                        "pointRadius": 0
                    }
                ]
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "backgroundColor": PROFESSIONAL_COLORS['background'],
                "plugins": {
                    "title": {
                        "display": True,
                        "text": f"{ticker} - Wolumen",
                        "color": PROFESSIONAL_COLORS['text'],
                        "font": {"size": 16, "weight": "bold"}
                    },
                    "legend": {
                        "display": True,
                        "position": "top",
                        "labels": {"color": PROFESSIONAL_COLORS['text']}
                    }
                },
                "scales": {
                    "x": {
                        "grid": {"color": PROFESSIONAL_COLORS['grid']},
                        "ticks": {"color": PROFESSIONAL_COLORS['text']}
                    },
                    "y": {
                        "grid": {"color": PROFESSIONAL_COLORS['grid']},
                        "ticks": {
                            "color": PROFESSIONAL_COLORS['text'],
                            "callback": {
                                "type": "function",
                                "body": """
                                function(value) {
                                    if (value >= 1000000) {
                                        return (value / 1000000).toFixed(1) + 'M';
                                    }
                                    if (value >= 1000) {
                                        return (value / 1000).toFixed(1) + 'K';
                                    }
                                    return value.toFixed(0);
                                }
                                """
                            }
                        },
                        "title": {
                            "display": True,
                            "text": "Wolumen",
                            "color": PROFESSIONAL_COLORS['text']
                        }
                    }
                }
            }
        }

        return chart_config

    except Exception as e:
        logger.error(f"Error generating volume chart for {ticker}: {e}")
        return None


def generate_comprehensive_chart_package(ticker: str, company_name: str, df: pd.DataFrame,
                                      ma5: Optional[float] = None, ma10: Optional[float] = None,
                                      ma20: Optional[float] = None, trend_label: str = "Boczny") -> List[Dict[str, Any]]:
    """
    Generate complete package of professional charts for a stock.

    Args:
        ticker: Stock ticker symbol
        company_name: Full company name
        df: DataFrame with OHLCV data
        ma5: 5-day moving average
        ma10: 10-day moving average
        ma20: 20-day moving average
        trend_label: Polish trend label

    Returns:
        List of Chart.js configurations
    """
    charts = []

    try:
        # 1. Main candlestick chart with MA and volume
        candlestick = generate_candlestick_chart(ticker, company_name, df, ma5, ma10, ma20, trend_label)
        if candlestick:
            charts.append(candlestick)

        # 2. RSI chart
        rsi = generate_rsi_chart(ticker, df)
        if rsi:
            charts.append(rsi)

        # 3. MACD chart
        macd = generate_macd_chart(ticker, df)
        if macd:
            charts.append(macd)

        # 4. Volume chart
        volume = generate_volume_chart(ticker, df)
        if volume:
            charts.append(volume)

        logger.info(f"Generated {len(charts)} professional charts for {ticker}")

    except Exception as e:
        logger.error(f"Error generating comprehensive charts for {ticker}: {e}")

    return charts


def save_professional_chart(chart_config: Dict[str, Any], ticker: str, chart_type: str,
                           output_dir: str = "professional_charts") -> Optional[str]:
    """
    Save professional chart configuration as JSON file.

    Args:
        chart_config: Chart.js configuration dictionary
        ticker: Stock ticker symbol
        chart_type: Type of chart (candlestick, rsi, macd, volume)
        output_dir: Output directory for chart files

    Returns:
        Path to saved file or None on failure
    """
    if chart_config is None:
        return None

    try:
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{ticker}_{chart_type}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chart_config, f, indent=2, ensure_ascii=False)

        logger.info(f"Professional chart saved: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Error saving professional chart for {ticker}: {e}")
        return None


def generate_charts_for_recommendations(stocks: List[Dict[str, Any]],
                                     max_stocks: int = 5) -> List[str]:
    """
    Generate professional charts for recommended stocks.

    Args:
        stocks: List of stock dictionaries with data
        max_stocks: Maximum number of stocks to process

    Returns:
        List of paths to generated chart files
    """
    chart_files = []
    processed = 0

    for stock in stocks[:max_stocks]:
        ticker = stock.get('ticker', 'UNKNOWN')
        company_name = stock.get('name', ticker)
        historical_prices = stock.get('historical_prices')

        if historical_prices is None or historical_prices.empty:
            continue

        # Get technical indicators
        ma5 = stock.get('ma5')
        ma10 = stock.get('ma10')
        ma20 = stock.get('ma20')
        trend_label = stock.get('trend_label', 'Boczny')

        # Generate complete chart package
        charts = generate_comprehensive_chart_package(
            ticker, company_name, historical_prices, ma5, ma10, ma20, trend_label
        )

        # Save all charts
        for i, chart in enumerate(charts):
            chart_types = ['candlestick', 'rsi', 'macd', 'volume']
            chart_type = chart_types[i] if i < len(chart_types) else 'unknown'

            file_path = save_professional_chart(chart, ticker, chart_type)
            if file_path:
                chart_files.append(file_path)

        processed += 1
        if processed >= max_stocks:
            break

    logger.info(f"Generated professional charts for {processed} stocks")
    return chart_files