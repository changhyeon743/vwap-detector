#!/bin/bash
# VWAP Backtest Runner
# Usage: ./run_backtest.sh [command]

cd "$(dirname "$0")"

case "$1" in
    backfill|bf)
        echo "📥 Backfilling historical data..."
        python3 -m backtest.run backfill --days ${2:-365} --timeframe ${3:-1h}
        ;;
    update|up)
        echo "🔄 Updating to latest data..."
        python3 -m backtest.run backfill --update --timeframe ${2:-1h}
        ;;
    test|bt)
        echo "📊 Running backtest..."
        python3 -m backtest.run backtest --timeframe ${2:-1h} --save
        ;;
    optimize|opt)
        echo "🔍 Running optimization..."
        python3 -m backtest.run optimize --trials ${2:-100} --timeframe ${3:-1h}
        ;;
    grid)
        echo "🔍 Running grid search..."
        python3 -m backtest.run optimize --grid --timeframe ${2:-1h}
        ;;
    info)
        echo "ℹ️ Database info..."
        python3 -m backtest.run info --timeframe ${2:-1h}
        ;;
    *)
        echo "VWAP Backtest Runner"
        echo ""
        echo "Commands:"
        echo "  backfill [days] [tf]  - Fetch historical data (default: 365 days, 1h)"
        echo "  update [tf]           - Update to latest data"
        echo "  test [tf]             - Run single backtest"
        echo "  optimize [trials] [tf]- Run Optuna optimization"
        echo "  grid [tf]             - Run grid search"
        echo "  info [tf]             - Show database info"
        echo ""
        echo "Examples:"
        echo "  ./run_backtest.sh backfill 180 5m"
        echo "  ./run_backtest.sh test 1h"
        echo "  ./run_backtest.sh optimize 200 1h"
        ;;
esac
