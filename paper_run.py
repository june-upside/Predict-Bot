#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import random
import time
import re
import requests
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from dotenv import load_dotenv
from py_clob_client.client import ClobClient

import bot
from paper_engine import PaperPortfolio


@dataclass
class PaperConfig:
    steps: int
    interval_sec: float
    starting_cash: float
    max_position_per_token: float
    close_edge_threshold: float
    synthetic_mode: bool

    @staticmethod
    def from_env() -> "PaperConfig":
        return PaperConfig(
            steps=int(os.getenv("PAPER_STEPS", "200")),
            interval_sec=float(os.getenv("PAPER_INTERVAL_SEC", "0.2")),
            starting_cash=float(os.getenv("PAPER_STARTING_CASH", "1000")),
            max_position_per_token=float(os.getenv("PAPER_MAX_POSITION", "100")),
            close_edge_threshold=float(os.getenv("PAPER_CLOSE_EDGE_THRESHOLD", "0.0")),
            synthetic_mode=os.getenv("PAPER_SYNTHETIC_MODE", "true").lower() in {"1", "true", "yes", "on"},
        )


class SyntheticFeed:
    """Fallback feed so you can test PnL pipeline without any API keys."""

    def __init__(self):
        self.rain_prob = 0.55
        self.yes_mid = 0.50

    def next(self) -> Tuple[float, float, float, float, float]:
        self.rain_prob = min(0.95, max(0.05, self.rain_prob + random.uniform(-0.03, 0.03)))
        self.yes_mid = min(0.95, max(0.05, self.yes_mid + random.uniform(-0.025, 0.025)))

        spread = 0.01
        yes_bid = max(0.001, self.yes_mid - spread / 2)
        yes_ask = min(0.999, self.yes_mid + spread / 2)

        no_mid = 1.0 - self.yes_mid + random.uniform(-0.01, 0.01)
        no_mid = min(0.95, max(0.05, no_mid))
        no_bid = max(0.001, no_mid - spread / 2)
        no_ask = min(0.999, no_mid + spread / 2)

        return self.rain_prob, yes_bid, yes_ask, no_bid, no_ask


def discover_weather_tokens(location_hint: str = "new york") -> Optional[Tuple[str, str, str]]:
    """Find an active weather-ish binary market from Gamma API.
    Returns (question, yes_token_id, no_token_id) or None.
    """
    kw = re.compile(r"\b(rain|snow|temperature|precipitation|weather|hurricane|storm)\b", re.I)
    loc = (location_hint or "").strip().lower()
    for offset in range(0, 20000, 500):
        r = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"limit": 500, "offset": offset, "active": "true", "closed": "false"},
            timeout=20,
        )
        r.raise_for_status()
        arr = r.json()
        if not arr:
            break
        for m in arr:
            q = m.get("question", "")
            if not kw.search(q):
                continue
            if loc and loc not in q.lower():
                continue
            toks_raw = m.get("clobTokenIds")
            toks = []
            if isinstance(toks_raw, str):
                try:
                    toks = json.loads(toks_raw)
                except Exception:
                    toks = []
            elif isinstance(toks_raw, list):
                toks = toks_raw
            if len(toks) >= 2:
                return q, str(toks[0]), str(toks[1])
    return None


class LiveFeed:
    def __init__(self, cfg: bot.Config):
        self.cfg = cfg
        self.weather = bot.WeatherClient(cfg.owm_api_key, timeout=cfg.request_timeout)
        # Public market-data client: no private key needed for paper simulation
        self.clob = ClobClient(host=cfg.polymarket_host, chain_id=cfg.polymarket_chain_id)

    def _best_bid_ask(self, token_id: str) -> Tuple[Optional[float], Optional[float]]:
        ob = self.clob.get_order_book(token_id)
        bids = ob.get("bids", []) if isinstance(ob, dict) else getattr(ob, "bids", [])
        asks = ob.get("asks", []) if isinstance(ob, dict) else getattr(ob, "asks", [])

        def _px(x):
            if isinstance(x, dict):
                return x.get("price")
            return getattr(x, "price", None)

        best_bid = float(_px(bids[0])) if bids and _px(bids[0]) is not None else None
        best_ask = float(_px(asks[0])) if asks and _px(asks[0]) is not None else None
        return best_bid, best_ask

    def next(self) -> Tuple[float, float, float, float, float]:
        rain_prob = self.weather.get_tomorrow_rain_probability(self.cfg.owm_lat, self.cfg.owm_lon, self.cfg.owm_units)
        yes_bid, yes_ask = self._best_bid_ask(self.cfg.yes_token_id)
        no_bid, no_ask = self._best_bid_ask(self.cfg.no_token_id)

        if yes_bid is None or yes_ask is None or no_bid is None or no_ask is None:
            raise RuntimeError("Order book is missing bids/asks")

        return rain_prob, yes_bid, yes_ask, no_bid, no_ask


def clamp_qty(port: PaperPortfolio, token_id: str, desired_qty: float, max_position: float) -> float:
    pos = port.positions.get(token_id)
    cur = pos.qty if pos else 0.0
    remaining = max(0.0, max_position - cur)
    return max(0.0, min(desired_qty, remaining))


def run():
    load_dotenv()
    cfg = bot.Config.from_env()
    pcfg = PaperConfig.from_env()

    portfolio = PaperPortfolio(starting_cash=pcfg.starting_cash)

    use_synth = pcfg.synthetic_mode
    if not use_synth:
        # Weather API key is optional now (Open-Meteo fallback), token IDs are required unless auto-discovered.
        if not (cfg.yes_token_id and cfg.no_token_id):
            hint = os.getenv("MARKET_LOCATION_HINT", "new york")
            discovered = discover_weather_tokens(hint)
            if discovered:
                q, yid, nid = discovered
                cfg.yes_token_id = yid
                cfg.no_token_id = nid
                print(f"[INFO] Auto-discovered market: {q}")
                print(f"[INFO] YES={yid} NO={nid}")
            else:
                print("[WARN] No active weather market token IDs found -> switching to synthetic mode")
                use_synth = True

    feed = SyntheticFeed() if use_synth else LiveFeed(cfg)

    snapshots = []
    equity_curve = []

    for step in range(1, pcfg.steps + 1):
        rain_prob, yes_bid, yes_ask, no_bid, no_ask = feed.next()
        marks: Dict[str, float] = {
            cfg.yes_token_id: (yes_bid + yes_ask) / 2,
            cfg.no_token_id: (no_bid + no_ask) / 2,
        }

        sum_arb = bot.detect_binary_sum_arb(yes_ask, no_ask, cfg.max_sum_arbitrage_threshold)
        edges = bot.compute_forecast_edges(rain_prob, yes_ask, no_ask)

        # Entry rules
        if sum_arb:
            qy = clamp_qty(portfolio, cfg.yes_token_id, cfg.max_order_size, pcfg.max_position_per_token)
            qn = clamp_qty(portfolio, cfg.no_token_id, cfg.max_order_size, pcfg.max_position_per_token)
            if qy > 0:
                portfolio.buy(step, cfg.yes_token_id, qy, bot.maker_buy_price_from_ask(yes_ask, cfg.maker_price_improvement), "sum_arb_yes")
            if qn > 0:
                portfolio.buy(step, cfg.no_token_id, qn, bot.maker_buy_price_from_ask(no_ask, cfg.maker_price_improvement), "sum_arb_no")
        else:
            if edges["yes_edge"] >= cfg.min_edge:
                qy = clamp_qty(portfolio, cfg.yes_token_id, cfg.max_order_size, pcfg.max_position_per_token)
                if qy > 0:
                    portfolio.buy(step, cfg.yes_token_id, qy, bot.maker_buy_price_from_ask(yes_ask, cfg.maker_price_improvement), "edge_yes")
            if edges["no_edge"] >= cfg.min_edge:
                qn = clamp_qty(portfolio, cfg.no_token_id, cfg.max_order_size, pcfg.max_position_per_token)
                if qn > 0:
                    portfolio.buy(step, cfg.no_token_id, qn, bot.maker_buy_price_from_ask(no_ask, cfg.maker_price_improvement), "edge_no")

        # Exit rules (simple)
        yes_pos = portfolio.positions.get(cfg.yes_token_id)
        if yes_pos and yes_pos.qty > 0 and edges["yes_edge"] <= pcfg.close_edge_threshold:
            portfolio.sell(step, cfg.yes_token_id, yes_pos.qty, yes_bid, "close_yes_edge_reverted")

        no_pos = portfolio.positions.get(cfg.no_token_id)
        if no_pos and no_pos.qty > 0 and edges["no_edge"] <= pcfg.close_edge_threshold:
            portfolio.sell(step, cfg.no_token_id, no_pos.qty, no_bid, "close_no_edge_reverted")

        eq = portfolio.equity(marks)
        equity_curve.append(eq)

        snapshots.append({
            "step": step,
            "rain_prob": rain_prob,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
            "yes_edge": edges["yes_edge"],
            "no_edge": edges["no_edge"],
            "sum_arb": sum_arb,
            "cash": portfolio.cash,
            "equity": eq,
            "unrealized_pnl": portfolio.unrealized_pnl(marks),
            "realized_proxy": portfolio.realized_pnl(),
        })

        if pcfg.interval_sec > 0:
            time.sleep(pcfg.interval_sec)

    # Metrics
    start_eq = pcfg.starting_cash
    end_eq = equity_curve[-1] if equity_curve else start_eq
    total_return = (end_eq - start_eq) / start_eq if start_eq else 0.0

    peak = -1e18
    max_dd = 0.0
    for x in equity_curve:
        peak = max(peak, x)
        dd = (peak - x) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "synthetic" if use_synth else "live-data-paper-trade",
        "steps": pcfg.steps,
        "starting_cash": start_eq,
        "ending_equity": end_eq,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "num_trades": len(portfolio.trades),
        "portfolio": portfolio.to_state(),
    }

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    json_path = out_dir / f"paper-report-{ts}.json"
    csv_path = out_dir / f"paper-snapshots-{ts}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(snapshots[0].keys()) if snapshots else ["step"])
        writer.writeheader()
        writer.writerows(snapshots)

    print("=== PAPER RUN DONE ===")
    print(f"Mode: {report['mode']}")
    print(f"Trades: {report['num_trades']}")
    print(f"Start: {start_eq:.2f} | End: {end_eq:.2f}")
    print(f"Return: {total_return*100:.2f}% | MaxDD: {max_dd*100:.2f}%")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    run()
