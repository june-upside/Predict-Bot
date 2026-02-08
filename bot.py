#!/usr/bin/env python3
"""
Polymarket Weather Arbitrage Bot (Scaffold)
"""

import os
import time
import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import requests
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("weather-arb-bot")


@dataclass
class Config:
    polymarket_host: str
    polymarket_chain_id: int
    polymarket_private_key: str
    polymarket_funder: Optional[str]
    polymarket_api_key: Optional[str]
    polymarket_api_secret: Optional[str]
    polymarket_api_passphrase: Optional[str]
    owm_api_key: str
    owm_lat: float
    owm_lon: float
    owm_units: str
    yes_token_id: str
    no_token_id: str
    simulation_mode: bool
    loop_seconds: int
    request_timeout: int
    min_edge: float
    max_order_size: float
    maker_price_improvement: float
    max_sum_arbitrage_threshold: float

    @staticmethod
    def from_env() -> "Config":
        def _get_bool(name: str, default: str = "true") -> bool:
            return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}

        return Config(
            polymarket_host=os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com"),
            polymarket_chain_id=int(os.getenv("POLYMARKET_CHAIN_ID", "137")),
            polymarket_private_key=os.getenv("POLYMARKET_PRIVATE_KEY", ""),
            polymarket_funder=os.getenv("POLYMARKET_FUNDER"),
            polymarket_api_key=os.getenv("POLYMARKET_API_KEY"),
            polymarket_api_secret=os.getenv("POLYMARKET_API_SECRET"),
            polymarket_api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE"),
            owm_api_key=os.getenv("OPENWEATHERMAP_API_KEY", ""),
            owm_lat=float(os.getenv("OPENWEATHERMAP_LAT", "40.7128")),
            owm_lon=float(os.getenv("OPENWEATHERMAP_LON", "-74.0060")),
            owm_units=os.getenv("OPENWEATHERMAP_UNITS", "metric"),
            yes_token_id=os.getenv("WEATHER_MARKET_YES_TOKEN_ID", ""),
            no_token_id=os.getenv("WEATHER_MARKET_NO_TOKEN_ID", ""),
            simulation_mode=_get_bool("SIMULATION_MODE", "true"),
            loop_seconds=int(os.getenv("LOOP_SECONDS", "30")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "15")),
            min_edge=float(os.getenv("MIN_EDGE", "0.05")),
            max_order_size=float(os.getenv("MAX_ORDER_SIZE", "20.0")),
            maker_price_improvement=float(os.getenv("MAKER_PRICE_IMPROVEMENT", "0.001")),
            max_sum_arbitrage_threshold=float(os.getenv("MAX_SUM_ARBITRAGE_THRESHOLD", "0.995")),
        )


class RateLimiter:
    def __init__(self, min_interval_sec: float):
        self.min_interval_sec = min_interval_sec
        self._last_call = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)
        self._last_call = time.time()


class WeatherClient:
    BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"

    def __init__(self, api_key: str, timeout: int = 15, min_interval_sec: float = 1.0):
        self.api_key = api_key
        self.timeout = timeout
        self.rl = RateLimiter(min_interval_sec=min_interval_sec)

    def get_tomorrow_rain_probability(self, lat: float, lon: float, units: str = "metric") -> float:
        self.rl.wait()
        params = {"lat": lat, "lon": lon, "appid": self.api_key, "units": units}
        resp = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        forecast_entries = data.get("list", [])
        if not forecast_entries:
            raise ValueError("No forecast data from OpenWeatherMap.")
        first_ts = forecast_entries[0]["dt"]
        first_day = time.gmtime(first_ts).tm_yday
        tomorrow_day = (first_day + 1) % 366
        pops = [float(e.get("pop", 0.0)) for e in forecast_entries if time.gmtime(e["dt"]).tm_yday == tomorrow_day]
        if not pops:
            pops = [float(e.get("pop", 0.0)) for e in forecast_entries[:8]]
        rain_prob = sum(pops) / len(pops)
        return max(0.0, min(1.0, rain_prob))


class PolymarketCLOB:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = self._build_client()
        self.rl = RateLimiter(min_interval_sec=0.35)

    def _build_client(self) -> ClobClient:
        if not self.cfg.polymarket_private_key:
            raise ValueError("Missing POLYMARKET_PRIVATE_KEY")
        client = ClobClient(host=self.cfg.polymarket_host, key=self.cfg.polymarket_private_key, chain_id=self.cfg.polymarket_chain_id)
        if self.cfg.polymarket_api_key and self.cfg.polymarket_api_secret and self.cfg.polymarket_api_passphrase:
            api_creds = {"key": self.cfg.polymarket_api_key, "secret": self.cfg.polymarket_api_secret, "passphrase": self.cfg.polymarket_api_passphrase}
        else:
            api_creds = client.create_or_derive_api_creds()
        client.set_api_creds(api_creds)
        return client

    def get_best_bid_ask(self, token_id: str) -> Tuple[Optional[float], Optional[float]]:
        self.rl.wait()
        ob = self.client.get_order_book(token_id)
        bids = ob.get("bids", []) if isinstance(ob, dict) else getattr(ob, "bids", [])
        asks = ob.get("asks", []) if isinstance(ob, dict) else getattr(ob, "asks", [])
        best_bid = float(bids[0]["price"]) if bids else None
        best_ask = float(asks[0]["price"]) if asks else None
        return best_bid, best_ask

    def place_maker_limit_order(self, token_id: str, side: str, price: float, size: float, simulation_mode: bool = True) -> Dict[str, Any]:
        price = max(0.001, min(0.999, round(price, 4)))
        size = round(max(0.0, size), 4)
        payload = {"token_id": token_id, "side": side, "price": price, "size": size, "order_type": "GTC", "maker_intent": True}
        if simulation_mode:
            logger.info("[SIMULATION] Would place order: %s", json.dumps(payload))
            return {"simulated": True, "order": payload}
        self.rl.wait()
        order_args = OrderArgs(token_id=token_id, side=side, price=price, size=size)
        signed_order = self.client.create_order(order_args)
        resp = self.client.post_order(signed_order, OrderType.GTC)
        return {"simulated": False, "response": resp}


def detect_binary_sum_arb(yes_ask: Optional[float], no_ask: Optional[float], threshold: float) -> bool:
    return yes_ask is not None and no_ask is not None and (yes_ask + no_ask) < threshold


def compute_forecast_edges(forecast_prob_yes: float, yes_ask: Optional[float], no_ask: Optional[float]) -> Dict[str, float]:
    return {
        "yes_edge": (forecast_prob_yes - yes_ask) if yes_ask is not None else float("-inf"),
        "no_edge": ((1.0 - forecast_prob_yes) - no_ask) if no_ask is not None else float("-inf"),
    }


def maker_buy_price_from_ask(ask: float, improvement: float) -> float:
    return max(0.001, ask - improvement)


class WeatherArbBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.weather = WeatherClient(cfg.owm_api_key, timeout=cfg.request_timeout)
        self.pm = PolymarketCLOB(cfg)

    def validate_config(self):
        required = []
        if not self.cfg.owm_api_key:
            required.append("OPENWEATHERMAP_API_KEY")
        if not self.cfg.yes_token_id:
            required.append("WEATHER_MARKET_YES_TOKEN_ID")
        if not self.cfg.no_token_id:
            required.append("WEATHER_MARKET_NO_TOKEN_ID")
        if required:
            raise ValueError("Missing required env vars: " + ", ".join(required))
        if self.cfg.polymarket_chain_id != 137:
            raise ValueError("This scaffold assumes Polygon mainnet (chain_id=137).")

    def run_once(self):
        rain_prob = self.weather.get_tomorrow_rain_probability(self.cfg.owm_lat, self.cfg.owm_lon, self.cfg.owm_units)
        yes_bid, yes_ask = self.pm.get_best_bid_ask(self.cfg.yes_token_id)
        no_bid, no_ask = self.pm.get_best_bid_ask(self.cfg.no_token_id)
        logger.info("YES bid/ask=%s/%s NO bid/ask=%s/%s forecast=%.3f", yes_bid, yes_ask, no_bid, no_ask, rain_prob)

        sum_arb = detect_binary_sum_arb(yes_ask, no_ask, self.cfg.max_sum_arbitrage_threshold)
        edges = compute_forecast_edges(rain_prob, yes_ask, no_ask)

        if sum_arb and yes_ask is not None and no_ask is not None:
            self.pm.place_maker_limit_order(self.cfg.yes_token_id, "BUY", maker_buy_price_from_ask(yes_ask, self.cfg.maker_price_improvement), self.cfg.max_order_size, self.cfg.simulation_mode)
            self.pm.place_maker_limit_order(self.cfg.no_token_id, "BUY", maker_buy_price_from_ask(no_ask, self.cfg.maker_price_improvement), self.cfg.max_order_size, self.cfg.simulation_mode)
            return

        if yes_ask is not None and edges["yes_edge"] >= self.cfg.min_edge:
            self.pm.place_maker_limit_order(self.cfg.yes_token_id, "BUY", maker_buy_price_from_ask(yes_ask, self.cfg.maker_price_improvement), self.cfg.max_order_size, self.cfg.simulation_mode)

        if no_ask is not None and edges["no_edge"] >= self.cfg.min_edge:
            self.pm.place_maker_limit_order(self.cfg.no_token_id, "BUY", maker_buy_price_from_ask(no_ask, self.cfg.maker_price_improvement), self.cfg.max_order_size, self.cfg.simulation_mode)

    def run_forever(self):
        self.validate_config()
        while True:
            try:
                self.run_once()
            except Exception as e:
                logger.exception("loop error: %s", e)
            time.sleep(self.cfg.loop_seconds)


if __name__ == "__main__":
    load_dotenv()
    cfg = Config.from_env()
    bot = WeatherArbBot(cfg)
    bot.run_forever()
