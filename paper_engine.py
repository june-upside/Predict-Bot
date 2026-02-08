from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class Position:
    qty: float = 0.0
    avg_price: float = 0.0


@dataclass
class Trade:
    step: int
    token_id: str
    side: str  # BUY / SELL
    qty: float
    price: float
    notional: float
    reason: str


class PaperPortfolio:
    def __init__(self, starting_cash: float = 1000.0):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []

    def _get_pos(self, token_id: str) -> Position:
        if token_id not in self.positions:
            self.positions[token_id] = Position()
        return self.positions[token_id]

    def buy(self, step: int, token_id: str, qty: float, price: float, reason: str) -> bool:
        notional = qty * price
        if qty <= 0 or price <= 0:
            return False
        if self.cash < notional:
            return False

        pos = self._get_pos(token_id)
        new_qty = pos.qty + qty
        if new_qty <= 0:
            return False

        pos.avg_price = ((pos.qty * pos.avg_price) + (qty * price)) / new_qty
        pos.qty = new_qty
        self.cash -= notional
        self.trades.append(Trade(step, token_id, "BUY", qty, price, notional, reason))
        return True

    def sell(self, step: int, token_id: str, qty: float, price: float, reason: str) -> bool:
        if qty <= 0 or price <= 0:
            return False
        pos = self._get_pos(token_id)
        if pos.qty < qty:
            return False

        notional = qty * price
        pos.qty -= qty
        self.cash += notional
        self.trades.append(Trade(step, token_id, "SELL", qty, price, notional, reason))

        if pos.qty == 0:
            pos.avg_price = 0.0
        return True

    def unrealized_pnl(self, marks: Dict[str, float]) -> float:
        pnl = 0.0
        for token_id, pos in self.positions.items():
            if pos.qty <= 0:
                continue
            mark = marks.get(token_id, pos.avg_price)
            pnl += pos.qty * (mark - pos.avg_price)
        return pnl

    def realized_pnl(self) -> float:
        # cash change minus still-open cost basis is a robust realized proxy
        open_cost = 0.0
        for p in self.positions.values():
            open_cost += p.qty * p.avg_price
        return (self.cash + open_cost) - self.starting_cash

    def equity(self, marks: Dict[str, float]) -> float:
        pos_value = 0.0
        for token_id, pos in self.positions.items():
            if pos.qty <= 0:
                continue
            pos_value += pos.qty * marks.get(token_id, pos.avg_price)
        return self.cash + pos_value

    def to_state(self) -> dict:
        return {
            "cash": self.cash,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "trades": [asdict(t) for t in self.trades],
        }
