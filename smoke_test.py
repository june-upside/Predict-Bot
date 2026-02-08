import bot


def run():
    assert bot.detect_binary_sum_arb(0.49, 0.49, 0.995) is True
    assert bot.detect_binary_sum_arb(0.51, 0.50, 0.995) is False

    edges = bot.compute_forecast_edges(0.80, 0.50, 0.55)
    assert round(edges['yes_edge'], 2) == 0.30
    assert round(edges['no_edge'], 2) == -0.35

    p = bot.maker_buy_price_from_ask(0.51, 0.001)
    assert round(p, 3) == 0.509

    print('SMOKE_TEST_OK')


if __name__ == '__main__':
    run()
