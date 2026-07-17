from config import Config


def test_ashare_config_present():
    assert Config.ASHARE_INITIAL_CAPITAL == 1_000_000.0
    assert Config.ASHARE_TOP_K == 10
    assert Config.ASHARE_COMMISSION_RATE == 0.00025
    assert Config.ASHARE_MIN_COMMISSION == 5.0
    assert Config.ASHARE_STAMP_TAX == 0.001
    assert Config.ASHARE_LIMIT_PCT == 0.10
    assert Config.ASHARE_LOT_SIZE == 100
    assert isinstance(Config.ASHARE_CACHE_DIR, str)
