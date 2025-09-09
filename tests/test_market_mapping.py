import pytest
from core.data.market_mapping import to_internal, normalize_book, normalize_market


def test_book_normalization_variants():
    assert normalize_book("DK") == "draftkings"
    assert normalize_book("Draft Kings") == "draftkings"
    assert normalize_book("FanDuel") == "fanduel"
    assert normalize_book("BetMGM") == "betmgm"


def test_market_normalization_variants():
    assert normalize_market("Passing Yards") == "player_pass_yds"
    assert normalize_market("pass yds") == "player_pass_yds"
    assert normalize_market("Passing TDs") == "player_pass_tds"
    assert normalize_market("Interceptions") == "player_ints"
    assert normalize_market("Rushing + Receiving Yards") == "player_rush_rec_yds"


def test_to_internal_ok():
    book, market = to_internal("DraftKings", "Passing Yards")
    assert book == "draftkings"
    assert market == "player_pass_yds"


def test_to_internal_unknown_market():
    with pytest.raises(ValueError):
        to_internal("DraftKings", "Some Unknown Market Name")
