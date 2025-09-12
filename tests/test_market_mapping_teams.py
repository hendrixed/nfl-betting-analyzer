import pytest
from core.data.market_mapping import normalize_team_name


def test_normalize_team_kc_variants():
    assert normalize_team_name("Kansas City Chiefs") == "KC"
    assert normalize_team_name("Kansas City") == "KC"
    assert normalize_team_name("Chiefs") == "KC"
    assert normalize_team_name("kc") == "KC"


def test_normalize_team_raiders_variants():
    assert normalize_team_name("Las Vegas Raiders") == "LV"
    assert normalize_team_name("Raiders") == "LV"
    assert normalize_team_name("LVR") == "LV"
    assert normalize_team_name("lv") == "LV"


def test_normalize_team_washington_variants():
    assert normalize_team_name("Washington Commanders") == "WAS"
    assert normalize_team_name("Washington Football Team") == "WAS"
    assert normalize_team_name("Washington") == "WAS"
    assert normalize_team_name("was") == "WAS"


def test_normalize_team_la_variants():
    assert normalize_team_name("Los Angeles Rams") == "LAR"
    assert normalize_team_name("Rams") == "LAR"
    assert normalize_team_name("Los Angeles Chargers") == "LAC"
    assert normalize_team_name("Chargers") == "LAC"


def test_normalize_team_sf_variants():
    assert normalize_team_name("San Francisco 49ers") == "SF"
    assert normalize_team_name("49ers") == "SF"
    assert normalize_team_name("Niners") == "SF"
