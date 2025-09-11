# Odds Providers

This project supports two odds providers for generating `odds.csv` snapshots under `data/snapshots/YYYY-MM-DD/`.

Keep this doc in sync with:
- `core/data/ingestion_adapters.py` (providers and normalization)
- `docs/SNAPSHOT_SCHEMAS.md` (canonical headers)
- `nfl_cli.py` (`odds-snapshot` command)

## Providers

- Mock Provider (default)
  - Adapter: `MockOddsAdapter`
  - Source: synthetic offers generated via `core/data/odds_snapshot.py:write_mock_odds_snapshot`
  - Purpose: deterministic, schema-correct data for development and tests
  - No API keys required

- TheOddsAPI (live)
  - Adapter: `TheOddsAPIAdapter`
  - API: https://the-odds-api.com/
  - Purpose: fetch real odds for supported player markets
  - Requires environment variables:
    - `THEODDSAPI_KEY` (required in order to fetch)
    - `THEODDSAPI_BASE` (optional, default: `https://api.the-odds-api.com/v4`)

## Output Files and Schemas

Both providers write to the current snapshot directory:

- `odds.csv` — canonical current odds
- `odds_history.csv` — header ensured; historical writing may be extended later

Canonical headers are defined in `docs/SNAPSHOT_SCHEMAS.md` and mirrored by `SNAPSHOT_MIN_COLUMNS` in `core/data/ingestion_adapters.py`.

- `odds.csv` headers:
  - `["timestamp","book","market","player_id","team_id","line","over_odds","under_odds"]`

- `odds_history.csv` headers:
  - `["ts_utc","book","market","selection_id","line","price","event_id","is_closing"]`

Empty/header-only files are acceptable. Tests validate these headers exactly.

## CLI Usage

Mock provider (default):

```bash
python nfl_cli.py odds-snapshot --provider mock --max-offers 100
```

Live provider (requires key):

```bash
export THEODDSAPI_KEY=your_api_key
python nfl_cli.py odds-snapshot --provider live --books DK,FanDuel --markets "Passing Yards,Receptions" --max-offers 250
```

Notes:
- `--books` and `--markets` are optional comma-delimited filters. Provider-specific naming is normalized to canonical `book` and `market` values when possible.
- The command will always ensure `odds_history.csv` exists with the correct header in the snapshot directory.

## Normalization

The adapters attempt to map provider-specific fields to the canonical schema. For example:
- `book` is lowercased (e.g., `DraftKings` → `draftkings`)
- `market` is mapped to provider keys when available and lowercased
- `player_id` uses provider `description/name` when an explicit ID is not present
- Prices are mapped to `over_odds` / `under_odds` when the provider exposes side-specific outcomes

When fields are unavailable, adapters fill them with `null` and preserve column order as defined by `SNAPSHOT_MIN_COLUMNS`.

## Testing

- `tests/test_betting_props.py::test_mock_provider_schema_matches_snapshot_mapping` ensures `odds.csv` header exactly matches `SNAPSHOT_MIN_COLUMNS["odds.csv"]` for the mock provider.
- `/betting/props` endpoint reads the latest snapshot `odds.csv` and returns filtered offers using canonical `book` and `market` (see `core/data/market_mapping.py`).
