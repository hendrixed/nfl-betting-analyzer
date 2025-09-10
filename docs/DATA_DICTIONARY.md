A. League & reference

Teams, divisions, conferences, aliases

Stadiums (surface, roof/dome, elevation, city/lat/lon)

Coaches (HC/OC/DC with start/end dates)

Officials & referee crews

Players (canonical IDs, name variants, team, pos, handedness, draft, height/weight, DOB)

teams.csv (keywords only)

| team_id | abbr | name | conf | div | hc | oc | dc | bye_week | alias[] |

stadiums.csv

| stadium_id | name | team_id | city | state | lat | lon | surface | roof | elevation_ft |

players.csv

| player_id | name | name_alt[] | pos | team_id | status | height_in | weight_lb | dob | draft_team | draft_year | draft_pick | hand | college |

coaches.csv

| team_id | season | hc | oc | dc | scheme_off | scheme_def |

officials.csv

| game_id | crew_id | referee | umpire | down_judge | line_judge | side_judge | back_judge | field_judge |

B. Rosters / depth / availability

Weekly rosters

Depth charts (off/def/ST with slot/package/role)

Transactions (sign/waive/elevations)

Injury reports (practice status, game designation)

Gameday inactives

rosters.csv

| season | week | team_id | player_id | pos | jersey | status | last_updated |

depth_charts.csv

| season | week | team_id | side | pos | slot | player_id | role | package | depth_rank |

injuries.csv

| report_date | team_id | player_id | pos | practice_status | designation | game_status | return_date |

inactives.csv

| season | week | team_id | player_id | pos | reason | declared_time |

transactions.csv

| date | team_id | player_id | type | detail |

C. Schedule / games / context

Full schedule (TZ-aware kickoffs)

Game metadata: stadium roof state at kickoff, broadcaster/network, referee crew

Derived: rest days, travel miles/tz delta, short week, bye windows

schedules.csv

| season | week | game_id | kickoff_utc | home_id | away_id | network | referee_crew | stadium_id |

games.csv

| game_id | roof_state | field_type | attendance | duration | closing_spread | closing_total |

team_context.csv (derived)

| season | week | team_id | opp_id | rest_days | travel_miles | tz_delta | pace_sn | pace_all | PROE | lead_pct | trail_pct | neutral_pct |

D. Play-by-play & drives

PBP with EPA, WPA, SR, air_yards, YAC, pressure, blitz, personnel, formation, motion, shotgun, no-huddle, penalties

Drive summaries (start yardline, time, result, points)

pbp.csv

| play_id | game_id | q | clock | offense | defense | yardline | down | distance | yards_gained | play_type | epa | wpa | success | air_yards | yac | pressure | blitz | personnel_off | formation | motion | shotgun | no_huddle | penalty_yards |

drives.csv

| drive_id | game_id | offense | start_q | start_clock | start_yardline | end_q | end_clock | result | plays | yards | time_elapsed | points |

E. Participation & usage

Snap counts & %

Routes run, route participation %

Carry share, target share

Red-zone & goal-line shares

Alignments: slot, wide, inline, backfield

Pass-block snaps

snaps.csv

| season | week | team_id | player_id | off_snaps | def_snaps | st_snaps | off_pct | def_pct | st_pct |

routes.csv

| season | week | team_id | player_id | routes_run | route_participation |

usage_shares.csv

| season | week | team_id | player_id | carry_share | target_share | rz_touch_share | gl_carry_share | pass_block_snaps | align_slot | align_wide | align_inline | align_backfield |

F. Box & advanced stats (by game)

QB: pass att/comp/yds/TD/INT/sacks, rush att/yds/TD, fumbles, aDOT, air yards

RB: rush att/yds/TD, targets, receptions, rec yds/TD, YAC, missed tackles

WR/TE: targets, receptions, yards, TD, air yards, YAC, drops, aDOT, alignments

K: FG att/made by distance bin, XP att/made

DST: sacks, INT, fumbles, TDs, points allowed buckets

box_passing.csv

| game_id | player_id | att | comp | yds | td | int | sacks | sack_yards | ypa | air_yards | aDOT | fumbles |

box_rushing.csv

| game_id | player_id | att | yds | td | long | ypc | fumbles |

box_receiving.csv

| game_id | player_id | targets | rec | yds | td | air_yards | yac | aDOT | drops | long |

box_defense.csv

| game_id | player_id | tackles | assists | sacks | tfl | qb_hits | ints | pbu | td |

kicking.csv

| game_id | player_id | fg_0_39 | fg_40_49 | fg_50p | fg_att | fg_made | xp_att | xp_made |

G. Team rates & splits

Pace (situation-neutral and overall), PROE

Script splits (lead/trail/tie)

Red-zone & goal-to-go efficiency

3rd/4th down aggressiveness

Opponent allowed metrics vs position

team_splits.csv

| season | week | team_id | pace_sn | pace_all | proe | rz_eff | g2g_eff | third_conv | fourth_att | vs_pos_rb_yds | vs_pos_wr_yds | vs_pos_te_yds |

H. Weather (forecast + realized)

weather.csv

| game_id | stadium_id | temp_f | humidity | wind_mph | wind_dir | precip_type | precip_prob | conditions | timestamp_utc |

I. Odds / props (live + history)

Book, market, selection, line, price (American), time series (open, live, closing)

Player props: pass yards/TD/INT/comp/att, rush att/yds/TD, rec/rec yds/TD, longest comp/rush/rec, QB rush yards, tackles+assists, sacks, interceptions, kicker FG made, kicking points

Team/game markets: spread, total, team totals, 1H/2H, alt lines

odds.csv (normalized)

| ts_utc | book | market | selection_id | selection_name | team_id | player_id | line | price | source |

odds_history.csv

| ts_utc | book | market | selection_id | line | price | event_id | is_closing |

J. Fantasy scaffolding

fantasy_scoring.json (per scoring system)

PPR definitions, bonuses, DST scoring, fumble rules

K. Modeling artifacts & evaluation

models/streamlined/ .pkl & sidecar .json:

| model_id | position | target | algo | version | train_start | train_end | features[] | r2 | mae | rmse | calibration_png |

reports/backtests/ <market>/:

| hit_rate | roi | brier | crps | n_bets | start_date | end_date | plot_paths[] |

