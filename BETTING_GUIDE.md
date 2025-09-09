# Betting Guide

This guide explains how to use model outputs to inform betting decisions. Content is plain ASCII for compatibility.

## Key Concepts

- Markets: Player props such as Passing Yards, Rushing Yards, Receptions, Receiving Yards, Touchdowns, Fantasy Points (PPR).
- Line: The threshold set by a sportsbook for a prop (for example, 275.5 passing yards).
- Odds: The payout multiplier, typically American odds (e.g., -110, +120).
- Edge: Model predicted probability or value minus implied probability from odds.
- Value Rating: Heuristic grade (for example, STRONG, GOOD, FAIR, PASS) derived from edge and confidence.

## Using Predictions

1. Generate player predictions via the CLI or API.
2. Compare predicted distributions (point estimate and confidence interval) to the sportsbook line.
3. Estimate probability that the outcome exceeds the line (Over) or falls short (Under).
4. Convert odds to implied probability and compute expected value.

## Example Workflow

- Model predicts Player A Receiving Yards mean = 72, CI = [60, 84].
- Sportsbook line is 65.5 at -110.
- Approximate probability over 65.5 using normal or bootstrap from historical residuals.
- Compute EV = p(win)*payout - (1-p(win)).
- If EV is positive and confidence meets threshold, consider it a value bet.

## Risk Management

- Bankroll: Fixed fraction Kelly or flat staking between 0.5 percent and 2.0 percent per wager.
- Correlation: Avoid overexposure to correlated props (same team, same game).
- Injury and News: Verify late-breaking news before placing wagers.

## Limitations

- Models are approximations and can be wrong. Past performance is not indicative of future results.
- Lines and odds move. Always use current numbers from the sportsbook.

## Glossary

- PPR: Point Per Reception fantasy scoring format.
- CI: Confidence Interval, a range representing prediction uncertainty.
- Implied Probability: Probability implied by betting odds.
