class UltimateEnhancedPredictor:
    """Stub predictor for tests; patched in unit tests."""
    def generate_ultimate_prediction(self, player_id: str):
        class _Pred:
            def __init__(self):
                self.final_prediction = {"fantasy_points_ppr": 0.0}
                self.confidence_score = 0.0
                self.market_edge = 0.0
        return _Pred()
