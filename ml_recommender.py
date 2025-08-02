# ml_recommender.py
import numpy as np

class MLRecommender:
    def __init__(self):
        pass

    def recommend_portfolio(self, portfolio_holdings):
        # Dummy recommendation: equal weight for each holding.
        num_assets = len(portfolio_holdings)
        if num_assets == 0:
            return []
        equal_weight = 1.0 / num_assets
        recommendations = []
        for holding in portfolio_holdings:
            recommendations.append({
                'ticker': holding['ticker'],
                'recommended_weight': equal_weight
            })
        return recommendations
