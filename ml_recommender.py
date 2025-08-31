# # ml_recommender.py
# import numpy as np

# class MLRecommender:
#     def __init__(self):
#         pass

#     def recommend_portfolio(self, portfolio_holdings):
#         # Dummy recommendation: equal weight for each holding.
#         num_assets = len(portfolio_holdings)
#         if num_assets == 0:
#             return []
#         equal_weight = 1.0 / num_assets
#         recommendations = []
#         for holding in portfolio_holdings:
#             recommendations.append({
#                 'ticker': holding['ticker'],
#                 'recommended_weight': equal_weight
#             })
#         return recommendations
# ml_recommender.py - Enhanced version with realistic recommendations
import numpy as np
import pandas as pd
import random
from datetime import datetime

class MLRecommender:
    def __init__(self):
        # Sample stock universe with sectors and risk levels
        self.stock_universe = {
            # Technology
            'AAPL': {'sector': 'Technology', 'risk_level': 'Medium', 'name': 'Apple Inc.'},
            'MSFT': {'sector': 'Technology', 'risk_level': 'Medium', 'name': 'Microsoft Corp.'},
            'GOOGL': {'sector': 'Technology', 'risk_level': 'Medium', 'name': 'Alphabet Inc.'},
            'META': {'sector': 'Technology', 'risk_level': 'High', 'name': 'Meta Platforms'},
            'NVDA': {'sector': 'Technology', 'risk_level': 'High', 'name': 'NVIDIA Corp.'},
            'AMZN': {'sector': 'Technology', 'risk_level': 'Medium', 'name': 'Amazon.com Inc.'},
            
            # Finance
            'JPM': {'sector': 'Finance', 'risk_level': 'Medium', 'name': 'JPMorgan Chase'},
            'BAC': {'sector': 'Finance', 'risk_level': 'Medium', 'name': 'Bank of America'},
            'GS': {'sector': 'Finance', 'risk_level': 'Medium', 'name': 'Goldman Sachs'},
            'WFC': {'sector': 'Finance', 'risk_level': 'Medium', 'name': 'Wells Fargo'},
            
            # Healthcare
            'JNJ': {'sector': 'Healthcare', 'risk_level': 'Low', 'name': 'Johnson & Johnson'},
            'PFE': {'sector': 'Healthcare', 'risk_level': 'Low', 'name': 'Pfizer Inc.'},
            'UNH': {'sector': 'Healthcare', 'risk_level': 'Low', 'name': 'UnitedHealth Group'},
            'ABBV': {'sector': 'Healthcare', 'risk_level': 'Medium', 'name': 'AbbVie Inc.'},
            
            # Consumer
            'KO': {'sector': 'Consumer', 'risk_level': 'Low', 'name': 'Coca-Cola Co.'},
            'PG': {'sector': 'Consumer', 'risk_level': 'Low', 'name': 'Procter & Gamble'},
            'WMT': {'sector': 'Consumer', 'risk_level': 'Low', 'name': 'Walmart Inc.'},
            'HD': {'sector': 'Consumer', 'risk_level': 'Medium', 'name': 'Home Depot'},
            
            # Energy
            'XOM': {'sector': 'Energy', 'risk_level': 'High', 'name': 'Exxon Mobil'},
            'CVX': {'sector': 'Energy', 'risk_level': 'High', 'name': 'Chevron Corp.'},
            
            # Automotive
            'TSLA': {'sector': 'Automotive', 'risk_level': 'High', 'name': 'Tesla Inc.'},
            'F': {'sector': 'Automotive', 'risk_level': 'High', 'name': 'Ford Motor Co.'},
            'GM': {'sector': 'Automotive', 'risk_level': 'High', 'name': 'General Motors'}
        }
        
        # Sector target allocations for diversification
        self.sector_targets = {
            'conservative': {
                'Technology': 0.25, 'Finance': 0.20, 'Healthcare': 0.25,
                'Consumer': 0.20, 'Energy': 0.05, 'Automotive': 0.05
            },
            'moderate': {
                'Technology': 0.35, 'Finance': 0.15, 'Healthcare': 0.20,
                'Consumer': 0.15, 'Energy': 0.10, 'Automotive': 0.05
            },
            'aggressive': {
                'Technology': 0.45, 'Finance': 0.10, 'Healthcare': 0.15,
                'Consumer': 0.10, 'Energy': 0.10, 'Automotive': 0.10
            }
        }

    def get_recommendations(self, current_tickers, recommendation_type, risk_preference):
        """
        Generate recommendations based on type and risk preference
        
        Args:
            current_tickers: List of current portfolio tickers
            recommendation_type: Type of recommendation to generate
            risk_preference: 'conservative', 'moderate', or 'aggressive'
        """
        try:
            if recommendation_type == 'portfolio_optimization':
                return self._get_optimization_recommendations(current_tickers, risk_preference)
            elif recommendation_type == 'similar_assets':
                return self._get_similar_assets(current_tickers, risk_preference)
            elif recommendation_type == 'sector_diversification':
                return self._get_sector_diversification(current_tickers, risk_preference)
            elif recommendation_type == 'risk_assessment':
                return self._get_risk_assessment(current_tickers, risk_preference)
            else:
                return None
                
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return None

    def _get_optimization_recommendations(self, current_tickers, risk_preference):
        """Generate portfolio optimization recommendations"""
        try:
            # Simulate optimized weights based on risk preference
            num_assets = len(current_tickers)
            
            if risk_preference == 'conservative':
                # More equal weighting for conservative
                base_weight = 1.0 / num_assets
                weights = [base_weight + random.uniform(-0.05, 0.05) for _ in range(num_assets)]
            elif risk_preference == 'moderate':
                # Moderate concentration
                weights = [random.uniform(0.1, 0.4) for _ in range(num_assets)]
            else:  # aggressive
                # Allow higher concentration
                weights = [random.uniform(0.05, 0.5) for _ in range(num_assets)]
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Create optimized weights dictionary
            optimized_weights = dict(zip(current_tickers, weights))
            
            # Simulate improvement metrics
            return_improvement = random.uniform(0.01, 0.03)  # 1-3% improvement
            risk_reduction = random.uniform(0.005, 0.02)     # 0.5-2% risk reduction
            sharpe_improvement = random.uniform(0.1, 0.3)    # 0.1-0.3 Sharpe improvement
            
            return {
                'optimized_weights': optimized_weights,
                'improvement': {
                    'return_improvement': return_improvement,
                    'risk_reduction': risk_reduction,
                    'sharpe_improvement': sharpe_improvement
                },
                'recommendation_type': 'Portfolio Optimization',
                'confidence': random.uniform(0.75, 0.95)
            }
            
        except Exception as e:
            print(f"Error in optimization recommendations: {e}")
            return None

    def _get_similar_assets(self, current_tickers, risk_preference):
        """Find similar assets to current holdings"""
        try:
            similar_assets = []
            
            for ticker in current_tickers:
                if ticker in self.stock_universe:
                    current_sector = self.stock_universe[ticker]['sector']
                    current_risk = self.stock_universe[ticker]['risk_level']
                    
                    # Find similar assets in same sector or risk level
                    candidates = []
                    for candidate_ticker, info in self.stock_universe.items():
                        if candidate_ticker != ticker and candidate_ticker not in current_tickers:
                            similarity_score = 0
                            
                            # Same sector adds similarity
                            if info['sector'] == current_sector:
                                similarity_score += 0.6
                            
                            # Same risk level adds similarity
                            if info['risk_level'] == current_risk:
                                similarity_score += 0.4
                            
                            # Risk preference adjustment
                            if risk_preference == 'conservative' and info['risk_level'] == 'Low':
                                similarity_score += 0.2
                            elif risk_preference == 'aggressive' and info['risk_level'] == 'High':
                                similarity_score += 0.2
                            
                            if similarity_score > 0.3:  # Minimum similarity threshold
                                candidates.append({
                                    'ticker': candidate_ticker,
                                    'name': info['name'],
                                    'similarity_score': min(similarity_score, 1.0),
                                    'risk_level': info['risk_level'],
                                    'sector': info['sector']
                                })
                    
                    # Sort by similarity and take top 3
                    candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
                    
                    similar_assets.append({
                        'base_asset': ticker,
                        'recommendations': candidates[:3]
                    })
            
            return {
                'similar_assets': similar_assets,
                'recommendation_type': 'Similar Assets',
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error in similar assets recommendations: {e}")
            return None

    def _get_sector_diversification(self, current_tickers, risk_preference):
        """Analyze sector diversification and suggest improvements"""
        try:
            # Analyze current sector allocation
            current_sectors = {}
            total_holdings = len(current_tickers)
            
            for ticker in current_tickers:
                if ticker in self.stock_universe:
                    sector = self.stock_universe[ticker]['sector']
                    current_sectors[sector] = current_sectors.get(sector, 0) + 1
            
            # Convert to percentages
            current_allocation = {
                sector: count / total_holdings
                for sector, count in current_sectors.items()
            }
            
            # Get target allocation based on risk preference
            target_allocation = self.sector_targets.get(risk_preference, self.sector_targets['moderate'])
            
            # Find underweight sectors
            recommendations = []
            for sector, target_weight in target_allocation.items():
                current_weight = current_allocation.get(sector, 0)
                
                if current_weight < target_weight:
                    # Find a good stock in this sector
                    sector_stocks = [
                        ticker for ticker, info in self.stock_universe.items()
                        if info['sector'] == sector and ticker not in current_tickers
                    ]
                    
                    if sector_stocks:
                        # Filter by risk preference
                        if risk_preference == 'conservative':
                            preferred_stocks = [t for t in sector_stocks if self.stock_universe[t]['risk_level'] == 'Low']
                        elif risk_preference == 'aggressive':
                            preferred_stocks = [t for t in sector_stocks if self.stock_universe[t]['risk_level'] == 'High']
                        else:
                            preferred_stocks = sector_stocks
                        
                        if not preferred_stocks:
                            preferred_stocks = sector_stocks
                        
                        suggested_ticker = random.choice(preferred_stocks)
                        
                        recommendations.append({
                            'sector': sector,
                            'current_weight': current_weight,
                            'target_weight': target_weight,
                            'suggested_weight': target_weight - current_weight,
                            'suggested_ticker': suggested_ticker,
                            'reason': f"Underweight in {sector} sector"
                        })
            
            return {
                'sector_analysis': {
                    'current_allocation': current_allocation,
                    'target_allocation': target_allocation,
                    'recommendations': recommendations
                },
                'recommendation_type': 'Sector Diversification',
                'risk_preference': risk_preference
            }
            
        except Exception as e:
            print(f"Error in sector diversification recommendations: {e}")
            return None

    def _get_risk_assessment(self, current_tickers, risk_preference):
        """Assess portfolio risk and provide recommendations"""
        try:
            # Analyze current portfolio risk
            risk_levels = {'Low': 0, 'Medium': 0, 'High': 0}
            sectors = set()
            
            for ticker in current_tickers:
                if ticker in self.stock_universe:
                    risk_level = self.stock_universe[ticker]['risk_level']
                    risk_levels[risk_level] += 1
                    sectors.add(self.stock_universe[ticker]['sector'])
            
            total_holdings = len(current_tickers)
            
            # Calculate risk metrics
            high_risk_pct = risk_levels['High'] / total_holdings if total_holdings > 0 else 0
            low_risk_pct = risk_levels['Low'] / total_holdings if total_holdings > 0 else 0
            
            # Determine overall risk level
            if high_risk_pct > 0.6:
                overall_risk = 'High'
            elif high_risk_pct > 0.3 or low_risk_pct < 0.3:
                overall_risk = 'Medium'
            else:
                overall_risk = 'Low'
            
            # Calculate concentration risk (simplified)
            concentration_risk = max(risk_levels.values()) / total_holdings if total_holdings > 0 else 0
            
            # Calculate diversification score
            sector_count = len(sectors)
            diversification_score = min(10, sector_count * 2)  # Max 10, 2 points per sector
            
            # Generate recommendations
            recommendations = []
            
            # Risk alignment recommendations
            if risk_preference == 'conservative' and high_risk_pct > 0.2:
                recommendations.append("Consider reducing high-risk holdings (currently {:.1%})".format(high_risk_pct))
                recommendations.append("Add more low-risk, dividend-paying stocks")
            
            elif risk_preference == 'aggressive' and low_risk_pct > 0.4:
                recommendations.append("Portfolio may be too conservative for aggressive risk preference")
                recommendations.append("Consider adding growth stocks or emerging market exposure")
            
            # Concentration risk recommendations
            if concentration_risk > 0.4:
                recommendations.append("High concentration in single risk category - consider diversifying")
            
            # Diversification recommendations
            if sector_count < 3:
                recommendations.append("Add holdings from more sectors to improve diversification")
            
            if total_holdings < 5:
                recommendations.append("Consider adding more holdings to reduce single-stock risk")
            
            return {
                'risk_analysis': {
                    'overall_risk': overall_risk,
                    'risk_distribution': {
                        'High Risk': f"{high_risk_pct:.1%}",
                        'Medium Risk': f"{risk_levels['Medium']/total_holdings:.1%}" if total_holdings > 0 else "0%",
                        'Low Risk': f"{low_risk_pct:.1%}"
                    },
                    'concentration_risk': concentration_risk,
                    'diversification_score': diversification_score,
                    'sector_count': sector_count,
                    'recommendations': recommendations
                },
                'recommendation_type': 'Risk Assessment',
                'alignment_with_preference': self._assess_risk_alignment(overall_risk, risk_preference)
            }
            
        except Exception as e:
            print(f"Error in risk assessment: {e}")
            return None

    def _assess_risk_alignment(self, portfolio_risk, risk_preference):
        """Assess how well portfolio risk aligns with user preference"""
        alignment_matrix = {
            ('Low', 'conservative'): 'Excellent',
            ('Low', 'moderate'): 'Good',
            ('Low', 'aggressive'): 'Poor',
            ('Medium', 'conservative'): 'Fair',
            ('Medium', 'moderate'): 'Excellent',
            ('Medium', 'aggressive'): 'Good',
            ('High', 'conservative'): 'Poor',
            ('High', 'moderate'): 'Fair',
            ('High', 'aggressive'): 'Excellent'
        }
        
        return alignment_matrix.get((portfolio_risk, risk_preference), 'Unknown')

    def recommend_portfolio(self, portfolio_holdings):
        """Legacy method for backward compatibility"""
        try:
            num_assets = len(portfolio_holdings)
            if num_assets == 0:
                return []
            
            # Simple equal weight recommendation
            equal_weight = 1.0 / num_assets
            recommendations = []
            
            for holding in portfolio_holdings:
                recommendations.append({
                    'ticker': holding['ticker'],
                    'recommended_weight': equal_weight,
                    'current_investment': holding.get('investment', 0),
                    'reason': 'Equal weight diversification'
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Error in recommend_portfolio: {e}")
            return []