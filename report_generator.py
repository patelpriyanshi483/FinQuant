# 

import io
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

class EnhancedPortfolioReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1f4e79')
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#2c5282'),
            borderWidth=1,
            borderColor=colors.HexColor('#e2e8f0'),
            borderPadding=5
        ))
        
        # Metric style
        self.styles.add(ParagraphStyle(
            name='MetricStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            leftIndent=20
        ))
        
        # Summary style
        self.styles.add(ParagraphStyle(
            name='Summary',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=10,
            alignment=TA_LEFT,
            textColor=colors.HexColor('#2d3748')
        ))

    def generate_pdf_report(self, portfolio, analysis):
        """Generate comprehensive PDF portfolio report"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        
        try:
            # Report Header
            story.extend(self._create_header(portfolio, analysis))
            
            # Executive Summary
            story.extend(self._create_executive_summary(portfolio, analysis))
            
            # Portfolio Composition
            story.extend(self._create_portfolio_composition(analysis))
            
            # Performance Metrics
            story.extend(self._create_performance_metrics(analysis))
            
            # Risk Analysis
            story.extend(self._create_risk_analysis(analysis))
            
            # Benchmark Comparison
            if self._has_benchmark_data(analysis):
                story.extend(self._create_benchmark_comparison(analysis))
            
            # Individual Asset Analysis
            story.extend(self._create_individual_asset_analysis(analysis))
            
            # Charts and Visualizations
            story.extend(self._create_charts(analysis))
            
            # Correlation Analysis
            if self._has_correlation_data(analysis):
                story.extend(self._create_correlation_analysis(analysis))
            
            # Recommendations
            story.extend(self._create_recommendations(analysis))
            
            # Footer
            story.extend(self._create_footer())
            
            # Build PDF
            doc.build(story)
            pdf_data = buffer.getvalue()
            buffer.close()
            
            return pdf_data
            
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            # Return a simple error report
            return self._generate_error_report(portfolio, str(e))

    def _create_header(self, portfolio, analysis):
        """Create report header section"""
        elements = []
        
        # Title
        title = f"Portfolio Analysis Report: {portfolio.get('name', 'Portfolio')}"
        elements.append(Paragraph(title, self.styles['CustomTitle']))
        elements.append(Spacer(1, 20))
        
        # Report details table
        report_data = [
            ['Report Generated:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
            ['Analysis Period:', f"{analysis.get('analysis_period', {}).get('start', 'N/A')} to {analysis.get('analysis_period', {}).get('end', 'N/A')}"],
            ['Portfolio ID:', str(portfolio.get('id', 'N/A'))],
            ['Total Investment:', f"₹{analysis.get('original_investment', 0):,.2f}"],
            ['Number of Assets:', str(len(analysis.get('tickers', [])))]
        ]
        
        table = Table(report_data, colWidths=[2*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f7fafc')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0'))
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 30))
        
        return elements

    def _create_executive_summary(self, portfolio, analysis):
        """Create executive summary section"""
        elements = []
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        if not analysis.get('success', False):
            error_msg = f"Analysis could not be completed: {analysis.get('error', 'Unknown error')}"
            elements.append(Paragraph(error_msg, self.styles['Summary']))
            return elements
        
        metrics = analysis.get('metrics', {})
        
        # Performance summary
        annual_return = metrics.get('expected_return', 0)
        volatility = metrics.get('volatility', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        
        summary_text = f"""
        This portfolio analysis covers {len(analysis.get('tickers', []))} assets with a total investment 
        of ₹{analysis.get('original_investment', 0):,.2f} over the period from 
        {analysis.get('analysis_period', {}).get('start', 'N/A')} to 
        {analysis.get('analysis_period', {}).get('end', 'N/A')}.
        
        <b>Key Performance Highlights:</b><br/>
        • Expected Annual Return: {annual_return:.2%}<br/>
        • Annual Volatility: {volatility:.2%}<br/>
        • Sharpe Ratio: {sharpe_ratio:.3f}<br/>
        • Maximum Drawdown: {max_drawdown:.2%}<br/>
        """
        
        # Add benchmark comparison if available
        alpha = metrics.get('alpha', 0)
        beta = metrics.get('beta', 0)
        if beta != 0:
            summary_text += f"""
            • Alpha vs Benchmark: {alpha:.2%}<br/>
            • Beta vs Benchmark: {beta:.3f}<br/>
            """
        
        # Performance interpretation
        if sharpe_ratio > 1.5:
            performance_note = "The portfolio demonstrates excellent risk-adjusted returns."
        elif sharpe_ratio > 1.0:
            performance_note = "The portfolio shows good risk-adjusted performance."
        elif sharpe_ratio > 0.5:
            performance_note = "The portfolio shows moderate risk-adjusted performance."
        else:
            performance_note = "The portfolio shows below-average risk-adjusted performance."
        
        summary_text += f"<br/><b>Overall Assessment:</b> {performance_note}"
        
        elements.append(Paragraph(summary_text, self.styles['Summary']))
        elements.append(Spacer(1, 20))
        
        return elements

    def _create_portfolio_composition(self, analysis):
        """Create portfolio composition section"""
        elements = []
        elements.append(Paragraph("Portfolio Composition", self.styles['SectionHeader']))
        
        tickers = analysis.get('tickers', [])
        weights = analysis.get('weights', [])
        original_investment = analysis.get('original_investment', 0)
        
        if not tickers or not weights:
            elements.append(Paragraph("No portfolio composition data available.", self.styles['Summary']))
            return elements
        
        # Create composition table
        composition_data = [['Asset', 'Weight', 'Investment Amount', 'Percentage']]
        
        for i, (ticker, weight) in enumerate(zip(tickers, weights)):
            investment_amount = weight * original_investment
            composition_data.append([
                ticker,
                f"{weight:.1%}",
                f"₹{investment_amount:,.2f}",
                f"{weight*100:.1f}%"
            ])
        
        table = Table(composition_data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0'))
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        
        return elements

    def _create_performance_metrics(self, analysis):
        """Create detailed performance metrics section"""
        elements = []
        elements.append(Paragraph("Performance Metrics", self.styles['SectionHeader']))
        
        metrics = analysis.get('metrics', {})
        
        # Return metrics
        elements.append(Paragraph("<b>Return Analysis</b>", self.styles['MetricStyle']))
        return_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Expected Annual Return', f"{metrics.get('expected_return', 0):.2%}", self._interpret_return(metrics.get('expected_return', 0))],
            ['Annual Volatility', f"{metrics.get('volatility', 0):.2%}", self._interpret_volatility(metrics.get('volatility', 0))],
            ['Win Rate', f"{metrics.get('win_rate', 0):.1%}", f"{'Strong' if metrics.get('win_rate', 0) > 0.55 else 'Average' if metrics.get('win_rate', 0) > 0.45 else 'Weak'} daily performance"]
        ]
        
        return_table = Table(return_data, colWidths=[2*inch, 1.2*inch, 2.3*inch])
        return_table.setStyle(self._get_table_style())
        elements.append(return_table)
        elements.append(Spacer(1, 15))
        
        # Risk-adjusted metrics
        elements.append(Paragraph("<b>Risk-Adjusted Performance</b>", self.styles['MetricStyle']))
        risk_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.3f}", self._interpret_sharpe(metrics.get('sharpe_ratio', 0))],
            ['Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.3f}", self._interpret_sortino(metrics.get('sortino_ratio', 0))],
            ['Calmar Ratio', f"{metrics.get('calmar_ratio', 0):.3f}", self._interpret_calmar(metrics.get('calmar_ratio', 0))]
        ]
        
        risk_table = Table(risk_data, colWidths=[2*inch, 1.2*inch, 2.3*inch])
        risk_table.setStyle(self._get_table_style())
        elements.append(risk_table)
        elements.append(Spacer(1, 20))
        
        return elements

    def _create_risk_analysis(self, analysis):
        """Create comprehensive risk analysis section"""
        elements = []
        elements.append(Paragraph("Risk Analysis", self.styles['SectionHeader']))
        
        metrics = analysis.get('metrics', {})
        
        # Risk metrics table
        risk_data = [
            ['Risk Metric', 'Value', 'Assessment'],
            ['Maximum Drawdown', f"{metrics.get('max_drawdown', 0):.2%}", self._interpret_max_drawdown(metrics.get('max_drawdown', 0))],
            ['Value at Risk (5%)', f"{metrics.get('var_5', 0):.2%}", "Daily loss not exceeded 95% of the time"],
            ['Skewness', f"{metrics.get('skewness', 0):.3f}", self._interpret_skewness(metrics.get('skewness', 0))],
            ['Kurtosis', f"{metrics.get('kurtosis', 0):.3f}", self._interpret_kurtosis(metrics.get('kurtosis', 0))]
        ]
        
        risk_table = Table(risk_data, colWidths=[2*inch, 1.2*inch, 2.3*inch])
        risk_table.setStyle(self._get_table_style())
        elements.append(risk_table)
        
        # Risk interpretation
        max_dd = metrics.get('max_drawdown', 0)
        var_5 = metrics.get('var_5', 0)
        
        risk_summary = f"""
        <b>Risk Assessment:</b><br/>
        The portfolio's maximum drawdown of {max_dd:.2%} indicates the worst peak-to-trough decline 
        experienced during the analysis period. The 5% Value at Risk of {var_5:.2%} suggests that 
        on any given day, there's only a 5% chance of losing more than {abs(var_5):.2%} of the portfolio value.
        """
        
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(risk_summary, self.styles['Summary']))
        elements.append(Spacer(1, 20))
        
        return elements

    def _create_benchmark_comparison(self, analysis):
        """Create benchmark comparison section"""
        elements = []
        elements.append(Paragraph("Benchmark Comparison", self.styles['SectionHeader']))
        
        metrics = analysis.get('metrics', {})
        alpha = metrics.get('alpha', 0)
        beta = metrics.get('beta', 0)
        
        # Benchmark metrics table
        benchmark_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Alpha (Annual)', f"{alpha:.2%}", self._interpret_alpha(alpha)],
            ['Beta', f"{beta:.3f}", self._interpret_beta(beta)]
        ]
        
        benchmark_table = Table(benchmark_data, colWidths=[2*inch, 1.2*inch, 2.3*inch])
        benchmark_table.setStyle(self._get_table_style())
        elements.append(benchmark_table)
        
        # Benchmark interpretation
        benchmark_summary = f"""
        <b>Benchmark Analysis:</b><br/>
        Alpha of {alpha:.2%} represents the excess return generated above the benchmark after adjusting for risk. 
        Beta of {beta:.3f} indicates the portfolio's sensitivity to market movements. 
        {self._get_beta_explanation(beta)}
        """
        
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(benchmark_summary, self.styles['Summary']))
        elements.append(Spacer(1, 20))
        
        return elements

    def _create_individual_asset_analysis(self, analysis):
        """Create individual asset analysis section"""
        elements = []
        elements.append(Paragraph("Individual Asset Analysis", self.styles['SectionHeader']))
        
        individual_metrics = analysis.get('individual_metrics', {})
        
        if not individual_metrics:
            elements.append(Paragraph("No individual asset data available.", self.styles['Summary']))
            return elements
        
        # Individual assets table
        asset_data = [['Asset', 'Weight', 'Annual Return', 'Volatility', 'Sharpe Ratio', 'Contribution']]
        
        portfolio_return = analysis.get('metrics', {}).get('expected_return', 0)
        
        for ticker, asset_metrics in individual_metrics.items():
            weight = asset_metrics.get('weight', 0)
            return_contrib = weight * asset_metrics.get('expected_return', 0)
            contribution_pct = (return_contrib / portfolio_return * 100) if portfolio_return != 0 else 0
            
            asset_data.append([
                ticker,
                f"{weight:.1%}",
                f"{asset_metrics.get('expected_return', 0):.2%}",
                f"{asset_metrics.get('volatility', 0):.2%}",
                f"{asset_metrics.get('sharpe_ratio', 0):.3f}",
                f"{contribution_pct:.1f}%"
            ])
        
        asset_table = Table(asset_data, colWidths=[1*inch, 0.8*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        asset_table.setStyle(self._get_table_style())
        elements.append(asset_table)
        
        # Asset performance insights
        best_performer = max(individual_metrics.items(), key=lambda x: x[1].get('expected_return', 0))
        worst_performer = min(individual_metrics.items(), key=lambda x: x[1].get('expected_return', 0))
        highest_vol = max(individual_metrics.items(), key=lambda x: x[1].get('volatility', 0))
        
        insights = f"""
        <b>Asset Performance Insights:</b><br/>
        • Best Performer: {best_performer[0]} with {best_performer[1].get('expected_return', 0):.2%} annual return<br/>
        • Highest Volatility: {highest_vol[0]} with {highest_vol[1].get('volatility', 0):.2%} annual volatility<br/>
        • Portfolio concentration in top 3 holdings: {sum(sorted([m.get('weight', 0) for m in individual_metrics.values()], reverse=True)[:3]):.1%}
        """
        
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(insights, self.styles['Summary']))
        elements.append(Spacer(1, 20))
        
        return elements

    def _create_correlation_analysis(self, analysis):
        """Create correlation analysis section"""
        elements = []
        elements.append(Paragraph("Asset Correlation Analysis", self.styles['SectionHeader']))
        
        correlation_matrix = analysis.get('correlation_matrix')
        
        if correlation_matrix is None or correlation_matrix.empty:
            elements.append(Paragraph("Correlation data not available.", self.styles['Summary']))
            return elements
        
        # Find highest and lowest correlations
        corr_values = []
        tickers = list(correlation_matrix.columns)
        
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                corr_val = correlation_matrix.iloc[i, j]
                if not np.isnan(corr_val):
                    corr_values.append((tickers[i], tickers[j], corr_val))
        
        if corr_values:
            corr_values.sort(key=lambda x: x[2], reverse=True)
            highest_corr = corr_values[0]
            lowest_corr = corr_values[-1]
            avg_corr = np.mean([x[2] for x in corr_values])
            
            corr_summary = f"""
            <b>Correlation Insights:</b><br/>
            • Highest correlation: {highest_corr[0]} & {highest_corr[1]} ({highest_corr[2]:.3f})<br/>
            • Lowest correlation: {lowest_corr[0]} & {lowest_corr[1]} ({lowest_corr[2]:.3f})<br/>
            • Average correlation: {avg_corr:.3f}<br/>
            • Diversification level: {self._assess_diversification(avg_corr)}
            """
            
            elements.append(Paragraph(corr_summary, self.styles['Summary']))
        
        elements.append(Spacer(1, 20))
        return elements

    def _create_recommendations(self, analysis):
        """Create recommendations section"""
        elements = []
        elements.append(Paragraph("Portfolio Recommendations", self.styles['SectionHeader']))
        
        metrics = analysis.get('metrics', {})
        individual_metrics = analysis.get('individual_metrics', {})
        
        recommendations = []
        
        # Risk-based recommendations
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        volatility = metrics.get('volatility', 0)
        
        if sharpe_ratio < 0.8:
            recommendations.append("Consider rebalancing to improve risk-adjusted returns (Sharpe ratio below 0.8)")
        
        if abs(max_drawdown) > 0.20:
            recommendations.append("High maximum drawdown suggests need for better risk management or diversification")
        
        if volatility > 0.25:
            recommendations.append("High portfolio volatility - consider adding less volatile assets or reducing position sizes")
        
        # Diversification recommendations
        weights = analysis.get('weights', [])
        if weights:
            max_weight = max(weights)
            if max_weight > 0.4:
                recommendations.append("Portfolio is heavily concentrated in single asset - consider diversification")
            
            # Check for very small positions
            small_positions = sum(1 for w in weights if w < 0.02)
            if small_positions > 0:
                recommendations.append(f"{small_positions} positions are very small (<2%) - consider consolidating")
        
        # Performance-based recommendations
        alpha = metrics.get('alpha', 0)
        if alpha < -0.02:  # Negative alpha > 2%
            recommendations.append("Portfolio is underperforming benchmark - review asset selection and allocation")
        
        # Individual asset recommendations
        if individual_metrics:
            poor_performers = [ticker for ticker, metrics_dict in individual_metrics.items() 
                             if metrics_dict.get('sharpe_ratio', 0) < 0]
            if poor_performers:
                recommendations.append(f"Consider reviewing positions in: {', '.join(poor_performers)} (negative Sharpe ratios)")
        
        # Default positive recommendations if none found
        if not recommendations:
            recommendations.extend([
                "Portfolio shows balanced risk-return characteristics",
                "Continue monitoring performance and rebalance quarterly",
                "Consider tax-loss harvesting opportunities during rebalancing"
            ])
        
        # Format recommendations
        rec_text = "<b>Key Recommendations:</b><br/>"
        for i, rec in enumerate(recommendations[:6], 1):  # Limit to top 6 recommendations
            rec_text += f"• {rec}<br/>"
        
        elements.append(Paragraph(rec_text, self.styles['Summary']))
        elements.append(Spacer(1, 20))
        
        return elements

    def _create_charts(self, analysis):
        """Create charts section with performance visualization"""
        elements = []
        elements.append(Paragraph("Performance Visualization", self.styles['SectionHeader']))
        
        try:
            portfolio_returns = analysis.get('portfolio_returns')
            benchmark_returns = analysis.get('benchmark_returns')
            
            if portfolio_returns is not None and not portfolio_returns.empty:
                # Create cumulative performance chart
                chart_buffer = self._create_performance_chart(portfolio_returns, benchmark_returns)
                if chart_buffer:
                    elements.append(Image(chart_buffer, width=6*inch, height=4*inch))
                    elements.append(Spacer(1, 10))
            
            # Create allocation pie chart
            weights = analysis.get('weights', [])
            tickers = analysis.get('tickers', [])
            
            if weights and tickers:
                pie_chart_buffer = self._create_allocation_chart(tickers, weights)
                if pie_chart_buffer:
                    elements.append(Image(pie_chart_buffer, width=5*inch, height=4*inch))
                    elements.append(Spacer(1, 20))
            
        except Exception as e:
            print(f"Error creating charts: {e}")
            elements.append(Paragraph("Charts could not be generated due to data limitations.", self.styles['Summary']))
        
        return elements

    def _create_performance_chart(self, portfolio_returns, benchmark_returns=None):
        """Create cumulative performance chart"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Calculate cumulative returns
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            plt.plot(cumulative_returns.index, cumulative_returns.values, 
                    label='Portfolio', linewidth=2, color='#3182ce')
            
            if benchmark_returns is not None and not benchmark_returns.empty:
                # Align benchmark with portfolio
                aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
                if not aligned_data.empty and aligned_data.shape[1] == 2:
                    bench_aligned = aligned_data.iloc[:, 1]
                    bench_cumulative = (1 + bench_aligned).cumprod()
                    plt.plot(bench_cumulative.index, bench_cumulative.values, 
                            label='Benchmark', linewidth=2, color='#e53e3e', linestyle='--')
            
            plt.title('Cumulative Portfolio Performance', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error creating performance chart: {e}")
            plt.close()
            return None

    def _create_allocation_chart(self, tickers, weights):
        """Create portfolio allocation pie chart"""
        try:
            plt.figure(figsize=(8, 8))
            
            # Prepare data
            labels = tickers
            sizes = weights
            
            # Create color palette
            colors_list = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            # Create pie chart
            wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                              colors=colors_list, startangle=90)
            
            plt.title('Portfolio Asset Allocation', fontsize=14, fontweight='bold')
            plt.axis('equal')
            
            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            print(f"Error creating allocation chart: {e}")
            plt.close()
            return None

    def _create_footer(self):
        """Create report footer"""
        elements = []
        elements.append(PageBreak())
        
        disclaimer = """
        <b>Important Disclaimer:</b><br/>
        This report is for informational purposes only and does not constitute investment advice. 
        Past performance does not guarantee future results. All investments carry risk of loss. 
        Please consult with a qualified financial advisor before making investment decisions.
        The analysis is based on historical data and may not reflect current market conditions.
        """
        
        elements.append(Paragraph(disclaimer, self.styles['Summary']))
        elements.append(Spacer(1, 20))
        
        footer_text = f"Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        elements.append(Paragraph(footer_text, self.styles['Normal']))
        
        return elements

    # Helper methods for interpretations
    def _interpret_return(self, return_val):
        if return_val > 0.15: return "Excellent"
        elif return_val > 0.10: return "Strong"
        elif return_val > 0.07: return "Good"
        elif return_val > 0.03: return "Moderate"
        else: return "Below average"

    def _interpret_volatility(self, vol):
        if vol < 0.10: return "Low risk"
        elif vol < 0.15: return "Moderate risk"
        elif vol < 0.25: return "High risk"
        else: return "Very high risk"

    def _interpret_sharpe(self, sharpe):
        if sharpe > 2.0: return "Exceptional"
        elif sharpe > 1.5: return "Excellent"
        elif sharpe > 1.0: return "Good"
        elif sharpe > 0.5: return "Acceptable"
        else: return "Poor"

    def _interpret_sortino(self, sortino):
        if sortino > 2.0: return "Excellent downside protection"
        elif sortino > 1.5: return "Good downside protection"
        elif sortino > 1.0: return "Moderate downside protection"
        else: return "Limited downside protection"

    def _interpret_calmar(self, calmar):
        if calmar > 1.0: return "Strong risk-adjusted performance"
        elif calmar > 0.5: return "Good risk-adjusted performance"
        elif calmar > 0.25: return "Moderate risk-adjusted performance"
        else: return "Weak risk-adjusted performance"

    def _interpret_max_drawdown(self, max_dd):
        max_dd = abs(max_dd)
        if max_dd < 0.05: return "Very low drawdown - excellent"
        elif max_dd < 0.10: return "Low drawdown - good"
        elif max_dd < 0.20: return "Moderate drawdown - acceptable"
        elif max_dd < 0.30: return "High drawdown - concerning"
        else: return "Very high drawdown - review strategy"

    def _interpret_skewness(self, skew):
        if abs(skew) < 0.5: return "Normal distribution"
        elif skew > 0.5: return "Positive skew - more upside potential"
        elif skew < -0.5: return "Negative skew - more downside risk"
        else: return "Slight skew"

    def _interpret_kurtosis(self, kurt):
        if abs(kurt) < 1: return "Normal tail risk"
        elif kurt > 1: return "Fat tails - higher extreme event risk"
        elif kurt < -1: return "Thin tails - lower extreme event risk"
        else: return "Moderate tail characteristics"

    def _interpret_alpha(self, alpha):
        if alpha > 0.03: return "Strong outperformance vs benchmark"
        elif alpha > 0.01: return "Modest outperformance vs benchmark"
        elif alpha > -0.01: return "Performance in line with benchmark"
        elif alpha > -0.03: return "Modest underperformance vs benchmark"
        else: return "Significant underperformance vs benchmark"

    def _interpret_beta(self, beta):
        if beta > 1.2: return "High market sensitivity"
        elif beta > 0.8: return "Moderate market sensitivity"
        elif beta > 0.5: return "Low market sensitivity"
        else: return "Very low market sensitivity"

    def _get_beta_explanation(self, beta):
        if beta > 1.1:
            return "The portfolio is more volatile than the market and amplifies market movements."
        elif beta < 0.9:
            return "The portfolio is less volatile than the market and dampens market movements."
        else:
            return "The portfolio moves roughly in line with the market."

    def _assess_diversification(self, avg_corr):
        if avg_corr < 0.3: return "Well diversified"
        elif avg_corr < 0.5: return "Moderately diversified"
        elif avg_corr < 0.7: return "Limited diversification"
        else: return "Poor diversification"

    def _get_table_style(self):
        """Standard table style for consistency"""
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ])

    def _has_benchmark_data(self, analysis):
        """Check if benchmark data is available"""
        metrics = analysis.get('metrics', {})
        return metrics.get('beta', 0) != 0 or metrics.get('alpha', 0) != 0

    def _has_correlation_data(self, analysis):
        """Check if correlation data is available"""
        correlation_matrix = analysis.get('correlation_matrix')
        return correlation_matrix is not None and not correlation_matrix.empty

    def _generate_error_report(self, portfolio, error_message):
        """Generate minimal error report"""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            story = []
            
            story.append(Paragraph(f"Portfolio Report: {portfolio.get('name', 'Portfolio')}", self.styles['CustomTitle']))
            story.append(Spacer(1, 30))
            story.append(Paragraph(f"Report Generation Error: {error_message}", self.styles['Summary']))
            story.append(Spacer(1, 20))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", self.styles['Normal']))
            
            doc.build(story)
            pdf_data = buffer.getvalue()
            buffer.close()
            
            return pdf_data
            
        except Exception as e:
            print(f"Error generating error report: {e}")
            return b"Error generating report"


# Updated report generation function for your app.py
def generate_portfolio_report(portfolio):
    """Enhanced portfolio report generation function"""
    try:
        print(f"🚀 Generating enhanced report for portfolio: {portfolio.get('name', 'Unknown')}")
        
        # Get portfolio holdings
        holdings = st.session_state.db_manager.get_portfolio_holdings(portfolio['id'])
        holdings = calculate_weights_from_investments(holdings)
        
        if not holdings:
            raise ValueError("No holdings found for this portfolio")
        
        # Extract tickers and weights
        tickers = [h['ticker'] for h in holdings]
        weights = [h['weight'] for h in holdings]
        
        print(f"📊 Portfolio composition: {len(tickers)} assets")
        print(f"🎯 Tickers: {tickers}")
        
        # Set analysis period (1 year)
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        print(f"📅 Analysis period: {start_date.date()} to {end_date.date()}")
        
        # Perform comprehensive analysis
        analysis = st.session_state.portfolio_analyzer.analyze_portfolio(
            tickers=tickers, 
            weights=weights, 
            start_date=start_date, 
            end_date=end_date, 
            benchmark_ticker='SPY'  # Use SPY as benchmark instead of ^GSPC for better compatibility
        )
        
        if not analysis:
            raise ValueError("Portfolio analysis failed - no data returned")
        
        # Generate enhanced PDF report
        report_generator = EnhancedPortfolioReportGenerator()
        pdf = report_generator.generate_pdf_report(portfolio, analysis)
        
        if not pdf:
            raise ValueError("PDF generation failed")
        
        print("✅ Enhanced portfolio report generated successfully!")
        return pdf
        
    except Exception as e:
        print(f"❌ Error in generate_portfolio_report: {e}")
        # Generate basic error report
        try:
            report_generator = EnhancedPortfolioReportGenerator()
            return report_generator._generate_error_report(portfolio, str(e))
        except:
            # Ultimate fallback
            return b"Error: Unable to generate portfolio report"


# Additional helper function to calculate investment amounts from weights
def calculate_weights_from_investments(holdings):
    """
    Enhanced function to calculate weights from investment amounts
    Handles both weight-based and dollar-amount-based portfolios
    """
    try:
        if not holdings:
            return []
        
        # Check if we have investment amounts or weights
        total_investment = sum(h.get('investment_amount', 0) for h in holdings)
        
        if total_investment > 0:
            # Calculate weights from investment amounts
            for holding in holdings:
                investment = holding.get('investment_amount', 0)
                holding['weight'] = investment / total_investment if total_investment > 0 else 0
        else:
            # Assume weights are already provided, normalize them
            total_weight = sum(h.get('weight', 0) for h in holdings)
            if total_weight > 0:
                for holding in holdings:
                    holding['weight'] = holding.get('weight', 0) / total_weight
            else:
                # Equal weights fallback
                equal_weight = 1.0 / len(holdings)
                for holding in holdings:
                    holding['weight'] = equal_weight
        
        print(f'📊 Calculated weights: {[(h['ticker'], f"{h['weight']:.1%}") for h in holdings]}')
        return holdings
        
    except Exception as e:
        print(f"Error calculating weights: {e}")
        return holdings


# Updated button handler for your app.py
def handle_report_generation(portfolio):
    """Enhanced report generation with better error handling and user feedback"""
    try:
        # Show progress indicator
        with st.spinner(f"Generating comprehensive report for {portfolio.get('name', 'portfolio')}..."):
            
            # Generate the enhanced report
            pdf = generate_portfolio_report(portfolio)
            
            if pdf and len(pdf) > 0:
                # Success - provide download button
                st.success("✅ Report generated successfully!")
                
                # Create filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                filename = f"{portfolio.get('name', 'Portfolio')}_{timestamp}_report.pdf"
                
                st.download_button(
                    label="📥 Download Comprehensive Report",
                    data=pdf,
                    file_name=filename,
                    mime='application/pdf',
                    help="Click to download your detailed portfolio analysis report"
                )
                
                # Show brief metrics preview
                st.info("📊 Report includes: Performance metrics, risk analysis, benchmark comparison, asset correlation, and personalized recommendations")
                
            else:
                st.error("❌ Report generation failed - empty report returned")
                
    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")
        st.info("💡 Tip: Ensure your portfolio has valid holdings with proper ticker symbols")


# Example usage in your app.py streamlit code:
"""
# Replace your existing button code with this enhanced version:

with col2:
    if st.button("📄 Enhanced Report", key=f"report_{portfolio['id']}"):
        handle_report_generation(portfolio)
"""

# Additional utility functions that might be helpful

def validate_portfolio_data(portfolio_data):
    """Validate portfolio data before report generation"""
    required_fields = ['id', 'name']
    
    for field in required_fields:
        if field not in portfolio_data:
            raise ValueError(f"Missing required field: {field}")
    
    return True

def get_report_summary_stats(analysis):
    """Extract key stats for display in the UI"""
    if not analysis or not analysis.get('success', False):
        return None
    
    metrics = analysis.get('metrics', {})
    
    return {
        'annual_return': f"{metrics.get('expected_return', 0):.2%}",
        'volatility': f"{metrics.get('volatility', 0):.2%}",
        'sharpe_ratio': f"{metrics.get('sharpe_ratio', 0):.3f}",
        'max_drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
        'total_assets': len(analysis.get('tickers', [])),
        'analysis_days': analysis.get('analysis_period', {}).get('days', 0)
    }