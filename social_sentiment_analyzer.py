"""
Social Sentiment and News Impact Analysis for NFL Betting
Twitter sentiment, news analysis, injury reports, and public perception tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import json
import re
from textblob import TextBlob
import sqlite3

logger = logging.getLogger(__name__)

@dataclass
class SentimentData:
    player_id: str
    sentiment_score: float  # -1 to 1
    mention_volume: int
    positive_mentions: int
    negative_mentions: int
    neutral_mentions: int
    trending_topics: List[str]
    news_impact_score: float

@dataclass
class NewsImpact:
    player_id: str
    headline: str
    impact_type: str  # 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
    impact_magnitude: float  # 0-1
    source_credibility: float
    recency_factor: float
    betting_relevance: float

class SocialSentimentAnalyzer:
    """Analyze social media sentiment and news impact on player performance."""
    
    def __init__(self, db_path: str = "data/sentiment_data.db"):
        self.db_path = db_path
        self._init_sentiment_database()
        
        # Sentiment analysis weights
        self.sentiment_weights = {
            'twitter_weight': 0.4,
            'news_weight': 0.4,
            'reddit_weight': 0.2,
            'recency_decay': 0.8,  # Older sentiment matters less
            'volume_threshold': 100  # Minimum mentions for reliability
        }
        
        # News source credibility scores
        self.source_credibility = {
            'espn.com': 0.9,
            'nfl.com': 0.95,
            'profootballtalk.com': 0.8,
            'bleacherreport.com': 0.7,
            'twitter.com': 0.5,
            'reddit.com': 0.4,
            'unknown': 0.3
        }
    
    def _init_sentiment_database(self):
        """Initialize sentiment tracking database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT,
                platform TEXT,
                sentiment_score REAL,
                mention_count INTEGER,
                timestamp DATETIME,
                trending_keywords TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sentiment_player_time 
            ON sentiment_data(player_id, timestamp)
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_impact (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT,
                headline TEXT,
                source TEXT,
                impact_type TEXT,
                impact_score REAL,
                credibility_score REAL,
                timestamp DATETIME
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_impact_player_time 
            ON news_impact(player_id, timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def analyze_player_sentiment(self, player_id: str) -> SentimentData:
        """Analyze overall sentiment for a player across platforms."""
        
        # Get recent sentiment data
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT platform, sentiment_score, mention_count, trending_keywords, timestamp
            FROM sentiment_data
            WHERE player_id = ?
            AND timestamp > datetime('now', '-7 days')
            ORDER BY timestamp DESC
        """
        
        results = conn.execute(query, (player_id,)).fetchall()
        conn.close()
        
        if not results:
            return self._generate_mock_sentiment(player_id)
        
        # Aggregate sentiment across platforms
        total_mentions = sum(row[2] for row in results)
        weighted_sentiment = 0
        trending_topics = []
        
        for platform, sentiment, mentions, keywords, timestamp in results:
            # Apply recency decay
            days_old = (datetime.now() - datetime.fromisoformat(timestamp)).days
            recency_factor = self.sentiment_weights['recency_decay'] ** days_old
            
            # Weight by platform and recency
            platform_weight = self.sentiment_weights.get(f'{platform}_weight', 0.3)
            weight = platform_weight * recency_factor * mentions
            
            weighted_sentiment += sentiment * weight
            
            if keywords:
                trending_topics.extend(keywords.split(','))
        
        # Normalize sentiment
        if total_mentions > 0:
            final_sentiment = weighted_sentiment / total_mentions
        else:
            final_sentiment = 0.0
        
        # Categorize mentions
        positive_mentions = sum(row[2] for row in results if row[1] > 0.1)
        negative_mentions = sum(row[2] for row in results if row[1] < -0.1)
        neutral_mentions = total_mentions - positive_mentions - negative_mentions
        
        # Get news impact
        news_impact = self._calculate_news_impact(player_id)
        
        return SentimentData(
            player_id=player_id,
            sentiment_score=final_sentiment,
            mention_volume=total_mentions,
            positive_mentions=positive_mentions,
            negative_mentions=negative_mentions,
            neutral_mentions=neutral_mentions,
            trending_topics=list(set(trending_topics))[:5],
            news_impact_score=news_impact
        )
    
    def _generate_mock_sentiment(self, player_id: str) -> SentimentData:
        """Generate mock sentiment data for demonstration."""
        
        # Extract player name for realistic sentiment
        player_name = player_id.split('_')[0] if '_' in player_id else player_id
        
        # Simulate sentiment based on player name hash for consistency
        name_hash = hash(player_name) % 1000
        
        # Generate realistic sentiment distribution
        base_sentiment = (name_hash - 500) / 1000  # -0.5 to 0.5
        mention_volume = 50 + (name_hash % 200)  # 50-250 mentions
        
        positive_pct = max(0.2, 0.5 + base_sentiment * 0.3)
        negative_pct = max(0.1, 0.3 - base_sentiment * 0.3)
        neutral_pct = 1.0 - positive_pct - negative_pct
        
        return SentimentData(
            player_id=player_id,
            sentiment_score=base_sentiment,
            mention_volume=mention_volume,
            positive_mentions=int(mention_volume * positive_pct),
            negative_mentions=int(mention_volume * negative_pct),
            neutral_mentions=int(mention_volume * neutral_pct),
            trending_topics=['injury_update', 'matchup', 'fantasy'],
            news_impact_score=abs(base_sentiment) * 0.5
        )
    
    def _calculate_news_impact(self, player_id: str) -> float:
        """Calculate news impact score for a player."""
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT impact_type, impact_score, credibility_score, timestamp
            FROM news_impact
            WHERE player_id = ?
            AND timestamp > datetime('now', '-3 days')
            ORDER BY timestamp DESC
        """
        
        results = conn.execute(query, (player_id,)).fetchall()
        conn.close()
        
        if not results:
            return 0.0
        
        total_impact = 0
        for impact_type, impact_score, credibility, timestamp in results:
            # Apply recency decay
            hours_old = (datetime.now() - datetime.fromisoformat(timestamp)).total_seconds() / 3600
            recency_factor = 0.9 ** (hours_old / 24)  # Decay over days
            
            # Weight by credibility and recency
            weighted_impact = impact_score * credibility * recency_factor
            
            if impact_type == 'NEGATIVE':
                weighted_impact *= -1
            
            total_impact += weighted_impact
        
        return max(-1.0, min(1.0, total_impact))  # Clamp to [-1, 1]
    
    def get_sentiment_multiplier(self, player_id: str) -> float:
        """Get sentiment-based multiplier for predictions."""
        
        sentiment_data = self.analyze_player_sentiment(player_id)
        
        # Base multiplier
        multiplier = 1.0
        
        # Sentiment impact (stronger for extreme sentiment with high volume)
        if sentiment_data.mention_volume >= self.sentiment_weights['volume_threshold']:
            sentiment_impact = sentiment_data.sentiment_score * 0.1  # Max 10% impact
            multiplier += sentiment_impact
        
        # News impact
        news_impact = sentiment_data.news_impact_score * 0.05  # Max 5% impact
        multiplier += news_impact
        
        # Trending topic boost
        if 'injury' in [topic.lower() for topic in sentiment_data.trending_topics]:
            multiplier *= 0.9  # Injury concerns reduce multiplier
        elif 'breakout' in [topic.lower() for topic in sentiment_data.trending_topics]:
            multiplier *= 1.05  # Breakout buzz increases multiplier
        
        return max(0.8, min(1.2, multiplier))  # Cap between 0.8 and 1.2
    
    def analyze_injury_sentiment(self, player_id: str) -> Dict[str, Any]:
        """Analyze injury-related sentiment and news."""
        
        injury_keywords = ['injury', 'hurt', 'questionable', 'doubtful', 'out', 'pain', 'rehab']
        
        # Mock injury sentiment analysis
        sentiment_data = self.analyze_player_sentiment(player_id)
        
        injury_mentions = 0
        injury_sentiment = 0.0
        
        # Check trending topics for injury keywords
        for topic in sentiment_data.trending_topics:
            if any(keyword in topic.lower() for keyword in injury_keywords):
                injury_mentions += 10  # Estimate mentions
                injury_sentiment -= 0.3  # Injury news is typically negative
        
        return {
            'injury_mention_count': injury_mentions,
            'injury_sentiment': injury_sentiment,
            'injury_concern_level': 'HIGH' if injury_mentions > 20 else 'MEDIUM' if injury_mentions > 5 else 'LOW',
            'injury_impact_multiplier': 1.0 + (injury_sentiment * 0.1)
        }


class NewsImpactAnalyzer:
    """Analyze news impact on player performance and betting lines."""
    
    def __init__(self):
        self.impact_keywords = {
            'positive': ['breakout', 'healthy', 'ready', 'explosive', 'dominant', 'record'],
            'negative': ['injury', 'suspended', 'benched', 'struggling', 'limited', 'concern'],
            'neutral': ['practice', 'meeting', 'interview', 'comment', 'statement']
        }
    
    def analyze_headline_impact(self, headline: str, player_id: str) -> NewsImpact:
        """Analyze the impact of a news headline on a player."""
        
        headline_lower = headline.lower()
        player_name = player_id.split('_')[0] if '_' in player_id else player_id
        
        # Check if headline is about the player
        if player_name.lower() not in headline_lower:
            betting_relevance = 0.1
        else:
            betting_relevance = 0.8
        
        # Determine impact type and magnitude
        positive_score = sum(1 for word in self.impact_keywords['positive'] if word in headline_lower)
        negative_score = sum(1 for word in self.impact_keywords['negative'] if word in headline_lower)
        
        if positive_score > negative_score:
            impact_type = 'POSITIVE'
            impact_magnitude = min(positive_score * 0.3, 1.0)
        elif negative_score > positive_score:
            impact_type = 'NEGATIVE'
            impact_magnitude = min(negative_score * 0.3, 1.0)
        else:
            impact_type = 'NEUTRAL'
            impact_magnitude = 0.1
        
        # Source credibility (mock)
        source_credibility = 0.7  # Default credibility
        
        # Recency factor (assume recent)
        recency_factor = 1.0
        
        return NewsImpact(
            player_id=player_id,
            headline=headline,
            impact_type=impact_type,
            impact_magnitude=impact_magnitude,
            source_credibility=source_credibility,
            recency_factor=recency_factor,
            betting_relevance=betting_relevance
        )
    
    def get_news_multiplier(self, player_id: str, recent_headlines: List[str]) -> float:
        """Get news-based multiplier for predictions."""
        
        if not recent_headlines:
            return 1.0
        
        total_impact = 0
        for headline in recent_headlines:
            news_impact = self.analyze_headline_impact(headline, player_id)
            
            impact_value = news_impact.impact_magnitude * news_impact.source_credibility
            impact_value *= news_impact.recency_factor * news_impact.betting_relevance
            
            if news_impact.impact_type == 'NEGATIVE':
                impact_value *= -1
            
            total_impact += impact_value
        
        # Convert to multiplier
        multiplier = 1.0 + (total_impact * 0.05)  # Max 5% impact per headline
        
        return max(0.85, min(1.15, multiplier))  # Cap between 0.85 and 1.15


class PublicPerceptionTracker:
    """Track public perception and contrarian betting opportunities."""
    
    def __init__(self):
        self.perception_thresholds = {
            'overhyped': 0.8,      # 80%+ positive sentiment
            'undervalued': -0.3,   # Negative sentiment
            'contrarian_opportunity': 0.75  # High public confidence
        }
    
    def identify_contrarian_opportunities(self, sentiment_data: List[SentimentData]) -> List[Dict[str, Any]]:
        """Identify contrarian betting opportunities based on public perception."""
        
        opportunities = []
        
        for data in sentiment_data:
            # Overhyped players (fade the public)
            if (data.sentiment_score > self.perception_thresholds['overhyped'] and 
                data.mention_volume > 200):
                opportunities.append({
                    'player_id': data.player_id,
                    'opportunity_type': 'FADE_OVERHYPED',
                    'sentiment_score': data.sentiment_score,
                    'mention_volume': data.mention_volume,
                    'recommendation': 'CONSIDER_UNDER_BETS',
                    'confidence': 'MEDIUM'
                })
            
            # Undervalued players (contrarian value)
            elif (data.sentiment_score < self.perception_thresholds['undervalued'] and 
                  data.mention_volume > 100):
                opportunities.append({
                    'player_id': data.player_id,
                    'opportunity_type': 'CONTRARIAN_VALUE',
                    'sentiment_score': data.sentiment_score,
                    'mention_volume': data.mention_volume,
                    'recommendation': 'CONSIDER_OVER_BETS',
                    'confidence': 'HIGH'
                })
        
        return opportunities


# Example usage and integration
def main():
    """Example usage of social sentiment analysis."""
    
    print("Social Sentiment Analysis Demo")
    print("=" * 35)
    
    # Initialize analyzers
    sentiment_analyzer = SocialSentimentAnalyzer()
    news_analyzer = NewsImpactAnalyzer()
    perception_tracker = PublicPerceptionTracker()
    
    # Example sentiment analysis
    player_id = "pmahomes_qb"
    print(f"\nSentiment Analysis for {player_id}:")
    
    sentiment_data = sentiment_analyzer.analyze_player_sentiment(player_id)
    print(f"Sentiment Score: {sentiment_data.sentiment_score:.3f}")
    print(f"Mention Volume: {sentiment_data.mention_volume}")
    print(f"Trending Topics: {sentiment_data.trending_topics}")
    
    # Sentiment multiplier
    multiplier = sentiment_analyzer.get_sentiment_multiplier(player_id)
    print(f"Sentiment Multiplier: {multiplier:.3f}")
    
    # Example news impact
    print(f"\nNews Impact Analysis:")
    sample_headlines = [
        "Patrick Mahomes looks explosive in practice",
        "Chiefs QB dealing with minor ankle concern"
    ]
    
    news_multiplier = news_analyzer.get_news_multiplier(player_id, sample_headlines)
    print(f"News Multiplier: {news_multiplier:.3f}")
    
    # Injury sentiment
    injury_analysis = sentiment_analyzer.analyze_injury_sentiment(player_id)
    print(f"Injury Concern Level: {injury_analysis['injury_concern_level']}")
    
    print("\nSocial sentiment analysis ready!")


if __name__ == "__main__":
    main()
