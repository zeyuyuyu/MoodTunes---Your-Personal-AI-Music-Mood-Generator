import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict, List, Tuple

class MoodAnalyzer:
    def __init__(self):
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()
        
    def analyze_mood(self, text: str) -> Dict[str, float]:
        """
        Analyzes the mood of input text using VADER sentiment analysis.
        Returns compound, positive, negative, and neutral scores.
        """
        return self.sia.polarity_scores(text)
    
    def get_music_parameters(self, mood_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Converts sentiment scores into music parameters.
        Returns tempo, energy, and valence parameters.
        """
        compound = mood_scores['compound']
        pos = mood_scores['pos']
        neg = mood_scores['neg']
        
        # Map compound score to tempo (60-180 BPM)
        tempo = 120 + (compound * 60)
        
        # Map positive sentiment to energy (0-1)
        energy = 0.5 + (pos * 0.5)
        
        # Map compound score to valence (0-1)
        valence = (compound + 1) / 2  # Normalize from [-1,1] to [0,1]
        
        return {
            'tempo': max(60, min(180, tempo)),
            'energy': max(0, min(1, energy)),
            'valence': max(0, min(1, valence))
        }
    
    def get_genre_weights(self, mood_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Determines appropriate music genres based on mood analysis.
        Returns weighted genre preferences.
        """
        compound = mood_scores['compound']
        pos = mood_scores['pos']
        neg = mood_scores['neg']
        
        genres = {
            'pop': 0.5 + (pos * 0.3),
            'rock': 0.3 + (neg * 0.4),
            'classical': 0.4 - (abs(compound) * 0.2),
            'electronic': 0.3 + (pos * 0.4),
            'jazz': 0.4 + (compound * 0.2),
            'ambient': 0.3 + ((1 - abs(compound)) * 0.4)
        }
        
        # Normalize weights
        total = sum(genres.values())
        return {k: v/total for k, v in genres.items()}
    
    def generate_playlist_parameters(self, text: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Generates complete playlist parameters from input text.
        Returns music parameters and genre weights.
        """
        mood_scores = self.analyze_mood(text)
        music_params = self.get_music_parameters(mood_scores)
        genre_weights = self.get_genre_weights(mood_scores)
        
        return music_params, genre_weights