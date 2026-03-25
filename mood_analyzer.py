import numpy as np
from transformers import pipeline
from typing import Dict, List, Tuple

class MoodAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline('sentiment-analysis')
        self.mood_features = {
            'happy': {
                'tempo_range': (120, 160),
                'key': ['C major', 'G major', 'D major'],
                'mode': 'major',
                'energy': (0.7, 1.0)
            },
            'sad': {
                'tempo_range': (60, 90),
                'key': ['A minor', 'D minor', 'E minor'],
                'mode': 'minor', 
                'energy': (0.2, 0.5)
            },
            'relaxed': {
                'tempo_range': (70, 100),
                'key': ['F major', 'Bb major'],
                'mode': 'major',
                'energy': (0.3, 0.6)
            },
            'energetic': {
                'tempo_range': (125, 180),
                'key': ['E major', 'A major'],
                'mode': 'major',
                'energy': (0.8, 1.0)
            }
        }

    def analyze_text(self, text: str) -> Tuple[str, float]:
        """Analyze text input to determine emotional mood and confidence score."""
        result = self.sentiment_analyzer(text)[0]
        sentiment = result['label'].lower()
        confidence = result['score']
        
        return self._map_sentiment_to_mood(sentiment), confidence

    def _map_sentiment_to_mood(self, sentiment: str) -> str:
        """Map raw sentiment to musical mood category."""
        sentiment_mood_map = {
            'positive': 'happy',
            'negative': 'sad',
            'neutral': 'relaxed'
        }
        return sentiment_mood_map.get(sentiment, 'relaxed')

    def get_musical_features(self, mood: str) -> Dict:
        """Get musical features corresponding to detected mood."""
        features = self.mood_features.get(mood, self.mood_features['relaxed'])
        
        # Add some controlled randomization
        tempo = np.random.uniform(*features['tempo_range'])
        key = np.random.choice(features['key'])
        energy = np.random.uniform(*features['energy'])
        
        return {
            'tempo': round(tempo, 1),
            'key': key,
            'mode': features['mode'],
            'energy': round(energy, 2)
        }

    def suggest_music_parameters(self, text: str) -> Dict:
        """Main method to analyze text and return musical parameters."""
        mood, confidence = self.analyze_text(text)
        musical_features = self.get_musical_features(mood)
        
        return {
            'mood': mood,
            'confidence': round(confidence, 3),
            'musical_features': musical_features
        }
