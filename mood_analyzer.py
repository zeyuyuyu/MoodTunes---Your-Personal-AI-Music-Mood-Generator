import numpy as np
from sklearn.ensemble import RandomForestRegressor

class MoodAnalyzer:
    def __init__(self):
        self.model = RandomForestRegressor()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_mood(self, audio_features):
        mood_scores = self.model.predict([audio_features])
        return mood_scores[0]

# Example usage
analyzer = MoodAnalyzer()
X_train = [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], ...]
y_train = [2.5, 3.0, ...]
analyzer.train(X_train, y_train)

new_audio_features = [6.2, 2.8, 4.8, 1.8]
predicted_mood = analyzer.predict_mood(new_audio_features)
print(f"Predicted mood score: {predicted_mood}")