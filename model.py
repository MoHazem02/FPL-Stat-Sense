import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import requests

class FantasyPremierLeaguePredictor:
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api"
        self.features = [
            'total_points', 'minutes', 'goals_scored', 'assists',
            'clean_sheets', 'goals_conceded', 'own_goals',
            'penalties_saved', 'penalties_missed', 'yellow_cards',
            'red_cards', 'saves', 'bonus', 'influence', 'creativity',
            'threat', 'form'
        ]
        self.player_data = None
        self.selected_features = None

    def fetch_player_data(self):
        try:
            bootstrap_url = f"{self.base_url}/bootstrap-static/"
            response = requests.get(bootstrap_url)
            data = response.json()
            players = pd.DataFrame(data['elements'])
            self.player_data = players[[
                'web_name', 'team', 'element_type', 'total_points',
                'minutes', 'goals_scored', 'assists', 'clean_sheets',
                'goals_conceded', 'own_goals', 'penalties_saved',
                'penalties_missed', 'yellow_cards', 'red_cards',
                'saves', 'bonus', 'influence', 'creativity',
                'threat', 'form'
            ]].copy()
            numeric_columns = [
                'total_points', 'minutes', 'goals_scored', 'assists',
                'clean_sheets', 'goals_conceded', 'own_goals',
                'penalties_saved', 'penalties_missed', 'yellow_cards',
                'red_cards', 'saves', 'bonus'
            ]
            for col in numeric_columns:
                self.player_data[col] = pd.to_numeric(self.player_data[col], errors='coerce')
            return self.player_data
        except Exception as e:
            print(f"Error fetching player data: {e}")
            return None

    def perform_statistical_analysis(self):
        features = [
            'minutes', 'goals_scored', 'assists', 'clean_sheets',
            'influence', 'creativity', 'threat', 'form'
        ]
        # Drop rows with missing values
        clean_data = self.player_data.dropna(subset=features + ['total_points'])

        # Ensure all columns are numeric
        for col in features + ['total_points']:
            clean_data[col] = pd.to_numeric(clean_data[col], errors='coerce')

        # Separate predictors (X) and target (y)
        X = clean_data[features]
        y = clean_data['total_points']

        # Add constant term for intercept
        X = sm.add_constant(X)

        # Check for empty data
        if X.empty or y.empty:
            raise ValueError("Input data for OLS regression is empty. Check for missing values.")

        # Fit the OLS model
        model = sm.OLS(y, X).fit()

        # Display summary
        print(model.summary())

        # Select significant features based on p-values
        significant_features = [features[i] for i in range(len(features)) if model.pvalues[i + 1] < 0.05]
        self.selected_features = significant_features

        print("\nSelected Features Based on Statistical Analysis:", self.selected_features)

    def prepare_training_data(self):
        if self.player_data is None:
            raise ValueError("Player data not fetched. Call fetch_player_data() first.")
        if not self.selected_features:
            raise ValueError("Selected features not determined. Call perform_statistical_analysis() first.")
        clean_data = self.player_data.dropna(subset=self.selected_features + ['total_points'])
        X = clean_data[self.selected_features]
        y = clean_data['total_points']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    def train_model(self):
        X_train, X_test, y_train, y_test, scaler = self.prepare_training_data()
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Model Performance:\nMAE: {mae:.2f}, RMSE: {rmse:.2f}")
        return rf_model, scaler

    def display_feature_importances(self, model):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': self.selected_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        print("\nFeature Importances:\n", feature_importance_df)

    def get_top_predicted_players(self, model, scaler, n=10):
        players = self.player_data.copy()
        players['predicted_points'] = players.apply(
            lambda row: self.predict_player_points(
                model,
                scaler,
                row[self.selected_features]
            ) if not pd.isnull(row[self.selected_features]).any() else np.nan,
            axis=1
        )
        top_players = players.dropna(subset=['predicted_points']) \
            .sort_values('predicted_points', ascending=False) \
            .head(n)
        return top_players[['web_name', 'predicted_points', 'total_points']]

    def predict_player_points(self, model, scaler, player_features):
        player_features_scaled = scaler.transform([player_features])
        predicted_points = model.predict(player_features_scaled)
        return predicted_points[0]

def main():
    predictor = FantasyPremierLeaguePredictor()
    predictor.fetch_player_data()
    print("Performing Statistical Analysis for Feature Selection...")
    predictor.perform_statistical_analysis()
    print("\nTraining Random Forest Model...")
    model, scaler = predictor.train_model()
    predictor.display_feature_importances(model)
    print("\nGetting Top Predicted Players...")
    top_players = predictor.get_top_predicted_players(model, scaler)
    print("\nTop Predicted Players:\n", top_players)

if __name__ == "__main__":
    main()
