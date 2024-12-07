import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import requests
import matplotlib.pyplot as plt
import seaborn as sns


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
        clean_data = self.player_data.dropna(subset=features + ['total_points'])

        for col in features + ['total_points']:
            clean_data[col] = pd.to_numeric(clean_data[col], errors='coerce')

        X = clean_data[features]
        y = clean_data['total_points']
        X = sm.add_constant(X)

        if X.empty or y.empty:
            raise ValueError("Input data for OLS regression is empty. Check for missing values.")

        model = sm.OLS(y, X).fit()

        # Display summary
        print(model.summary())

        # Create a DataFrame with feature names and their corresponding p-values
        p_values_df = pd.DataFrame({
            'Feature': features,
            'P-Value': model.pvalues[1:]  # exclude constant
        })

        # Plot p-values for feature selection
        plt.figure(figsize=(10, 6))
        sns.barplot(x='P-Value', y='Feature', data=p_values_df, palette='viridis')
        plt.axvline(x=0.05, color='r', linestyle='--')  # Threshold line at p-value = 0.05
        plt.title('P-Values for Feature Selection')
        plt.xlabel('P-Value')
        plt.ylabel('Feature')
        plt.show()

        # Select features with p-values below 0.05 (significant)
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
        return rf_model, scaler, y_test, y_pred

    def display_feature_importances(self, model):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': self.selected_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        print("\nFeature Importances:\n", feature_importance_df)

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Feature Importances')
        plt.gca().invert_yaxis()  # Invert y-axis so the most important feature is at the top
        plt.show()

    def plot_actual_vs_predicted(self, y_test, y_pred):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        plt.xlabel('Actual Points')
        plt.ylabel('Predicted Points')
        plt.title('Actual vs Predicted Points')
        plt.show()

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
    model, scaler, y_test, y_pred = predictor.train_model()

    # Display feature importances
    predictor.display_feature_importances(model)

    # Plot Actual vs Predicted
    predictor.plot_actual_vs_predicted(y_test, y_pred)

    print("\nGetting Top Predicted Players...")
    top_players = predictor.get_top_predicted_players(model, scaler)
    print("\nTop Predicted Players:\n", top_players)


if __name__ == "__main__":
    main()
