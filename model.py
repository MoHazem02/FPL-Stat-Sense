import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import requests

class FantasyPremierLeaguePredictor:
    def __init__(self):
        # API endpoint for FPL data
        self.base_url = "https://fantasy.premierleague.com/api"
        
        # Key features to consider for prediction
        self.features = [
            'total_points', 'minutes', 'goals_scored', 'assists', 
            'clean_sheets', 'goals_conceded', 'own_goals', 
            'penalties_saved', 'penalties_missed', 'yellow_cards', 
            'red_cards', 'saves', 'bonus', 'influence', 'creativity', 
            'threat', 'form'
        ]
        
        # Dataframes to store data
        self.player_data = None
        self.historical_data = None
        
    def fetch_player_data(self):
        """
        Fetch current player data from FPL API
        """
        try:
            # Fetch bootstrap data (contains player information)
            bootstrap_url = f"{self.base_url}/bootstrap-static/"
            response = requests.get(bootstrap_url)
            data = response.json()
            
            # Convert players to DataFrame
            players = pd.DataFrame(data['elements'])
            
            # Select and preprocess relevant columns
            self.player_data = players[[
                'web_name', 'team', 'element_type', 'total_points', 
                'minutes', 'goals_scored', 'assists', 'clean_sheets', 
                'goals_conceded', 'own_goals', 'penalties_saved', 
                'penalties_missed', 'yellow_cards', 'red_cards', 
                'saves', 'bonus', 'influence', 'creativity', 
                'threat', 'form'
            ]].copy()
            
            # Convert string columns to numeric, handling potential errors
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
    
    def prepare_training_data(self):
        """
        Prepare data for model training
        """
        if self.player_data is None:
            raise ValueError("Player data not fetched. Call fetch_player_data() first.")
        
        # Drop rows with missing data
        clean_data = self.player_data.dropna(subset=self.features)
        
        # Separate features and target
        X = clean_data[self.features[1:]]  # Exclude total_points from features
        y = clean_data['total_points']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def train_model(self):
        """
        Train Random Forest Regression model
        """
        X_train, X_test, y_train, y_test, scaler = self.prepare_training_data()
        
        # Initialize and train Random Forest Regressor
        rf_model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = rf_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print("Model Performance:")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        
        return rf_model, scaler
    
    def predict_player_points(self, model, scaler, player_features):
        """
        Predict points for a specific player
        
        :param model: Trained RandomForestRegressor
        :param scaler: StandardScaler used for training
        :param player_features: List of feature values for a player
        :return: Predicted points
        """
        # Scale the input features
        player_features_scaled = scaler.transform([player_features])
        
        # Predict points
        predicted_points = model.predict(player_features_scaled)
        
        return predicted_points[0]
    
    def get_top_predicted_players(self, model, scaler, n=10):
        """
        Get top N players with highest predicted points
        
        :param model: Trained RandomForestRegressor
        :param scaler: StandardScaler used for training
        :param n: Number of top players to return
        :return: DataFrame of top players
        """
        if self.player_data is None:
            raise ValueError("Player data not fetched. Call fetch_player_data() first.")
        
        # Create a copy to avoid modifying original data
        players = self.player_data.copy()
        
        # Predict points for each player
        players['predicted_points'] = players.apply(
            lambda row: self.predict_player_points(
                model, 
                scaler, 
                row[self.features[1:]]
            ) if not pd.isnull(row[self.features[1:]]).any() else np.nan, 
            axis=1
        )
        
        # Sort and return top N players
        top_players = players.dropna(subset=['predicted_points']) \
            .sort_values('predicted_points', ascending=False) \
            .head(n)
        
        return top_players[['web_name', 'predicted_points', 'total_points']]

def main():
    # Create predictor instance
    predictor = FantasyPremierLeaguePredictor()
    
    # Fetch player data
    predictor.fetch_player_data()
    
    # Train the model
    model, scaler = predictor.train_model()
    
    # Get top 10 predicted players
    top_players = predictor.get_top_predicted_players(model, scaler)
    
    print("\nTop 10 Players by Predicted Points:")
    print(top_players)

if __name__ == "__main__":
    main()

# Note: Requires 'requests' and 'scikit-learn' libraries
# Install with: pip install requests scikit-learn pandas numpy