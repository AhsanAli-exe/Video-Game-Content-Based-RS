import pandas as pd
import numpy as np
from games_contentBased import getRecs,df

def calculate_precision(actual_genres, recommended_games, k=10):
    """
    Calculate precision@k for genre-based recommendations
    Precision = (Number of relevant recommendations) / (Number of recommendations)
    """
    if not recommended_games:
        return 0.0
        
    relevant_count = 0
    for game in recommended_games[:k]:
        recGameData = df[df['Title'] == game['title']]
        if not recGameData.empty:
            recGenres = recGameData['genres_list'].iloc[0]
            if any(genre in actual_genres for genre in recGenres):
                relevant_count += 1
    
    return relevant_count/min(k,len(recommended_games))

def evalPrecision(sample_size=50,k_values=[5,10]):
    """
    Evaluate precision@k for a sample of games
    """
    # Sample random games from the dataset
    if sample_size > len(df):
        sample_size = len(df)
    
    sample_games = df.sample(sample_size)['Title'].tolist()
    
    results = {f'precision@{k}': [] for k in k_values}
    
    print(f"Evaluating precision on {sample_size} random games...")
    
    for i, game in enumerate(sample_games):
        print(f"Processing game {i+1}/{sample_size}: {game}")
        game_data = df[df['Title'] == game]
        if game_data.empty:
            continue
            
        actual_genres = game_data['genres_list'].iloc[0]
        recommendations = getRecs(game,numRecs=max(k_values))
        
        # Calculate precision for each k value
        for k in k_values:
            precision = calculate_precision(actual_genres, recommendations, k)
            results[f'precision@{k}'].append(precision)
    avg_results = {metric: np.mean(values) for metric, values in results.items()}
    
    return avg_results, results


print("=" * 80)
print("EVALUATING CONTENT-BASED RECOMMENDATION SYSTEM")
print("=" * 80)
k_values = [5, 10]
# Number of random games to evaluate
sample_size = 20

avg_results,detailed_results = evalPrecision(sample_size, k_values)

# Print results
print("\nEvaluation Results:")
print("-" * 50)
for metric, value in avg_results.items():
    print(f"{metric}: {value:.4f}")

# Optional: Save detailed results to CSV
results_df = pd.DataFrame(detailed_results)
results_df.to_csv('precision_evaluation_results.csv', index=False)
print("\nDetailed results saved to 'precision_evaluation_results.csv'")

# Calculate precision distribution
print("\nPrecision Distribution:")
for k in k_values:
    precision_values = detailed_results[f'precision@{k}']
    print(f"\nPrecision@{k} Distribution:")
    print(f"  Min: {min(precision_values):.4f}")
    print(f"  Max: {max(precision_values):.4f}")
    print(f"  Median: {np.median(precision_values):.4f}")
    print(f"  Std Dev: {np.std(precision_values):.4f}")

print("\nEvaluation complete!")
