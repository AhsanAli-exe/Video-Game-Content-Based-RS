import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from difflib import get_close_matches
import json
import re

# Load the games dataset
df = pd.read_csv('modified_games.csv')

# Clean and parse genres from string format to list
def parse_genres(text):
    if not isinstance(text, str):
        return []
    if text.startswith('[') and text.endswith(']'):
        try:
            text = text.replace("'", "\"")
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except:
            genres = re.findall(r"'([^']*)'", text)
            if genres:
                return genres
    if ',' in text:
        return [g.strip() for g in text.split(',')]
    return [text]

# Clean and prepare data
print("Preparing game data...")
df['genres_list'] = df['Genres'].apply(parse_genres)
df['developers_list'] = df['Developers'].apply(parse_genres)
df['game_content'] = df['Summary'].fillna('') 

# Create TF-IDF matrix for game content
print("Building recommendation system...")
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['game_content'])
sparse.save_npz('tfidf_matrix.npz', tfidf_matrix)

def get_recommendations(game_title, num_recommendations=10):
    # Check if game exists
    if game_title not in df['Title'].values:
        print(f"Game '{game_title}' not found in the dataset.")
        return []
    
    # Get target game information
    game_idx = df[df['Title'] == game_title].index[0]
    target_game = df.iloc[game_idx]
    target_genres = target_game['genres_list']
    target_developers = target_game['developers_list']
    
    print(f"\nFinding games similar to: {game_title}")
    print(f"Genres: {target_genres}")
    print(f"Developers: {target_developers}")
    
    # Find similar games based on genres and developers
    exact_genre_matches = []
    similar_genre_matches = []
    developer_matches = []
    other_games = []
    
    for idx, row in df.iterrows():
        if idx == game_idx:  # Skip the input game
            continue
        
        # Check for genre and developer matches
        current_genres = row['genres_list']
        current_developers = row['developers_list']
        
        # Count matching genres and developers
        genre_overlap = set(current_genres).intersection(set(target_genres))
        dev_overlap = set(current_developers).intersection(set(target_developers))
        
        # Calculate match scores
        genre_match_ratio = len(genre_overlap) / max(len(target_genres), 1)
        
        # Categorize matches
        if len(genre_overlap) > 0 and len(dev_overlap) > 0:
            # Both genre and developer match
            exact_genre_matches.append((idx, 0.9))
        elif genre_match_ratio == 1.0:
            # Perfect genre match
            exact_genre_matches.append((idx, 0.8))
        elif len(genre_overlap) > 0:
            # Partial genre match
            similar_genre_matches.append((idx, 0.7 * genre_match_ratio))
        elif len(dev_overlap) > 0:
            # Only developer match
            developer_matches.append((idx, 0.6))
        else:
            # No direct match
            other_games.append(idx)
    
    # Combine all matches
    candidates = exact_genre_matches + similar_genre_matches + developer_matches
    
    # If we need more candidates, add text similarity matches
    if len(candidates) < num_recommendations * 2:
        # Calculate text similarity for other games
        text_sim = cosine_similarity(tfidf_matrix[game_idx:game_idx+1], tfidf_matrix).flatten()
        
        # Find top text matches excluding games already in candidates
        top_indices = text_sim.argsort()[::-1][1:20]  # Get top 20 similar games
        for idx in top_indices:
            if idx not in [c[0] for c in candidates] and idx != game_idx:
                candidates.append((idx, 0.4 * text_sim[idx]))
    
    # Sort candidates by score and take top recommendations
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates[:num_recommendations]
    
    # Convert to recommendation objects
    recommendations = []
    for idx, score in top_candidates:
        game = df.iloc[idx]
        recommendations.append({
            'title': game['Title'],
            'genres': game['Genres'],
            'developers': game['Developers'],
            'summary': game['Summary'],
            'similarity': score
        })
    
    return recommendations

def print_recommendations(recommendations):
    if not recommendations:
        print("No recommendations found.")
        return
        
    print("\nRecommended Games:")
    print("-" * 100)
    
    for i, rec in enumerate(recommendations):
        try:
            print(f"{i+1}. {rec['title']} (Similarity: {rec['similarity']:.2f})")
            print(f"   Genres: {rec['genres']}")
            print(f"   Developer: {rec['developers']}")
            
            summary = rec['summary']
            if isinstance(summary, str) and len(summary) > 150:
                summary = summary[:150] + "..."
            print(f"   Summary: {summary}")
        except KeyError as e:
            print(f"   Error: Missing field in recommendation: {e}")
        print("-" * 100)

def find_game(game_title):
    # Handle approximate matching if exact title not found
    if game_title not in df['Title'].values:
        close_matches = get_close_matches(game_title, df['Title'].values, n=5, cutoff=0.6)
        if close_matches:
            print(f"Game '{game_title}' not found. Did you mean one of these?")
            for i, match in enumerate(close_matches):
                print(f"{i+1}. {match}")
            choice = input("Enter the number of your choice (or any other key to try again): ")
            try:
                if 1 <= int(choice) <= len(close_matches):
                    game_title = close_matches[int(choice)-1]
                else:
                    return None
            except:
                return None
        else:
            print(f"No games found similar to '{game_title}'")
            return None
    
    return game_title

def show_game_recommendations(game_title):
    game_title = find_game(game_title)
    if not game_title:
        return
    
    try:
        # Display game info
        game_data = df[df['Title'] == game_title]
        print(f"\nSelected game: {game_title}")
        print(f"Genres: {game_data['Genres'].values[0]}")
        print(f"Developer: {game_data['Developers'].values[0]}")
        
        summary = game_data['Summary'].values[0]
        if isinstance(summary, str) and len(summary) > 200:
            summary = summary[:200] + "..."
        print(f"Summary: {summary}")
        
        # Get and show recommendations
        recommendations = get_recommendations(game_title)
        print_recommendations(recommendations)
    except Exception as e:
        print(f"Error: {str(e)}")

def browse_by_genre():
    # Get all unique genres
    all_genres = set()
    for genres_list in df['genres_list']:
        for genre in genres_list:
            all_genres.add(genre)
    
    # Display genres
    sorted_genres = sorted(list(all_genres))
    print("\nAvailable Genres:")
    for i, genre in enumerate(sorted_genres):
        print(f"{i+1}. {genre}")
    
    # Let user select a genre
    try:
        choice = int(input("\nEnter the number of the genre you're interested in: "))
        if 1 <= choice <= len(sorted_genres):
            selected_genre = sorted_genres[choice-1]
            print(f"\nShowing games in the '{selected_genre}' genre:")
            
            # Find games with the selected genre
            genre_games = []
            for _, row in df.iterrows():
                if selected_genre in row['genres_list']:
                    genre_games.append(row['Title'])
            
            # Show sample of games in this genre
            if genre_games:
                print(f"Found {len(genre_games)} games in the '{selected_genre}' genre.")
                sample_size = min(10, len(genre_games))
                sample_games = genre_games[:sample_size]
                
                for i, game in enumerate(sample_games):
                    print(f"{i+1}. {game}")
                
                # Let user pick a game for recommendations
                game_choice = int(input("\nEnter the number of a game to get recommendations: "))
                if 1 <= game_choice <= len(sample_games):
                    selected_game = sample_games[game_choice-1]
                    show_game_recommendations(selected_game)
            else:
                print(f"No games found in the '{selected_genre}' genre.")
        else:
            print("Invalid choice.")
    except ValueError:
        print("Please enter a valid number.")
    except Exception as e:
        print(f"Error: {str(e)}")

def run_game_recommender():
    print("\n" + "="*80)
    print("Game Recommendation System")
    print("="*80)
    
    while True:
        print("\nOptions:")
        print("1. Get recommendations for a specific game")
        print("2. View random game recommendations")
        print("3. Browse games by genre")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            game_title = input("Enter a game title: ")
            show_game_recommendations(game_title)
        elif choice == '2':
            random_game = df['Title'].sample().values[0]
            print(f"\nRandom game selected: {random_game}")
            show_game_recommendations(random_game)
        elif choice == '3':
            browse_by_genre()
        elif choice == '4':
            print("Thank you for using the Game Recommendation System!")
            break
        else:
            print("Invalid choice. Please try again.")

# Main program
if __name__ == "__main__":
    print("\nGame Recommendation System")
    print("="*80)
    print("Finding similar games based on genres, developers, and content")
    print(f"Total games in database: {len(df)}")
    
    run_game_recommender()
