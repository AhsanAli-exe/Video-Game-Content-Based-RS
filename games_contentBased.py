import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from difflib import get_close_matches
import json
import re

df = pd.read_csv('modified_games.csv')
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

print("Preparing game data...")
df['genres_list'] = df['Genres'].apply(parse_genres)
df['developers_list'] = df['Developers'].apply(parse_genres)
df['game_content'] = df['Summary'].fillna('') 

#Creating TF-IDF matrix for game content
print("Building recommendation system...")
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['game_content'])
sparse.save_npz('tfidf_matrix.npz', tfidf_matrix)

def getRecs(gameTitle,numRecs=10):
    # Check if game exists
    if gameTitle not in df['Title'].values:
        print(f"Game '{gameTitle}' not found in the dataset.")
        return []
    
    gameIndex = df[df['Title'] == gameTitle].index[0]
    target_game = df.iloc[gameIndex]
    target_genres = target_game['genres_list']
    target_developers = target_game['developers_list']
    
    print(f"\nFinding games similar to: {gameTitle}")
    print(f"Genres: {target_genres}")
    print(f"Developers: {target_developers}")
    
    # Find similar games based on genres and developers
    exactGenreMatches = []
    similarGenreMatches = []
    developerMatches = []
    otherGames = []
    
    for index,row in df.iterrows():
        if index == gameIndex:  # Skip the input game
            continue
        current_genres = row['genres_list']
        current_developers = row['developers_list']
        genreOverlap = set(current_genres).intersection(set(target_genres))
        devOverlap = set(current_developers).intersection(set(target_developers))
        
        # Calculate match scores
        genre_match_ratio = len(genreOverlap) / max(len(target_genres), 1)
        
        # Categorize matches
        if len(genreOverlap) > 0 and len(devOverlap) > 0:
            # Both genre and developer match
            exactGenreMatches.append((index, 0.9))
        elif genre_match_ratio == 1.0:
            # Perfect genre match
            exactGenreMatches.append((index, 0.8))
        elif len(genreOverlap) > 0:
            # Partial genre match
            similarGenreMatches.append((index, 0.7 * genre_match_ratio))
        elif len(devOverlap) > 0:
            # Only developer match
            developerMatches.append((index, 0.6))
        else:
            # No direct match
            otherGames.append(index)
    
    candidates = exactGenreMatches + similarGenreMatches + developerMatches
    
    if len(candidates)<numRecs * 2:
        # Calculate text similarity for other games
        text_sim = cosine_similarity(tfidf_matrix[gameIndex:gameIndex+1],tfidf_matrix).flatten()
        
        #Finding top text matches excluding games already in candidates
        top_indices = text_sim.argsort()[::-1][1:20]  # Get top 20 similar games
        for index in top_indices:
            if index not in [c[0] for c in candidates] and index != gameIndex:
                candidates.append((index, 0.4 * text_sim[index]))
    
    # sorting candidatex by score and take top recommendations
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates[:numRecs]
    recommendations = []
    for index, score in top_candidates:
        game = df.iloc[index]
        recommendations.append({
            'title': game['Title'],
            'genres': game['Genres'],
            'developers': game['Developers'],
            'summary': game['Summary'],
            'similarity': score
        })
    
    return recommendations

def printRecs(recommendations):
    if not recommendations:
        print("No recommendations found.")
        return
        
    print("\nRecommended Games:")
    print("-" * 100)
    
    for i,rec in enumerate(recommendations):
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

def findGame(gameTitle):
    if gameTitle not in df['Title'].values:
        closeMatches = get_close_matches(gameTitle, df['Title'].values, n=5, cutoff=0.6)
        if closeMatches:
            print(f"Game '{gameTitle}' not found. Did you mean one of these?")
            for i, match in enumerate(closeMatches):
                print(f"{i+1}. {match}")
            choice = input("Enter the number of your choice (or any other key to try again): ")
            try:
                if 1 <= int(choice) <= len(closeMatches):
                    gameTitle = closeMatches[int(choice)-1]
                else:
                    return None
            except:
                return None
        else:
            print(f"No games found similar to '{gameTitle}'")
            return None
    
    return gameTitle

def showRecs(gameTitle):
    gameTitle = findGame(gameTitle)
    if not gameTitle:
        return
    
    try:
        # Display game info
        game_data = df[df['Title'] == gameTitle]
        print(f"\nSelected game: {gameTitle}")
        print(f"Genres: {game_data['Genres'].values[0]}")
        print(f"Developer: {game_data['Developers'].values[0]}")
        
        summary = game_data['Summary'].values[0]
        if isinstance(summary, str) and len(summary) > 200:
            summary = summary[:200] + "..."
        print(f"Summary: {summary}")
        
        # Get and show recommendations
        recommendations = getRecs(gameTitle)
        printRecs(recommendations)
    except Exception as e:
        print(f"Error: {str(e)}")


def gameRecommender():
    print("\n" + "="*80)
    print("Game Recommendation System")
    print("="*80)
    
    while True:
        print("\nOptions:")
        print("1. Get recommendations for a specific game")
        print("2. View random game recommendations")
        print("3. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            gameTitle = input("Enter a game title: ")
            showRecs(gameTitle)
        elif choice == '2':
            randomGame = df['Title'].sample().values[0]
            print(f"\nRandom game selected: {randomGame}")
            showRecs(randomGame)
        elif choice == '3':
            print("Thank you for using the Game Recommendation System!")
            break
        else:
            print("Invalid choice. Please try again.")


print("\nGame Recommendation System")
print("="*80)
print("Finding similar games based on genres, developers, and content")
print(f"Total games in database: {len(df)}")

gameRecommender()
