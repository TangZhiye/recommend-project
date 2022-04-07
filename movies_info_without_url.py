import pandas as pd

movies_data = './/ml-latest-small//movies.csv'
ratings_data = './/ml-latest-small//ratings.csv'
movies_df = pd.read_csv(movies_data)
ratings_df = pd.read_csv(ratings_data)

#Every genre is separated by a | so we simply have to call the split function on |
movies_df['genres'] = movies_df.genres.str.split('|')

# First let's make a copy of the movies_df
movies_with_genres = movies_df[['movieId','title','genres']].copy(deep=True)

# Let's iterate through movies_df, then append the movie genres as columns of 1s or 0s.
# 1 if that column contains movies in the genre at the present index and 0 if not.
genre_list = [] # store the occurred genres

for index, row in movies_df.iterrows():
    for genre in row['genres']:
        movies_with_genres.at[index, genre] = 1
        if genre not in genre_list:
            genre_list.append(genre)

print(genre_list)
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
movies_with_genres = movies_with_genres.fillna(0)

# extract the release year
movies_with_genres['release_year'] = movies_with_genres['title'].map(lambda x: str(x).strip()[-5:-1] if str(x)[-1:]==')' else 'unknown' ) 
movies_with_genres['title'] = movies_with_genres['title'].map(lambda x: str(x).strip()[:-6] if str(x).strip()[-1:]==')' else x ) 

movies_with_genres.rename(columns={'(no genres listed)':'unknown','Sci-Fi':'Sci_Fi','Film-Noir':'Film_Noir'}, inplace = True)
movies_with_genres.insert(3,'release_year',movies_with_genres.pop('release_year'))
movies_with_genres.insert(4,'unknown',movies_with_genres.pop('unknown'))
print(movies_with_genres.columns)

movies_with_genres.to_csv('./data/movies_info_without_url.csv', index=False)