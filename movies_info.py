import pandas as pd 

movies_info_without_url = './data/movies_info_without_url.csv'
poster_url = './data/poster_url.csv'
movies_info_df = pd.read_csv(movies_info_without_url)
poster_url_df = pd.read_csv(poster_url,header=None)

# concat the movies_info_without_url and poster_url
poster_url_df = poster_url_df.rename(columns={1:'poster_url'})
movies_info = pd.concat([movies_info_df, poster_url_df['poster_url']], axis=1)
# rename for query
movies_info = movies_info.rename(columns={'movieId': 'movie_id', 'title': 'movie_title', 'unknown': 'Other', 'Film-Noir':'Film_Noir','Sci-Fi':'Sci_Fi'})
movies_info.insert(3,'genres',movies_info.pop('genres'))

# fill in the 4 null poster_url by hand
movies_info.loc[[421],['poster_url']] = "https://m.media-amazon.com/images/M/MV5BMGMyZjAzZGItY2ZkZi00Y2YwLTk4M2QtNTRmNWEzYjU1NDVlXkEyXkFqcGdeQXVyNjg4NzYzMzA@._V1_QL75_UY281_CR1,0,190,281_.jpg"
movies_info.loc[[585],['poster_url']] = "https://m.media-amazon.com/images/M/MV5BYWQ0MTJlYTYtNDU3ZS00ZmE4LWJjYTAtYzEzODAyZmM5ZjJhXkEyXkFqcGdeQXVyMTI4ODc2NDY@._V1_QL75_UY281_CR111,0,190,281_.jpg"
movies_info.loc[[5620],['poster_url']] = "https://m.media-amazon.com/images/M/MV5BZDY3NjhiZDItODg1NC00OTAxLThiMjUtNDRkMTkwZjI1NjQ3XkEyXkFqcGdeQXVyNTY2MzQ3MDE@._V1_QL75_UY281_CR93,0,190,281_.jpg"
movies_info.loc[[7600],['poster_url']] = "https://m.media-amazon.com/images/M/MV5BMjExMTM5NjQzMF5BMl5BanBnXkFtZTgwNjY0OTAxMzE@._V1_QL75_UY281_CR93,0,190,281_.jpg"

movies_info.to_csv('movie_info_latest.csv', index=False)