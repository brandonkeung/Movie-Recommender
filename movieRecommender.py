import os # access my csv files
import numpy as np # linear algebra(used for normalization)
import pandas as pd # To create and utilize datsets
import scipy as sp # to pivot my table
from sklearn.metrics.pairwise import cosine_similarity # ML model


#default theme and settings
pd.options.display.max_columns

# Storing paths to both datasets
movie_path = "movies.csv"
rating_path = "ratings.csv"
rating_df = pd.read_csv(rating_path)
movie_df = pd.read_csv(movie_path)

# prints the head of both datasets
print("Movie and Rating Heads:")
print(rating_df.head())
print(movie_df.head())

print("-"*50, '\n\n')

# Prints info the table such as number of columns and types of values stored
print("Movie and Ratings Info: ")
print("Movies:\n")
print(movie_df.info())
print("\n","*"*50,"\nRating:\n")
print(rating_df.info())

print("-"*50, '\n\n')

# Shows info with regard to any missing values
print("Movie missing values: ")
print(movie_df.isnull().sum())
print("*"*50)
print("Rating missing values (%):")
print(rating_df.isnull().sum())
# Shows that there is no missing values in either datasets so no need for cleanup

# If i needed to cleanup, assuming -1 represents the user did not input anything for a rating, 
# I would use the following code to :
'''
rating_df['rating'] = rating_df['rating'].apply(lambda x: np.nan if x==-1 else x)
print(rating_df.head(20))
'''

# Merging both datasets using movieID
rated_movies = rating_df.merge(movie_df, left_on = 'movieId', right_on = 'movieId', suffixes= ['_user', ''])
print(rated_movies)

# Gets rid of the column for timestamp
rated_movies =rated_movies[['userId', 'genres', 'title', 'rating']] 
print("Filtered out column for timestamp\n")
print(rated_movies)
print("-"*50, '\n\n')
print("Rated_movies Info:")
print(rated_movies.info())

# Pivot table so users are rows annd movie titles are columns
# By doing this, we can see individual users along with all of their ratings
pivot_df = rated_movies.pivot_table(index = ['userId'], columns=['title'], values = 'rating')
print(pivot_df.head())

# Fill all NaN values with -1 and then drop the columns with all -1(no user rated the movie)
pivot_n = pivot_df.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)       # normalization
pivot_n.fillna(-1, inplace=True)        # replace NaN with -1
pivot_n = pivot_n.T     # transpose pivot for dropping -1's
pivot_n = pivot_n.loc[:, (pivot_n != -1).any(axis=0)]   # dropping columns with the value of 0(meaning unrated movies)
print(pivot_n)
piv_sparse = sp.sparse.csr_matrix(pivot_n.values)       # convert to sparse matrix format for similarity computation
print(pivot_n)

movie_similarity = cosine_similarity(piv_sparse)        # generate model based on cosine similarity comparisons

mov_sim_df = pd.DataFrame(movie_similarity, index = pivot_n.index, columns = pivot_n.index)     # df of movie similarities

    
def movieRecommendation(mov_name, num_recommendations):
    number = 1
    print('Recommended because you watched {}:\n'.format(mov_name))
    for movie in mov_sim_df.sort_values(by = mov_name, ascending = False).index[1:num_recommendations+1]:
        print(f'#{number}: {movie}, {round(mov_sim_df[movie][mov_name]*100,2)}% match')
        number +=1  

if __name__ == "__main__":
    print("-"*50, "\n\n")
    movieRecommendation('Star Wars: Episode IV - A New Hope (1977)', 5)
    print("-"*50, "\n\n")
    movieRecommendation('Star Wars: Episode IV - A New Hope (1977)', 10)
    print("-"*50, "\n\n")
    movieRecommendation('A Quiet Place (2018)', 10)
    print("-"*50, "\n\n")