import requests
from bs4 import BeautifulSoup

def get_top_five_movies():
    url = "https://www.imdb.com/chart/top/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    top_five_movies = []
    movie_rows = soup.select(".lister-list tr")
    for row in movie_rows[:5]:
        title_column = row.select_one(".titleColumn")
        title = title_column.a.text
        year = title_column.span.text.strip("()")
        rating = row.select_one(".imdbRating strong").text
        director = row.select_one(".titleColumn a")["title"].split(", ")[0]
        
        movie_info = f"Title: {title}\nYear: {year}\nRating: {rating}\nDirector: {director}\n"
        top_five_movies.append(movie_info)

    for movie in top_five_movies:
        print(movie)

    return top_five_movies

get_top_five_movies()
