🎬 Movie Recommendation System

A content-based movie recommendation system that recommends movies based on their similarity to a selected movie.

📌 Project Overview

This project uses movie information such as genres, keywords, cast, crew, and overview to find movies that are similar to a user's selected movie.

The recommendation system uses text processing and cosine similarity to calculate how similar movies are to each other.

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Gradio
- Google Colab

📊 Dataset

The project uses the TMDB 5000 Movie Dataset.

The dataset contains information about movies including:

- Movie title
- Genres
- Keywords
- Cast
- Crew
- Overview
- Popularity
- Budget

⚙️ How It Works

The recommendation system follows these steps:

1. Load the movie datasets using Pandas.
2. Check and handle missing values and duplicate records.
3. Select the relevant movie features.
4. Extract useful information from genres, keywords, cast, and crew.
5. Combine the selected information into a `tags` column.
6. Convert the text data into numerical vectors using `CountVectorizer`.
7. Calculate similarity between movies using cosine similarity.
8. Create a recommendation function that returns the five most similar movies.
9. Build a simple interactive interface using Gradio.

🎯 Example

If the user selects a movie such as **Avatar**, the system returns five movies with the highest similarity scores according to the content-based recommendation model.

💻 Interface

The project includes a Gradio interface where users can select a movie and receive recommendations.

🚀 Future Improvements

- Add movie posters using the TMDB API.
- Improve the user interface.
- Deploy the recommendation system as a web application.
- Explore additional recommendation techniques.

📚 What I Learned

Through this project, I gained practical experience with:

- Data cleaning and preprocessing
- Feature extraction
- Text vectorization
- Cosine similarity
- Content-based recommendation systems
- Building a basic interactive ML interface
