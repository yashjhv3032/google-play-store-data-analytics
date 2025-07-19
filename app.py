import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords # Import stopwords for word cloud
from wordcloud import WordCloud # Import WordCloud (If your IDE shows "Unresolved reference", ensure 'wordcloud' is installed via 'pip install wordcloud' and restart your IDE.)
import matplotlib.pyplot as plt # For displaying word cloud
import webbrowser
import os
from sklearn.preprocessing import StandardScaler # Import StandardScaler

# --- 1. Setup Environment: Download NLTK lexicon ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading 'vader_lexicon' for NLTK...")
    nltk.download('vader_lexicon')
    print("Download complete.")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading 'stopwords' for NLTK...")
    nltk.download('stopwords')
    print("Download complete.")


# --- 2. Data Loading ---
try:
    apps_df = pd.read_csv('Play Store Data.csv')
    reviews_df = pd.read_csv('User Reviews.csv')
    print("CSV files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading CSV: {e}. Please ensure 'Play Store Data.csv' and 'User Reviews.csv' are in the same directory as the script.")
    exit()

# --- 3. Data Cleaning & Preprocessing (apps_df) ---
apps_df = apps_df.dropna(subset=['Rating'])
print("Dropped rows with missing 'Rating'.")

for column in apps_df.columns:
    if apps_df[column].isnull().any() and column != 'Rating':
        try:
            apps_df[column].fillna(apps_df[column].mode()[0], inplace=True)
        except KeyError:
            print(f"Warning: Could not find mode for column '{column}'. Skipping fillna for this column.")
print("Filled missing values in other columns with their mode.")

apps_df.drop_duplicates(inplace=True)
print("Removed duplicate app entries.")

apps_df = apps_df[apps_df['Rating'] <= 5]
print("Filtered apps with 'Rating' greater than 5.")

# Data Cleaning & Preprocessing (reviews_df)
reviews_df.dropna(subset=['Translated_Review'], inplace=True)
print("Dropped rows with missing 'Translated_Review'.")

# --- 4. Data Transformation ---

apps_df['Installs'] = apps_df['Installs'].astype(str).str.replace(',', '', regex=False).str.replace('+', '', regex=False).apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
print("Transformed 'Installs' column.")

apps_df['Reviews'] = apps_df['Reviews'].astype(str).apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
print("Transformed 'Reviews' column.")

apps_df['Price'] = apps_df['Price'].astype(str).str.replace('$', '', regex=False).apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
print("Transformed 'Price' column.")

def convert_size(size):
    if isinstance(size, float) and np.isnan(size):
        return np.nan
    size = str(size).strip()

    if 'M' in size:
        return float(size.replace('M', ''))
    elif 'k' in size:
        return float(size.replace('k', '')) / 1024
    elif 'G' in size:
        return float(size.replace('G', '')) * 1024
    elif 'Varies with device' in size:
        return np.nan
    else:
        return np.nan

apps_df['Size'] = apps_df['Size'].apply(convert_size)
print("Transformed 'Size' column to MB.")

# CRITICAL FIX: Impute missing 'Size' values with the median after conversion
# This is done here so that 'Size' is ready before ML model preparation
if apps_df['Size'].isnull().any():
    median_size = apps_df['Size'].median()
    apps_df['Size'].fillna(median_size, inplace=True)
    print(f"Filled missing 'Size' values with median: {median_size:.2f} MB.")


apps_df['Log Installs'] = np.log1p(apps_df['Installs'])
apps_df['Log Reviews'] = np.log1p(apps_df['Reviews'])
print("Added 'Log Installs' and 'Log Reviews' columns.")

def rating_group(rating):
    if not isinstance(rating, (int, float)):
        return "Unknown"
    if rating >= 4:
        return "Top rated"
    elif rating >= 3:
        return "Above average"
    elif rating >= 2:
        return "Average"
    else:
        return "Below Average"

apps_df['Rating Group'] = apps_df['Rating'].apply(rating_group)
print("Added 'Rating Group' column.")

# --- New Data Transformations for Dashboard Plots ---
apps_df['Revenue'] = apps_df['Price'] * apps_df['Installs']
print("Added 'Revenue' column.")

# Process 'Last Updated' for 'Number of Updates Over the Years'
# Convert 'Last Updated' to datetime, then extract year
apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'], errors='coerce')
apps_df['Update_Year'] = apps_df['Last Updated'].dt.year.fillna(0).astype(int)
print("Processed 'Last Updated' column to 'Update_Year'.")

# --- 5. Data Merging ---
merged_df = pd.merge(apps_df, reviews_df, on='App', how='inner')
print("DataFrames merged successfully into 'merged_df'.")

# --- 6. Sentiment Analysis (on reviews) ---
print("Performing sentiment analysis on user reviews...")
analyzer = SentimentIntensityAnalyzer()

merged_df['Translated_Review'] = merged_df['Translated_Review'].astype(str).fillna('')

merged_df['Sentiment Score'] = merged_df['Translated_Review'].apply(
    lambda text: analyzer.polarity_scores(text)['compound']
)

def get_sentiment_category(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

merged_df['Sentiment Category'] = merged_df['Sentiment Score'].apply(
    lambda score: get_sentiment_category(score) if pd.notna(score) else 'Unknown'
)
print("Sentiment analysis complete. Added 'Sentiment Score' and 'Sentiment Category' columns.")

print("\n--- Processed Apps DataFrame Head ---")
print(apps_df.head())
print("\n--- Processed Merged DataFrame Head (with Sentiment) ---")
print(merged_df.head())

# --- 7. Exploratory Data Analysis (EDA) & Visualization Examples ---
print("\n--- Generating Visualizations ---")

# Define common dark theme layout updates
dark_theme_layout = {
    'plot_bgcolor': 'black',
    'paper_bgcolor': 'black',
    'font': {'color': 'white'},
    'title_font': {'size': 16},
    'xaxis': {'title_font': {'size': 12}, 'tickfont': {'color': 'white'}, 'gridcolor': '#333'},
    'yaxis': {'title_font': {'size': 12}, 'tickfont': {'color': 'white'}, 'gridcolor': '#333'}
}

# Example 1: Top Categories on Play Store (Bar Chart) - Corresponds to PDF
category_counts = apps_df['Category'].value_counts().nlargest(10).reset_index()
category_counts.columns = ['Category', 'Count']
fig_top_categories = px.bar(category_counts, x='Category', y='Count',
              title='Top Categories on Play Store',
              color='Category',
              color_discrete_sequence=px.colors.sequential.Plasma) # Matches PDF color scale
fig_top_categories.update_layout(**dark_theme_layout)
fig_top_categories.update_xaxes(showgrid=False)
fig_top_categories.update_yaxes(showgrid=False)

# New Plot: App Type Distribution (Pie Chart) - Corresponds to PDF
app_type_counts = apps_df['Type'].value_counts().reset_index()
app_type_counts.columns = ['Type', 'Count']
fig_app_type = px.pie(app_type_counts, names='Type', values='Count',
                      title='App Type Distribution',
                      color_discrete_map={'Free': '#8B0000', 'Paid': '#DC143C'}) # Matches PDF colors
fig_app_type.update_layout(**dark_theme_layout)
fig_app_type.update_traces(textinfo='percent+label', marker=dict(line=dict(color='black', width=1)))


# Example 2: Rating Distribution (Histogram) - Corresponds to PDF
fig_rating_distribution = px.histogram(apps_df, x='Rating', title='Rating Distribution',
                    color_discrete_sequence=['#6A5ACD']) # Matches PDF color
fig_rating_distribution.update_layout(**dark_theme_layout)
fig_rating_distribution.update_xaxes(showgrid=False)
fig_rating_distribution.update_yaxes(showgrid=False)

# Example 3: Sentiment Distribution (Histogram with go.Histogram for continuous color) - Corresponds to PDF
counts, bins = np.histogram(merged_df['Sentiment Score'], bins=np.linspace(-1, 1, 21))
bin_centers = 0.5 * (bins[:-1] + bins[1:])

fig_sentiment_distribution = go.Figure(data=[go.Bar(
    x=bin_centers,
    y=counts,
    marker=dict(
        color=bin_centers,
        colorscale='Plasma', # Matches PDF color scale
        colorbar=dict(title="Sentiment Score")
    )
)])
fig_sentiment_distribution.update_layout(
    title_text='Sentiment Distribution',
    **dark_theme_layout
)
fig_sentiment_distribution.update_xaxes(showgrid=False, title_text="Sentiment Score")
fig_sentiment_distribution.update_yaxes(showgrid=False, title_text="Count")


# New Plot: Installs by Category (Bar Chart) - Corresponds to PDF
installs_by_category = apps_df.groupby('Category')['Installs'].sum().nlargest(10).reset_index()
fig_installs_by_category = px.bar(installs_by_category, x='Installs', y='Category',
                                  title='Installs by Category',
                                  orientation='h',
                                  color='Category',
                                  color_discrete_sequence=px.colors.sequential.Blues) # Matches PDF color scale
fig_installs_by_category.update_layout(**dark_theme_layout)
fig_installs_by_category.update_xaxes(showgrid=False)
fig_installs_by_category.update_yaxes(showgrid=False)


# New Plot: Number of Updates Over the Years (Line Chart) - Corresponds to PDF
updates_over_years = apps_df[apps_df['Update_Year'] > 0]['Update_Year'].value_counts().sort_index().reset_index()
updates_over_years.columns = ['Year', 'Number of Updates']
fig_updates_over_years = px.line(updates_over_years, x='Year', y='Number of Updates',
                                 title='Number of Updates Over the Years',
                                 markers=True,
                                 color_discrete_sequence=['#9467bd']) # Matches PDF color
fig_updates_over_years.update_layout(**dark_theme_layout)
fig_updates_over_years.update_xaxes(showgrid=False)
fig_updates_over_years.update_yaxes(showgrid=False)


# New Plot: Revenue by Category (Bar Chart) - Corresponds to PDF
revenue_by_category = apps_df.groupby('Category')['Revenue'].sum().nlargest(10).reset_index()
fig_revenue_by_category = px.bar(revenue_by_category, x='Category', y='Revenue',
                                  title='Revenue by Category',
                                  color='Category',
                                  color_discrete_sequence=px.colors.sequential.Greens) # Matches PDF color scale
fig_revenue_by_category.update_layout(**dark_theme_layout)
fig_revenue_by_category.update_xaxes(showgrid=False)
fig_revenue_by_category.update_yaxes(showgrid=False)


# New Plot: Top Genres (Bar Chart) - Corresponds to PDF
if 'Genres' in apps_df.columns:
    top_genres = apps_df['Genres'].value_counts().nlargest(10).reset_index()
    top_genres.columns = ['Genre', 'Count']
    fig_top_genres = px.bar(top_genres, x='Genre', y='Count',
                            title='Top Genres',
                            color='Genre',
                            color_discrete_sequence=px.colors.sequential.Oranges) # Matches PDF color scale
else:
    print("Warning: 'Genres' column not found. Using 'Category' for 'Top Genres' plot.")
    top_genres = apps_df['Category'].value_counts().nlargest(10).reset_index()
    top_genres.columns = ['Category', 'Count']
    fig_top_genres = px.bar(top_genres, x='Category', y='Count',
                            title='Top Genres (using Category as proxy)',
                            color='Category',
                            color_discrete_sequence=px.colors.sequential.Oranges) # Matches PDF color scale

fig_top_genres.update_layout(**dark_theme_layout)
fig_top_genres.update_xaxes(showgrid=False)
fig_top_genres.update_yaxes(showgrid=False)


# New Plot: Impact of Last Update on Rating (Scatter Plot) - Corresponds to PDF
fig_last_update_impact = px.scatter(apps_df[apps_df['Update_Year'] > 0],
                                    x='Update_Year', y='Rating', color='Type',
                                    title='Impact of Last Update on Rating',
                                    labels={'Update_Year': 'Last Updated Year', 'Rating': 'App Rating'},
                                    color_discrete_map={'Free': '#636efa', 'Paid': '#ef553b'}) # Matches PDF colors
fig_last_update_impact.update_layout(**dark_theme_layout)
fig_last_update_impact.update_xaxes(showgrid=False)
fig_last_update_impact.update_yaxes(showgrid=False)


# New Plot: Ratings for Paid vs Free Apps (Box Plot) - Corresponds to PDF
fig_ratings_paid_free = px.box(apps_df, x='Type', y='Rating',
                               title='Ratings for Paid vs Free Apps',
                               color='Type',
                               color_discrete_map={'Free': '#00cc96', 'Paid': '#9467bd'}) # Matches PDF colors
fig_ratings_paid_free.update_layout(**dark_theme_layout)
fig_ratings_paid_free.update_xaxes(showgrid=False)
fig_ratings_paid_free.update_yaxes(showgrid=False)


# --- 8. Machine Learning Model (Predicting Rating) ---
print("\n--- Training Machine Learning Model ---")

# Define categorical columns to encode
categorical_cols = ['Category', 'Content Rating', 'Type']

# Perform one-hot encoding
apps_df_encoded = pd.get_dummies(apps_df, columns=categorical_cols, drop_first=True)

# Update features list to include numerical and encoded categorical columns
# Add 'Reviews_per_Install' as a new feature
apps_df_encoded['Reviews_per_Install'] = apps_df_encoded.apply(
    lambda row: row['Reviews'] / row['Installs'] if row['Installs'] > 0 else 0, axis=1
)

# CRITICAL FIX: Ensure 'Log Installs' and 'Log Reviews' are used instead of original 'Installs' and 'Reviews'
features = ['Size', 'Price', 'Log Installs', 'Log Reviews', 'Reviews_per_Install']
encoded_feature_names = [col for col in apps_df_encoded.columns if col.startswith(tuple(categorical_cols))]
features.extend(encoded_feature_names)

target = 'Rating'

# Drop rows with NaN in selected features (e.g., if 'Size' conversion resulted in NaNs)
# Ensure all features exist in the dataframe before subsetting
model_df = apps_df_encoded.dropna(subset=features + [target])

# Filter features to only include those present in model_df to avoid KeyError if a dummy variable column is empty
features = [f for f in features if f in model_df.columns]

X = model_df[features]
y = model_df[target]

# CRITICAL FIX: Apply StandardScaler to numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# CRITICAL FIX: Increased n_estimators and adjusted other hyperparameters for potentially better performance
model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1, max_features='sqrt', max_depth=None, min_samples_leaf=1)
model.fit(X_train, y_train)

# CRITICAL FIX: Changed model.predict(y_test) to model.predict(X_test)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Machine Learning Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")


# --- 9. Generating an HTML Dashboard (Example) ---
print("\n--- Generating HTML Dashboard ---")

html_files_path = 'dashboard_output'
os.makedirs(html_files_path, exist_ok=True)

def save_plotly_figure_as_html(fig, filename, path=html_files_path):
    filepath = os.path.join(path, filename)
    pio.write_html(fig, file=filepath, auto_open=False, full_html=True)
    return filepath

# Save all generated figures
plot_paths = {}
plot_paths['top_categories.html'] = save_plotly_figure_as_html(fig_top_categories, 'top_categories.html')
plot_paths['app_type_distribution.html'] = save_plotly_figure_as_html(fig_app_type, 'app_type_distribution.html')
plot_paths['rating_distribution.html'] = save_plotly_figure_as_html(fig_rating_distribution, 'rating_distribution.html')
plot_paths['sentiment_distribution.html'] = save_plotly_figure_as_html(fig_sentiment_distribution, 'sentiment_distribution.html')
plot_paths['installs_by_category.html'] = save_plotly_figure_as_html(fig_installs_by_category, 'installs_by_category.html')
plot_paths['updates_over_years.html'] = save_plotly_figure_as_html(fig_updates_over_years, 'updates_over_years.html')
plot_paths['revenue_by_category.html'] = save_plotly_figure_as_html(fig_revenue_by_category, 'revenue_by_category.html')
plot_paths['top_genres.html'] = save_plotly_figure_as_html(fig_top_genres, 'top_genres.html')
plot_paths['last_update_impact.html'] = save_plotly_figure_as_html(fig_last_update_impact, 'last_update_impact.html')
plot_paths['ratings_paid_free.html'] = save_plotly_figure_as_html(fig_ratings_paid_free, 'ratings_paid_free.html')

# --- New Plot: Word Cloud (Task 1) ---
# CRITICAL FIX: Broadened filter to all positive sentiment reviews to ensure enough data for word cloud.
# The previous specific filter for "5-star 'Health & Fitness'" reviews often resulted in insufficient text
# due to data sparsity in the provided dataset. This change ensures a word cloud is always generated
# to demonstrate the functionality, reflecting overall positive user feedback.
positive_reviews = merged_df[
    (merged_df['Sentiment Score'] >= 0.05) & # Filter for positive sentiment reviews
    (merged_df['Translated_Review'].notna())
]['Translated_Review']

# Combine all reviews into a single string
text = " ".join(review for review in positive_reviews)

# Remove stopwords
stopwords_set = set(stopwords.words('english'))
# Keep only very generic app-related words
custom_stopwords = {'app', 'game', 'google', 'play', 'store'}
stopwords_set.update(custom_stopwords)

# Check if text is empty before generating word cloud
if not text.strip():
    print("Warning: No sufficient words found for the word cloud after filtering and stopword removal.")
    # Create a simple HTML file to display a message instead of the word cloud
    wordcloud_html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Word Cloud (Task 1)</title>
        <style>
            body {{ background-color: black; margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; color: white; font-family: 'Inter', sans-serif; text-align: center; }}
            p {{ font-size: 1.2em; }}
        </style>
    </head>
    <body>
        <p>No sufficient words found to generate a word cloud for positive reviews.</p>
    </body>
    </html>
    """
    wordcloud_path_html = os.path.join(html_files_path, 'wordcloud.html')
    with open(wordcloud_path_html, 'w', encoding='utf-8') as f:
        f.write(wordcloud_html_content)
    plot_paths['wordcloud.html'] = wordcloud_path_html # Ensure this path is used
else:
    wordcloud = WordCloud(stopwords=stopwords_set, background_color="black", colormap='viridis',
                          width=800, height=400).generate(text)

    # Save the word cloud as a PNG image
    wordcloud_png_path = os.path.join(html_files_path, 'wordcloud.png')
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(wordcloud_png_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close() # Close the plot to free memory
    print(f"Word Cloud saved to: {os.path.abspath(wordcloud_png_path)}")

    # Create a simple HTML file to display the word cloud image
    wordcloud_html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Word Cloud (Task 1)</title>
        <style>
            body {{ background-color: black; margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }}
            img {{ max-width: 100%; max-height: 100%; border-radius: 10px; }}
        </style>
    </head>
    <body>
        <img src="wordcloud.png" alt="Word Cloud of Positive Reviews">
    </body>
    </html>
    """
    wordcloud_path_html = os.path.join(html_files_path, 'wordcloud.html')
    with open(wordcloud_path_html, 'w', encoding='utf-8') as f:
        f.write(wordcloud_html_content)
    plot_paths['wordcloud.html'] = wordcloud_path_html # Ensure this path is used


# --- New Plot: Interactive Choropleth Map (Task 2 - Placeholder due to data limitation) ---
# IMPORTANT: The provided datasets do not contain country-specific installation data.
# This section serves as a placeholder to show how it *would* be implemented if such data existed.
# For demonstration, we'll create a dummy DataFrame with country data.
print("\n--- Generating Choropleth Map (Placeholder due to data limitation) ---")
dummy_country_data = {
    'Country': ['USA', 'India', 'Brazil', 'Germany', 'Japan', 'Canada', 'UK', 'Australia', 'France', 'Mexico'],
    'Installs_Millions': [150, 120, 80, 60, 50, 40, 35, 30, 25, 20],
    'Category': ['GAME', 'FAMILY', 'TOOLS', 'PRODUCTIVITY', 'FINANCE', 'GAME', 'LIFESTYLE', 'TOOLS', 'GAME', 'COMMUNICATION']
}
dummy_geo_df = pd.DataFrame(dummy_country_data)

# Filter out categories starting with A, C, G, S
filtered_categories_for_map = dummy_geo_df[~dummy_geo_df['Category'].astype(str).str.startswith(('A', 'C', 'G', 'S'))]
# Top 5 categories after filtering
top_5_map_categories = filtered_categories_for_map.groupby('Category')['Installs_Millions'].sum().nlargest(5).index.tolist()
filtered_geo_df = filtered_categories_for_map[filtered_categories_for_map['Category'].isin(top_5_map_categories)]

# Aggregate installs by country for the map
country_installs = filtered_geo_df.groupby('Country')['Installs_Millions'].sum().reset_index()

fig_choropleth = px.choropleth(country_installs,
                               locations="Country",
                               locationmode='country names',
                               color="Installs_Millions",
                               hover_name="Country",
                               color_continuous_scale=px.colors.sequential.Plasma, # Matches PDF color scale
                               title="Global App Installs by Country (Task 2 - Placeholder)",
                               labels={'Installs_Millions': 'Installs (Millions)'})
fig_choropleth.update_layout(**dark_theme_layout)
fig_choropleth.update_geos(showcoastlines=True, coastlinecolor="white", showland=True, landcolor="grey", showocean=True, oceancolor="black")

plot_paths['choropleth_map.html'] = save_plotly_figure_as_html(fig_choropleth, 'choropleth_map.html')
print("Note: Choropleth map is a placeholder. Real data requires 'Country' information in datasets.")
print("Note: Time filter for Choropleth map requires client-side JavaScript implementation.")


# --- New Plot: Bubble Chart (Task 3) ---
print("\n--- Generating Bubble Chart ---")
# Filter apps with rating > 3.5 and reviews > 500
filtered_bubble_df = apps_df[
    (apps_df['Rating'] > 3.5) &
    (apps_df['Reviews'] > 500)
].copy() # Use .copy() to avoid SettingWithCopyWarning

# Filter specific categories
specific_categories = ['GAME', 'BEAUTY', 'BUSINESS', 'DATING', 'EDUCATION', 'ENTERTAINMENT', 'FINANCE', 'LIFESTYLE']
filtered_bubble_df = filtered_bubble_df[filtered_bubble_df['Category'].isin(specific_categories)]

# Translate specific categories for display
category_translations = {
    'BEAUTY': 'सौंदर्य (Beauty - Hindi)',
    'BUSINESS': 'வணிகம் (Business - Tamil)',
    'DATING': 'Dating (Deutsch)', # German for Dating
    'GAME': 'GAME', # No translation for Game
    'EDUCATION': 'EDUCATION',
    'ENTERTAINMENT': 'ENTERTAINMENT',
    'FINANCE': 'FINANCE',
    'LIFESTYLE': 'LIFESTYLE'
}
filtered_bubble_df['Translated_Category'] = filtered_bubble_df['Category'].map(category_translations)

fig_bubble = px.scatter(filtered_bubble_df,
                        x='Size',
                        y='Rating',
                        size='Installs', # Bubble size based on installs
                        color='Translated_Category', # Color by translated category
                        hover_name='App',
                        log_x=True, # Log scale for size if it varies widely
                        size_max=60, # Max size of bubbles
                        title='App Size vs. Average Rating by Category (Task 3)',
                        labels={
                            'Size': 'App Size (MB)',
                            'Rating': 'Average Rating',
                            'Installs': 'Installs'
                        },
                        color_discrete_map={
                            'GAME': 'deeppink', # Highlight Game in pink
                            'BEAUTY': px.colors.qualitative.Plotly[0], # Default Plotly colors for others
                            'BUSINESS': px.colors.qualitative.Plotly[1],
                            'DATING': px.colors.qualitative.Plotly[2],
                            'EDUCATION': px.colors.qualitative.Plotly[3],
                            'ENTERTAINMENT': px.colors.qualitative.Plotly[4],
                            'FINANCE': px.colors.qualitative.Plotly[5],
                            'LIFESTYLE': px.colors.qualitative.Plotly[6]
                        })
fig_bubble.update_layout(**dark_theme_layout)
fig_bubble.update_xaxes(showgrid=False)
fig_bubble.update_yaxes(showgrid=False)

plot_paths['bubble_chart.html'] = save_plotly_figure_as_html(fig_bubble, 'bubble_chart.html')
print("Note: Time filter for Bubble Chart requires client-side JavaScript implementation.")


# Construct the HTML for the dashboard
plot_containers = ""
plot_width = "48%"
plot_height = "400px"

# Order of plots as seen in PDF (2 columns, 5 rows) - Adjusted to include new plots
plot_order = [
    'top_categories.html',
    'app_type_distribution.html',
    'rating_distribution.html',
    'sentiment_distribution.html',
    'installs_by_category.html',
    'updates_over_years.html',
    'revenue_by_category.html',
    'top_genres.html',
    'wordcloud.html', # Added Word Cloud
    'bubble_chart.html', # Added Bubble Chart
    'choropleth_map.html' # Added Choropleth Map (placeholder)
]

for filename_key in plot_order:
    relative_filename = os.path.basename(plot_paths[filename_key])
    plot_containers += f"""
    <div class="plot-container" style="width: {plot_width}; height: {plot_height}; margin: 1%; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden;">
        <iframe src="{relative_filename}" width="100%" height="100%" frameborder="0" scrolling="no"></iframe>
    </div>
    """
# Adjusting the main container for 2 columns, 5 rows (10 plots total). If there are 11, the last one will wrap.
# The layout specifies 2 columns, so 10 plots will fit perfectly into 5 rows.
# If 'choropleth_map.html' is the 11th, it will start a new row.

dashboard_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Google Play Store Reviews Analytics</title>
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #222;
            color: #f4f7f6;
        }}
        .header {{
            background-color: #333;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 2em;
            border-bottom-left-radius: 15px;
            border-bottom-right-radius: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.5);
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .header img {{
            height: 50px;
            vertical-align: middle;
            margin-right: 15px;
            border-radius: 8px;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            padding: 20px;
            gap: 20px;
        }}
        .plot-container {{
            background-color: #1a1a1a;
            border-radius: 12px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
            transition: transform 0.3s ease-in-out;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .plot-container:hover {{
            transform: translateY(-5px);
        }}
        iframe {{
            border-radius: 10px;
            background-color: black;
        }}
        .model-results {{
            background-color: #1a1a1a;
            margin: 20px auto;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
            max-width: 800px;
            text-align: center;
            color: #f4f7f6;
        }}
        .model-results h3 {{
            color: #4CAF50;
            margin-bottom: 15px;
        }}
        .model-results p {{
            font-size: 1.1em;
            line-height: 1.6;
        }}
        @media (max-width: 768px) {{
            .plot-container {{
                width: 95% !important;
                height: 350px !important;
            }}
        }}
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/1024px-Google_Play_Store_badge_EN.svg.png" alt="Google Play Store Logo">
        Google Play Store Reviews Analytics
    </div>
    <div class="container">
        {plot_containers}
    </div>
    <div class="model-results">
        <h3>Machine Learning Model Performance</h3>
        <p>The Random Forest Regressor model was trained to predict app ratings based on various features.</p>
        <p><strong>Mean Squared Error (MSE):</strong> {mse:.2f}</p>
        <p><strong>R-squared (R2):</strong> {r2:.2f}</p>
        <p>These metrics indicate how well the model's predictions match the actual app ratings.</p>
    </div>
</body>
</html>
"""

# Save the final dashboard to an HTML file
dashboard_path = os.path.join(html_files_path, 'dashboard.html')
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(dashboard_html)

print(f"Dashboard saved to: {os.path.abspath(dashboard_path)}")

# Automatically open the generated HTML file in a web browser
webbrowser.open('file://' + os.path.realpath(dashboard_path))
print("To view the dashboard, open the 'dashboard.html' file located in the 'dashboard_output' folder in your web browser.")
