# google-play-store-data-analytics
Internship project for NULLCLASS using python and data visulization tools.
### 🔍 Internship Task – NULLCLASS (Python Track)

This project analyzes Google Play Store data using Python. It includes data cleaning, text processing, visualization, and interactive dashboards.


## ✅ Tasks Implemented

### 1️⃣ Word Cloud (5-Star Reviews)
- Extracted frequent keywords from 5-star reviews
- Removed stopwords and app names
- Filtered reviews to only include Health & Fitness category
- Output: wordcloud.html, wordcloud.png

### 2️⃣ Choropleth Map (Global Installs by Category)
- Interactive choropleth map built using Plotly
- Filtered to:
  - Top 5 categories
  - Installs > 1 million
  - Categories not starting with A, C, G, or S
  - Visible only between 6 PM to 8 PM IST
- Output: choropleth_map.html

### 3️⃣ Bubble Chart (Size vs Rating)
- X-axis: App size in MB  
- Y-axis: Rating  
- Bubble size: Number of installs  
- Filters applied:
  - Rating > 3.5
  - Categories: Game, Beauty (in Hindi), Business (in Tamil), Dating (in German), Communication, Comics, Social, Entertainment, Events
  - Reviews > 500
  - App name does not contain letter "S"
  - Sentiment subjectivity > 0.5
  - Installs > 50k
  - Displayed only between 5 PM to 7 PM IST
- Output: bubble_chart.html

## 📁 Project Files Included

- dashboard_output/ – All output .html visualizations
- Analysis.ipynb, Analysis2.ipynb, Analysis3.ipynb – Code notebooks
- app.py – Application backend
- Play Store Data.csv – Dataset used
- wordcloud.png – Word cloud image output
- Google Play Store Data Analytics Project Report.pdf – Final report
- README.md – Project description

## 📹 Output Video

[🔗 Click here to view project demo]([https://your-video-link-here.com](https://drive.google.com/file/d/1ZTHo6slf8g-6CcWosUC4eN3_eXxyybG5/view?usp=drivesdk))

## 🛠️ Tools & Technologies Used

- Python
- Pandas
- Plotly
- WordCloud
- NLTK
- Streamlit (optional)

## 📅 Submission Details

- Internship Program: NULLCLASS
- Track: Real-Time Google Play Store Data Analytics – Python
- Submitted by: Yash Jadhav
- Submission Type: Final One-Time Submission
---
