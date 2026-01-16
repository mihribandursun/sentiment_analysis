import pandas as pd
from sqlalchemy import create_engine, text

csv_path = r"C:\Users\nurha\OneDrive\Desktop\sentiment_analysis_project\data\processed\restaurant_reviews_processed.csv"

engine = create_engine("postgresql+psycopg2://postgres:Hazal123@localhost:5432/sentiment_db")

df = pd.read_csv(csv_path)

# RESTAURANTS Table 
restaurants_cols = [
    "vendor_id", "restaurant_name", "link", "img", "district",
    "cuisine_type", "price_range", "min_order", "delivery_time", "delivery_type", "restaurant_slug"
]

restaurants_df = df[restaurants_cols].drop_duplicates(subset=["vendor_id"])  # Her restoran bir kez
restaurants_df.reset_index(drop=True, inplace=True)

# REVIEWS Table
reviews_cols = [
    "vendor_id", "uuid", "reviewer_name", "reviewer_id",
    "overall_score", "restaurant_food_score", "rider_score",
    "text", "created_at", "like_count", "replies_count", "product_names",
    "delivery_time_min", "sentiment_label", "sentiment", "has_rider_score", "review_length"
]

reviews_df = df[reviews_cols].copy()
reviews_df.reset_index(drop=True, inplace=True)

# CLEAR AND ADD TABLES
with engine.begin() as conn:
    # First, clean up the comments (the reviews table is linked to a foreign key).
    conn.execute(text("TRUNCATE TABLE reviews CASCADE"))
    # Then clean the restaurants
    conn.execute(text("TRUNCATE TABLE restaurants CASCADE"))

# Add data to table
restaurants_df.to_sql("restaurants", engine, if_exists="append", index=False)
reviews_df.to_sql("reviews", engine, if_exists="append", index=False)

print("The CSV file has been successfully uploaded to the database!")
