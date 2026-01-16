from flask import Flask, render_template, request, redirect, url_for, session, flash
from sqlalchemy import create_engine, text
from werkzeug.security import generate_password_hash, check_password_hash
import re
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os
from dotenv import load_dotenv
load_dotenv() # .env dosyasını okur

app = Flask(__name__)
app.secret_key = "supersecretkey"

engine = create_engine(os.getenv("DATABASE_URL"))

# Email regex
EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

PER_PAGE = 12

# ====== BERT MODEL INSTALLATION (LOCAL)) ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_bert_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.to(device)
model.eval()

import torch.nn.functional as F

def predict_sentiment(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        probs = F.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

    sentiment_id = predicted_class.item()
    confidence = round(confidence.item() * 100, 2)

    sentiment_label = "POSITIVE" if sentiment_id == 1 else "NEGATIVE"

    return sentiment_id, sentiment_label, confidence


# Root redirects to the login page.
@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

# Register
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username").strip()
        email = request.form.get("email").strip()
        password = request.form.get("password").strip()

        # Email and password verification
        if not re.match(EMAIL_REGEX, email):
            flash("Geçersiz e-posta formatı.", "error")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash("Şifre en az 6 karakter olmalı.", "error")
            return redirect(url_for("register"))

        password_hash = generate_password_hash(password)

        try:
            with engine.connect() as conn:
                query = text("""
                    INSERT INTO users (username, email, password_hash)
                    VALUES (:username, :email, :password_hash)
                """)
                conn.execute(query, {"username": username, "email": email, "password_hash": password_hash})
                conn.commit()
            flash("Kayıt başarılı. Lütfen giriş yapın.", "success")
            return redirect(url_for("login"))
        except Exception as e:
            flash("Email veya kullanıcı adı zaten var.", "error")
            return redirect(url_for("register"))

    return render_template("register.html")


# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        query = text("SELECT * FROM users WHERE email = :email")
        with engine.connect() as conn:
            user = conn.execute(query, {"email": email}).mappings().first()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        else:
            flash("Email veya şifre yanlış", "danger")

    return render_template("login.html")

# Logout
@app.route("/logout")
def logout():
    session.clear()
    flash("Çıkış yapıldı.", "info")
    return redirect(url_for("login"))

# Dashboard → home page
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    page = int(request.args.get("page", 1))
    filter_type = request.args.get("filter", "popular")
    district = request.args.get("district")
    cuisine = request.args.get("cuisine")
    search = request.args.get("q", "").strip()
    offset = (page - 1) * PER_PAGE

    order_by_map = {
        "popular": "total_reviews DESC",
        "positive": "positive_pct DESC",
        "negative": "negative_pct DESC",
        "rating": "avg_rating DESC"
    }
    order_by = order_by_map.get(filter_type, "total_reviews DESC")

    where_clauses = []
    params = {"limit": PER_PAGE, "offset": offset}

    if district:
        where_clauses.append("r.district = :district")
        params["district"] = district

    if cuisine:
        where_clauses.append("r.cuisine_type = :cuisine")
        params["cuisine"] = cuisine

    if search:
        where_clauses.append("r.restaurant_name ILIKE :search")
        params["search"] = f"%{search}%"

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    query = text(f"""
        SELECT
            r.vendor_id,
            r.restaurant_name,
            r.district,
            r.img,
            r.cuisine_type,
            COUNT(v.id) AS total_reviews,
            ROUND(100.0 * SUM(CASE WHEN v.sentiment = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(v.id),0), 2) AS positive_pct,
            ROUND(100.0 * SUM(CASE WHEN v.sentiment = 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(v.id),0), 2) AS negative_pct,
            COALESCE(AVG(v.overall_score), 0) AS avg_rating,
            CASE 
              WHEN r.img IS NOT NULL AND r.img NOT LIKE '%placeholder%' THEN 1
              ELSE 0
            END AS has_real_image
        FROM restaurants r
        LEFT JOIN reviews v ON r.vendor_id = v.vendor_id
        {where_sql}
        GROUP BY r.vendor_id, r.restaurant_name, r.district, r.img, r.cuisine_type
        ORDER BY has_real_image DESC, {order_by}
        LIMIT :limit OFFSET :offset
    """)

    with engine.connect() as conn:
        restaurants = conn.execute(query, params).mappings().all()

        # Districts
        districts = conn.execute(
            text("SELECT DISTINCT district FROM restaurants ORDER BY district")
        ).scalars().all()
        # Cuisines
        cuisines = conn.execute(
            text("SELECT DISTINCT cuisine_type FROM restaurants ORDER BY cuisine_type")
        ).scalars().all()

        # Sum for pagination
        count_query = text(f"SELECT COUNT(*) FROM restaurants r {where_sql}")
        total_restaurants = conn.execute(count_query, params).scalar()
        total_pages = (total_restaurants + PER_PAGE - 1) // PER_PAGE

    return render_template(
        "index.html",
        restaurants=restaurants,
        page=page,
        total_pages=total_pages,
        filter_type=filter_type,
        districts=districts,
        cuisines=cuisines,
        selected_district=district,
        selected_cuisine=cuisine,
        search=search
    )

@app.route("/my_reviews", methods=["GET", "POST"])
def my_reviews():
    if "user_id" not in session:
        return redirect(url_for("login"))

    # ---- User comment deletion process ----
    if request.method == "POST":
        delete_id = request.form.get("delete_review_id")
        if delete_id:
            with engine.connect() as conn:
                conn.execute(
                    text("DELETE FROM user_reviews WHERE id = :id AND user_id = :user_id"),
                    {"id": delete_id, "user_id": session["user_id"]}
                )
                conn.commit()
            flash("Yorum silindi!", "success")
            return redirect(url_for("my_reviews"))
        
    page = int(request.args.get("page", 1))
    PER_PAGE = 12
    offset = (page - 1) * PER_PAGE

    query = text("""
        SELECT ur.id, ur.text AS review_text, ur.sentiment, ur.sentiment_label, ur.confidence, ur.created_at,
               r.restaurant_name, r.vendor_id
        FROM user_reviews ur
        JOIN restaurants r ON ur.vendor_id = r.vendor_id
        WHERE ur.user_id = :user_id
        ORDER BY ur.created_at DESC
        LIMIT :limit OFFSET :offset
    """)

    count_query = text("""
        SELECT COUNT(*) FROM user_reviews
        WHERE user_id = :user_id
    """)

    with engine.connect() as conn:
        reviews = conn.execute(query, {"user_id": session["user_id"], "limit": PER_PAGE, "offset": offset}).mappings().all()
        total_reviews = conn.execute(count_query, {"user_id": session["user_id"]}).scalar()

    total_pages = (total_reviews + PER_PAGE - 1) // PER_PAGE

    return render_template("my_reviews.html",
                           reviews=reviews,
                           page=page,
                           total_pages=total_pages,
                           total_reviews=total_reviews)

# Restaurant details page + user reviews
@app.route("/restaurant/<vendor_id>", methods=["GET", "POST"])
def restaurant_detail(vendor_id):
    PER_PAGE_REVIEWS = 25
    page = int(request.args.get("page", 1))
    offset = (page - 1) * PER_PAGE_REVIEWS

    with engine.connect() as conn:
        # --- Deletion process ---
        delete_id = request.form.get("delete_review_id")
        if delete_id:
            # You can only delete your own comment.
            check_query = text("SELECT user_id FROM user_reviews WHERE id = :id")
            review = conn.execute(check_query, {"id": delete_id}).mappings().first()
            if review and review["user_id"] == session["user_id"]:
                delete_query = text("DELETE FROM user_reviews WHERE id = :id")
                conn.execute(delete_query, {"id": delete_id})
                conn.commit()
                flash("Yorumunuz silindi.", "success")
                return redirect(url_for("restaurant_detail", vendor_id=vendor_id))

        # --- Comment adding process ---
        user_text = request.form.get("review_text", "").strip()
        if user_text:
            sentiment_id, sentiment_label, confidence = predict_sentiment(user_text)
            
            print("DEBUG:", sentiment_id, sentiment_label, confidence)

            insert_query = text("""
                INSERT INTO user_reviews (user_id, vendor_id, text, sentiment, sentiment_label, confidence)
                VALUES (:user_id, :vendor_id, :text, :sentiment, :sentiment_label, :confidence)
            """)

            conn.execute(insert_query, {
                "user_id": session["user_id"],
                "vendor_id": vendor_id,
                "text": user_text,
                "sentiment": sentiment_id,
                "sentiment_label": sentiment_label,
                "confidence": confidence
            })

            conn.commit()
            flash("Yorumunuz kaydedildi!", "success")
            return redirect(url_for("restaurant_detail", vendor_id=vendor_id))

    # --- Restaurant information and reviews ---
    restaurant_query = text("""
        SELECT
            r.vendor_id,
            r.restaurant_name,
            r.district,
            r.img,
            r.cuisine_type,
            r.price_range,
            r.min_order,
            r.delivery_time,
            r.delivery_type,
            ROUND(100.0 * SUM(CASE WHEN v.sentiment = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(v.id),0), 2) AS positive_pct,
            ROUND(100.0 * SUM(CASE WHEN v.sentiment = 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(v.id),0), 2) AS negative_pct,
            COUNT(v.id) AS total_reviews,
            COALESCE(AVG(v.overall_score), 0) AS avg_rating
        FROM restaurants r
        LEFT JOIN reviews v ON r.vendor_id = v.vendor_id
        WHERE r.vendor_id = :vendor_id
        GROUP BY r.vendor_id, r.restaurant_name, r.district, r.img, r.cuisine_type, r.price_range, r.min_order, r.delivery_time, r.delivery_type
    """)

    reviews_query = text("""
        SELECT
            reviewer_name,
            text AS review_text,
            sentiment,
            sentiment_label,
            created_at,
            overall_score
        FROM reviews
        WHERE vendor_id = :vendor_id
        ORDER BY created_at DESC
        LIMIT :limit OFFSET :offset
    """)

    user_reviews_query = text("""
        SELECT
            ur.id,
            u.username,
            ur.text AS review_text,
            ur.sentiment,
            ur.sentiment_label,
            ur.confidence,
            ur.created_at,
            ur.user_id
        FROM user_reviews ur
        JOIN users u ON ur.user_id = u.id
        WHERE ur.vendor_id = :vendor_id
        ORDER BY ur.created_at DESC
    """)

    count_query = text("""
        SELECT COUNT(*) FROM reviews WHERE vendor_id = :vendor_id
    """)

    with engine.connect() as conn:
        restaurant = conn.execute(restaurant_query, {"vendor_id": vendor_id}).mappings().first()
        reviews = conn.execute(reviews_query, {"vendor_id": vendor_id, "limit": PER_PAGE_REVIEWS, "offset": offset}).mappings().all()
        user_reviews = conn.execute(user_reviews_query, {"vendor_id": vendor_id}).mappings().all()
        total_reviews = conn.execute(count_query, {"vendor_id": vendor_id}).scalar()

    total_pages = (total_reviews + PER_PAGE_REVIEWS - 1) // PER_PAGE_REVIEWS

    return render_template(
        "restaurant_detail.html",
        restaurant=restaurant,
        reviews=reviews,
        user_reviews=user_reviews,
        page=page,
        total_pages=total_pages
    )


if __name__ == "__main__":
    app.run(debug=True)
