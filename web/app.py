import os
import sys
import webbrowser

# Add the parent directory to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request, redirect, url_for, session
from functools import wraps

from src.predict import predict
from src.data_access import (
    get_places_summary,
    get_place_detail,
    get_place_reviews,
    search_places,
    places_by_novelty
)
from src.auth import init_db, create_user, get_user_by_email, get_user_by_id, verify_password

app = Flask(__name__)
app.secret_key = "dev"
init_db()

# ✅ ONLY 4 TYPES
NOVELTY_TYPES = ["experience", "arousal", "relaxation", "boredom alleviation"]

# ================= AUTH DECORATOR =================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# ================= USER =================
@app.context_processor
def inject_user():
    uid = session.get("user_id")
    return {"current_user": get_user_by_id(uid) if uid else None}

# ================= HOME =================
@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    result = None

    if request.method == "POST":
        text = request.form.get("prompt") or request.form.get("review_text")

        if text:
            raw_prediction = predict(text)
            print("Raw Prediction:", raw_prediction)

            novelty_type = raw_prediction.strip().lower()
            text_lower = text.lower()

            if "forest" in text_lower:
                novelty_type = "arousal"
            elif "peace" in text_lower:
                novelty_type = "relaxation"
            elif "bored" in text_lower:
                novelty_type = "boredom alleviation"
            elif "explor" in text_lower:
                novelty_type = "experience"
            else:
                if "relax" in novelty_type:
                    novelty_type = "relaxation"
                elif "arousal" in novelty_type or "thrill" in novelty_type:
                    novelty_type = "arousal"
                elif "experience" in novelty_type:
                    novelty_type = "experience"
                else:
                    novelty_type = "boredom alleviation"

            print("Final Mapped Type:", novelty_type)

            related_places = places_by_novelty(novelty_type, limit=10)
            print("Places Found:", len(related_places))

            result = {
                "novelty_type": novelty_type,
                "places": related_places
            }

    top_places = get_places_summary(limit=10)
    return render_template("index.html", result=result, top_places=top_places)

# ================= GENRES =================
@app.route("/genres")
@login_required
def genres():
    return render_template("genre.html", types=NOVELTY_TYPES)

# ================= NOVELTY =================
@app.route("/novelty/<novelty>")
@login_required
def novelty_page(novelty):
    novelty = novelty.strip().lower()

    if novelty not in NOVELTY_TYPES:
        return redirect(url_for("genres"))

    places = places_by_novelty(novelty, limit=20)
    return render_template("novelty.html", novelty=novelty, places=places)

# ================= PLACE =================
@app.route("/place/<place_id>/<city>")
@login_required
def place_detail(place_id, city):
    info = get_place_detail(place_id, city)
    reviews = get_place_reviews(place_id, city)
    return render_template("place_detail.html", info=info, reviews=reviews)

# ================= SEARCH =================
@app.route("/search")
@login_required
def search():
    q = request.args.get("q", "")
    items = search_places(q) if q else []
    return render_template("search.html", items=items, q=q)

# ================= AUTH =================
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        user = get_user_by_email(request.form["email"])
        if not user or not verify_password(user, request.form["password"]):
            error = "Invalid credentials"
        else:
            session["user_id"] = user["id"]
            return redirect(url_for("index"))
    return render_template("login.html", error=error)

@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        if request.form["password"] != request.form["confirm"]:
            error = "Passwords mismatch"
        else:
            ok = create_user(
                request.form["email"],
                request.form["name"],
                request.form["password"]
            )
            if ok:
                return redirect(url_for("login"))
            else:
                error = "Email exists"
    return render_template("register.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# ================= TRENDS =================
@app.route("/trends")
@login_required
def trends():
    from src.data_access import load_dataset
    df = load_dataset()

    if df is None:
        return render_template("trends.html", items=[], stats={})

    stats = {
        "novelty_dist": df["novelty_type"].value_counts().to_dict(),
        "avg_ratings": df.groupby("novelty_type")["rating"].mean().to_dict(),
        "top_countries": df["country"].value_counts().head(5).to_dict()
    }

    items = df.groupby(["place_id", "place_name", "city", "country"]).agg(
        recent_reviews=("review_text", "count"),
        avg_rating=("rating", "mean")
    ).reset_index().sort_values(by="recent_reviews", ascending=False).head(10).to_dict(orient="records")

    return render_template("trends.html", items=items, stats=stats)

# ================= RUN APP =================
if __name__ == "__main__":
    print("Starting Flask app on http://127.0.0.1:5010 ...")

    # ✅ Prevent double opening (important fix)
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open("http://127.0.0.1:5010")

    app.run(debug=True, host='0.0.0.0', port=5010, use_reloader=True)