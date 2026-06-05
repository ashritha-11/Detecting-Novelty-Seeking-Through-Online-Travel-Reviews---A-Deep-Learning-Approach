import os
import sys
sys.path.append(os.getcwd())

print("Testing imports...")
try:
    from src.predict import predict
    print("✅ src.predict imported")
    from src.data_access import (
        get_places_summary,
        get_place_detail,
        get_place_reviews,
        search_places,
        places_by_novelty
    )
    print("✅ src.data_access imported")
    from src.auth import init_db, create_user, get_user_by_email, get_user_by_id, verify_password
    print("✅ src.auth imported")
    from flask import Flask
    print("✅ flask imported")
except Exception as e:
    print(f"❌ Error during imports: {e}")
    import traceback
    traceback.print_exc()
