import os
import sys

# Add the project root to sys.path so modules like src and web can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.app import app

# Vercel needs the 'app' variable to be the Flask instance
if __name__ == "__main__":
    app.run()
