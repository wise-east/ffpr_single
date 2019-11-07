# app/wsgi.py

from app import app, socketio
socketio.run(app)
