import os
import json
from flask import Flask, render_template, jsonify, redirect, url_for
from flask_cors import CORS
from fono.config import APP_SECRET
from fono.route import bp

def create_app():
    app = Flask(__name__, instance_relative_config=True, static_url_path="")
    CORS(app)
    app.config.from_mapping(
        SECRET_KEY=APP_SECRET
    )

    app.register_blueprint(bp)
    return app