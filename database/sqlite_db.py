import sqlite3 as sl
import os

DATABASE_DIR = os.path.dirname(__file__)

conn = sl.connect(DATABASE_DIR+'/music_classification.db')

