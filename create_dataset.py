import json
import os
import random
import re
import shutil
import string
from datetime import datetime

import cv2
import psycopg2
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    'dbname': 'time_traveler',
    'user': 'postgres',
    'password': 'm06Ar14u',
    'host': '192.168.1.201',
    'port': 5432,
}

ROOT_DIR = '/Volumes/TTBS/time_traveler'
SAMPLE_COUNT = 2
ANNOTATIONS_DIR = "dataset/annotations"

def random_json_filename(length=10):
    chars = string.ascii_letters + string.digits
    name = ''.join(random.choices(chars, k=length))
    return f"{name}.json"


def set_annotation_file(annotation: dict):
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    output_path = os.path.join(ANNOTATIONS_DIR, random_json_filename())

    # Write to JSON file
    with open(output_path, "w") as json_file:
        json.dump(annotation, json_file, indent=4)

    print(f"Annotation saved to {output_path}")


def get_video_length(filename):
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        duration = frame_count / fps
        return round(duration, 2)
    else:
        return None  # Could not determine FPS


def get_db_filenames(show_id: int, limit: int = 10):
    query = f"""WITH ranked AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY show_id ORDER BY episode_airdate) AS rn
        FROM episodes
        WHERE (end_point != FLOOR(end_point) OR start_point != FLOOR(start_point))  AND episode_airdate < '1990-01-01')
        SELECT * FROM ranked WHERE rn <= {SAMPLE_COUNT};"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(query, (show_id, limit))
    result = cur.fetchall()
    cur.close()
    conn.close()
    return result


def get_model_annotations(show_id: int):
    db_filenames = get_db_filenames(show_id)
    for x in db_filenames:
        year = int(x['episode_airdate'].strftime("%y"))
        decade = f"{(year // 10) % 10}0s"
        end_point = get_video_length(f"{ROOT_DIR}/{decade}/{year}/{x['episode_file']}")
        outro = [x['end_point'], end_point] if x['end_point'] < (end_point - 2) else None
        intro = [0, x['start_point']] if x['start_point'] > 3 else None

        bumpers = []
        if intro:
            bumpers.append(intro)
        if outro:
            bumpers.append(outro)

        annotation = {
            "file_path": f"{ROOT_DIR}/{decade}/{year}/{x['episode_file']}",
            "bumpers": bumpers if bumpers else None,
            "commercials": None,
            "content": [[300.0, 320.0], [600.0, 610.0]]
            # "commercials": [[300.0, 360.0], [600.0, 660.0]]
        }
        set_annotation_file(annotation)


if __name__ == "__main__":
    get_model_annotations(5)
