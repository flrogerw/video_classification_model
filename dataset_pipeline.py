import cv2
import psycopg2
import json
import os
import random
import string
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

# Load .env file
load_dotenv()

DB_CONFIG = {
    'dbname': os.getenv("DB_NAME"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'host': os.getenv("DB_HOST"),
    'port': os.getenv("DB_PORT"),
}


def random_json_filename(length=10):
    chars = string.ascii_letters + string.digits
    name = ''.join(random.choices(chars, k=length))
    return f"{name}.json"


def set_annotation_file(annotation: dict):
    os.makedirs(os.getenv("ANNOTATIONS_DIR"), exist_ok=True)
    output_path = os.path.join(os.getenv("ANNOTATIONS_DIR"), random_json_filename())

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
        SELECT * FROM ranked WHERE rn <= {int(os.getenv("SAMPLE_COUNT"))};"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(query, (show_id, limit))
    result = cur.fetchall()
    cur.close()
    conn.close()
    return result


def get_model_annotations(show_id: int):
    db_filenames = get_db_filenames(show_id=show_id)
    for x in db_filenames:
        year = int(x['episode_airdate'].strftime("%y"))
        decade = f"{(year // 10) % 10}0s"
        end_point = get_video_length(f'{os.getenv("ROOT_DIR")}/{decade}/{year}/{x["episode_file"]}')
        outro = [x['end_point'], end_point] if x['end_point'] < (end_point - 2) else None
        intro = [0, x['start_point']] if x['start_point'] > 3 else None

        bumpers = []
        if intro:
            bumpers.append(intro)
        if outro:
            bumpers.append(outro)

        content = []

        # Content after intro (30 seconds)
        if intro and intro[1]:
            start_content = intro[1]
            end_content = min(start_content + int(os.getenv("CONTENT_BUFFER")), end_point)
            content.append([start_content, end_content])
        else:
            # No intro, use first 30 seconds of video
            content.append([0, min(int(os.getenv("CONTENT_BUFFER")), end_point)])

        # Content before outro (30 seconds)
        if outro and outro[0]:
            end_content = outro[0]
            start_content = max(end_content - int(os.getenv("CONTENT_BUFFER")), 0)
            content.append([start_content, end_content])
        else:
            # No outro, use last 30 seconds of video
            start_last = max(end_point - int(os.getenv("CONTENT_BUFFER")), 0)
            content.append([start_last, end_point])

        annotation = {
            "show_id": x['show_id'],
            "file_path": f"{os.getenv('ROOT_DIR')}/{decade}/{year}/{x['episode_file']}",
            "bumpers": bumpers if bumpers else None,
            "commercials": None,
            "content": content
            # "commercials": [[300.0, 360.0], [600.0, 660.0]]
        }
        set_annotation_file(annotation)


if __name__ == "__main__":
    get_model_annotations(5)
