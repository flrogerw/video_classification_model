import os
import psycopg2
from tasks import process_video
from psycopg2.extras import RealDictCursor

ROOT_DIR = os.getenv("DIR_ROOT")

db_config = {
    'dbname': os.getenv("DB_NAME"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'host': os.getenv("DB_HOST"),
    'port': os.getenv("DB_PORT"),
}


def get_file_path(airdate, episode_file):
    year = int(airdate.strftime("%y"))
    decade = f"{(year // 10) % 10}0s"
    return f'{ROOT_DIR}/{decade}/{year}/{episode_file}'


def get_video_jobs():
    video_jobs = []  # always define up front
    try:
        query = "SELECT episode_id, episode_file, episode_airdate FROM episodes WHERE processed = false AND episode_airdate < '1990-01-01' LIMIT 100"
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if rows:
            video_jobs = [
                (get_file_path(row['episode_airdate'], row['episode_file']), row['episode_id'])
                for row in rows
            ]
        else:
            print("Database query returned no results")

    except Exception as e:
        print(f"Database query failed: {e}")

    return video_jobs


# --- 2. Submit jobs to Celery ---
async_results = []


def make_callback(video_path):
    """Builds an on_message callback bound to this video."""

    def handle_progress(message):
        if message['status'] == "PROGRESS":
            print(f"[{video_path}] Progress:", message['result'])
        elif message['status'] == "FAILURE":
            print(f"[{video_path}] Failed:", message)

    return handle_progress


for video_path, episode_id in get_video_jobs():
    print(video_path, episode_id)
    task = process_video.apply_async(args=[video_path], kwargs={"episode_id": episode_id, "dev_mode": True})
    async_results.append((task, video_path))

# --- 3. Collect results (event-driven per task) ---
final_results = []
for task, video_path in async_results:
    result = task.get(on_message=make_callback(video_path), propagate=True)
    final_results.append(result)
    print(f"[{video_path}] Final:", result)

print("All jobs finished:", final_results)
