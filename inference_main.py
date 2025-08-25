import os
from pprint import pprint

import psycopg2
from psycopg2.extras import RealDictCursor
from concurrent.futures import ThreadPoolExecutor, as_completed

from inference_tasks import process_video  # this loads your worker function

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


def get_video_jobs(limit=100):
    video_jobs = []
    try:
        query = """
            SELECT episode_id, episode_file, episode_airdate
            FROM episodes
            -- WHERE processed = false
            WHERE show_id = 97
              AND episode_airdate < '1990-01-01'
            LIMIT %s
        """
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, (limit,))
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


def run_all_jobs(max_workers=2, dev_mode=True):
    jobs = get_video_jobs()
    if not jobs:
        return []

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(process_video, video_path, episode_id=episode_id, dev_mode=dev_mode): (video_path, episode_id)
            for video_path, episode_id in jobs
        }

        for future in as_completed(future_to_job):
            video_path, episode_id = future_to_job[future]
            try:
                result = future.result()
                results.append(result)
                pprint(result)
            except Exception as exc:
                print(f"[{video_path}] Generated an exception: {exc}")

    print("All jobs finished:", results)
    return results

if __name__ == "__main__":
    run_all_jobs(max_workers=1, dev_mode=True)
