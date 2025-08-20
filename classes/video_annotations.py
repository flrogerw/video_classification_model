"""
Video Annotation Generator

This module connects to a PostgreSQL database to retrieve episode metadata,
analyzes video files to calculate durations, and generates annotation files
(bumpers, commercials, content) for use in video processing pipelines.

Configuration is managed via environment variables in a `.env` file.
"""
import contextlib
import io
import os
import shutil

import cv2
import psycopg2
import json
import random
import string
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from typing import Optional, List, Dict, Any

# Load environment variables once at the top
load_dotenv()


class VideoAnnotationGenerator:
    """
    Handles database queries, video duration analysis, and annotation file creation
    for a CLIP-based classifier dataset.

    Attributes:
        db_config (dict): PostgreSQL connection details loaded from environment variables.
        annotations_dir (str): Path where annotation JSON files will be stored.
        root_dir (str): Root directory containing the video files.
        sample_count (int): Number of episode samples per show.
        content_buffer (int): Number of seconds to include for intro/outro content segments.
    """

    def __init__(self) -> None:
        self.db_config = {
            'dbname': os.getenv("DB_NAME"),
            'user': os.getenv("DB_USER"),
            'password': os.getenv("DB_PASSWORD"),
            'host': os.getenv("DB_HOST"),
            'port': os.getenv("DB_PORT"),
        }
        self.annotations_dir = os.getenv("ANNOTATIONS_DIR", "annotations")
        self.root_dir = os.getenv("ROOT_DIR", ".")
        self.sample_count = int(os.getenv("SAMPLE_COUNT", 10))
        self.content_buffer = int(os.getenv("CONTENT_BUFFER", 30))

    @staticmethod
    def _random_json_filename(length: int = 10) -> str:
        """Generate a random JSON filename."""
        chars = string.ascii_letters + string.digits
        name = ''.join(random.choices(chars, k=length))
        return f"{name}.json"

    def _set_annotation_file(self, annotation: Dict[str, Any]) -> None:
        """Save an annotation dictionary as a JSON file."""
        try:
            os.makedirs(self.annotations_dir, exist_ok=True)
            output_path = os.path.join(self.annotations_dir, self._random_json_filename())

            with open(output_path, "w") as json_file:
                json.dump(annotation, json_file, indent=4)

            print(f"Annotation saved to {output_path}")
        except Exception as e:
            print(f"Error saving annotation file: {e}")

    @staticmethod
    def get_video_length(filename: str) -> Optional[float]:
        """Return video duration in seconds, or None if not available."""
        try:
            stderr_capture = io.StringIO()
            with contextlib.redirect_stderr(stderr_capture):
                cap = cv2.VideoCapture(filename)
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Get any warnings from OpenCV
                warning_msg = stderr_capture.getvalue()
                if warning_msg:
                    print(f"OpenCV Warning: {warning_msg.strip()}")
                    # Handle warning here, e.g., retry with fallback settings

            if fps <= 0:
                raise ValueError("Invalid FPS value")
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
            cap.release()
            return round(duration, 2)
        except Exception as e:
            print(f"Error probing video {filename}: {e}")
            return None

    def update_episode_start_end(self, start: float, end: float, filename: str) -> None:
        query = f"""UPDATE episodes SET start_point = %s, end_point = %s, processed = true WHERE episode_file = %s"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, (start, end, filename))
            conn.commit()
            cur.close()
            conn.close()
            print(f"Updated {os.path.basename(filename)}: Start: {start} End: {end}")

        except Exception as e:
            print(f"Database query failed: {e}")
            return None

    def _empty_annotations_directory(self):
        if not os.path.exists(self.annotations_dir):
            print(f"Directory does not exist: {self.annotations_dir}")
            return

        for filename in os.listdir(self.annotations_dir):
            file_path = os.path.join(self.annotations_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symlink
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and its contents
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

    def get_show_episode_filename(self, show_id: int) -> List[Dict[str, Any]]:
        """Fetch episode records from the database for a given show ID. Used by inference"""
        query = f"""SELECT * FROM episodes WHERE show_id = %s;"""
        query2 = f"""WITH ranked AS (
                          SELECT
                              e.*,
                              ROW_NUMBER() OVER (PARTITION BY show_id ORDER BY random()) AS rn
                          FROM episodes e
                          WHERE show_id = 5
                      )
                      SELECT *
                     FROM ranked
                     WHERE rn <= 1 ORDER BY show_id;"""

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, (show_id,))
            result = cur.fetchall()
            cur.close()
            conn.close()
            return result
        except Exception as e:
            print(f"Database query failed: {e}")
            return []

    def _get_db_filenames(self) -> List[Dict[str, Any]]:
        """Fetch episode records from the database."""
        query = f"""
            WITH ranked AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY show_id ORDER BY episode_airdate) AS rn
                FROM episodes
                WHERE (end_point != FLOOR(end_point) OR start_point != FLOOR(start_point))
                -- AND episode_airdate < '1990-01-01'
                --AND show_id in (6)
            )
            SELECT * FROM ranked WHERE rn <= {self.sample_count};
        """

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, ())
            result = cur.fetchall()
            cur.close()
            conn.close()
            return result
        except Exception as e:
            print(f"Database query failed: {e}")
            return []

    def get_inference_filenames(self, show_id: int) -> list[str]:
        filenames: list = []
        db_filenames: list = self.get_show_episode_filename(show_id)
        for record in db_filenames:
            year = int(record['episode_airdate'].strftime("%y"))
            decade = f"{(year // 10) % 10}0s"
            filenames.append(f'{self.root_dir}/{decade}/{year}/{record["episode_file"]}')

        print(f"Total Records Processed: {len(db_filenames)}")
        return filenames

    def get_training_annotations(self, segments: tuple) -> None:

        file_path, segs = segments
        print(file_path, segs)

        bumpers = []
        end_point = self.get_video_length(file_path)

        if end_point is None:
            print(f"Skipping file {file_path} (no duration found).")
            return
        try:
            for timestamp in sorted(segs):
                closer = min((0, end_point), key=lambda v: abs(timestamp - v))
                print(closer)
                if closer == 0:
                    bumpers.append((0, round(timestamp, 2)))
                else:
                    bumpers.append((round(timestamp, 2), end_point))

        except Exception as e:
            print(f"Error processing training record {file_path}: {e}")

        content = []
        if bumpers and bumpers[0][0] == 0:
            start_content = bumpers[0][1]
            end_content = min(start_content + self.content_buffer, end_point)
            content.append([start_content, end_content])
        else:
            content.append([0, min(self.content_buffer, end_point)])

        if bumpers and bumpers[-1][1] == end_point:
            end_content = bumpers[-1][0]
            start_content = max(end_content - self.content_buffer, 0)
            content.append([start_content, end_content])
        else:
            start_last = max(end_point - self.content_buffer, 0)
            content.append([start_last, end_point])

        annotation = {
            "file_path": file_path,
            "bumpers": bumpers if bumpers else None,
            "commercials": None,
            "content": content,
            "video_duration": end_point
        }
        # if annotation['bumpers'] or annotation['commercials']:
        self._set_annotation_file(annotation)

    def get_model_annotations(self) -> None:
        """Generate annotations for episodes from a given show ID."""
        self._empty_annotations_directory()

        db_filenames = self._get_db_filenames()
        for record in db_filenames:
            try:
                year = int(record['episode_airdate'].strftime("%y"))
                decade = f"{(year // 10) % 10}0s"
                file_path = f'{self.root_dir}/{decade}/{year}/{record["episode_file"]}'
                end_point = self.get_video_length(file_path)

                if end_point is None:
                    print(f"Skipping file {file_path} (no duration found).")
                    continue

                outro = [record['end_point'], end_point] if record['end_point'] < end_point else None
                intro = [0, record['start_point']] if record['start_point'] > 0 else None

                bumpers = [seg for seg in (intro, outro) if seg]

                content = []
                if intro and intro[1]:
                    start_content = intro[1]
                    end_content = min(start_content + self.content_buffer, end_point)
                    content.append([start_content, end_content])
                else:
                    content.append([0, min(self.content_buffer, end_point)])

                if outro and outro[0]:
                    end_content = outro[0]
                    start_content = max(end_content - self.content_buffer, 0)
                    content.append([start_content, end_content])
                else:
                    start_last = max(end_point - self.content_buffer, 0)
                    content.append([start_last, end_point])

                annotation = {
                    "show_id": record['show_id'],
                    "file_path": file_path,
                    "bumpers": bumpers if bumpers else None,
                    "commercials": None,
                    "content": content,
                    "video_duration": end_point
                }
                if annotation['bumpers'] or annotation['commercials']:
                    self._set_annotation_file(annotation)

            except Exception as e:
                print(f"Error processing record {record}: {e}")

        print(f"Total Records Processed: {len(db_filenames)}")
