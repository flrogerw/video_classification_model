"""
Module: video_contact_sheet
---------------------------
This module defines the VideoContactSheet class for extracting frames from
a video file and compiling them into a visual contact sheet. It supports
extracting specific frames by timestamp or evenly spaced frames over an interval.
"""

import math
from typing import Union, List, Optional

import cv2
from PIL import Image
import matplotlib.pyplot as plt


class VideoContactSheet:
    """
    A class for extracting frames from a video and compiling them into
    a contact sheet image.

    Attributes:
        video_path (str): Path to the video file.
        cols (int): Number of columns in the contact sheet.
        frames (list[Image.Image]): List of extracted PIL image frames.
    """

    def __init__(self, video_path: str, cols: int = 4) -> None:
        """
        Initialize the VideoContactSheet.

        Args:
            video_path: Path to the input video file.
            cols: Number of columns in the contact sheet.
        """
        self.video_path = video_path
        self.cols = cols
        self.frames: List[Image.Image] = []

    @staticmethod
    def frame_to_image(frame) -> Image.Image:
        """
        Convert an OpenCV BGR frame to a PIL Image in RGB format.

        Args:
            frame: OpenCV frame (numpy array).

        Returns:
            PIL Image object.
        """
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def get_frame_at(self, timestamps: Union[float, int, List[Union[float, int]]]) -> None:
        """
        Grab one or more frames at specific timestamps (in seconds)
        and append them to the contact sheet frames list.

        Args:
            timestamps: A single float/int or a list of floats/ints (seconds).
        """
        try:
            # Normalize to list of timestamps
            if isinstance(timestamps, (float, int)):
                timestamps = [timestamps]

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {self.video_path}")

            for ts in timestamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000)
                ret, frame = cap.read()
                if ret:
                    self.frames.append(self.frame_to_image(frame))
                else:
                    print(f"[WARNING] Could not read frame at {ts} seconds.")

            cap.release()
        except Exception as e:
            print(f"[ERROR] Failed to get frame(s) at {timestamps}: {e}")

    def extract_frames_interval(self, start_sec: float, end_sec: float, interval_sec: float) -> None:
        """
        Extract frames at a fixed interval between start and end times.

        Args:
            start_sec: Start time in seconds.
            end_sec: End time in seconds.
            interval_sec: Interval between frames in seconds.
        """
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {self.video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps else 0

            # Adjust invalid end time
            if end_sec > duration or end_sec <= 0:
                end_sec = duration

            self.frames.clear()
            t = start_sec

            while t <= end_sec:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if not ret:
                    break
                self.frames.append(self.frame_to_image(frame))
                t += interval_sec

            cap.release()
        except Exception as e:
            print(f"[ERROR] Failed to extract frames from {start_sec} to {end_sec} every {interval_sec}s: {e}")

    def make_contact_sheet(self) -> Optional[Image.Image]:
        """
        Create a contact sheet from the collected frames.

        Returns:
            A PIL Image representing the contact sheet, or None if no frames exist.
        """
        try:
            if not self.frames:
                raise ValueError("No frames extracted to make contact sheet.")

            w, h = self.frames[0].size
            rows = math.ceil(len(self.frames) / self.cols)
            sheet = Image.new("RGB", (self.cols * w, rows * h))

            for idx, frame in enumerate(self.frames):
                x = (idx % self.cols) * w
                y = (idx // self.cols) * h
                sheet.paste(frame, (x, y))

            return sheet
        except Exception as e:
            print(f"[ERROR] Failed to create contact sheet: {e}")
            return None

    def show_contact_sheet(self) -> None:
        """
        Display the generated contact sheet using matplotlib.
        """
        try:
            sheet = self.make_contact_sheet()
            if sheet is None:
                return
            plt.imshow(sheet)
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"[ERROR] Failed to display contact sheet: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    video_path = "/Volumes/TTBS/time_traveler/70s/70/Night_Gallery_The_Dead_ManThe_Housekeeper.mp4"
    vcs = VideoContactSheet(video_path, cols=5)

    # Extract specific frame
    vcs.get_frame_at(5.5)

    # Extract frames at interval
    # vcs.extract_frames_interval(start_sec=0, end_sec=22.7, interval_sec=0.3)

    # Display contact sheet
    vcs.show_contact_sheet()
