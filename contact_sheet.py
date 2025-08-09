from typing import Union

import cv2
from PIL import Image
import math
import matplotlib.pyplot as plt

class VideoContactSheet:
    def __init__(self, video_path: str, cols: int = 4):
        self.video_path = video_path
        self.cols = cols
        self.frames = []

    @staticmethod
    def frame_to_image(frame):
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def get_frame_at(self, timestamps: Union[float, list[float]]):
        """
        Grab one or more frames at specific timestamps (in seconds)
        and append them to the contact sheet frames list.
        :param timestamps: A single float or a list of floats (seconds)
        """
        if isinstance(timestamps, (float, int)):
            timestamps = [timestamps]

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ret, frame = cap.read()
            if ret:
                self.frames.append(self.frame_to_image(frame))
            else:
                print(f"Could not read frame at {ts} seconds.")

        cap.release()
    def extract_frames_interval(self, start_sec: float, end_sec: float, interval_sec: float):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else 0

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

    def make_contact_sheet(self):
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

    def show_contact_sheet(self):
        sheet = self.make_contact_sheet()
        plt.imshow(sheet)
        plt.axis('off')
        plt.show()



# --- Example Usage ---
if __name__ == "__main__":
    video_path = "/Volumes/TTBS/time_traveler/70s/70/Night_Gallery_The_Dead_ManThe_Housekeeper.mp4"
    vcs = VideoContactSheet(video_path, cols=5)

    # Extract a time range
    #vcs.extract_frames(start_sec=10, end_sec=20)

    # Add an extra single frame at a specific timestamp
    vcs.get_frame_at(5.5)  # adds this frame to the contact sheet

    #vcs.extract_frames_interval(start_sec=0, end_sec=22.7, interval_sec=0.3)
    vcs.show_contact_sheet()

