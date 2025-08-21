from time import sleep

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you have PyQt5/PySide2 installed
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import math

from matplotlib.widgets import Button
from sklearn.metrics import confusion_matrix
from typing import Union, List, Optional
from PIL import Image

from classes.video_annotations import VideoAnnotationGenerator

"""
Module: video_contact_sheet
---------------------------
This module defines the VideoContactSheet class for extracting frames from
a video file and compiling them into a visual contact sheet. It supports
extracting specific frames by timestamp or evenly spaced frames over an interval.

    vcs = VideoContactSheet(video_path, cols=5)

    # Extract specific frame
    vcs.get_frame_at(5.5)

    # Extract frames at interval
    # vcs.extract_frames_interval(start_sec=0, end_sec=22.7, interval_sec=0.3)

    # Display contact sheet
    vcs.show_contact_sheet()
"""


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
        self.metadata: list[float] = []
        self.clip_metadata: list[float] = []
        self.annotations: list = []
        self.generator = VideoAnnotationGenerator()

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

    def extract_frames_intervals(self, segments: list[tuple[float, float]], interval_sec: float,
                                 on_click: bool = False) -> None:
        """
        Extract frames at a fixed interval for multiple (start, end) segments.

        Args:
            segments: List of (start_sec, end_sec) tuples.
            interval_sec: Interval between frames in seconds.
        """
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {self.video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps else 0

            self.frames.clear()
            self.clip_metadata.clear()
            if not on_click:
                self.metadata.clear()

            for i, (start_sec, end_sec) in enumerate(segments):
                # Adjust invalid times
                if end_sec > duration or end_sec <= 0:
                    end_sec = duration
                if start_sec < 0:
                    start_sec = 0

                t = start_sec
                while t <= end_sec:
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                    ret, frame = cap.read()
                    if not ret:
                        break

                    timestamp_text = f"{t:0>8.3f}s"
                    if not on_click:
                        self.metadata.append((t, start_sec, end_sec))
                    else:
                        self.clip_metadata.append((t, start_sec, end_sec))

                    cv2.putText(
                        frame,
                        timestamp_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    self.frames.append(self.frame_to_image(frame))
                    t += interval_sec
                self.frames.append(None)
                self.metadata.append(None)

                # Insert a marker for a line break between segments
                if i < len(segments) - 1:
                    self.frames.append(None)
                    self.metadata.append(None)

            cap.release()

        except Exception as e:
            print(f"[ERROR] Failed to extract frames from segments {segments} every {interval_sec}s: {e}")

    def make_contact_sheet(self, return_grid: bool = False) -> Optional[tuple[Image.Image, tuple[int, int]]]:
        """
        Create a contact sheet from the collected frames, inserting a line break between segments.

        Args:
            return_grid: If True, also return (rows, cols) grid shape.

        Returns:
            If return_grid is False:
                A PIL Image representing the contact sheet, or None if no frames exist.
            If return_grid is True:
                (sheet, (rows, cols)), or None if no frames exist.
        """
        try:
            if not self.frames:
                raise ValueError("No frames extracted to make contact sheet.")

            w, h = self.frames[0].size

            # Count actual frames to compute rows
            num_actual_frames = len([f for f in self.frames if f is not None])
            # Estimate rows; we'll adjust y dynamically for line breaks
            rows = math.ceil(num_actual_frames / self.cols) + len([f for f in self.frames if f is None])
            sheet = Image.new("RGB", (self.cols * w, rows * h), color="white")

            x_offset = 0
            y_offset = 0
            col_count = 0

            for frame in self.frames:
                if frame is None:
                    # Line break between segments
                    x_offset = 0
                    y_offset += h
                    col_count = 0
                    continue

                sheet.paste(frame, (x_offset, y_offset))

                col_count += 1
                x_offset += w

                if col_count >= self.cols:
                    # Move to next row
                    x_offset = 0
                    y_offset += h
                    col_count = 0

            if return_grid:
                return sheet, (rows, self.cols)
            else:
                return sheet

        except Exception as e:
            print(f"[ERROR] Failed to create contact sheet: {e}")
            return None

    def save_contact_sheet(self, output_path: str) -> None:
        """
        Generate and save the contact sheet to disk without displaying it.
        """
        try:
            sheet, grid_shape = self.make_contact_sheet(return_grid=True)
            if sheet is None:
                return

            # Create a figure for saving (no interactive buttons)
            fig, ax = plt.subplots(figsize=(16, 12))
            ax.imshow(np.array(sheet))
            ax.axis('off')
            ax.set_title(self.video_path, fontsize=12, pad=20)

            # Save the contact sheet
            fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)

        except Exception as e:
            print(f"[ERROR] Failed to save contact sheet: {e}")

    def show_contact_sheet(self, image_click: bool = False) -> None:
        """
        Display the generated contact sheet using matplotlib.
        """
        try:
            sheet, grid_shape = self.make_contact_sheet(return_grid=True)
            if sheet is None:
                return

            # Create a bigger figure (width, height) in inches
            fig, ax = plt.subplots(figsize=(12, 10))
            plt.imshow(np.array(sheet))
            plt.axis('off')
            plt.title(self.video_path, fontsize=12, pad=20)

            def on_content_click(event):
                self.generator.get_training_annotations((self.video_path, []))
                plt.close(fig)

            def on_skip_click(event):
                plt.close(fig)

            def on_clear_click(event):
                self.annotations.clear()

            def on_button_click(event):
                if self.annotations:
                    self.generator.get_training_annotations((self.video_path, self.annotations))
                    self.annotations.clear()
                plt.close(fig)

            if not image_click:
                button_ax = plt.axes([0.8, 0.02, 0.1, 0.05])  # [left, bottom, width, height]
                btn = Button(button_ax, "Process")
                btn.on_clicked(on_button_click)

                button_content = plt.axes([0.65, 0.02, 0.1, 0.05])  # [left, bottom, width, height]
                btn_o = Button(button_content, "Content Only")
                btn_o.on_clicked(on_content_click)

                button_skip = plt.axes([0.35, 0.02, 0.1, 0.05])  # [left, bottom, width, height]
                btn_s = Button(button_skip, "Skip")
                btn_s.on_clicked(on_skip_click)

                button_clear = plt.axes([0.2, 0.02, 0.1, 0.05])  # [left, bottom, width, height]
                btn_c = Button(button_clear, "Clear")
                btn_c.on_clicked(on_clear_click)

            # Compute cell width/height
            nrows, ncols = grid_shape
            w, h = sheet.size  # <-- use PIL .size
            cell_w = w / ncols
            cell_h = h / nrows

            def on_click(event):
                if event.inaxes != ax:  # click outside image
                    return

                # Compute which cell was clicked
                col = int(event.xdata // cell_w)
                row = int(event.ydata // cell_h)
                idx = row * ncols + col
                if 0 <= idx < len(self.metadata):
                    meta = self.metadata[idx]
                    self.extract_frames_intervals([(meta, meta + 8)], 0.2, on_click=True)
                    self.show_contact_sheet(image_click=True)

            def second_click(event):
                if event.inaxes != ax:  # click outside image
                    return

                # Compute which cell was clicked
                col = int(event.xdata // cell_w)
                row = int(event.ydata // cell_h)
                idx = row * ncols + col
                if 0 <= idx < len(self.clip_metadata):
                    meta = self.clip_metadata[idx]
                    self.annotations.append(meta)
                    sleep(0.5)
                    plt.close(fig)

            if image_click:
                fig.canvas.mpl_connect("button_press_event", second_click)
            else:
                fig.canvas.mpl_connect("button_press_event", on_click)
            plt.show()

        except Exception as e:
            print(f"[ERROR] Failed to display contact sheet: {e}")


class ConfusionMatrix:
    """
    Class to compute and plot a confusion matrix using true and predicted labels.
    """

    def __init__(self, target_classes: Optional[List[str]] = None):
        """
        Initialize the plotter.

        Args:
            target_classes: List of class names or labels for axis ticks.
        """
        self.target_classes = target_classes or []
        self.true_labels = []
        self.predicted_labels = []

    def set_labels(self, true_labels: List[int], predicted_labels: List[int]) -> None:
        """
        Set the true and predicted labels.

        Args:
            true_labels: List of ground truth labels.
            predicted_labels: List of predicted labels by the model.
        """
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels

    def plot(self, epoch: int = 0) -> None:
        """
        Plot the confusion matrix heatmap.

        Args:
            epoch: Epoch number for the plot title.
        """
        if not self.true_labels or not self.predicted_labels:
            print("True labels or predicted labels are empty. Cannot plot confusion matrix.")
            return

        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.target_classes,
            yticklabels=self.target_classes
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
        plt.show()


class ConfidenceGraph:
    """
    Class to plot accuracy and confidence over training epochs.
    """

    def __init__(self, accuracy: List[float] = None, confidence: List[float] = None):
        """
        Initialize with optional accuracy and confidence data.

        Args:
            accuracy: List of accuracy values per epoch.
            confidence: List of confidence values per epoch.
        """
        self.accuracy = accuracy or []
        self.confidence = confidence or []

    def set_data(self, accuracy: List[float], confidence: List[float]) -> None:
        """
        Set the accuracy and confidence data.

        Args:
            accuracy: List of accuracy values.
            confidence: List of confidence values.
        """
        self.accuracy = accuracy
        self.confidence = confidence

    def plot(self) -> None:
        """
        Plot the accuracy and confidence graph.
        """
        if len(self.accuracy) != len(self.confidence):
            print(f"Data length mis-match: Accuracy:{len(self.accuracy)}, Confidence: {len(self.confidence)}")
            return

        epochs = np.arange(1, len(self.accuracy) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.accuracy, marker='o', label='Accuracy', color='blue')
        plt.plot(epochs, self.confidence, marker='o', label='Confidence', color='orange')
        plt.axvline(8, linestyle='--', color='gray', alpha=0.5, label='Potential Early Stop')
        plt.title('Accuracy vs Confidence Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.ylim(0.5, 1.0)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()
