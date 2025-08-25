from classes.video_annotations import VideoAnnotationGenerator
from classes.video_utils import VideoContactSheet


root_dir = "/Volumes/TTBS/time_traveler"

generator = VideoAnnotationGenerator()

db_filenames: list = generator.get_show_episode_filename(show_id=97)
for record in db_filenames:
    year = int(record['episode_airdate'].strftime("%y"))
    decade = f"{(year // 10) % 10}0s"
    filename = f'{root_dir}/{decade}/{year}/{record["episode_file"]}'

    vcs = VideoContactSheet(filename, cols=6)
    duration = generator.get_video_length(filename)
    samples = [(0,30), (duration - 30,duration)]

    vcs.extract_frames_intervals(segments=samples, interval_sec=3.0)
    vcs.show_contact_sheet()