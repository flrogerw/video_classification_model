from celery import Celery

celery_app = Celery(
    "video_tasks",
    broker="redis://192.168.1.201:6379/0",
    backend="redis://192.168.1.201:6379/0"
)

celery_app.conf.update(
    task_track_started=True,
    result_expires=3600,
)
