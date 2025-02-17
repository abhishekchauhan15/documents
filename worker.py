from celery import Worker
import datetime

class CustomWorker(Worker):
    def __init__(self, *args, **kwargs):
        self.startup_time = datetime.datetime.now(datetime.UTC)
        super().__init__(*args, **kwargs)

# Update Celery app configuration
celery_app.Worker = CustomWorker 