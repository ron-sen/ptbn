

from celery import Celery
from app.config import settings
from celery.schedules import crontab


# broker - task messages queues up (redis)
celery_app = Celery(
    "ptbn",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND, # task results are stored 
    include=["app.task.tasks" , "app.task.indicator_task" , "app.task.fuzzy_task"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started = True # shows "started"
)

# scheuling by celery beats 
# beats is the seaparate process that just fires task on schedule by pushing them onto the same queue workers consumes from  , it only triggers it 

celery_app.conf.beat_schedule = {
    "fetch-market-candles-1h" :{
        "task" : "app.tasks.fetch_market_data", # matches name in @celery
        "schedule" : 30.0 , # every 30 sec 
        "args" : ("1m",) ,
    },
}


    