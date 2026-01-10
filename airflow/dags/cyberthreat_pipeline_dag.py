from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="cyberthreat_insight_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False
):

    stages = ["dev", "inference", "production", "attack", "dashboard"]

    previous = None
    for stage in stages:
        task = BashOperator(
            task_id=f"run_{stage}",
            bash_command=f"python /app/main.py --stage {stage}"
        )

        if previous:
            previous >> task
        previous = task
