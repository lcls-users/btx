from datetime import datetime
import os
from airflow import DAG
import importlib
jid = importlib.import_module("btx-dev.plugins.jid")
JIDSlurmOperator = jid.JIDSlurmOperator

# DAG SETUP
description='BTX index run DAG'
dag_name = os.path.splitext(os.path.basename(__file__))[0]

dag = DAG(
    dag_name,
    start_date=datetime( 2022,4,1 ),
    schedule_interval=None,
    description=description,
  )


# Tasks SETUP
task_id='find_peaks'
find_peaks = JIDSlurmOperator(task_id=task_id, dag=dag)

task_id='post_to_elog'
elog1 = JIDSlurmOperator(task_id = task_id, dag=dag)

task_id='index'
index = JIDSlurmOperator(task_id=task_id, dag=dag)

# Draw the DAG
find_peaks >> elog1 >> index
