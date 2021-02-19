
import datetime
import logging
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from google.cloud import bigquery
from airflow.contrib.operators import bigquery_operator

dag = DAG(
        'Earnings_ml_model_log_regression',
        schedule_interval='@weekly',
        start_date=datetime.datetime.now()
)

def read_raw_storage():
    client = bigquery.Client()
    dataset_id = 'machine_learning_02'
    dataset_ref = client.dataset(dataset_id)

    #Job Configuration Parameters
    job_config = bigquery.LoadJobConfig()
    job_config.autodetect = True
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
    job_config.source_format = bigquery.SourceFormat.CSV
    data_uri = "gs://crazy-hippo-01/dataset/census_train.csv"

    load_job = client.load_table_from_uri(data_uri, dataset_ref.table("income_model"), job_config=job_config)
                                            
    logging.info("Starting job {}".format(load_job.job_id))

    results = load_job.result()  # Waits for table load to complete.
    logging.info("Job finished.")

    destination_table = client.get_table("machine_learning_02.income_model")
    logging.info("Loaded {} rows.".format(destination_table.num_rows))                                      

read_raw_storage = PythonOperator(
    task_id="read_raw_storage",
    python_callable=read_raw_storage,
    dag=dag
)

bq_clean_table = bigquery_operator.BigQueryOperator(
    task_id='bq_clean_table',
    bql="""
    SELECT age, workclass, gender, occupation, education_num, marital_status, relationship, capital_gain, income_bracket
    FROM `crazy-hippo-01.machine_learning_02.income_model` 
    WHERE workclass IS NOT NULL AND workclass != "Never-worked"
    """,
    use_legacy_sql=False,
    destination_dataset_table="machine_learning_02.bq_clean_table",
    write_disposition="WRITE_TRUNCATE",
    location="US"
)

model_training = bigquery_operator.BigQueryOperator(
    task_id='model_training',
    bql="""
    CREATE OR REPLACE MODEL machine_learning_02.income_model_log_classifier
    OPTIONS(input_label_cols=['income_bracket'], model_type='logistic_reg')
    AS 
    SELECT *
    FROM `crazy-hippo-01.machine_learning_02.bq_clean_table`
    """,
    use_legacy_sql=False,
    write_disposition="WRITE_TRUNCATE",
    location="US"
)

model_evaluation = bigquery_operator.BigQueryOperator(
    task_id='model_evaluation',
    bql="""
    INSERT machine_learning_02.evaluation_log_regression (Accuracy, f1_Score, Precision, Recall, ROC_AUC) 
    SELECT ROUND(precision,2) as Accuracy, ROUND(recall,2) as f1_Score, ROUND(accuracy,2) as Precision, ROUND(f1_score,2) as Recall, ROUND(roc_auc,2) as ROC_AUC
    FROM
    ML.EVALUATE(MODEL `machine_learning_02.income_model_log_classifier`)
    """,
    use_legacy_sql=False,
    write_disposition="WRITE_TRUNCATE",
    location="US"
)


# Configure Task Dependencies
read_raw_storage >> bq_clean_table
bq_clean_table >> model_training
model_training >> model_evaluation
