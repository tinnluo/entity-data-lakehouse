.PHONY: dbt-run dbt-test airflow-up airflow-down eval clickhouse-up clickhouse-down

dbt-run:
	cd dbt && dbt run --profiles-dir .

dbt-test:
	cd dbt && dbt test --profiles-dir .

airflow-up:
	docker compose up airflow

airflow-down:
	docker compose down

eval:
	python3 evals/run_evals.py

clickhouse-up:
	USE_CLICKHOUSE=true docker compose --profile clickhouse up --build

clickhouse-down:
	docker compose --profile clickhouse down
