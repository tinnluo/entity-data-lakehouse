.PHONY: dbt-run dbt-test airflow-up airflow-down

dbt-run:
	cd dbt && dbt run --profiles-dir .

dbt-test:
	cd dbt && dbt test --profiles-dir .

airflow-up:
	docker compose up airflow

airflow-down:
	docker compose down
