-- Assert that owned_capacity_mw and average_ownership_pct are non-negative,
-- and that average_ownership_pct does not exceed 100 (it is stored as a percent, not a fraction).
-- Returns failing rows; dbt passes when this query returns zero rows.
select *
from {{ source('lakehouse', 'mart_owner_infrastructure_exposure_snapshot') }}
where
    owned_capacity_mw < 0
    or average_ownership_pct < 0
    or average_ownership_pct > 100
