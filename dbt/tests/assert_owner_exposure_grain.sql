-- Assert that (snapshot_date, owner_entity_id, asset_country, asset_sector) is unique
-- in the owner infrastructure exposure snapshot mart.
-- Returns failing rows; dbt passes when this query returns zero rows.
select
    snapshot_date,
    owner_entity_id,
    asset_country,
    asset_sector,
    count(*) as n
from {{ source('lakehouse', 'mart_owner_infrastructure_exposure_snapshot') }}
group by snapshot_date, owner_entity_id, asset_country, asset_sector
having count(*) > 1
