select
    owner_entity_id,
    asset_country,
    asset_sector,
    asset_count,
    controlled_asset_count,
    owned_capacity_mw,
    average_ownership_pct,
    relationship_count,
    snapshot_date,
    change_status_vs_prior_snapshot
from {{ source('lakehouse', 'mart_owner_infrastructure_exposure_snapshot') }}
