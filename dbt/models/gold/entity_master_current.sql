select
    entity_id,
    entity_name,
    normalized_name,
    country_code,
    entity_type,
    registry_entity_id,
    lei,
    source_entity_id,
    current_observation_source,
    current_snapshot_date,
    entity_resolution_method
from {{ source('lakehouse', 'dw_entity_master_current') }}
