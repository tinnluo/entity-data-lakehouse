-- Assert that no two current ownership rows share the same business_key_hash.
-- Returns failing rows; dbt passes when this query returns zero rows.
select
    business_key_hash,
    count(*) as n
from {{ source('lakehouse', 'dw_ownership_current') }}
where is_current_flag = 'Y'
group by business_key_hash
having count(*) > 1
