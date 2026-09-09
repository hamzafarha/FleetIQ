# FleetIQ — Data Quality Assessment

**Dataset:** NYC TLC Yellow Taxi completed-trip records  
**Files assessed:** January, April, July, and October 2025  
**Rows assessed:** 15,773,441  
**Modelling scope:** Case B — trip duration from pickup to drop-off  
**Assessment date:** 2026-09-09

## Decision

The data is **usable with documented cleaning** for Case B trip-duration prediction. It is **not suitable** for Driver-to-Passenger ETA (Case A): it has no ride-request event, driver assignment, driver position, or verified driver-arrival event.

Raw Parquet files remain immutable. This assessment defines what a later cleaning step may do; it does not remove, impute, or overwrite data.

## Target definition and availability

The candidate target is:

`trip_duration_seconds = tpep_dropoff_datetime - tpep_pickup_datetime`

Both timestamps are present in all assessed files. The target can therefore be derived, but invalid and zero-length durations must not enter training.

| Check | Finding | Status | Cleaning decision |
| --- | ---: | --- | --- |
| Missing pickup timestamp | 0 | PASS | Retain. |
| Missing drop-off timestamp | 0 | PASS | Retain. |
| Drop-off before pickup | 290 (0.0018%) | CRITICAL | Exclude: a negative target is physically impossible. |
| Zero-duration trip | 160,562 (1.0179%) | WARNING | Exclude from the supervised target after documenting the rule. |
| Trip duration over 24 hours | 111 | WARNING | Flag for review; decide a capped maximum only in the cleaning policy. |
| Unexpected historical timestamp | One 2009 pickup in the July 2025 file | WARNING | Exclude after confirming it is not a valid corrected record. |

## Completeness

Core timestamps, location IDs, trip distance, and fare amount have no missing values. Missingness is concentrated in fields that should not be silently imputed without source-system confirmation.

| Field group | Missing-rate range across months | Decision |
| --- | ---: | --- |
| `passenger_count` | 15.543%–26.642% | Keep missingness explicit; create a missing indicator or exclude this feature. Do not remove trips solely for this. |
| `RatecodeID`, `store_and_fwd_flag`, `congestion_surcharge`, `Airport_fee` | Same month-level pattern as `passenger_count` | Treat as optional until their semantics and prediction-time availability are confirmed. |
| Target timestamps and location IDs | 0% | Required and usable. |

## Validity and consistency

| Check | Finding | Status | Cleaning decision |
| --- | ---: | --- | --- |
| Exact duplicate rows | 1 total (July) | WARNING | Inspect the record; deduplicate only with a documented deterministic rule. |
| Negative trip distance | 0 | PASS | Retain validation in the pipeline. |
| Zero trip distance | 431,705 (2.7369%) | WARNING | Flag. Do not automatically delete; inspect jointly with duration and zones. |
| Distance above 100 miles | 971 total | WARNING | Flag as implausible for ordinary NYC trips; retain only if the later policy supports it. |
| Maximum observed distance | 276,333–397,994 miles by month | CRITICAL | Do not use raw distance as a model feature until outlier handling is applied. |
| Negative fare | 899,233 (5.7009%) | WARNING | Exclude from features: fare is post-trip information and may represent adjustments/refunds. |
| Pickup/drop-off zone IDs outside lookup | 0 | PASS | Use the supplied zone lookup for later geographic features. |

## Leakage assessment

At prediction time, the model may use information known at trip start: pickup zone, destination zone, pickup timestamp, and any feature explicitly available then.

The following fields must be excluded from Case B model inputs because they are known only during or after the trip, or require confirmation: `tpep_dropoff_datetime`, `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tolls_amount`, `improvement_surcharge`, `total_amount`, `congestion_surcharge`, and `Airport_fee`.

`trip_distance` is reported after a trip by this source. It must not be used as a prediction-time feature unless replaced by an independently available routing-distance estimate.

## Schema governance issue

The repository configuration currently describes the Kaggle NYC Taxi Trip Duration CSV schema (latitude/longitude fields), while this assessment covers TLC Parquet files with zone IDs. Before feature engineering, select one prototype dataset contract and align `config/config.yaml`, loading code, and notebooks to it. Mixing the two schemas would make the pipeline non-reproducible.

## Approved input set for the next stage

Subject to the cleaning rules above, use:

| Role | Fields |
| --- | --- |
| Target construction only | `tpep_pickup_datetime`, `tpep_dropoff_datetime` |
| Candidate prediction-time features | `tpep_pickup_datetime`, `PULocationID`, `DOLocationID`, optionally `VendorID` and `passenger_count` with a documented missing-value policy |
| Exclude as leakage/post-trip data | Drop-off timestamp, monetary fields, surcharge fields, and observed `trip_distance` |

## Exit criteria for Data Cleaning

The next stage is complete only when it produces a versioned, non-raw dataset and a before/after log showing:

1. rows excluded for non-positive duration;
2. handling of the anomalous historical timestamp and any chosen maximum-duration rule;
3. the policy for zero/extreme observed distance;
4. the policy for missing or zero passenger count; and
5. assertions that retained records have valid timestamp order, a positive target, and valid zone IDs.

## Final quality verdict

**Proceed to Data Cleaning for Case B only.** Do not start Driver-to-Passenger ETA modelling from these TLC records, and do not start feature engineering until the prototype schema contract is aligned.
