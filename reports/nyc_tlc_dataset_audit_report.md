# NYC TLC Dataset Audit Report

**Project:** FleetIQ smart ETA prediction  
**Source:** `notebooks/01_nyc_tlc_dataset_audit.ipynb`  
**Audit scope:** Raw NYC TLC Yellow Taxi trip records, before cleaning or modeling

## Executive conclusion

The acquired data is technically usable for **Case B: passenger trip-duration prediction**, but only after a documented cleaning step. All four inspected files contain the required pickup and drop-off timestamps, and the core predictors and location identifiers are present. However, the candidate duration target contains invalid, zero-length, and unusually long trips, so it should not be modeled directly from the raw files.

The data is **not suitable for Case A: driver-to-passenger ETA prediction**. NYC TLC records completed trips, not ride requests or dispatch events. The audit found no request timestamp, driver assignment, driver GPS position, driver trajectory, or verified driver-arrival timestamp.

## Data covered

The audit successfully loaded four quarterly 2025 Yellow Taxi Parquet files and the 265-row taxi-zone lookup table.

| Month | Rows | Columns | In-memory size |
|---|---:|---:|---:|
| 2025-01 | 3,475,226 | 20 | 616.31 MB |
| 2025-04 | 3,970,553 | 20 | 700.97 MB |
| 2025-07 | 3,898,963 | 20 | 680.73 MB |
| 2025-10 | 4,428,699 | 20 | 777.90 MB |
| **Total** | **15,773,441** | | |

The Parquet files were inspected independently to preserve month-level differences and avoid an unnecessary full concatenation.

## Main findings

### 1. Schema and required fields

The inspected files have a consistent 20-column schema. The fields needed for trip-duration modeling are available in every month:

- `tpep_pickup_datetime`
- `tpep_dropoff_datetime`
- `PULocationID`
- `DOLocationID`
- `trip_distance`
- `passenger_count`
- `fare_amount`

Pickup and drop-off timestamps, location identifiers, trip distance, and fare amount have no missing values in the audit output.

### 2. Missing values are concentrated in several related fields

Five fields have severe missingness in every inspected month: `passenger_count`, `RatecodeID`, `store_and_fwd_flag`, `congestion_surcharge`, and `Airport_fee`. The missing rows and rates are identical within each month:

| Month | Missing rows | Missing rate |
|---|---:|---:|
| 2025-01 | 540,149 | 15.543% |
| 2025-04 | 745,730 | 18.782% |
| 2025-07 | 1,038,755 | 26.642% |
| 2025-10 | 990,887 | 22.374% |

This pattern should be investigated before using passenger count or these related fields as predictors. The audit does not assume that missing values should simply be imputed or removed.

### 3. Exact duplicates are negligible

Only one exact duplicate was found, in July 2025, representing 0.00003% of that month. No exact duplicates were found in January, April, or October. The single duplicate should be investigated as a possible ingestion artifact before deduplication is introduced.

### 4. Trip duration is available but needs cleaning

The candidate target is computed as:

`trip_duration = tpep_dropoff_datetime - tpep_pickup_datetime`

No pickup or drop-off timestamps were null or invalid. The issue is the validity of the resulting duration:

| Month | Negative durations | Zero-duration trips | Trips over 24 hours | Median duration |
|---|---:|---:|---:|---:|
| 2025-01 | 124 | 1,927 | 20 | 11.70 min |
| 2025-04 | 163 | 34,606 | 25 | 12.93 min |
| 2025-07 | 1 | 56,063 | 31 | 13.53 min |
| 2025-10 | 2 | 67,966 | 35 | 14.55 min |

The July data also contains a pickup timestamp from 2009, despite being labeled as a July 2025 file. The January audit includes an extreme negative duration of about -51,472 minutes, and other months also contain extreme positive values. These observations make an explicit validity and outlier policy necessary.

**Interpretation:** Case B is conditionally suitable. The target can be constructed, but suspicious records must be handled before training.

### 5. Trip distance contains suspicious values

Trip distance is complete and non-negative in all four months, with medians between 1.67 and 1.91 miles. However, zero-distance trips are common and values over 100 miles occur in every file:

| Month | Zero distance | Distance over 100 miles | Maximum reported distance |
|---|---:|---:|---:|
| 2025-01 | 90,893 | 162 | 276,423.57 mi |
| 2025-04 | 91,439 | 306 | 386,088.43 mi |
| 2025-07 | 123,774 | 274 | 397,994.37 mi |
| 2025-10 | 125,599 | 229 | 276,333.48 mi |

The maximum values are implausible for ordinary NYC taxi trips and should be reviewed separately from legitimate long journeys.

### 6. Passenger counts need a defined policy

Passenger count is missing for 15.5% to 26.6% of rows depending on the month. Zero passenger counts are also present, while counts above six are rare:

| Month | Missing | Zero | Greater than 6 |
|---|---:|---:|---:|
| 2025-01 | 540,149 | 24,656 | 18 |
| 2025-04 | 745,730 | 23,101 | 14 |
| 2025-07 | 1,038,755 | 20,544 | 18 |
| 2025-10 | 990,887 | 22,310 | 22 |

These values may reflect source-system conventions rather than simple errors. Their treatment should be based on the TLC definitions and the intended modeling use.

### 7. Location identifiers pass referential-integrity checks

Every observed pickup and drop-off location identifier matched `taxi_zone_lookup.csv` in all four months. The lookup is therefore suitable for later geographic feature construction, subject to retaining the validation step in the processing pipeline.

## Feasibility by FleetIQ use case

| Use case | Result | Reason |
|---|---|---|
| Case B: passenger trip duration | **Conditionally suitable** | Both timestamps are present, but duration anomalies require cleaning. |
| Case A: driver-to-passenger ETA | **Not suitable with TLC alone** | No request, dispatch, driver-location, assignment, trajectory, or arrival-event data. |

The pickup timestamp in a completed trip is not equivalent to a ride-request timestamp or a verified driver-arrival timestamp. The pickup zone identifies the passenger origin, but it cannot provide the driver's approach state.

## Recommended next steps

1. Create a separate cleaning pipeline that preserves the raw Parquet files.
2. Standardize timestamp types and document rules for negative, zero, and extremely long durations.
3. Investigate date-range anomalies, especially the 2009 timestamp in the July file.
4. Define a transparent policy for zero and extreme trip distances.
5. Decide how missing and zero passenger counts will be represented, using the source definitions rather than an arbitrary imputation rule.
6. Investigate the one exact duplicate before deciding whether to deduplicate.
7. Derive `trip_duration` only in processed data, then run exploratory analysis and leakage checks before modeling.
8. Obtain operator dispatch data for Case A, including request time, driver assignment, driver location or trajectory, and arrival events.

## Audit status

| Category | Status |
|---|---|
| File integrity | PASS |
| Schema | PASS |
| Missing values | WARNING |
| Duplicates | WARNING |
| Timestamps | WARNING |
| Trip duration | WARNING |
| Distance | PASS with outlier review required |
| Passenger count | PASS for availability, with missing-value policy required |
| Locations | PASS |
| Case B feasibility | WARNING: conditionally suitable |
| Case A feasibility | CRITICAL: unavailable from TLC alone |

This report summarizes the executed audit notebook. The audit itself does not remove rows, impute values, overwrite raw data, or train a model.