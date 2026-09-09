# Quick Dataset Audit Report

## Source
This report summarizes the NYC TLC dataset audit performed in the dataset audit notebook and the existing project audit report.

## Data Covered
The audit inspected four Yellow Taxi Parquet files for 2025:
- January 2025: 3,475,226 rows
- April 2025: 3,970,553 rows
- July 2025: 3,898,963 rows
- October 2025: 4,428,699 rows

Total rows audited: 15,773,441.

## Main Results

### 1. Dataset Suitability
The dataset is usable for the project’s trip-duration prediction case, but only after a defined cleaning process. It is not suitable for a driver-to-passenger ETA case because TLC trip records do not contain request, dispatch, driver assignment, GPS, or arrival-event timestamps.

### 2. Schema and Required Signals
The schema is consistent across all monthly files with 20 columns. Core trip fields are present in each file, including pickup timestamp, drop-off timestamp, pickup/drop-off zones, trip distance, passenger count, and fare amount. The key timestamps and location identifiers are available for downstream modeling.

### 3. Missing Values
Missingness is concentrated in several fields, especially passenger_count, RatecodeID, store_and_fwd_flag, congestion_surcharge, and Airport_fee. These missing patterns are consistent across months and require a documented policy before modeling.

### 4. Duplicates
Exact duplicates are negligible, with only one exact duplicate in the July file. This is very small and should be reviewed as an ingestion artifact, but it does not block the dataset.

### 5. Trip Duration Quality
Trip duration can be derived from pickup and drop-off timestamps, but the raw duration field contains several quality issues:
- Negative durations
- Zero-duration trips
- Trips lasting over 24 hours
- A suspicious timestamp in the July file from 2009

These findings indicate that duration must be cleaned before training.

### 6. Distance Quality
Trip distance appears complete and non-negative, but the dataset contains many zero-distance trips and extreme values over 100 miles, including implausible values up to hundreds of thousands of miles. These records should be reviewed before use.

### 7. Location Validation
Pickup and drop-off location IDs match the taxi-zone lookup table in every month. The lookup is therefore valid for later geographic feature construction.

## Overall Verdict
The dataset passes the audit for trip-duration prediction with cleaning, but not for driver-to-passenger ETA prediction.

## Suggested Next Steps
1. Store the raw Parquet files unchanged.
2. Create an explicit cleaning pipeline for duration, distance, and missing-value handling.
3. Keep the zone lookup validation step in the processing workflow.
4. Define a passenger-count policy using source definitions before feature engineering.
