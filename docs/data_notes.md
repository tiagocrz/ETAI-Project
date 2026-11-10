# Data notes

Raw profile of the four input files. These are observations only: no cleaning decisions yet.
Generated 2026-11-10 with a scratch pandas script (pandas 3.0.3).

> **Location:** `data/raw/` 

## Files at a glance

| File | Rows | Cols | Notes |
|---|---:|---:|---|
| `train.csv` | 6,534 | 40 | Holds both targets. 610 customers. No fully duplicated rows. |
| `test.csv` | 986 | 38 | Same columns minus the 2 targets. 206 customers. No fully duplicated rows. |
| `clientes.csv` | 1,143 | 13 | One row per `customer_id` (no duplicate ids). Covers 100% of train and test customers. |
| `sample_submission.csv` | 986 | 2 | `ID`, `repeat_purchase_90d`. Its IDs are exactly the same set as `test.ID`. |

- `ID` ranges overlap between train (10000000–10007674) and test (10000006–10007675), but no ID appears in both. The IDs look like one sequence split into two parts.
- Dtypes differ between files. Count columns (`N_ Detalhes`, `prior_purchase_count`, `purchase_count_*`, ...) are `float64` in train only because train has NaNs. They are `int64` in test.

## Targets (train only)

| | Count | Share |
|---|---:|---:|
| `repeat_purchase_90d = 1` | 3,654 | 55.9% |
| `repeat_purchase_90d = 0` | 2,880 | 44.1% |

- The classes are only mildly imbalanced.
- `spend_next_90d` is NaN for every row with target 0, as expected. It is **also NaN for 229 rows with target 1**, so the regression has 3,425 usable rows.
- Among returners, `spend_next_90d` runs from 1.27 to 25,654 (mean 1,922, median 986, std 2,571). It is strongly right-skewed, with no zeros or negatives.
- The positive rate by purchase year moves between 0.48 and 0.63 with no clear trend.
- `sample_submission` holds 741 ones and 245 zeros (75%). This is probably a placeholder, not a hint about the test distribution.

## Customers and time

| | Train | Test |
|---|---|---|
| Customers | 610 | 206 |
| Rows per customer: mean / median / max | 10.7 / 4 / 277 | 4.8 / 3 / 43 |
| Customers with a single row | 157 | 46 |
| `purchase_date` range | 2015-07-11 → 2025-02-15 | 2025-05-24 → 2026-10-06 |
| `purchase_date` missing | 65 (1.0%) | 0 |

- **The test set is strictly later than train**, with a gap of about 3 months between them. A time-ordered validation split is worth comparing against `StratifiedGroupKFold`.
- 184 of the 206 test customers (932 of 986 rows) also appear in train. 22 test customers (54 rows) are new.
- Train rows per year range from 335 (2015) to 782 (2021). 2025 has only 78 rows. 186 train rows fall within 90 days of the last train date.

## Missingness patterns (train)

- **65 rows (0.99%) are NaN in all 37 non-key columns.** Only `ID`, `customer_id` and `repeat_purchase_90d` are filled. That's why so many columns show exactly 0.99% missing. Test has no such rows.
- About 260 rows (3.98%) are missing `payment_type`, `salesperson`, `transport_method`, `warehouse` and `spend_365d` together. Test has the same block at 3.04%.
- The `prior_*` and `days_since_previous_purchase` columns are missing for 10–23% of rows. This is mostly structural: 603 rows have `prior_purchase_count == 0`, while 861 rows have `days_since_previous_purchase` NaN. That leaves about 258 NaNs not explained by "first purchase".
- **Negative values** where they shouldn't be possible: `days_since_previous_purchase` (16), `prior_mean_purchase_value` (16), `prior_mean_gap_days` (16), `spend_365d` (15).

## Columns by group

The columns below are profiled on train. Missing % and unique counts for test are shown where they differ in a useful way.

### Monetary (current purchase)

| Column | Dtype | Missing % | Unique | Notes |
|---|---|---:|---:|---|
| `Total a Pagar` | float | 0.99 | 6,122 | Amount to pay. Range 0.93–22,297, median 477. |
| `Total Bruto` | float | 0.99 | 6,038 | Gross total |
| `Total Liquido` | float | 0.99 | 6,093 | Net total |
| `Total Imp_` | float | 0.99 | 5,407 | Tax |
| `Total Portes` | float | 0.99 | 215 | Shipping. Mostly 0. |
| `Total Desc_ Global` | float | 0.99 | 109 | Global discount. Mostly 0. |
| `Total Desc_ Linha` | float | 0.99 | 3,520 | Line discounts |
| `Custo` | float | 0.99 | 5,740 | Cost |

### Details (current purchase)

| Column | Dtype | Missing % | Unique | Notes |
|---|---|---:|---:|---|
| `N_ Detalhes` | float | 0.99 | 171 | Number of line items |
| `invoice_count_current_purchase` | float | 0.99 | 5 | Test has only 3 distinct values |
| `purchase_date` | str | 0.99 | 1,492 | ISO `YYYY-MM-DD`. Test: 223 unique dates. |

### Categoricals (current purchase)

| Column | Missing % | Train levels | Test levels | Test-only levels (rows) | Notes |
|---|---:|---:|---:|---|---|
| `payment_type` | 3.98 | 9 | 11 | 3 (3 rows) | Payment 7 = 70% of train |
| `salesperson` | 3.96 | 9 | 3 | 0 | Master 1 = 62%, Master 4 = 22% |
| `commercial_zone` | 0.99 | 5 | 5 | **2 (209 rows)** | Zone 4 = 96% of train. Zones 1 and 5 are test-only and cover 21% of test. |
| `transport_method` | 3.99 | 80 | 29 | 2 (63 rows) | High cardinality |
| `warehouse` | 3.98 | 2 | 2 | 1 (7 rows) | Warehouse 1 = 96% of train |
| `document_series` | 0.99 | 3 | 1 | 0 | Train split 2,738 / 1,903 / 1,828. Test is all Series 2. |

Encoders must handle unseen levels, `commercial_zone` above all.

### History (timing and value before this purchase)

| Column | Dtype | Missing % (train / test) | Unique | Notes |
|---|---|---:|---:|---|
| `days_since_previous_purchase` | float | 13.18 / 5.27 | 495 | 16 negatives |
| `customer_tenure_days` | float | 0.99 / 0 | 2,121 | |
| `prior_purchase_count` | float | 0.99 / 0 | 277 | 603 zeros in train, 22 in test |
| `prior_total_spend` | float | 0.99 / 0 | 5,864 | |
| `prior_mean_purchase_value` | float | 13.19 / 5.27 | 5,652 | 16 negatives |
| `prior_std_purchase_value` | float | 17.03 / 4.67 | 5,420 | |
| `prior_min_purchase_value` | float | 10.22 / 2.23 | 1,157 | |
| `prior_max_purchase_value` | float | 10.22 / 2.23 | 1,156 | |
| `prior_mean_gap_days` | float | 20.02 / 7.71 | 3,874 | 16 negatives |
| `prior_std_gap_days` | float | 22.62 / 7.00 | 4,911 | |

### Rolling frequency and spend

| Column | Dtype | Missing % (train / test) | Unique |
|---|---|---:|---:|
| `purchase_count_30d` | float | 0.99 / 0 | 12 |
| `spend_30d` | float | 0.99 / 0 | 2,105 |
| `purchase_count_90d` | float | 0.99 / 0 | 23 |
| `spend_90d` | float | 0.99 / 0 | 3,523 |
| `purchase_count_180d` | float | 0.99 / 0 | 38 |
| `spend_180d` | float | 0.99 / 0 | 5,021 |
| `purchase_count_365d` | float | 0.99 / 0 | 61 |
| `spend_365d` | float | 3.98 / 3.04 | 5,434 |

`spend_365d` has 15 negatives and follows the 3.98% categorical missing block, not the 0.99% one.

### Other

| Column | Dtype | Missing % | Unique | Notes |
|---|---|---:|---:|---|
| `ID` | int | 0 | 6,534 | Row key. Not a feature. |
| `customer_id` | int | 0 | 610 | Grouping variable and join key to `clientes` |
| `random_noise` | float | 0.99 | 6,469 | Pearson r with the target is −0.012 and with spend is 0.002. Looks like pure noise. Keep it as a sanity check for feature selection. |
| `repeat_purchase_90d` | int | 0 | 2 | Classification target |
| `spend_next_90d` | float | 47.58 | 3,355 | Regression target. **Never a feature.** |

## `clientes.csv`

| Column | Dtype | Missing % | Unique | Notes |
|---|---|---:|---:|---|
| `customer_id` | int | 0 | 1,143 | Key |
| `country_or_region` | str | 0 | 11 | |
| `commercial_region` | str | 9.97 | 37 | |
| `island` | str | 50.57 | 10 | |
| `municipality` | str | 13.91 | 183 | |
| `city` | str | 47.77 | 417 | |
| `postal_area` | str | 6.21 | 271 | |
| `postal_code` | str | 14.87 | 828 | Near-unique |
| `street` | str | 52.32 | 511 | |
| `address` | str | 0.52 | 1,120 | Near-unique, so effectively an ID |
| `customer_type` | float | **100** | 0 | Entirely empty |
| `company_name` | str | 0 | 1,141 | Near-unique, so effectively an ID |
| `customer_since_year` | int | 0 | 12 | The only clearly numeric field |

Values are anonymised strings with a code prefix, e.g. `Country CT11: ...` and `Postal code PC828: ...`.

## Open questions for the cleaning step

1. Drop the 65 all-NaN train rows? Each has a target but no features.
2. What should the regression do with the 229 returners that have no `spend_next_90d`?
3. Are the ~16 negative history values (and 15 negative `spend_365d`) errors to set to NaN, or a meaningful encoding?
4. Why do ~258 rows have `days_since_previous_purchase` NaN while `prior_purchase_count > 0`?
5. How should unseen test categories be handled, especially `commercial_zone`, where 21% of test is unseen?
6. Which `clientes` columns are usable? Likely `country_or_region`, `commercial_region` and `customer_since_year`. The near-unique ones are not, and `customer_type` is empty.
