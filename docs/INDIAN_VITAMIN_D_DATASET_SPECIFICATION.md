
# BalAarogya Indian Vitamin-D Dataset Specification

## 1. Purpose

This document defines the data structure for the future Indian paired
Vitamin-D dataset used by BalAarogya.

The eventual dataset is intended to combine:

- child profile/context
- growth measurements
- sun-exposure information
- nutrition and feeding information
- breastfeeding / early-life factors
- supplementation
- smartphone optical assessment
- laboratory serum 25(OH)D ground truth

This specification defines data collection and model-development structure.
It does not define a clinical diagnostic rule.

---

## 2. Core distinction

### Model inputs

Information available before prediction:

- child profile
- growth
- sun exposure
- nutrition
- breastfeeding / early-life factors
- supplementation
- relevant context
- smartphone optical features

### Ground truth

Laboratory-measured serum:

`25(OH)D`

stored as:

`lab_25ohd_nmol_l`

The laboratory value is NOT a normal user input during screening.

### Model output

The deployed model may produce:

- predicted 25(OH)D
- risk probability
- risk category
- uncertainty
- screening-oriented recommendation

Only validated outputs and decision rules will be exposed as clinical
claims.

---

## 3. Evidence classification

Each variable will be classified as one of:

- CNNS_CHILD_LEVEL
  Available from the underlying CNNS child-level dataset.
- CNNS_REPORTED
  Reported in the CNNS publication/report, but not necessarily available
  as an individual-level variable from the published material.
- BMC_REPORTED
  Used or discussed in the BMC Pediatrics analysis.
- BALAAAROGYA_COLLECTED
  Information that our application should collect prospectively.
- OPTICAL
  Smartphone/image-derived information produced by our optical pipeline.
- GROUND_TRUTH
  Laboratory measurement used as the prediction target.

---

## 4. Child profile and context

| Variable              | Classification                           | Role                     |
| --------------------- | ---------------------------------------- | ------------------------ |
| age_months            | CNNS_CHILD_LEVEL / BALAAAROGYA_COLLECTED | Model input              |
| sex                   | CNNS_CHILD_LEVEL / BALAAAROGYA_COLLECTED | Model input              |
| state                 | CNNS_CHILD_LEVEL / BALAAAROGYA_COLLECTED | Context/model candidate  |
| district              | BALAAAROGYA_COLLECTED                    | Context                  |
| residence_type        | CNNS_CHILD_LEVEL / BALAAAROGYA_COLLECTED | Model candidate          |
| season                | CNNS_CHILD_LEVEL / BALAAAROGYA_COLLECTED | Model candidate          |
| socioeconomic context | CNNS_CHILD_LEVEL / BMC_REPORTED          | Research/model candidate |

Socioeconomic and demographic variables must not automatically be included
in the final predictive model merely because an association was observed.
Their usefulness, fairness implications and deployment suitability must be
evaluated separately.

---

## 5. Growth

| Variable            | Classification                           | Role                   |
| ------------------- | ---------------------------------------- | ---------------------- |
| height_cm           | CNNS_CHILD_LEVEL / BALAAAROGYA_COLLECTED | Model input            |
| weight_kg           | CNNS_CHILD_LEVEL / BALAAAROGYA_COLLECTED | Model input            |
| bmi                 | Derived                                  | Model candidate        |
| height_for_age_z    | Derived                                  | Future model candidate |
| weight_for_age_z    | Derived                                  | Future model candidate |
| weight_for_height_z | Derived                                  | Future model candidate |
| bmi_for_age_z       | Derived                                  | Future model candidate |

WHO-standardized z-scores must not be fabricated. The reference standard and
calculation method must be frozen and tested before these features are used.

---

## 6. Sun exposure

The CNNS/BMC material does not provide individual-level duration of sunlight
exposure sufficient for direct reconstruction.

Therefore these variables will be collected prospectively by BalAarogya:

| Variable                    | Classification        | Role                     |
| --------------------------- | --------------------- | ------------------------ |
| outdoor_time_bucket         | BALAAAROGYA_COLLECTED | Model candidate          |
| outdoor_frequency           | BALAAAROGYA_COLLECTED | Model candidate          |
| typical_outdoor_time_of_day | BALAAAROGYA_COLLECTED | Model candidate          |
| clothing_coverage           | BALAAAROGYA_COLLECTED | Model candidate          |
| sun-avoidant behaviour      | BALAAAROGYA_COLLECTED | Research/model candidate |

These variables must not be represented as variables directly obtained from
CNNS.

---

## 7. Nutrition

CNNS provides information relevant to child feeding, food consumption and
dietary/nutritional patterns.

BalAarogya will collect a simplified structured questionnaire suitable for
community screening:

| Variable                      | Classification                        | Role            |
| ----------------------------- | ------------------------------------- | --------------- |
| diet_type                     | CNNS-informed / BALAAAROGYA_COLLECTED | Model candidate |
| dietary_diversity             | CNNS-informed / BALAAAROGYA_COLLECTED | Model candidate |
| vitamin_d_rich_food_frequency | BALAAAROGYA_COLLECTED                 | Model candidate |
| egg_consumption               | BALAAAROGYA_COLLECTED                 | Model candidate |
| dairy_consumption             | BALAAAROGYA_COLLECTED                 | Model candidate |
| fortified_food_consumption    | BALAAAROGYA_COLLECTED                 | Model candidate |

The application questionnaire must not claim that every simplified category
is an exact CNNS variable.

---

## 8. Breastfeeding and early-life factors

CNNS includes age-specific infant and young-child feeding information,
including breastfeeding and complementary-feeding-related information.

BalAarogya will retain:

| Variable                          | Classification                        | Role                     |
| --------------------------------- | ------------------------------------- | ------------------------ |
| breastfeeding_status              | CNNS-informed / BALAAAROGYA_COLLECTED | Model candidate          |
| breastfeeding_duration_months     | CNNS-informed / BALAAAROGYA_COLLECTED | Model candidate          |
| maternal_sun_exposure             | BALAAAROGYA_COLLECTED                 | Research/model candidate |
| complementary_feeding information | CNNS-informed / BALAAAROGYA_COLLECTED | Research/model candidate |

---

## 9. Supplementation

| Variable                 | Classification                        | Role            |
| ------------------------ | ------------------------------------- | --------------- |
| vitamin_d_supplement_use | CNNS-informed / BALAAAROGYA_COLLECTED | Model candidate |
| supplement_frequency     | BALAAAROGYA_COLLECTED                 | Model candidate |
| recent_supplement_use    | BALAAAROGYA_COLLECTED                 | Model candidate |

Supplementation variables must clearly distinguish current/recent use from
historical use.

---

## 10. Optical assessment

Smartphone optical information is BalAarogya's novel modality.

Pipeline:

image
→ image quality assessment
→ skin-region detection
→ normalization/calibration
→ optical feature extraction

Candidate features currently include:

- RGB statistics
- HSV statistics
- LAB statistics
- derived color ratios/statistics

These are computational image features.

They must NOT be described as independently validated Vitamin-D biomarkers.

---

## 11. Laboratory ground truth

For future paired Indian participants:

`lab_25ohd_nmol_l`

will represent laboratory-measured serum 25(OH)D.

Additional metadata:

- lab_assay_method
- lab_sample_date

The laboratory measurement is the prediction target and is not required from
the ordinary screening user.

---

## 12. Paired-record requirement

A research participant used for optical/model development should ideally
have, from the same assessment episode:

- child profile
- growth
- relevant questionnaire data
- smartphone image
- laboratory 25(OH)D

This pairing is essential for training and validating the eventual
multimodal model.

---

## 13. What must NOT be done

We must not:

1. Create individual laboratory values from published prevalence tables.
2. Convert population-level odds ratios into individual labels.
3. Assign regional mean 25(OH)D values to individual children.
4. Treat CNNS aggregate results as paired optical data.
5. Claim CNNS contains sunlight-duration measurements when it does not.
6. Use optical skin colour/features as a standalone diagnostic test.
7. Hard-code a clinical risk category without an appropriate validated
   decision framework.
8. Present the current NHANES development model as an Indian clinical model.

---

## 14. Dataset development stages

### Stage A — Methodological development

Existing NHANES data:

structured features
→ development baseline
→ software/inference validation

This is not Indian validation.

### Stage B — Indian structured dataset

Obtain legitimate individual-level Indian data where permitted and evaluate
structured predictors against laboratory 25(OH)D.

### Stage C — Indian paired optical dataset

Prospectively collect:

structured information

+ smartphone image
+ laboratory 25(OH)D

from the same child/assessment episode.

### Stage D — Multimodal model

Train and internally validate:

structured features + optical features
→ laboratory 25(OH)D

### Stage E — Independent validation

Evaluate the frozen model on an independent Indian cohort.

---

## 15. Source references

Primary references for this specification:

1. Comprehensive National Nutrition Survey (CNNS), 2016–18,
   Micronutrients chapter.
2. Mustafa, M. & Shekhar, C. (2021),
   BMC Pediatrics analysis of Vitamin-D deficiency using CNNS data.

The CNNS report states that serum 25(OH)D was used to assess Vitamin-D
status and reports Vitamin-D deficiency prevalence by age group and
population characteristics.

The BMC Pediatrics analysis uses CNNS data to examine demographic,
socioeconomic and geographic correlates of serum 25(OH)D.

The BMC analysis also identifies the absence of information on time spent
in sunlight as an important limitation of the available CNNS data.
