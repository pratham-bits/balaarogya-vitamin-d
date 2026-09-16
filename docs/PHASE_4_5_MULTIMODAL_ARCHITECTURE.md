
# Phase 4.5 — Multimodal Feature Integration

## Objective

Define a modular feature-integration architecture for Vitamin-D risk
screening that combines structured child/growth information,
sun/nutrition/supplementation information, and smartphone-derived
optical features.

The integration is designed for future Indian paired data containing
questionnaire + growth + smartphone image + laboratory 25(OH)D.

---

## Modalities

### 1. Profile and Growth

Candidate fields:

- age_months
- sex
- height_cm
- weight_kg
- bmi
- height_for_age_z
- weight_for_age_z
- weight_for_height_z
- bmi_for_age_z

These fields belong to the future Indian multimodal model contract.
Their inclusion in the final predictive model will be determined
empirically.

---

### 2. Sun Exposure, Nutrition and Supplementation

Candidate fields:

#### Sun exposure

- outdoor_time_bucket
- outdoor_frequency
- typical_outdoor_time_of_day
- clothing_coverage

#### Nutrition

- diet_type
- dietary_diversity
- vitamin_d_rich_food_frequency
- egg_consumption
- dairy_consumption
- fortified_food_consumption

#### Breastfeeding / maternal context

- breastfeeding_status
- breastfeeding_duration_months
- maternal_sun_exposure

#### Supplementation

- vitamin_d_supplement_use
- supplement_frequency
- recent_supplement_use

Only features with a defensible individual-level relationship to
Vitamin-D status will be retained in the final model.

---

### 3. Optical

The optical modality uses the frozen 18-feature contract:

1. r_mean
2. g_mean
3. b_mean
4. r_std
5. g_std
6. b_std
7. r_g_ratio
8. r_b_ratio
9. g_b_ratio
10. h_mean
11. s_mean
12. v_mean
13. l_mean
14. a_mean
15. b_lab_mean
16. l_std
17. a_std
18. b_lab_std

These are computational candidate optical features and are not
considered clinically validated Vitamin-D biomarkers.

---

## Fusion Strategy

The initial prototype will use feature-level fusion.

Conceptually:

X_multimodal =
    [X_profile_growth,
     X_behavior_nutrition,
     X_optical]

The modality boundaries will remain explicit in the implementation
so that modality-specific ablation experiments can be performed.

---

## Planned Ablation Experiments

1. Profile/Growth only
2. Sun/Nutrition/Supplementation only
3. Optical only
4. Profile/Growth + Sun/Nutrition/Supplementation
5. Profile/Growth + Optical
6. Sun/Nutrition/Supplementation + Optical
7. All modalities

The purpose is to determine whether optical information provides
incremental predictive value beyond structured information.

---

## Missing Optical Data

An unusable or unavailable optical assessment must not automatically
remove the child from the dataset.

Optical availability must remain explicitly represented.

Examples:

- profile/growth available + questionnaire available + optical available
- profile/growth available + questionnaire available + optical unavailable

The integration layer must preserve this distinction.

---

## Model Development Strategy

The initial multimodal implementation will use conventional
machine-learning models rather than a deep multimodal neural network.

Reason:

- current NHANES methodological dataset is small
- no paired Indian smartphone-image + laboratory dataset currently exists
- deep multimodal architectures would be difficult to justify with
  the current sample size

The architecture should remain extensible to future neural or
representation-learning approaches once sufficiently large paired
Indian data are collected.

---

## Scientific Constraint

The system is a screening/risk-estimation prototype.

Optical features must not be interpreted as direct measurements
of Vitamin-D concentration.

The laboratory serum 25(OH)D value remains the ground-truth target
for future paired-data validation.
