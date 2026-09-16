# Phase 3R.3 — Age & Growth Representation

## Frozen product representation

Canonical child-level fields:

- `age_months`
- `sex`
- `height_cm`
- `weight_kg`

## Derived growth representation

Future WHO-compatible standardized features:

- `height_for_age_z`
- `weight_for_age_z`
- `weight_for_height_z`
- `bmi_for_age_z`

These are intentionally placeholders until the exact WHO reference tables,
age/measurement applicability, and calculation method are frozen.

## Current NHANES development dataset

The existing NHANES baseline remains unchanged:

- `RIDAGEYR`
- `RIAGENDR`
- `BMXWT`

No current NHANES columns are renamed or silently converted.

## Architecture

`Child Profile → Shared Growth Representation → Milestone + Vitamin-D`

The shared child profile prevents the Milestone and Vitamin-D modules from
using conflicting age/growth representations.

## Safety rule

This implementation performs data validation and representation only.
It does not produce a clinical growth classification or Vitamin-D decision.
