DATASET INFORMATION

| Variable | Dataset | Meaning               | Unit     | Coding  | Missing handling | Role             |
| -------- | ------- | --------------------- | -------- | ------- | ---------------- | ---------------- |
| RIDAGEYR | DEMO    | Age                   | years    | numeric | —               | Primary          |
| RIAGENDR | DEMO    | Sex                   | —       | 1/2     | —               | Primary          |
| BMXWT    | BMX     | Weight                | kg       | numeric | impute           | Primary          |
| DBQ197   | DBQ     | Milk frequency        | category | TBD     | TBD              | Primary          |
| DR1TVD   | DR1TOT  | Dietary Vitamin D     | µg      | numeric | TBD              | Secondary        |
| DS1DS    | DS1TOT  | Supplement use        | —       | 1/2     | TBD              | Secondary        |
| DS1DSCNT | DS1TOT  | Number of supplements | count    | numeric | TBD              | Secondary        |
| LBXVIDMS | VID     | Total 25(OH)D         | nmol/L   | numeric | none             | **Target** |
