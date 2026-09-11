from nhanes_loader import load_xpt


def main() -> None:

    print("=" * 70)
    print("NHANES CENTRALIZED LOADER TEST")
    print("=" * 70)

    dbq = load_xpt(
        "DBQ_J.XPT",
        ["SEQN", "DBQ197"],
    )

    dr1 = load_xpt(
        "DR1TOT_J.XPT",
        ["SEQN", "DR1TVD"],
    )

    ds1 = load_xpt(
        "DS1TOT_J.XPT",
        ["SEQN", "DS1DS", "DS1DSCNT", "DS1TVD"],
    )

    print("\n--- DBQ197 ---")
    print(
        dbq["DBQ197"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\n--- DR1TVD ---")
    print(
        dr1["DR1TVD"]
        .describe()
    )

    print("\n--- DS1DS ---")
    print(
        ds1["DS1DS"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\n--- DS1DSCNT ---")
    print(
        ds1["DS1DSCNT"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\n--- DS1TVD ---")
    print(
        ds1["DS1TVD"]
        .describe()
    )

    print("\n" + "=" * 70)
    print("LOADER TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()