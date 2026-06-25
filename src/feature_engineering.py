from __future__ import annotations

import pandas as pd


def add_house_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if {"YearBuilt", "YearRemodAdd"}.issubset(enriched.columns):
        enriched["HouseAgeAtRemodel"] = enriched["YearRemodAdd"] - enriched["YearBuilt"]
    if {"LivingArea", "Bedrooms"}.issubset(enriched.columns):
        enriched["AreaPerBedroom"] = enriched["LivingArea"] / enriched["Bedrooms"].replace(0, 1)
    elif {"GrLivArea", "BedroomAbvGr"}.issubset(enriched.columns):
        enriched["AreaPerBedroom"] = enriched["GrLivArea"] / enriched["BedroomAbvGr"].replace(0, 1)
    if {"GarageArea", "GarageCars"}.issubset(enriched.columns):
        enriched["GarageAreaPerCar"] = enriched["GarageArea"] / enriched["GarageCars"].replace(0, 1)
    return enriched
