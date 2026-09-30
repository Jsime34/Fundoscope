"""Orchestration.

Combines the pure domain modules with the adapters (db, sentinel): e.g. "analyze a parcel" =
check the DB cache, fetch missing Sentinel-2 dates, compute campaigns, classify, flag, store.
Used by both the CLI and the API worker so the logic lives in one place. Weeks 3-10.
"""
