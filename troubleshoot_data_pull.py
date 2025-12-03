"""
Troubleshoot pybaseball data access issues
"""
import pandas as pd
from pybaseball import playerid_lookup, statcast_pitcher, cache

# Enable caching
cache.enable()

print("="*80)
print("TROUBLESHOOTING DATA ACCESS")
print("="*80)

# Try different spellings for Jhoan Duran
print("\n1. Testing Jhoan Duran player lookup...")
print("-" * 60)

# Try variations
variations = [
    ('duran', 'jhoan'),
    ('duran', 'johan'),
    ('duran', 'juan'),
]

duran_id = None
for last, first in variations:
    print(f"Trying: {first} {last}")
    result = playerid_lookup(last, first)
    if len(result) > 0:
        print(f"✓ Found! Results:")
        print(result)
        # Look for recent players
        recent = result[result['mlb_played_last'] >= 2023.0]
        if len(recent) > 0:
            duran_id = recent.iloc[0]['key_mlbam']
            print(f"✓ Using MLB ID: {duran_id}")
            break
    else:
        print(f"  No results for {first} {last}")

if duran_id is None:
    print("\n⚠️  Could not find Jhoan Duran. Will proceed with Skubal only.")

# Skubal ID from earlier lookup
skubal_id = 669373

print("\n2. Testing data pull with 2024 season data...")
print("-" * 60)
print("Note: Using 2024 data as 2025 may not be fully available\n")

# Try pulling a small date range from 2024
test_start = '2024-04-01'
test_end = '2024-04-07'

print(f"Testing Skubal data pull ({test_start} to {test_end})...")
try:
    skubal_test = statcast_pitcher(test_start, test_end, skubal_id)
    if skubal_test is not None and len(skubal_test) > 0:
        print(f"✓ SUCCESS! Retrieved {len(skubal_test)} pitches")
        print(f"  Columns: {len(skubal_test.columns)}")
        print(f"  Date range: {skubal_test['game_date'].min()} to {skubal_test['game_date'].max()}")

        # Save sample
        skubal_test.to_csv('data/skubal_sample_2024.csv', index=False)
        print(f"  Saved to: data/skubal_sample_2024.csv")
    else:
        print("❌ No data returned")
except Exception as e:
    print(f"❌ Error: {e}")

if duran_id:
    print(f"\nTesting Duran data pull ({test_start} to {test_end})...")
    try:
        duran_test = statcast_pitcher(test_start, test_end, duran_id)
        if duran_test is not None and len(duran_test) > 0:
            print(f"✓ SUCCESS! Retrieved {len(duran_test)} pitches")
            print(f"  Columns: {len(duran_test.columns)}")
            print(f"  Date range: {duran_test['game_date'].min()} to {duran_test['game_date'].max()}")

            # Save sample
            duran_test.to_csv('data/duran_sample_2024.csv', index=False)
            print(f"  Saved to: data/duran_sample_2024.csv")
        else:
            print("❌ No data returned")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "="*80)
print("TROUBLESHOOTING COMPLETE")
print("="*80)
