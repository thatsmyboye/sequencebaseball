"""
Comprehensive Data Import Troubleshooting Script
Tests multiple approaches to access Statcast data
"""
import pandas as pd
from pybaseball import statcast, statcast_pitcher, playerid_lookup, cache
from datetime import datetime
import requests

print("="*80)
print("STATCAST DATA IMPORT TROUBLESHOOTING")
print("="*80)

# Enable caching
cache.enable()

# Test 1: Check internet connectivity
print("\n1. TESTING INTERNET CONNECTIVITY")
print("-" * 60)
test_urls = [
    ('Google', 'https://www.google.com'),
    ('MLB.com', 'https://www.mlb.com'),
    ('Baseball Savant', 'https://baseballsavant.mlb.com'),
]

for name, url in test_urls:
    try:
        response = requests.get(url, timeout=5)
        print(f"✓ {name:20s} - Status {response.status_code}")
    except requests.exceptions.ProxyError as e:
        print(f"❌ {name:20s} - Proxy Error: {str(e)[:60]}...")
    except requests.exceptions.ConnectionError as e:
        print(f"❌ {name:20s} - Connection Error: {str(e)[:60]}...")
    except Exception as e:
        print(f"❌ {name:20s} - {type(e).__name__}: {str(e)[:60]}...")

# Test 2: Player ID lookup variations
print("\n2. TESTING PLAYER ID LOOKUPS")
print("-" * 60)

# Skubal (we know this works)
print("\nA. Tarik Skubal:")
try:
    skubal = playerid_lookup('skubal', 'tarik')
    if len(skubal) > 0:
        print(f"✓ Found: {skubal['name_first'].values[0]} {skubal['name_last'].values[0]}")
        print(f"  MLB ID: {skubal['key_mlbam'].values[0]}")
        print(f"  Active: {skubal['mlb_played_first'].values[0]} - {skubal['mlb_played_last'].values[0]}")
    else:
        print("❌ Not found")
except Exception as e:
    print(f"❌ Error: {e}")

# Duran - try multiple variations
print("\nB. Jhoan Duran (multiple attempts):")
duran_variations = [
    ('duran', 'jhoan'),
    ('duran', 'johan'),
    ('duran', 'juan'),
    ('duran', 'j'),  # Just first initial
]

duran_found = False
for last, first in duran_variations:
    try:
        result = playerid_lookup(last, first)
        if len(result) > 0:
            print(f"✓ Found with '{first} {last}':")
            for idx, row in result.iterrows():
                print(f"  - {row['name_first']} {row['name_last']}")
                print(f"    MLB ID: {row['key_mlbam']}")
                print(f"    Active: {row['mlb_played_first']} - {row['mlb_played_last']}")
                # Check if this is our pitcher (recent player)
                if row['mlb_played_last'] >= 2023.0:
                    print(f"    ⭐ Likely candidate (recent player)")
                    duran_found = True
        else:
            print(f"  '{first} {last}' - No results")
    except Exception as e:
        print(f"  '{first} {last}' - Error: {e}")

if not duran_found:
    print("\n  Trying broader search...")
    try:
        # Search for all Durans
        all_durans = playerid_lookup('duran', '')
        print(f"  Found {len(all_durans)} players with last name 'Duran'")
        if len(all_durans) > 0:
            # Filter for recent players
            recent = all_durans[all_durans['mlb_played_last'] >= 2020.0]
            print(f"  Recent players (2020+):")
            for idx, row in recent.iterrows():
                print(f"    - {row['name_first']} {row['name_last']} (ID: {row['key_mlbam']})")
                print(f"      Active: {row['mlb_played_first']} - {row['mlb_played_last']}")
    except Exception as e:
        print(f"  Broad search error: {e}")

# Test 3: Try different data pull methods
print("\n3. TESTING DATA RETRIEVAL METHODS")
print("-" * 60)

skubal_id = 669373  # We know this ID

# Method 1: statcast_pitcher (what we've been using)
print("\nA. Method: statcast_pitcher() with date range")
try:
    test_data = statcast_pitcher('2024-04-01', '2024-04-03', skubal_id)
    if test_data is not None and len(test_data) > 0:
        print(f"✓ Success! Retrieved {len(test_data)} pitches")
        print(f"  Columns: {len(test_data.columns)}")
        print(f"  Sample pitch types: {test_data['pitch_name'].value_counts().head(3).to_dict()}")
    else:
        print("❌ No data returned")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}")
    print(f"   {str(e)[:100]}")

# Method 2: statcast general (no pitcher filter)
print("\nB. Method: statcast() with date range (all pitchers)")
try:
    test_data = statcast('2024-04-01', '2024-04-01')  # Just one day
    if test_data is not None and len(test_data) > 0:
        print(f"✓ Success! Retrieved {len(test_data)} total pitches")
        # Filter for Skubal
        skubal_subset = test_data[test_data['pitcher'] == skubal_id]
        print(f"  Skubal pitches in dataset: {len(skubal_subset)}")
    else:
        print("❌ No data returned")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}")
    print(f"   {str(e)[:100]}")

# Method 3: Try with environment variable to bypass proxy
print("\nC. Method: Attempt proxy bypass")
import os
original_proxies = {}
for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    original_proxies[var] = os.environ.get(var)
    if var in os.environ:
        del os.environ[var]

try:
    test_data = statcast_pitcher('2024-04-01', '2024-04-03', skubal_id)
    if test_data is not None and len(test_data) > 0:
        print(f"✓ Success with proxy bypass! Retrieved {len(test_data)} pitches")
    else:
        print("❌ Still no data with proxy bypass")
except Exception as e:
    print(f"❌ Error even with proxy bypass: {type(e).__name__}")

# Restore proxy settings
for var, value in original_proxies.items():
    if value is not None:
        os.environ[var] = value

# Test 4: Check pybaseball cache
print("\n4. CHECKING PYBASEBALL CACHE")
print("-" * 60)
try:
    from pybaseball.cache import config
    print(f"Cache enabled: {config.enabled}")
    print(f"Cache directory: {config.cache_directory}")

    # Check if cache directory exists and has files
    import pathlib
    cache_path = pathlib.Path(config.cache_directory)
    if cache_path.exists():
        cache_files = list(cache_path.glob('*'))
        print(f"Cached files: {len(cache_files)}")
        if cache_files:
            print("Recent cache files:")
            for f in sorted(cache_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                size_mb = f.stat().st_size / (1024*1024)
                print(f"  - {f.name} ({size_mb:.2f} MB)")
    else:
        print("Cache directory does not exist")
except Exception as e:
    print(f"Error checking cache: {e}")

# Summary and recommendations
print("\n" + "="*80)
print("TROUBLESHOOTING SUMMARY")
print("="*80)

print("\n📊 FINDINGS:")
print("-" * 60)

print("\n1. Network Access:")
print("   If Baseball Savant showed a 403/Proxy error:")
print("   → Network restrictions are blocking MLB data access")
print("   → This is likely a firewall or corporate proxy")

print("\n2. Player Lookup:")
print("   Skubal: Should work (MLB ID: 669373)")
print("   Duran: May require exact spelling or broader search")

print("\n3. Recommended Solutions:")
print("   A. Use VPN or different network (if available)")
print("   B. Use cached data if available")
print("   C. Use sample data generator (already implemented)")
print("   D. Download CSV files manually from Baseball Savant website")

print("\n4. Manual Download Instructions:")
print("   a. Visit: https://baseballsavant.mlb.com/statcast_search")
print("   b. Set filters:")
print("      - Pitcher Name: Tarik Skubal (or Jhoan Duran)")
print("      - Season: 2024 (or 2025 when available)")
print("      - Game Type: Regular Season")
print("   c. Click 'Download CSV'")
print("   d. Save to data/ directory as:")
print("      - data/skubal_statcast_2024.csv")
print("      - data/duran_statcast_2024.csv")

print("\n5. Alternative: Use Sample Data")
print("   → Sample data generator already created realistic data")
print("   → Run: python generate_sample_data.py")
print("   → All visualization functions work with sample data")

print("\n" + "="*80)
print("Would you like to:")
print("  1. Continue with sample data (recommended)")
print("  2. Try manual CSV download")
print("  3. Attempt alternate network configuration")
print("="*80)
