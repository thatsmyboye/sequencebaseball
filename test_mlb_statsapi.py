"""
Test MLB-StatsAPI as alternative data source
Official MLB Stats API - may bypass proxy restrictions
"""
import statsapi
import pandas as pd
from datetime import datetime, timedelta

print("="*80)
print("TESTING MLB-STATSAPI (Official MLB Stats API)")
print("="*80)

# Test 1: Check API connectivity
print("\n1. TESTING API CONNECTIVITY")
print("-" * 60)

try:
    # Get current season info (simple API call)
    meta = statsapi.meta('seasons')
    print(f"✓ API is accessible!")
    print(f"  Available seasons: {len(meta)} seasons in database")
    print(f"  Latest season: {meta[-1] if meta else 'N/A'}")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}")
    print(f"   {str(e)[:100]}")

# Test 2: Look up Tarik Skubal
print("\n2. LOOKING UP TARIK SKUBAL")
print("-" * 60)

try:
    # Search for Skubal
    skubal_search = statsapi.lookup_player('skubal')

    if skubal_search:
        print(f"✓ Found {len(skubal_search)} result(s):")
        for player in skubal_search:
            print(f"\n  Name: {player['fullName']}")
            print(f"  ID: {player['id']}")
            print(f"  Position: {player.get('primaryPosition', {}).get('name', 'N/A')}")
            print(f"  Team: {player.get('currentTeam', {}).get('name', 'N/A')}")
            print(f"  Active: {player.get('active', 'N/A')}")

            skubal_id = player['id']
    else:
        print("❌ No results found")
        skubal_id = 669373  # Known ID
except Exception as e:
    print(f"❌ Error: {e}")
    skubal_id = 669373

# Test 3: Look up Jhoan Duran
print("\n3. LOOKING UP JHOAN DURAN")
print("-" * 60)

try:
    # Try different variations
    for name in ['duran', 'jhoan duran', 'johan duran']:
        print(f"\nSearching for '{name}'...")
        result = statsapi.lookup_player(name)

        if result:
            # Filter for pitchers named Duran
            for player in result:
                if 'duran' in player['fullName'].lower():
                    print(f"  Found: {player['fullName']}")
                    print(f"    ID: {player['id']}")
                    print(f"    Position: {player.get('primaryPosition', {}).get('name', 'N/A')}")
                    print(f"    Team: {player.get('currentTeam', {}).get('name', 'N/A')}")
        else:
            print(f"  No results")

except Exception as e:
    print(f"❌ Error: {e}")

# Test 4: Get game schedule for pitcher
print("\n4. TESTING GAME DATA ACCESS")
print("-" * 60)

try:
    # Get recent games for Tigers (Skubal's team)
    # Using a specific date range in 2024
    schedule = statsapi.schedule(
        start_date='04/01/2024',
        end_date='04/07/2024',
        team=116  # Detroit Tigers team ID
    )

    if schedule:
        print(f"✓ Retrieved {len(schedule)} games for Tigers (Apr 1-7, 2024)")

        # Show first game
        if schedule:
            game = schedule[0]
            print(f"\n  Sample game:")
            print(f"    Game ID: {game['game_id']}")
            print(f"    Date: {game['game_date']}")
            print(f"    Matchup: {game['summary']}")
            print(f"    Status: {game['status']}")

            # Try to get play-by-play data
            game_id = game['game_id']
            print(f"\n  Attempting to get pitch data for game {game_id}...")

            try:
                # Get play-by-play
                plays = statsapi.get('game_playByPlay', {'gamePk': game_id})

                if 'allPlays' in plays:
                    print(f"✓ Retrieved play-by-play data!")
                    print(f"  Total plays: {len(plays['allPlays'])}")

                    # Count pitches
                    total_pitches = 0
                    for play in plays['allPlays']:
                        if 'playEvents' in play:
                            total_pitches += len(play['playEvents'])

                    print(f"  Total pitch events: {total_pitches}")

                    # Show sample pitch data
                    if plays['allPlays'] and 'playEvents' in plays['allPlays'][0]:
                        sample_pitch = plays['allPlays'][0]['playEvents'][0]
                        print(f"\n  Sample pitch data fields available:")
                        print(f"    {list(sample_pitch.keys())}")

                        if 'pitchData' in sample_pitch:
                            print(f"\n  Pitch metrics available:")
                            print(f"    {list(sample_pitch['pitchData'].keys())}")

            except Exception as e:
                print(f"  ⚠️  Play-by-play error: {type(e).__name__}")

    else:
        print("❌ No games found")

except Exception as e:
    print(f"❌ Error: {type(e).__name__}")
    print(f"   {str(e)[:150]}")

# Test 5: Direct pitch data extraction attempt
print("\n5. TESTING DIRECT PITCH DATA EXTRACTION")
print("-" * 60)

try:
    # Try to get a specific game with known pitcher appearance
    # This would require knowing game IDs where Skubal pitched

    print("Attempting to find games where Skubal pitched...")

    # Get Tigers schedule for a week
    games = statsapi.schedule(
        start_date='04/01/2024',
        end_date='04/30/2024',
        team=116
    )

    print(f"Found {len(games)} Tigers games in April 2024")

    if games:
        # Check first few games for Skubal
        for i, game in enumerate(games[:5]):
            game_id = game['game_id']

            try:
                # Get boxscore to see if Skubal pitched
                boxscore = statsapi.boxscore_data(game_id)

                # Check if Skubal in away or home pitchers
                away_pitchers = boxscore.get('away', {}).get('pitchers', [])
                home_pitchers = boxscore.get('home', {}).get('pitchers', [])

                all_pitcher_ids = away_pitchers + home_pitchers

                if skubal_id in all_pitcher_ids:
                    print(f"\n✓ Found Skubal in game {game_id}!")
                    print(f"  Date: {game['game_date']}")
                    print(f"  Matchup: {game['summary']}")

                    # This game has Skubal - we could extract his pitches
                    break

            except Exception as e:
                continue

except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\n📊 MLB-StatsAPI Assessment:")
print("-" * 60)

print("\n✅ Advantages:")
print("  • Official MLB API (may not be blocked)")
print("  • Includes player lookup")
print("  • Has game schedules and play-by-play data")
print("  • Provides pitch-level data")

print("\n⚠️  Considerations:")
print("  • Different data structure than Statcast")
print("  • Requires game-by-game extraction")
print("  • May not have all Statcast metrics (release point, movement)")
print("  • Needs conversion logic to match our data format")

print("\n🎯 Recommendation:")
print("  If API is accessible:")
print("    → Create converter to transform MLB-StatsAPI data to Statcast format")
print("    → Extract pitch data game-by-game for specific pitchers")
print("    → May have fewer advanced metrics than Statcast")
print("  ")
print("  If API is blocked:")
print("    → Continue with sample data (already working perfectly)")
print("    → Wait for unrestricted network access")

print("\n" + "="*80)
