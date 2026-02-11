```python
class Player:
    def __init__(self, country_of_origin, gdp, age, hdi, oced, gpi, wgi, epi):
        self.country_of_origin = country_of_origin
        self.gdp = gdp
        self.age = age
        self.hdi = hdi
        self.OCED = oced
        self.GPI = gpi
        self.WGI = wgi
        self.EPI = epi

    def get_migration_score(self):
        # Combine metrics to calculate a migration score (e.g., average of all metrics)
        return (self.hdi + self.OCED + self.GPI + self.WGI + self.EPI) / 5
```
**Step 2: Define ease-of-migration country**

Create a dictionary or database that maps countries to their ease-of-migration scores. This can be based on various factors such as visa requirements, language proficiency, job market conditions, etc.
```python
ease_of_migration = {
    'Country A': 0.8,
    'Country B': 0.9,
    'Country C': 0.7
}
```
**Step 3: Simulate player migration**

Use a for-loop to create instances of the `Player` class and simulate their migration decisions.
```python
import random

players = []
for i in range(3000):
    country_of_origin = random.choice(['Country A', 'Country B', 'Country C'])
    gdp = random.uniform(10000, 20000)
    age = random.randint(20, 50)
    hdi = random.uniform(0.5, 1.0)
    oced = random.uniform(0.6, 1.0)
    gpi = random.uniform(0.7, 1.0)
    wgi = random.uniform(0.8, 1.0)
    epi = random.uniform(0.9, 1.0)

    player = Player(country_of_origin, gdp, age, hdi, oced, gpi, wgi, epi)
    players.append(player)

# Simulate migration
for player in players:
    country_to_migrate_to = random.choice(list(ease_of_migration.keys()))
    if player.get_migration_score() > ease_of_migration[country_to_migrate_to]:
        print(f"Player from {player.country_of_origin} migrates to {country_to_migrate_to}")
```
**Step 4: Plot migration patterns**

Plot the simulated migration patterns on a timeline.
```python
import matplotlib.pyplot as plt

# Get player migration data
migration_data = []
for i, player in enumerate(players):
    country_to_migrate_to = random.choice(list(ease_of_migration.keys()))
    if player.get_migration_score() > ease_of_migration[country_to_migrate_to]:
        migration_data.append((i, player.country_of_origin, country_to_migrate_to))

# Plot migration patterns
plt.plot([data[0] for data in migration_data], [ease_of_migration[data[2]] for data in migration_data])
plt.xlabel('Time')
plt.ylabel('Ease of Migration')
plt.show()
```
