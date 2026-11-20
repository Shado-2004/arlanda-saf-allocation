# ADS-B track data attribution

The position, altitude and speed observations are attributed to **adsb.lol contributors**, using their 17 November 2026 `readsb-prod-0` archive in the replay format produced by **milleleaf/world-flights**. Airborne stretches were selected by matching registrations, recorded times and both airport positions. The track summary and waypoint weather table use this subset of the public track database.

- [adsb.lol archive for 17 November 2026](https://github.com/adsblol/globe_history_2026/releases/tag/v2026.11.17-planes-readsb-prod-0)
- [milleleaf dataset and replay schema](https://huggingface.co/datasets/milleleaf/world-flights)
- [Persistent mirror of the selected day](https://www.kaggle.com/datasets/helenkwok/world-flights-2026-11-17)
- [Open Data Commons Open Database Licence v1.0](https://opendatacommons.org/licenses/odbl/1-0/)

The selected track database and its derived tables are provided under ODbL 1.0. The replay source rounds altitudes and flags ground separately. The screening calculation treats published airborne altitude as pressure altitude because the replay omits the geometric-altitude flag. Original points can be retrieved from the cited source.
