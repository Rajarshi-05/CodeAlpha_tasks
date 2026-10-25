# Data Dictionary - Titanic passenger data

**Source:** the widely used Titanic passenger dataset (the 891-passenger training split popularised by Kaggle's
"Titanic - Machine Learning from Disaster" competition). The copy used here was downloaded from the public
`datasciencedojo/datasets` repository on GitHub. The file is `titanic.csv` (891 rows x 12 columns).

| Variable | Description | Type | Notes |
|---|---|---|---|
| `PassengerId` | Row identifier | ID | Unique, carries no information |
| `Survived` | Survival outcome (0 = did not survive, 1 = survived) | Binary (target) | 342 survived, 549 did not |
| `Pclass` | Ticket class (1 = upper, 2 = middle, 3 = lower) | Ordinal | Proxy for socio-economic status |
| `Name` | Passenger name including title (Mr, Mrs, Miss, ...) | Text | Title can be extracted |
| `Sex` | Sex | Nominal | `male` / `female` |
| `Age` | Age in years | Continuous | 177 missing; fractional for infants |
| `SibSp` | Number of siblings / spouses aboard | Discrete count | |
| `Parch` | Number of parents / children aboard | Discrete count | |
| `Ticket` | Ticket number | Text | Shared by passengers travelling together |
| `Fare` | Passenger fare paid | Continuous | Strongly right-skewed; 15 zero values |
| `Cabin` | Cabin number | Text | 687 missing (77%); first letter = deck |
| `Embarked` | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) | Nominal | 2 missing |

## Engineered features (created in the notebook)

| Feature | Definition |
|---|---|
| `Title` / `TitleGroup` | Extracted from `Name`; Mlle/Ms -> Miss, Mme -> Mrs; titles with < 10 holders grouped as `Rare` |
| `FamilySize` | `SibSp + Parch + 1` |
| `FamilyGroup` | Alone (1), Small (2-4), Large (5+) |
| `AgeGroup` | Child (<=12), Teen (13-18), Adult (19-35), Middle (36-60), Senior (61+) |
| `HasCabin` / `Deck` | Whether a cabin is recorded / first letter of the cabin (`Unknown` if missing) |
| `TicketGroupSize` | Number of passengers sharing the same ticket |
| `Fare_log` | `log(1 + Fare)` to reduce skew |
| `Age_was_missing` | 1 if `Age` was imputed |
