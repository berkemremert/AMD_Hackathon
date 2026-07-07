# Schema Inventory

We examined the local TriviaQA dataset, specifically the extracted `qa/` folder from `triviaqa-rc.tar.gz`. The dataset variant is the **RC (Reading Comprehension)** set, and we are looking specifically at `wikipedia-dev.json` to extract Wikipedia-sourced evidence.

Top-level keys in the JSON file:
- `Data`: List of question-answer entries
- `Domain`: Source domain (e.g., Wikipedia)
- `Split`: Dataset split (e.g., dev, train)
- `VerifiedEval`: Boolean/string indicating verified evaluation
- `Version`: Version string

Here are the fields from 5 sample records from the `Data` list:

```json
[
  {
    "Answer": {
      "Aliases": [
        "Sunset Blvd",
        "West Sunset Boulevard",
        "Sunset Boulevard",
        "Sunset Bulevard",
        "Sunset Blvd."
      ],
      "MatchedWikiEntityName": "Sunset Boulevard",
      "NormalizedAliases": [
        "sunset boulevard",
        "sunset bulevard",
        "west sunset boulevard",
        "sunset blvd"
      ],
      "NormalizedMatchedWikiEntityName": "sunset boulevard",
      "NormalizedValue": "sunset boulevard",
      "Type": "WikipediaEntity",
      "Value": "Sunset Boulevard"
    },
    "EntityPages": [
      {
        "DocSource": "TagMe",
        "Filename": "Andrew_Lloyd_Webber.txt",
        "Title": "Andrew Lloyd Webber"
      }
    ],
    "Question": "Which Lloyd Webber musical premiered in the US on 10th December 1993?",
    "QuestionId": "tc_33",
    "QuestionSource": "http://www.triviacountry.com/"
  },
  {
    "Answer": {
      "Aliases": [
        "Sir Henry Campbell-Bannerman",
        "Campbell-Bannerman",
        "Campbell Bannerman",
        "Sir Henry Campbell Bannerman",
        "Henry Campbell Bannerman",
        "Henry Campbell-Bannerman"
      ],
      "MatchedWikiEntityName": "Henry Campbell-Bannerman",
      "NormalizedAliases": [
        "henry campbell bannerman",
        "sir henry campbell bannerman",
        "campbell bannerman"
      ],
      "NormalizedMatchedWikiEntityName": "henry campbell bannerman",
      "NormalizedValue": "campbell bannerman",
      "Type": "WikipediaEntity",
      "Value": "Campbell-Bannerman"
    },
    "EntityPages": [
      {
        "DocSource": "TagMe",
        "Filename": "Prime_Minister_of_the_United_Kingdom.txt",
        "Title": "Prime Minister of the United Kingdom"
      },
      {
        "DocSource": "TagMe",
        "Filename": "Arthur_Balfour.txt",
        "Title": "Arthur Balfour"
      }
    ],
    "Question": "Who was the next British Prime Minister after Arthur Balfour?",
    "QuestionId": "tc_40",
    "QuestionSource": "http://www.triviacountry.com/"
  },
  {
    "Answer": {
      "Aliases": [
        "Internal exile",
        "Exiles",
        "Transported for life",
        "Exile (politics and government)",
        "Voluntary exile",
        "Sent into exile",
        "Exile and Banishment",
        "Self-exile",
        "Forced exile",
        "Exile",
        "Exile in Greek tragedy",
        "Banish",
        "Banishment"
      ],
      "MatchedWikiEntityName": "Exile",
      "NormalizedAliases": [
        "exiles",
        "voluntary exile",
        "forced exile",
        "banish",
        "self exile",
        "exile politics and government",
        "exile in greek tragedy",
        "sent into exile",
        "banishment",
        "transported for life",
        "exile",
        "internal exile",
        "exile and banishment"
      ],
      "NormalizedMatchedWikiEntityName": "exile",
      "NormalizedValue": "exile",
      "Type": "WikipediaEntity",
      "Value": "Exile"
    },
    "EntityPages": [
      {
        "DocSource": "TagMe",
        "Filename": "Kiss_You_All_Over.txt",
        "Title": "Kiss You All Over"
      }
    ],
    "Question": "Who had a 70s No 1 hit with Kiss You All Over?",
    "QuestionId": "tc_49",
    "QuestionSource": "http://www.triviacountry.com/"
  },
  {
    "Answer": {
      "Aliases": [
        "Cancer pathology",
        "Deaths by cancer",
        "Anti-cancer",
        "Cancer (disease)",
        "Cancerophobia",
        "Malignant lesion",
        "Cancer medication",
        "Malignant tumors",
        "Cancer signs",
        "Malignant neoplasm",
        "Invasive (cancer)",
        "Malignant Neoplasms",
        "Malignant growth",
        "Sporadic cancer",
        "Malignant cancer",
        "Tumour virus",
        "Cancer en cuirasse",
        "Microtumor",
        "Malignant neoplasms",
        "Malignant tumour",
        "Carcinophobia",
        "Malignacy",
        "Cancer patient",
        "Epithelial cancers",
        "Solid cancer",
        "Cancers",
        "Tumor medication",
        "Malignant neoplastic disease",
        "AIDS-related cancer",
        "Invasive cancer",
        "Cancer therapy",
        "Cancerous tumor",
        "Cancer",
        "Financial toxicity",
        "Cancer diagnosis",
        "Cancer (medicine)",
        "Malignant tumor",
        "Cancerous",
        "Borderline (cancer)",
        "Signs of cancer",
        "Malignancies",
        "Cancer aromatase"
      ],
      "MatchedWikiEntityName": "Cancer",
      "NormalizedAliases": [
        "aids related cancer",
        "sporadic cancer",
        "cancer disease",
        "malignant tumors",
        "cancers",
        "carcinophobia",
        "cancer",
        "cancer diagnosis",
        "malignant neoplastic disease",
        "malignant neoplasm",
        "tumour virus",
        "cancer medicine",
        "deaths by cancer",
        "malignant tumour",
        "epithelial cancers",
        "solid cancer",
        "cancerous",
        "borderline cancer",
        "invasive cancer",
        "anti cancer",
        "cancer pathology",
        "cancer signs",
        "cancer aromatase",
        "cancer therapy",
        "financial toxicity",
        "cancerophobia",
        "cancer en cuirasse",
        "cancer patient",
        "cancerous tumor",
        "malignant cancer",
        "malignant neoplasms",
        "tumor medication",
        "signs of cancer",
        "malignacy",
        "malignant tumor",
        "cancer medication",
        "microtumor",
        "malignancies",
        "malignant lesion",
        "malignant growth"
      ],
      "NormalizedMatchedWikiEntityName": "cancer",
      "NormalizedValue": "cancer",
      "Type": "WikipediaEntity",
      "Value": "Cancer"
    },
    "EntityPages": [
      {
        "DocSource": "TagMe",
        "Filename": "Kathleen_Ferrier.txt",
        "Title": "Kathleen Ferrier"
      }
    ],
    "Question": "What claimed the life of singer Kathleen Ferrier?",
    "QuestionId": "tc_56",
    "QuestionSource": "http://www.triviacountry.com/"
  },
  {
    "Answer": {
      "Aliases": [
        "Bacall",
        "Lauren Becal",
        "Lauren Bacall",
        "Lauren Becall",
        "Betty J. Perske",
        "Loren Bacall",
        "Betty Joan Perske",
        "Betty Perske",
        "Betty Joan Perski"
      ],
      "MatchedWikiEntityName": "Lauren Bacall",
      "NormalizedAliases": [
        "lauren becall",
        "loren bacall",
        "lauren becal",
        "lauren bacall",
        "betty j perske",
        "betty perske",
        "betty joan perske",
        "bacall",
        "betty joan perski"
      ],
      "NormalizedMatchedWikiEntityName": "lauren bacall",
      "NormalizedValue": "lauren bacall",
      "Type": "WikipediaEntity",
      "Value": "Lauren Bacall"
    },
    "EntityPages": [
      {
        "DocSource": "Search",
        "Filename": "Lauren_Bacall.txt",
        "Title": "Lauren Bacall",
        "originalUrl": "https://www.wikipedia.org/search-redirect.php?family=wikipedia&search=bacall&language=en&go=++%E2%86%92++&go=Go"
      }
    ],
    "Question": "Which actress was voted Miss Greenwich Village in 1942?",
    "QuestionId": "tc_106",
    "QuestionSource": "http://www.triviacountry.com/"
  }
]
```
