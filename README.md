# To Coup or Not to Coup?

## Overview
While many are fortunate enough to live in a nation with a stable government that (generally) works for its people, recent news headlines from places such as Egypt, Turkey, or Venezuela, make clear that not everyone is so lucky. Such circumstances can lead to extreme, often violent actions to overthrow the regime in power.  

Imagine if we could identify nations most at risk of coup attempts, then intervene before such catastrophic events ever occur. To this end, the goal of this project is to utilize supervised learning techniques to classify whether a sovereign nation will experience a coup attempt within the next year.

## Project Organization
    ├── README.md              <- The top-level README for developers using this project
    │
    ├── LICENSE                <- Copyright and permissive license
    │
    ├── Notebooks              <- Project source code in Jupyter notebook
    │   └── common7.py         <- Script of helper functions
    │
    ├── Reports                <- Various reports
    │   ├── proposal.pdf       <- Project proposal
    │   ├── summary.docx       <- Project summary
    │   └── presentation3.pptx <- Project presentation slide deck
    │
    ├── Images                 <- Images featured in presentation slide deck   

## Data
The data for this project was obtained from several sources, including:
* OEF Research’s REIGN (Rulers, Elections, and Irregular Governance) database
  * When and where coup attempts occurred
  * Whether a leader’s legitimacy stemmed from a career in the military
  * Number of years a leader has been in power
  * Whether an executive election is anticipated within the next few month
* World Bank’s World Development Indicators database
  * Life expectancy
  * Percentage of the population with access to basic resources such as food, water, sanitation, electricity, healthcare, and education
  * Government finances, such as tax revenues, overall revenue, reserves, natural resources rents, military spending, and debt forgiveness
* Penn World Table
  * Expenditure-side real GDP
  * Human capital index

Due to the scarcity of historic global development data, I decided to limit my observations years to 1990 to 2016, leaving me with a total of 5,138 observations.  

## Modeling
Throughout this project, I trained a suite of supervised learning models including:
* K-Nearest Neighbors
* Logistic Regression
* Decision Trees
* Random Forests
* Gradient Boosting
* Bagging

The Area Under the Curve (AUC) was used as the evaluation metric for each iteration of the above models.   

## LICENSE
MIT © Brooke Ann Coco

