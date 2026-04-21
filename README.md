# Doggy Day Out (DDO): Evaluating Effectiveness on Dog Adoption

## About <a name = "about"></a>
Animal shelters face a range of challenges when placing animals in permanent homes, including limited funding, space constraints, the need to screen potential adopters, and the individual characteristics of each animal. Increasing the adoption rate and reducing the length of stay are both practical goals and ethical responsibilities. To support these objectives, the Williamson County Regional Animal Shelter in Georgetown, TX, implemented a “Doggy Day Out” (DDO) program in 2023. This initiative allows members of the public to take a dog on a one-day outing, giving the animal a chance to socialize in a more typical environment such as a home or park.

This project evaluates whether participation in the DDO program is associated with improved adoption outcomes. This approach combines feature selection and classification methods to identify key factors and evaluate the program’s effectiveness.

[Learn more about the Doggy Day Out program](https://www.wilcotx.gov/379/Doggy-Day-Out)

[How do you DDO?](https://www.youtube.com/watch?v=Vbfd7KTQ4BU)

[Read my LinkedIn article: How Machine Learning Helped Me Rethink the Impact of a Doggy Day Out Program in Animal Shelters](https://www.linkedin.com/pulse/how-machine-learning-helped-me-rethink-impact-doggy-day-wackerle-vbegc)

## Project Overview

- **Goal:** Evaluate whether Doggy Day Out (DDO) improves dog adoption outcomes.
- **Methods:** Data cleaning, feature engineering to transform raw data into model ready features, mutual information feature selection, and classification models (SVM, Random Forest).
- **Performance Evaluation:** Models were evaluated on a held out test set using accuracy and F1 score. F1 macro was used as the primary metric because the classes are imbalanced.

## Results

- No statistically significant association was found between DDO participation and adoption as an outcome.
- **Usefulness of Results:** This allowed the shelter to embrace DDO as a community engagement program and better understand where it fits in supporting its mission. DDO is a way to enable the community to engage with the shelter and support unhoused pets in its care.

### DDO Feature Behavior
- Adding DDO participation as a feature did not meaningfully improve model performance, indicating that simply going on a DDO outing was not a strong standalone predictor of adoption. More traditional factors like length of stay, age, and size were shown to be key drivers of adoption outcomes. 

- In the top 20 features ranked by mutual information, the DDO indicator appears near the bottom, suggesting limited marginal relevance for predicting adoption.
  <img width="928" height="556" alt="DDO_top_20_MI" src="https://github.com/user-attachments/assets/def189c9-04c2-4e62-8661-6d81c366077d" />

- In the Random Forest model trained on all features, DDO is ranked third in importance, but its importance score is much smaller than that of the top two features. The top 15 features are shown here.
  <img width="928" height="560" alt="Random Forest feature importance with DDO" src="https://github.com/user-attachments/assets/11073891-5800-4b55-be42-aa28bfbc3d63" />

- For SVM, the DDO feature does not contribute meaningfully with a coefficient close to 0. This reinforces that the effect was small or indistinguishable from noise in this dataset.
  <img width="928" height="560" alt="SVM_DDO_top_20_MI" src="https://github.com/user-attachments/assets/d42ae374-9367-4ee9-9d9f-227d98835e2b" />

### Interpretation for the shelter

- The absence of a strong statistical link between DDO and adoption suggests the program should be used as a **welfare and outreach** initiative rather than a guaranteed adoption booster.

## Data

- **Intakes:** Status of the dogs during shelter intake.
- **Outcomes:** Status and outcome when the dogs leave the shelter.
- **Doggy Day Out (DDO):** Log of dogs that participated in DDO.

## Key Scripts

| Script                          | Description |
|----------------------------------|-------------|
| `DDO_data_cleaning.py` | Cleans and preprocesses shelter data from multiple CSV files (Intakes, Outcomes, DDO). Merges records as needed for analysis.<br><br>**Output Data files:**<br>- One row per animal ID per stay:<br>  &nbsp;&nbsp;-- `Output__Animal_ID_per_stay_n_ddo_cnt_df.csv`<br>  &nbsp;&nbsp;-- `Output__Animal_ID_per_stay_n_ddo_cnt_NO_PUPPIES_df.csv`<br>- One row per animal ID with cumulative sum of shelter stay time:<br>  &nbsp;&nbsp;-- `Output___per_Animal_ID_n_ddo_cnt_df.csv`<br>  &nbsp;&nbsp;-- `Output___per_Animal_ID_n_ddo_cnt_NO_PUPPIES_df.csv` |
| `DDO_dog_adoption_prediction_ML.py` | Builds and evaluates ML models (Random Forest, SVM) using features prepared from cleaning. Includes feature selection, PCA, model training, cross-validation, and visualization of results. |                                                                           |

## Getting Started <a name = "getting_started"></a>

### Installing

You can set up the required environment using either Conda or pip.

#### **Option 1: Using Conda**
1. Ensure you have **Conda** installed. If not, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).
2. Navigate to the root directory of the project:
   ```sh
   cd path/to/project
   ```
3. Create the environment:
   ```sh
   conda create -n DDO_env python=3.12
   ```
4. Activate the environment:
   ```sh
   conda activate DDO_env
   ```
5. Install the requirements from requirements.txt:
   ```sh
   pip install -r requirements.txt
   ```



#### **Option 2: Using pip**
1. Ensure you have **Python 3.12+** installed.
2. Navigate to the root directory of the project:
   ```sh
   cd path/to/project
   ```
3. Create a virtual environment:
   ```sh
   python -m venv venv
   ```
4. Activate the virtual environment:
   - **Windows**:
     ```sh
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```sh
     source venv/bin/activate
     ```
5. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Example Usage

```sh
# 1. Run data cleaning to prepare features and target columns
python DDO_data_cleaning.py

# 2. Run the ML pipeline to train models and evaluate results
python DDO_dog_adoption_prediction_ML.py
```


