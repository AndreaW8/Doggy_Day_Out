# Doggy Day Out: Evaluating the Effectiveness on Dog Adoption

## About <a name = "about"></a>
Animal shelters face a range of challenges when placing animals in permanent homes, including limited funding, space constraints, the need to screen potential adopters, and the individual characteristics of each animal. Increasing the adoption rate and reducing the length of stay are both practical goals and ethical responsibilities. To support these objectives, the Williamson County Regional Animal Shelter in Georgetown, TX, implemented a “Doggy Day Out” (DDO) program in 2023. This initiative allows members of the public to take a dog on a one-day outing, giving the animal a chance to socialize in a more typical environment such as a home or park.

This project evaluates whether participation in the DDO program is associated with improved adoption outcomes.  This approach combines feature selection and classification methods to identify key factors and evaluate the program’s effectiveness.

Learn more about the Doggy Day Out program:
https://www.wilcotx.gov/379/Doggy-Day-Out

How do you DDO?
https://www.youtube.com/watch?v=Vbfd7KTQ4BU

## Project Overview

- **Goal:** Evaluate whether Doggy Day Out (DDO) improves dog adoption outcomes.
- **Methods:** Data cleaning, feature engineering, classification models.
- **Results:** No statistically significant association found between DDO participation and adoption as an outcome
- **Usefulness of Results:** This allowed the shelter to embrace that DDO is a community engagement program and better know where it fits in to support its mission. DDO is a way to enable the community to engage with the shelter and support unhoused pets in the shelter's care.

## Getting Started <a name = "getting_started"></a>

### Installing

You can set up the required environment using either Conda or pip.

#### **Option 1: Using Conda**
1. Ensure you have **Conda** installed. If not, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).
2. Navigate to the root directory of the project:
   ```sh
   cd path/to/project
   ```
3. Activate the environment:
   ```sh
   conda activate DDO_env
   ```
4. Install the requirements from requirements.txt:
   ```sh
   pip install -r requirements.txt
   ```

#### **Option 2: Using pip**
1. Ensure you have **Python 3.8+** installed.
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

End with an example of getting some data out of the system or using it for a little demo.

## Key Scripts

| Script                          | Description |
|----------------------------------|-------------|
| `DDO_data_cleaning.py` | Cleans and preprocesses shelter data from multiple CSV files (Dog Intakes, Outcomes, DDO). Merges records as needed for analysis.<br><br>**Output Data files:**<br>- One row per animal ID per stay:<br>  &nbsp;&nbsp;-- `Output__Animal_ID_per_stay_n_ddo_cnt_df.csv`<br>  &nbsp;&nbsp;-- `Output__Animal_ID_per_stay_n_ddo_cnt_NO_PUPPIES_df.csv`<br>- One row per animal ID with cumulative sum of shelter stay time:<br>  &nbsp;&nbsp;-- `Output___per_Animal_ID_n_ddo_cnt_df.csv`<br>  &nbsp;&nbsp;-- `Output___per_Animal_ID_n_ddo_cnt_NO_PUPPIES_df.csv` |
| `DDO_dog_adoption_prediction_ML.py` | Builds and evaluates ML models (Random Forest, SVM) using features prepared from cleaning. Includes feature selection, PCA, model training, cross-validation, and visualization of results. |                                                                           |

## Data

- **Intakes:** Status of animals during shelter intake.
- **Outcomes:** Status and outcome when animals leave the shelter.
- **Dog Day Out:** Log of dogs that participated in DDO.

## Example Usage

```sh
# 1. Run data cleaning to prepare features and target columns
python DDO_data_cleaning.py

# 2. Run the ML pipeline to train models and evaluate results
python DDO_dog_adoption_prediction_ML.py
```


