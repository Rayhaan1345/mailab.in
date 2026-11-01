# Pima Indians Diabetes data (the clinic case)

`pima.json` contains the Pima Indians Diabetes dataset: 768 women of Pima heritage, all aged 21 or older, from a study near Phoenix, Arizona, in the United States. The file is a copy of `projects/diabetes-screening/primer/data.json` from Rayhaan's diabetes screening project, with the same 80/20 split that his XGBoost model uses.

This is a published research dataset of measurements on people, so it is the opposite of the made-up shark records. It does not come from Mumbai. The Mumbai clinic in the primer is a story built around it.

## Source

- Original owners: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), USA.
- Donor to the UCI Machine Learning Repository: Vincent Sigillito, Applied Physics Laboratory, Johns Hopkins University, received 9 May 1990. This is taken from the header of the UCI version of the file (read in an ARFF copy).
- First described in this paper: Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C. and Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. Proceedings of the Annual Symposium on Computer Application in Medical Care, pages 261 to 265. Open copy: https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/ (checked 31 Oct 2026).
- The UCI header says the women are at least 21 years old, of Pima Indian heritage, and live near Phoenix, Arizona. In that paper ADAP was trained on 576 women and tested on the other 192, with a sensitivity and specificity near 76%.

## What a row means

The label is 1 if the woman shows signs of diabetes by World Health Organization criteria: a 2-hour plasma glucose of at least 200 mg/dl at any survey visit, or a diagnosis found during routine care.

| Column | Meaning | Zeros in this file |
|---|---|---|
| Pregnancies | number of times pregnant | 111 (a valid value) |
| Glucose | plasma glucose 2 hours into an oral glucose tolerance test, mg/dl | 5 |
| BloodPressure | diastolic blood pressure, mm Hg | 35 |
| SkinThickness | triceps skin fold thickness, mm | 227 |
| Insulin | 2-hour serum insulin, mu U/ml | 374 |
| BMI | body mass index, kg per square metre | 11 |
| DiabetesPedigreeFunction | a score for diabetes in relatives | 0 |
| Age | years, 21 to 81 | 0 |

Zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are not possible measurements. They mean "not measured". The UCI header says there are no missing values, which is misleading for this reason. The zero counts above were counted in this file.

## Keys in `pima.json`

- `features`: the eight column names above, in order.
- `X`: 768 rows of eight numbers.
- `y`: 768 labels, 1 for diabetes (268 women, 34.9%) and 0 for no diabetes (500 women).
- `test`: 768 flags. A flag of 1 marks a woman in the held-out set (154 women, 55 with diabetes). A flag of 0 marks training (614 women, 213 with diabetes). The diabetes screening project describes this as an 80/20 split with random seed 42.
- `scale`: mean and standard deviation of Glucose and BMI, used for the figures.
- `depth`: accuracy against tree depth (train, held-out, and low and high bands over 50 splits, plus leaf counts), used by the overfitting figure.

## How chapters should label it

- Say that it is a published dataset of 768 women of Pima heritage, measured in the United States, and not data from Mumbai or from India.
- Do not present it as a picture of diabetes in any other group. The dataset shift lesson depends on this: a model trained on one population can fail on another.
- Say that zeros in five columns mean "not measured".
- Cite Smith et al. (1988) and the UCI repository where the data first appears on a page.
- Keep the clinic scenario clearly marked as a story. The sharks and the clinic are two different cases, and only the clinic data comes from people.
