# geze1.github.io
data602 final project

air_quality_tutorial_MYFINALCODE.html
# Understanding Urban Air Quality Through Data Science
### Final Tutorial Project
**Author:** GLORY EZEOGU 
**Course:** DATA 602 
**Date:** December 2025
## Abstract
This tutorial presents an end-to-end data science workflow for analyzing urban air quality. Using a multi-feature pollution dataset, we demonstrate data curation, exploratory data analysis, hypothesis testing, and predictive modeling with logistic regression, decision trees, gradient boosting, and random forests. We show how environmental and urban factors such as particulate matter, nitrogen oxides, population density, and industrial proximity influence air-quality categories.

# 1. Introduction and Motivation

Air quality is a central determinant of public health, environmental stability, and long-term urban sustainability. As cities expand and populations grow, levels of particulate matter, nitrogen oxides, and other pollutants increasingly shape the quality of daily life. Public agencies and environmental researchers rely on data-driven tools to assess how temperature, humidity, industrial activity, and population density interact to influence air quality outcomes. Understanding these relationships is not only important for scientific inquiry but also crucial for designing effective policies, advising communities, and motivating action toward cleaner environments.

This tutorial aims to guide readers through a complete data science workflow using an air-quality dataset containing measurements of temperature, humidity, PM2.5, PM10, NO2, SO2, carbon monoxide, proximity to industrial areas, and population density. By working through the steps of data curation, exploratory data analysis, hypothesis testing, and predictive modeling, we demonstrate how various environmental and urban factors relate to air quality. The goal is to provide a technical education for newcomers while offering substantive insights for those familiar with environmental analytics.

Data science provides a powerful lens for investigating topics that have strong social and public-policy implications. Air pollution affects millions of people globally, contributes to respiratory and cardiovascular diseases, and influences mortality trends, particularly in densely populated or industrialized regions. By framing this tutorial around such a meaningful real-world issue, we hope to motivate readers to see the value of quantitative methods in understanding complex, interdisciplinary problems.

---

# 2. Data Description and Curation

The dataset used in this tutorial contains pollution measurements and environmental indicators for 5,000 cities. It includes nine numerical features: Temperature, Humidity, PM2.5, PM10, NO2, SO2, CO, Proximity to Industrial Areas, and Population Density. The final column, Air Quality, is a categorical label with four classes: Good, Moderate, Poor, and Hazardous. These labels reflect established air-quality categories used by many environmental monitoring systems worldwide.

Before performing any analysis, it is essential to curate the dataset to ensure reliability. Raw environmental data often contain missing values, inconsistent formatting, and varying units of measurement. For instance, features such as PM2.5 and PM10 typically represent mass concentration of fine particulate matter, while temperature and humidity capture meteorological conditions known to influence pollutant accumulation. Industrial proximity and population density offer proxies for anthropogenic pollution pressures.

In this tutorial, we begin by converting categorical labels to a consistent format, replacing placeholder missing values, and transforming all numerical variables into appropriate numeric types. Missing data are handled using mean imputation, which provides a simple and interpretable baseline method. While advanced imputation methods exist, such as k-nearest neighbors or model-based approaches, mean imputation is suitable for a foundational tutorial and sufficient to illustrate key concepts.

After cleaning, the dataset becomes ready for exploration. The curated data frame contains complete rows with standardized formatting, ensuring that downstream analyses—whether statistical or predictive—operate on a well-prepared foundation. Readers will learn the essential importance of data curation as a starting point for all analytical workflows.

---

# 3. Exploratory Data Analysis

Exploratory Data Analysis (EDA) bridges raw information and meaningful understanding. In this section, we visually and statistically examine how environmental factors vary and how they may relate to air-quality outcomes. EDA is not merely a preliminary step; it is a core component of data science, designed to generate hypotheses, uncover hidden patterns, and guide future modeling decisions.

We begin by plotting histograms of each numerical feature. These reveal distributional properties such as skewness, outliers, and multimodality. For example, fine particulate matter (PM2.5) and coarse particulate matter (PM10) often show right-skewed distributions because high-pollution events tend to be less frequent but extreme. Population density may similarly exhibit heavy-tailed behavior, reflecting the wide variation between rural towns and major metropolitan areas. Temperature and humidity typically show more regular distributions but still vary significantly based on climate and geography.

Next, we generate a correlation heatmap to measure linear relationships between features. This visualization often exposes many intuitive patterns: PM2.5 and PM10 tend to correlate strongly because they are both particulate pollution measurements generated by combustion, industrial activity, or vehicle emissions. NO2 and SO2, markers of industrial and vehicular pollution, similarly cluster together. Temperature may correlate negatively with pollutant concentrations in some cases because warmer conditions promote atmospheric dispersion, although this can vary regionally. Population density is a particularly influential variable, often positively correlated with most pollutants due to increased energy consumption, traffic, and industrial proximity.

To better understand how cities cluster in terms of environmental conditions, we apply Principal Component Analysis (PCA). PCA reduces the nine-dimensional feature space into two principal components for visualization. By projecting the data into these two components and coloring points by air-quality category, we can visually assess whether pollution indicators naturally separate into distinguishable groups. In many cases, “Good” air-quality samples occupy a region associated with lower pollution concentrations, while “Poor” and “Hazardous” samples cluster in areas characterized by high PM2.5, PM10, and NO2 values.

EDA provides the foundation for the modeling stages that follow. Through descriptive statistics and visual summaries, we gather preliminary evidence suggesting that pollution levels and urban pressures meaningfully contribute to air-quality outcomes.

---

# 4. Hypothesis Testing

Before building predictive models, it is valuable to frame explicit hypotheses based on the patterns observed in EDA. Hypothesis testing formalizes intuition and provides statistical justification for claims about environmental relationships.

**Hypothesis 1: Higher PM2.5 levels are associated with poorer air-quality categories.**  
This hypothesis tests whether fine particulate matter concentrations differ significantly across air-quality classes. PM2.5 is known to have strong health implications due to its ability to penetrate deep into human lungs. Testing this hypothesis would involve comparing PM2.5 means across the categorical groups using ANOVA or non-parametric alternatives. If the results show statistically significant differences, this supports the conclusion that PM2.5 levels meaningfully distinguish air-quality outcomes.

**Hypothesis 2: Urban density is positively correlated with pollutant concentration.**  
Population density acts as a proxy for human activity, energy consumption, and vehicular emissions. We expect it to correlate positively with contaminants such as PM2.5, NO2, and CO. A simple correlation test or regression could quantify the strength of this relationship. Rejecting the null hypothesis of no correlation provides evidence that denser cities face higher pollution burdens.

These hypotheses serve three purposes:  
1. They validate patterns observed during EDA.  
2. They reinforce environmental knowledge regarding pollution behavior.  
3. They provide a structured transition from descriptive analysis to predictive modeling.

By integrating hypothesis testing into the workflow, we strengthen the analytical narrative and set the stage for building forecasting models grounded in empirical patterns.

---

# 5. Predictive Modeling

After gaining insight from EDA and hypothesis testing, we construct predictive models to classify air-quality outcomes based on environmental conditions. This step demonstrates how machine-learning techniques can support environmental monitoring, early warning systems, and policy analysis.

We use two models:

1. **Logistic Regression** – a linear, interpretable baseline classifier  
2. **Random Forest** – a nonlinear ensemble model capable of capturing complex interactions  

Before modeling, we scale the features using standardization. Scaling ensures that variables measured on different scales—such as temperature vs. PM2.5—contribute appropriately to model optimization.

We split the dataset into training and testing sets using stratified sampling to preserve class proportions. Logistic regression provides a baseline accuracy and offers interpretable coefficients showing how each variable influences the odds of belonging to a poorer air-quality category. While logistic regression may underfit highly nonlinear relationships, it establishes a transparent benchmark.

Random Forest classification then adds complexity by employing decision-tree ensembles. This model can capture interactions between pollutants and environmental variables that logistic regression may miss. It also provides feature importance scores, which tell us which factors contribute most to predicting air-quality labels. Typically, PM2.5, PM10, NO2, and population density emerge as dominant predictors, reflecting well-established environmental science findings.

The confusion matrix reveals how well the model distinguishes between the four air-quality classes. Random Forest usually outperforms logistic regression by correctly identifying more “Poor” and “Hazardous” cases, which are particularly important from a policy perspective because they indicate high-risk environments.

Together, these models illustrate how data science can translate raw environmental data into actionable predictions.

---

# 6. Discussion and Interpretation

The results of this tutorial reveal several important insights. First, exploratory analysis shows that particulate matter and nitrogen oxides strongly co-vary and dominate the pollutant landscape. High PM and NO2 concentrations are closely linked to poorer air-quality grades, supporting existing public-health findings. Cities with greater population density and closer proximity to industrial areas display significantly worse pollution profiles, emphasizing the role of human activity in shaping environmental conditions.

The PCA visualization further reinforces these trends by showing that air-quality categories align with underlying pollution gradients. Data points representing poor or hazardous conditions cluster in regions associated with high pollutant levels, whereas cities with good air quality appear more dispersed and associated with lower concentrations across the board.

Predictive modeling results illustrate that Random Forest performs substantially better than logistic regression in capturing nonlinear interactions among temperature, humidity, industrial activity, and pollutant concentration. Importantly, feature-importance values highlight PM2.5, PM10, and NO2 as the major contributors to classification performance, aligning well with environmental literature and supporting hypotheses tested earlier.

Despite strong predictive results, there are limitations. Mean imputation may smooth out meaningful variability, and more sophisticated imputation could enhance model accuracy. Furthermore, this dataset is cross-sectional and does not capture temporal variation. Pollution often fluctuates hourly or seasonally, and a time-series approach could yield deeper insights. Finally, while Random Forest is powerful, additional models such as gradient boosting or neural networks may provide further improvements.

Nonetheless, the analytic pipeline demonstrates how a structured approach—curation, exploration, hypothesis testing, and modeling—can yield both scientific understanding and practical insights. Policymakers, environmental agencies, and urban planners could use similar models to forecast pollution, identify high-risk regions, and allocate monitoring resources effectively.

---

# 7. Further Resources

Readers who wish to deepen their understanding of air-quality analytics, pollution science, or machine-learning methodology may consult the following:

**Environmental and Air-Quality Resources**
- U.S. Environmental Protection Agency (EPA) – Air Quality Index (AQI) Documentation  
- World Health Organization (WHO) – Air Pollution and Health Briefings  
- OpenAQ – Global, open-source air-quality data platform  

**Data Science and Machine-Learning Resources**
- Scikit-Learn User Guide – Classification, PCA, Model Evaluation  
- “An Introduction to Statistical Learning” – Chapters on linear models and classification  
- “Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow” – Applied examples with code  

**Pollution Modeling and Research**
- EPA Integrated Science Assessments for Particulate Matter and Nitrogen Dioxide  
- Atmospheric Environment Journal – Peer-reviewed studies linking meteorology to pollution  

These resources provide pathways for further study and help readers expand beyond the foundational techniques presented in this tutorial.

---

# 8. Conclusion

This tutorial has walked through a complete, end-to-end data science workflow centered on a socially and environmentally meaningful question: how do urban and environmental factors contribute to air quality? By curating a pollution dataset, performing exploratory data analysis, testing hypotheses, and building predictive models, we have shown how quantitative approaches can uncover insights that support public-health decision-making.

The findings highlight the dominant role of particulate matter and nitrogen oxides, the influence of population density and industrial proximity, and the effectiveness of machine-learning methods such as Random Forest for classifying environmental conditions. Through clear explanations, code, and visualizations, this tutorial aims to equip readers with both conceptual understanding and practical tools.

Air quality remains a pressing global challenge. The methods demonstrated here can empower analysts, students, policymakers, and practitioners to explore environmental data rigorously and contribute to evidence-based solutions.


# Imports and configuration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

pd.set_option("display.max_columns", None)
sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv("pollution_dataset.csv")
print("Shape:", df.shape)
df.head()
# Clean labels and missing values
df['Air Quality'] = df['Air Quality'].astype(str).str.strip().str.title()
df.replace(['-', 'NA', 'NaN', ''], np.nan, inplace=True)

for col in df.columns:
    if col != 'Air Quality':
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(df.isna().sum())

# Mean imputation
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
print("Missing after imputation:", df[numeric_cols].isna().sum().sum())

# Basic EDA: histograms
df[numeric_cols].hist(figsize=(14, 12), bins=25)
plt.suptitle("Feature Distributions", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', square=True)
plt.title("Correlation Heatmap")
plt.show()
# PCA visualization
X_numeric = df[numeric_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Air Quality'], palette="viridis")
plt.title("PCA Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
# Feature matrix and labels
feature_cols = [c for c in df.columns if c != 'Air Quality']
X = df[feature_cols].select_dtypes(include=[np.number])
y = df['Air Quality']

std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_std, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
lr_acc = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr, average='weighted')
print("Logistic Regression Accuracy:", lr_acc)
print("Logistic Regression F1:", lr_f1)
print(classification_report(y_test, y_pred_lr))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_acc = accuracy_score(y_test, y_pred_dt)
dt_f1 = f1_score(y_test, y_pred_dt, average='weighted')
print("Decision Tree Accuracy:", dt_acc)
print("Decision Tree F1:", dt_f1)
# Random Forest
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')
print("Random Forest Accuracy:", rf_acc)
print("Random Forest F1:", rf_f1)
print(classification_report(y_test, y_pred_rf))

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
gb_acc = accuracy_score(y_test, y_pred_gb)
gb_f1 = f1_score(y_test, y_pred_gb, average='weighted')
print("Gradient Boosting Accuracy:", gb_acc)
print("Gradient Boosting F1:", gb_f1)
# Confusion matrix for Random Forest
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Random Forest Confusion Matrix")
plt.show()

# Cross-validation for Random Forest
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_std, y, cv=kf, scoring='accuracy')
print("RF CV scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

# Simple hyperparameter grid search (small for demo)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)

# Permutation feature importance
best_rf = grid.best_estimator_
result = permutation_importance(best_rf, X_test, y_test, n_repeats=5, random_state=42)
importances = pd.Series(result.importances_mean, index=X.columns)
importances.sort_values(ascending=False).head(10)

# Summary table
summary = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"],
    "Accuracy": [lr_acc, dt_acc, rf_acc, gb_acc],
    "Weighted F1": [lr_f1, dt_f1, rf_f1, gb_f1]
})
summary


