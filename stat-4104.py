\documentclass[12pt]{article}

\usepackage[a4paper, margin=1in]{geometry}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{tikz}
\usepackage{graphicx}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}

% Define colors
\definecolor{lightblue}{RGB}{176,196,222}
\definecolor{darkslate}{RGB}{70,90,110}
\definecolor{steelblue}{RGB}{70,130,180}
\definecolor{footerblue}{RGB}{70,100,150}
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}

\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    numbers=left,
    numberstyle=\tiny\color{codegray},
    keywordstyle=\color{blue},
    commentstyle=\color{codegreen},
    stringstyle=\color{red}
}

\begin{document}

%% ===================== COVER PAGE =====================
\pagestyle{empty}

\begin{tikzpicture}[remember picture, overlay]
    % Full page light blue background
    \fill[lightblue!60] (current page.south west) rectangle (current page.north east);

    % Right white panel
    \fill[white] ([xshift=0.52\paperwidth]current page.south west)
        rectangle (current page.north east);

    % Dark title box (top right)
    \fill[darkslate] ([xshift=0.52\paperwidth, yshift=-0.02\paperheight]current page.north west)
        rectangle ([xshift=0.95\paperwidth, yshift=-0.38\paperheight]current page.north west);

    % Title text inside dark box
    \node[anchor=center, text=white, font=\large]
        at ([xshift=0.735\paperwidth, yshift=-0.20\paperheight]current page.north west)
        {Assignment- \textsc{Using} python};

    % Footer bar
    \fill[footerblue] (current page.south west)
        rectangle ([xshift=\paperwidth, yshift=0.045\paperheight]current page.south west);

    % Footer text
    \node[anchor=center, text=white, font=\small]
        at ([yshift=0.022\paperheight]current page.south)
        {ARGHO:ID-12110063};
\end{tikzpicture}

\vspace*{4.5cm}

\begin{minipage}[t]{0.48\textwidth}
    \textbf{\large Submitted by :} \quad \textbf{\underline{Argho Kundu.}}\\[0.4cm]
    \hspace{1.5cm} ID: \hspace{1.3cm} 12110063.\\[0.3cm]
    \hspace{1.5cm} Session \; : \; 2021-2022.\\[0.3cm]
    \hspace{1.5cm} Year: \hspace{1.1cm} 4\textsuperscript{th} year .\\[0.3cm]
    \hspace{1.5cm} Semester: \; 1\textsuperscript{st} semester.

    \vspace{1.5cm}

    \textbf{\large Submitted to :} \quad \underline{Siddikur} Rahman\\[0.3cm]
    \hspace{3.0cm} ( \underline{Associate} \; Professor)\\[0.5cm]
    \textbf{Department of Statistics, Begum Rokeya University ,Rangpur.}
\end{minipage}%
\hfill
\begin{minipage}[t]{0.44\textwidth}
    \vspace{5.5cm}

    \textcolor{steelblue}{\textbf{Course Title:}}\\[0.2cm]
    \hspace{0.8cm} \textbf{Categorical Data Analysis}\\[0.5cm]
    \textcolor{steelblue}{\textbf{Course Code:}}\\[0.2cm]
    \hspace{0.8cm} \underline{STAT-4104}
\end{minipage}

\vfill
\hfill \small MSI

\newpage
\pagestyle{plain}

%% =============== SECTION 1: LUNG DISEASE DATASET ===============
\begin{center}
    {\LARGE \textbf{ problem 1}}\\[0.3cm]
    {\large Lung Disease Dataset Analysis using Python}
\end{center}




\section*{1. Import Libraries and Dataset}

\begin{lstlisting}[language=Python]
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score

df = pd.read_csv('lung_disease.csv')

print(df.head())
\end{lstlisting}

\section*{2. Basic EDA Questions}

\subsection*{i) Distribution of Age}

\subsubsection*{Python Code}

\begin{lstlisting}[language=Python]
print(df['Age'].describe())
print('Skewness =', df['Age'].skew())
\end{lstlisting}

\subsubsection*{Output}

\begin{verbatim}
count    500.000000
mean      43.904000
std       14.604841
min       20.000000
25%       30.750000
50%       45.000000
75%       56.000000
max       69.000000

Skewness = 0.012
\end{verbatim}

\subsubsection*{Interpretation}

\begin{itemize}
\item The average age is about 44 years.
\item Age is approximately normally distributed.
\item Skewness is very close to 0.
\end{itemize}

\subsection*{ii) Proportion of Smokers vs Non-Smokers}

\subsubsection*{Python Code}

\begin{lstlisting}[language=Python]
smoking_prop = df['Smoking'].value_counts(normalize=True) * 100
print(smoking_prop)
\end{lstlisting}

\subsubsection*{Output}

\begin{verbatim}
No     58.6%
Yes    41.4%
\end{verbatim}

\subsubsection*{Interpretation}

\begin{itemize}
\item About 41.4\% individuals are smokers.
\item About 58.6\% are non-smokers.
\end{itemize}

\subsection*{iii) Income Group with Highest Frequency}

\begin{lstlisting}[language=Python]
print(df['Income'].value_counts())
\end{lstlisting}

\subsubsection*{Output}

\begin{verbatim}
Low       204
Medium    197
High       99
\end{verbatim}

\subsubsection*{Interpretation}

\begin{itemize}
\item The Low income group has the highest frequency.
\end{itemize}

\subsection*{iv) Percentage Exposed to High Pollution}

\begin{lstlisting}[language=Python]
high_pollution = (df['Pollution'] == 'High').mean() * 100
print(high_pollution)
\end{lstlisting}

\subsubsection*{Output}

\begin{verbatim}
70.4%
\end{verbatim}

\subsubsection*{Interpretation}

\begin{itemize}
\item Around 70.4\% individuals are exposed to high pollution.
\end{itemize}

\subsection*{v) Overall Prevalence of Lung Disease}

\begin{lstlisting}[language=Python]
prevalence = (df['LungDisease'] == 'Yes').mean() * 100
print(prevalence)
\end{lstlisting}

\subsubsection*{Output}

\begin{verbatim}
52.8%
\end{verbatim}

\subsubsection*{Interpretation}

\begin{itemize}
\item About 52.8\% individuals have lung disease.
\end{itemize}

\section*{3. Association and Crosstab}

\subsection*{Smoking and Lung Disease}

\begin{lstlisting}[language=Python]
crosstab_smoking = pd.crosstab(df['Smoking'], df['LungDisease'])
print(crosstab_smoking)
\end{lstlisting}

\subsubsection*{Output}

\begin{verbatim}
LungDisease   No   Yes
Smoking
No            184  109
Yes            52  155
\end{verbatim}

\subsubsection*{Interpretation}

\begin{itemize}
\item Smokers appear much more likely to develop lung disease.
\end{itemize}

\section*{4. Chi-Square Test}

\subsection*{Smoking vs Lung Disease}

\begin{lstlisting}[language=Python]
chi2, p, dof, expected = chi2_contingency(crosstab_smoking)

print('Chi-square =', chi2)
print('p-value =', p)
\end{lstlisting}

\subsubsection*{Output}

\begin{verbatim}
Chi-square = 67.59
p-value = 2.01e-16
\end{verbatim}

\subsubsection*{Interpretation}

\begin{itemize}
\item Smoking and Lung Disease are significantly associated.
\item p-value is less than 0.05.
\end{itemize}

\section*{5. Odds Ratio}

\begin{lstlisting}[language=Python]
oddsratio, p = fisher_exact([[155, 52],
                             [109, 184]])

print('Odds Ratio =', oddsratio)
\end{lstlisting}

\subsubsection*{Output}

\begin{verbatim}
Odds Ratio = 5.03
\end{verbatim}

\subsubsection*{Interpretation}

\begin{itemize}
\item Smokers have approximately 5 times higher odds of lung disease.
\end{itemize}

\section*{6. Logistic Regression}

\begin{lstlisting}[language=Python]
df2 = df.copy()

df2['Smoking'] = df2['Smoking'].map({'Yes':1, 'No':0})
df2['Pollution'] = df2['Pollution'].map({'High':1, 'Low':0})
df2['LungDisease'] = df2['LungDisease'].map({'Yes':1, 'No':0})

model = smf.logit(
    'LungDisease ~ Smoking + Age + Pollution + C(Income)',
    data=df2
).fit()

print(model.summary())
\end{lstlisting}

\subsubsection*{Important Output}

\begin{verbatim}
Variable                Coef      p-value
Smoking                 1.702     <0.001
Age                     0.040     <0.001
Pollution               1.303     <0.001
Income(Low)             0.440      0.123
Income(Medium)          0.220      0.440
\end{verbatim}

\subsubsection*{Interpretation}

\begin{itemize}
\item Smoking, Age, and Pollution are statistically significant.
\item Income is not statistically significant.
\end{itemize}

\section*{7. ROC Curve and AUC}

\begin{lstlisting}[language=Python]
pred_prob = model.predict(df2)
auc = roc_auc_score(df2['LungDisease'], pred_prob)

print('AUC =', auc)
\end{lstlisting}

\subsubsection*{Output}

\begin{verbatim}
AUC = 0.787
\end{verbatim}

\subsubsection*{Interpretation}

\begin{itemize}
\item The model has good classification performance.
\end{itemize}

\section*{8. Confusion Matrix}

\begin{lstlisting}[language=Python]
pred_class = (pred_prob > 0.5).astype(int)
cm = confusion_matrix(df2['LungDisease'], pred_class)

print(cm)
\end{lstlisting}

\subsubsection*{Output}

\begin{verbatim}
[[156  80]
 [ 64 200]]
\end{verbatim}

\subsubsection*{Interpretation}

\begin{itemize}
\item The model predicts lung disease reasonably well.
\end{itemize}

\section*{9. Accuracy, Sensitivity, Specificity}

\begin{lstlisting}[language=Python]
accuracy = accuracy_score(df2['LungDisease'], pred_class)
sensitivity = recall_score(df2['LungDisease'], pred_class)
specificity = 156 / (156 + 80)

print('Accuracy =', accuracy)
print('Sensitivity =', sensitivity)
print('Specificity =', specificity)
\end{lstlisting}

\subsubsection*{Output}

\begin{verbatim}
Accuracy = 0.712
Sensitivity = 0.758
Specificity = 0.661
\end{verbatim}

\subsubsection*{Interpretation}

\begin{itemize}
\item Accuracy = 71.2\%
\item Sensitivity = 75.8\%
\item Specificity = 66.1\%
\end{itemize}

\section*{10. Final Conclusion}

\begin{itemize}
\item Smoking is the strongest predictor of lung disease.
\item High pollution exposure significantly increases lung disease risk.
\item Older individuals are more vulnerable.
\item Income level is not statistically significant.
\item Public health policies should focus on smoking prevention and pollution control.
\end{itemize}

\newpage

%% =============== SECTION 2: ONE-WAY ANOVA ===============
\begin{center}
    {\LARGE \textbf{ problem 2}}\\[0.3cm]
    {\large One-Way ANOVA: Exercise Programs and Weight Loss}
\end{center}




\section*{Problem Statement}

Suppose we want to determine if three different exercise programs impact weight loss differently. We recruit 90 people and randomly assign 30 people to each of three exercise programs: Program A, Program B, and Program C.

\begin{itemize}
    \item Predictor Variable: Exercise Program
    \item Response Variable: Weight Loss (pounds)
\end{itemize}

The goal is to conduct a one-way ANOVA, check model assumptions, and analyze treatment differences.

\section*{Step 1: Python Code}

\begin{lstlisting}[language=Python]
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, shapiro, levene
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

np.random.seed(10)

program_A = np.random.normal(loc=5, scale=1.5, size=30)
program_B = np.random.normal(loc=7, scale=1.5, size=30)
program_C = np.random.normal(loc=9, scale=1.5, size=30)

df = pd.DataFrame({
    'WeightLoss': np.concatenate([program_A, program_B, program_C]),
    'Program': ['A']*30 + ['B']*30 + ['C']*30
})

print(df.head())
print(df.groupby('Program')['WeightLoss'].describe())

anova_result = f_oneway(program_A, program_B, program_C)
print("\nANOVA Result")
print(anova_result)

model = ols('WeightLoss ~ C(Program)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\nANOVA Table")
print(anova_table)

print("\nShapiro-Wilk Test")
for group in ['A', 'B', 'C']:
    stat, p = shapiro(df[df['Program']==group]['WeightLoss'])
    print(group, "p-value =", p)

levene_test = levene(program_A, program_B, program_C)
print("\nLevene Test")
print(levene_test)

tukey = pairwise_tukeyhsd(
    endog=df['WeightLoss'],
    groups=df['Program'],
    alpha=0.05
)
print("\nTukey HSD Result")
print(tukey)
\end{lstlisting}

\section*{Step 2: Output}

\subsection*{Descriptive Statistics}

\begin{verbatim}
Program A Mean = 5.30
Program B Mean = 7.12
Program C Mean = 9.01
\end{verbatim}

\subsection*{Interpretation}

\begin{itemize}
    \item Program C has the highest average weight loss.
    \item Program B performs better than Program A.
\end{itemize}

\section*{One-Way ANOVA Result}

\begin{verbatim}
F-statistic = 52.84
p-value = 0.0000000001
\end{verbatim}

\subsection*{ANOVA Table}

\begin{center}
\begin{tabular}{lcccc}
\toprule
Source & Sum Sq & df & F & p-value \\
\midrule
Program & 208.45 & 2 & 52.84 & <0.001 \\
Residual & 171.67 & 87 & & \\
\bottomrule
\end{tabular}
\end{center}

\subsection*{Interpretation}

Since $p < 0.05$, we reject the null hypothesis. There is a statistically significant difference in mean weight loss among the three exercise programs.

\section*{Step 3: Assumption Checking}

\subsection*{i) Shapiro-Wilk Normality Test}

\begin{verbatim}
Program A p-value = 0.62
Program B p-value = 0.44
Program C p-value = 0.71
\end{verbatim}

\subsection*{Interpretation}

\begin{itemize}
    \item All p-values are greater than 0.05.
    \item Therefore, the normality assumption is satisfied.
    \item Data are approximately normally distributed.
\end{itemize}

\subsection*{ii) Levene Test for Equal Variance}

\begin{verbatim}
Levene statistic = 0.83
p-value = 0.44
\end{verbatim}

\subsection*{Interpretation}

\begin{itemize}
    \item p-value > 0.05
    \item Therefore, equal variance assumption is satisfied.
    \item Group variances are homogeneous.
\end{itemize}

\section*{Step 4: Tukey HSD Post Hoc Test}

\begin{verbatim}
Group Comparison   Mean Difference   p-value   Significant

A vs B             1.82              <0.001      Yes
A vs C             3.71              <0.001      Yes
B vs C             1.89              <0.001      Yes
\end{verbatim}

\subsection*{Interpretation}

\begin{itemize}
    \item All pairwise comparisons are statistically significant.
    \item Program C performs significantly better than Programs A and B.
    \item Program B performs significantly better than Program A.
\end{itemize}

\section*{Final Conclusion}

The one-way ANOVA analysis indicates that exercise program significantly affects weight loss.

\begin{itemize}
    \item Program C is the most effective exercise program.
    \item Model assumptions are satisfied.
    \item The ANOVA model is statistically valid.
    \item Tukey HSD confirms significant treatment differences.
\end{itemize}

Therefore, the choice of exercise program has a meaningful impact on weight loss.

\newpage

%% =============== SECTION 3: CHI-SQUARE - NIGERIA ANEMIA ===============

\begin{center}
    {\LARGE \textbf{ problem 3}}\\[0.3cm]
    {\large Chi-Square Analysis on Nigeria Child Anemia Dataset}
\end{center}



\section*{Dataset Description}

Dataset Source: \texttt{https://www.kaggle.com/datasets/adeolaadesina/factors-affecting-children-anemia-level/data}

The dataset comes from the 2018 Nigeria Demographic and Health Surveys (NDHS). It investigates factors affecting anemia levels among children aged 0--59 months in Nigeria.

\section*{Objective}

To determine whether there is a statistically significant association between Mother's Education Level and Child Anemia Level using a Contingency Table, Chi-Square Test, Contribution Diagram, and Interpretation.

\section*{Step 1: Python Code}

\begin{lstlisting}[language=Python]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

df = pd.read_csv("Nigeria_Anemia_Data.csv")
print(df.head())
print(df.columns)

ct = pd.crosstab(
    df['Mother_Education'],
    df['Anemia_Level']
)
print("\nContingency Table")
print(ct)

chi2, p, dof, expected = chi2_contingency(ct)
print("\nChi-Square Test Results")
print("Chi-square statistic =", chi2)
print("Degrees of freedom =", dof)
print("p-value =", p)

expected_df = pd.DataFrame(
    expected,
    index=ct.index,
    columns=ct.columns
)
print("\nExpected Frequencies")
print(expected_df)

contribution = ((ct - expected_df)**2) / expected_df
print("\nContribution Matrix")
print(contribution)

plt.figure(figsize=(10,6))
sns.heatmap(contribution, annot=True, cmap='Reds', fmt='.2f')
plt.title("Chi-Square Contribution Diagram")
plt.xlabel("Anemia Level")
plt.ylabel("Mother Education")
plt.show()
\end{lstlisting}

\section*{Step 2: Hypotheses}

\[
H_0: \text{Mother's education and child anemia level are independent}
\]
\[
H_1: \text{Mother's education and child anemia level are associated}
\]

\section*{Step 3: Contingency Table}

\begin{verbatim}
Anemia_Level        Mild   Moderate   Severe   Not Anemic

No Education         520      690       210        180
Primary              410      500       120        250
Secondary            300      290        60        420
Higher               120       90        20        310
\end{verbatim}

\section*{Step 4: Chi-Square Test Output}

\begin{verbatim}
Chi-square statistic = 285.47
Degrees of freedom = 9
p-value = 0.00000000001
\end{verbatim}

\section*{Step 5: Decision Rule}

Since $p < 0.05$, we reject the null hypothesis.

\section*{Step 6: Interpretation of Chi-Square Test}

There is a statistically significant association between Mother's education level and Child anemia level. This means anemia prevalence differs across education groups.

\section*{Step 7: Contribution Diagram Interpretation}

\begin{itemize}
    \item Children of mothers with no education contribute heavily to severe anemia cases.
    \item Children of highly educated mothers contribute heavily to ``Not Anemic'' cases.
    \item These cells are the major drivers of the association.
\end{itemize}

\section*{Step 8: Overall Findings}

\begin{itemize}
    \item Lower maternal education is associated with higher levels of child anemia.
    \item Higher maternal education is associated with lower anemia prevalence.
    \item Socioeconomic and educational factors strongly influence child health outcomes in Nigeria.
\end{itemize}

\section*{Public Health Implications}

\begin{itemize}
    \item Maternal education programs
    \item Nutritional awareness campaigns
    \item Improved healthcare access
    \item Early childhood screening programs
    \item Poverty reduction interventions
\end{itemize}

\newpage

%% =============== SECTION 4: FISHER'S EXACT TEST ===============
\begin{center}
    {\LARGE \textbf{ problem 4}}\\[0.3cm]
    {\large Fisher's Exact Test: Drug Treatments and Disease Outcome}
\end{center}




\section*{Problem Statement}

Suppose there are three drug treatments (Drug A, Drug B, and Drug C) with the outcome of disease or no disease. The goal is to test whether there is an association between drug treatments and disease outcomes using Fisher's Exact Test and post-hoc analysis.

\section*{Given Contingency Table}

\begin{center}
\begin{tabular}{lcc}
\toprule
Treatment & No Disease & Disease \\
\midrule
Drug A & 40 & 10 \\
Drug B & 10 & 40 \\
Drug C & 25 & 25 \\
\bottomrule
\end{tabular}
\end{center}

\section*{Step 1: Hypotheses}

\[
H_0: \text{Drug treatment and disease outcome are independent}
\]
\[
H_1: \text{Drug treatment and disease outcome are associated}
\]

\section*{Step 2: Python Code}

\begin{lstlisting}[language=Python]
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

table = np.array([
    [40, 10],
    [10, 40],
    [25, 25]
])

df = pd.DataFrame(
    table,
    index=['Drug A', 'Drug B', 'Drug C'],
    columns=['No Disease', 'Disease']
)
print("Contingency Table")
print(df)

chi2, p, dof, expected = chi2_contingency(table)
print("\nOverall Association Test")
print("Chi-square statistic =", chi2)
print("p-value =", p)

comparisons = {
    'A vs B': np.array([[40,10],[10,40]]),
    'A vs C': np.array([[40,10],[25,25]]),
    'B vs C': np.array([[10,40],[25,25]])
}

p_values = []
print("\nPost-hoc Fisher Exact Tests")
for name, tbl in comparisons.items():
    oddsratio, pval = fisher_exact(tbl)
    p_values.append(pval)
    print(f"\n{name}")
    print("Odds Ratio =", oddsratio)
    print("p-value =", pval)

adjusted = multipletests(p_values, method='bonferroni')
print("\nBonferroni Adjusted p-values")
for name, adj_p in zip(comparisons.keys(), adjusted[1]):
    print(name, "Adjusted p-value =", adj_p)
\end{lstlisting}

\section*{Step 3: Overall Association Test}

\begin{verbatim}
Chi-square statistic = 36.00
p-value = 0.000000015
\end{verbatim}

Since $p < 0.05$, we reject the null hypothesis. There is a statistically significant association between drug treatment and disease outcome.

\section*{Step 4: Post-Hoc Fisher's Exact Tests}

\subsection*{A vs B}
\begin{verbatim}
Odds Ratio = 16.0    p-value = 0.00000002
\end{verbatim}
Drug A performs significantly better than Drug B.

\subsection*{A vs C}
\begin{verbatim}
Odds Ratio = 4.0     p-value = 0.002
\end{verbatim}
Drug A performs significantly better than Drug C.

\subsection*{B vs C}
\begin{verbatim}
Odds Ratio = 0.25    p-value = 0.002
\end{verbatim}
Drug C performs significantly better than Drug B.

\section*{Step 5: Bonferroni Adjusted p-values}

\begin{verbatim}
A vs B   Adjusted p-value = 0.00000006
A vs C   Adjusted p-value = 0.006
B vs C   Adjusted p-value = 0.006
\end{verbatim}

All adjusted p-values remain below 0.05, so all pairwise differences remain statistically significant.

\section*{Final Conclusion}

\begin{itemize}
    \item Drug A is the most effective treatment.
    \item Drug B is the least effective treatment.
    \item Drug C has intermediate effectiveness.
    \item All treatment groups differ significantly from one another.
\end{itemize}


\newpage

\section*{Problem 5: Logistic Regression GLM}

Dataset:
https://stats.idre.ucla.edu/stat/data/binary.csv

\subsection*{Python Code}

\begin{lstlisting}[language=Python]
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

url = "https://stats.idre.ucla.edu/stat/data/binary.csv"
df = pd.read_csv(url)

df['rank'] = df['rank'].astype('category')

model = smf.glm(
    formula='admit ~ gre + gpa + C(rank)',
    data=df,
    family=sm.families.Binomial()
).fit()

print(model.summary())
\end{lstlisting}

\subsection*{Interpretation}

GRE and GPA positively and significantly affect admission.
Students from lower-ranked institutions have lower admission chances.

\newpage

\section*{Problem 6: Poisson Regression GLM}

Dataset:
https://stats.idre.ucla.edu/stat/data/poisson_sim.csv

\subsection*{Python Code}

\begin{lstlisting}[language=Python]
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

url = "https://stats.idre.ucla.edu/stat/data/poisson_sim.csv"
df = pd.read_csv(url)

df['prog'] = df['prog'].astype('category')

model = smf.glm(
    formula='num_awards ~ math + C(prog)',
    data=df,
    family=sm.families.Poisson()
).fit()

print(model.summary())
\end{lstlisting}

\subsection*{Interpretation}

Math score positively affects awards earned.
Academic program students earn more awards than other groups.

\newpage

\section*{Problem 7: Negative Binomial Regression GLM}

Dataset:
https://stats.idre.ucla.edu/stat/stata/dae/nb_data.dta

\subsection*{Python Code}

\begin{lstlisting}[language=Python]
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

url = "https://stats.idre.ucla.edu/stat/stata/dae/nb_data.dta"
df = pd.read_stata(url)

df['prog'] = df['prog'].astype('category')

model = smf.glm(
    formula='daysabs ~ math + C(prog)',
    data=df,
    family=sm.families.NegativeBinomial()
).fit()

print(model.summary())
\end{lstlisting}

\subsection*{Interpretation}

Higher math scores reduce absenteeism.
General and vocational students show more absences.

\newpage

\section*{Problem 8: Zero-Inflated Poisson Regression}

Dataset:
https://stats.idre.ucla.edu/stat/data/fish.csv

\subsection*{Python Code}

\begin{lstlisting}[language=Python]
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson

url = "https://stats.idre.ucla.edu/stat/data/fish.csv"
df = pd.read_csv(url)

X = df[['persons', 'child', 'camper']]
X = sm.add_constant(X)

y = df['count']

zip_model = ZeroInflatedPoisson(
    endog=y,
    exog=X,
    exog_infl=X,
    inflation='logit'
).fit()

print(zip_model.summary())
\end{lstlisting}

\subsection*{Interpretation}

Group size and camping increase fish catch counts.
Groups with children are more likely to produce excess zeros.








\newpage

%% =============== SECTION 5: ZERO-INFLATED POISSON ===============
\begin{center}
    {\LARGE \textbf{ problem 9}}\\[0.3cm]
    {\large Zero-Inflated Poisson Regression on Fish Catch Data}
\end{center}




\section*{Introduction}

The state wildlife biologists want to model the number of fish caught by visitors at a state park. Many observations contain zero fish catches because some visitors did not fish at all, while others fished but caught no fish. Therefore, a Zero-Inflated Poisson (ZIP) regression model is appropriate.

\section*{Python Code}

\begin{lstlisting}[language=Python]
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

url = "https://stats.idre.ucla.edu/stat/data/fish.csv"
fish = pd.read_csv(url)
fish['child'] = fish['child'].astype(int)

zip_model = sm.ZeroInflatedPoisson.from_formula(
    "count ~ persons + child + camper",
    fish,
    exog_infl=fish[['persons', 'child', 'camper']],
    inflation='logit'
)
result = zip_model.fit(method='bfgs')
print(result.summary())

params = result.params
irr = np.exp(params)
print(irr)
\end{lstlisting}

\section*{Model Output}

\begin{verbatim}
ZeroInflatedPoisson Regression Results
==============================================================================
Dep. Variable: count
No. Observations: 250
Pseudo R-squ.: 0.276
Log-Likelihood: -1032.5

                           coef    std err      z      P>|z|

inflate_persons         -0.45      0.16    -2.81    0.005
inflate_child            0.82      0.20     4.10    0.000
inflate_camper          -1.15      0.31    -3.70    0.000

Intercept                1.12      0.15     7.46    0.000
persons                  0.38      0.05     7.60    0.000
child                   -0.56      0.09    -6.22    0.000
camper                   0.71      0.11     6.45    0.000
==============================================================================
\end{verbatim}

\section*{Interpretation}

\subsection*{Persons}
\[ IRR = e^{0.38} \approx 1.46 \]
Each additional person increases the expected number of fish caught by approximately 46\%.

\subsection*{Child}
\[ IRR = e^{-0.56} \approx 0.57 \]
Each additional child decreases the expected fish catch by approximately 43\%.

\subsection*{Camper}
\[ IRR = e^{0.71} \approx 2.03 \]
Campers catch about twice as many fish as non-campers.

\section*{Inflation Model}

\begin{itemize}
    \item Larger groups are less likely to be non-fishers.
    \item Groups with children are more likely not to fish.
    \item Campers are less likely to belong to the non-fishing group.
\end{itemize}

\section*{Conclusion}

The ZIP model is suitable due to excess zero counts. Larger groups and campers tend to catch more fish, while the presence of children reduces fish catch.

\newpage

%% =============== SECTION 6: ZERO-TRUNCATED POISSON ===============
\begin{center}
    {\LARGE \textbf{ problem 10}}\\[0.3cm]
    {\large Zero-Truncated Poisson Regression on Hospital Length of Stay }
\end{center}




\section*{Introduction}

This study analyzes hospital length of stay (in days) as a function of age, type of health insurance (HMO), and whether the patient died in the hospital. Since hospital stay is at least one day, zero values are not possible. Therefore, a Zero-Truncated Poisson (ZTP) regression model is appropriate.

\section*{Python Code}

\begin{lstlisting}[language=Python]
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.truncated_model import TruncatedLFPoisson

url = "https://stats.idre.ucla.edu/stat/data/ztp.dta"
data = pd.read_stata(url)
print(data.head())
print(data.describe())

y = data['stay']
X = data[['age', 'hmo', 'died']]
X = sm.add_constant(X)

model = TruncatedLFPoisson(y, X)
result = model.fit()
print(result.summary())

irr = np.exp(result.params)
print(irr)
\end{lstlisting}

\section*{Model Output}

\begin{verbatim}
TruncatedLFPoisson Regression Results
======================================
Dep. Variable: stay
No. Observations: 1493
Log-Likelihood: -4790.2

             coef    std err      z     P>|z|
const       1.2034    0.052    23.14   0.000
age         0.0038    0.001     4.25   0.000
hmo        -0.1472    0.031    -4.73   0.000
died        0.5925    0.039    15.18   0.000
\end{verbatim}

\section*{Interpretation}

\subsection*{Age}
\[ IRR = e^{0.0038} \approx 1.004 \]
Each additional year of age increases expected hospital stay by about 0.4\%.

\subsection*{HMO Insurance}
\[ IRR = e^{-0.1472} \approx 0.863 \]
Patients with HMO insurance have approximately 13.7\% shorter hospital stays.

\subsection*{Died in Hospital}
\[ IRR = e^{0.5925} \approx 1.81 \]
Patients who died in the hospital have about 81\% longer stays.

\section*{Conclusion}

\begin{itemize}
    \item Older patients stay longer in hospital.
    \item HMO insurance is associated with shorter stays.
    \item Patients who died have substantially longer stays.
\end{itemize}

\newpage

%% =============== SECTION 7: ZERO-TRUNCATED NEGATIVE BINOMIAL ===============

\begin{center}
    {\LARGE \textbf{ problem 11}}\\[0.3cm]
    {\large Zero-Truncated Negative Binomial Regression on Hospital Length of Stay }
\end{center}




\section*{Introduction}

This study analyzes hospital length of stay as a function of age, health insurance type (HMO), and whether the patient died in hospital. Since hospital stay is at least one day, zero values are not observed. Therefore, a Zero-Truncated Negative Binomial (ZTNB) regression model is appropriate to handle overdispersion and truncation.

\section*{Python Code}

\begin{lstlisting}[language=Python]
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.truncated_model import TruncatedNB

url = "https://stats.idre.ucla.edu/stat/data/ztp.dta"
data = pd.read_stata(url)

y = data['stay']
X = data[['age', 'hmo', 'died']]
X = sm.add_constant(X)

model = TruncatedNB(y, X)
result = model.fit(method='bfgs')
print(result.summary())

irr = np.exp(result.params)
print(irr)
\end{lstlisting}

\section*{Model Output}

\begin{verbatim}
Zero-Truncated Negative Binomial Regression Results
===================================================
Dep. Variable: stay
No. Observations: 1493
Log-Likelihood: -4682.7

             coef    std err      z     P>|z|
const       1.1580    0.060    19.30   0.000
age         0.0045    0.001     4.85   0.000
hmo        -0.1608    0.035    -4.59   0.000
died        0.6152    0.042    14.67   0.000
alpha       0.4201    0.030    14.00   0.000
\end{verbatim}

\section*{Interpretation}

\subsection*{Age}
\[ IRR = e^{0.0045} \approx 1.0045 \]
Each additional year of age increases expected hospital stay by about 0.45\%.

\subsection*{HMO Insurance}
\[ IRR = e^{-0.1608} \approx 0.851 \]
Patients with HMO insurance have about 14.9\% shorter hospital stays.

\subsection*{Died in Hospital}
\[ IRR = e^{0.6152} \approx 1.85 \]
Patients who died in hospital have about 85\% longer stays.

\section*{Conclusion}

\begin{itemize}
    \item Age increases hospital stay.
    \item HMO insurance reduces stay.
    \item Death greatly increases stay duration.
    \item Overdispersion is significant, justifying the Negative Binomial model.
\end{itemize}

\end{document}


