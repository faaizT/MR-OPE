# Marginal Density Ratio for Off-Policy Evaluation in Contextual Bandits

By [Muhammad Faaiz Taufiq](https://scholar.google.com/citations?user=oDL6ahoAAAAJ&hl=en), [Arnaud Doucet](https://scholar.google.com/citations?user=W4SZGV8AAAAJ&hl=en&oi=sra), [Rob Cornish](https://scholar.google.com/citations?user=ZDVQRN0AAAAJ&hl=en&oi=sra) and [Jean-Francois Ton](https://scholar.google.com/citations?user=WWVOu4kAAAAJ&hl=en&oi=sra).  

This project is the official implementation of "Marginal Density Ratio for Off-Policy Evaluation in Contextual Bandits" (NeurIPS 2023)

## Abstract
Off-Policy Evaluation (OPE) in contextual bandits is crucial for assessing new policies using existing data without costly experimentation. However, current OPE methods, such as Inverse Probability Weighting (IPW) and Doubly Robust (DR) estimators, suffer from high variance, particularly in cases of low overlap between target and behaviour policies or large action and context spaces. In this paper, we introduce a new OPE estimator for contextual bandits, the Marginal Ratio (MR) estimator, which focuses on the shift in the marginal distribution of outcomes $Y$ instead of the policies themselves. Through rigorous theoretical analysis, we demonstrate the benefits of the MR estimator compared to conventional methods like IPW and DR in terms of variance reduction. Additionally, we establish a connection between the MR estimator and the state-of-the-art Marginalized Inverse Propensity Score (MIPS) estimator, proving that MR achieves lower variance among a generalized family of MIPS estimators. We further illustrate the utility of the MR estimator in causal inference settings, where it exhibits enhanced performance in estimating Average Treatment Effects (ATE). Our experiments on synthetic and real-world datasets corroborate our theoretical findings and highlight the practical advantages of the MR estimator in OPE for contextual bandits.

## Installation
To install this package, run the following command:  
```pip install -r requirements.txt```

## Running the synthetic data experiments
To run the synthetic data experiments, run the following command:  
```python3 src/main_mips.py```  
This will run the synthetic data experiments provided in the main text.  

### Running the additional synthetic data experiments in the appendix
Tor run the additional synthetic data experiments provided in the appendix, run the following command:  
```python3 src/main_obp.py```  

## Running the ATE experiments
To run the ATE experiments, run the following command:  
```python3 src/ate_estimation_obp.py```  

## Running the classification data experiments
To run the classification data experiments, run the following command:  
```python3 src/main_obp_multiclass.py```  

## Configuring the parameters  
To configure the different parameters used in any experiment, refer to the ```args``` in the corresponding experiment main file.