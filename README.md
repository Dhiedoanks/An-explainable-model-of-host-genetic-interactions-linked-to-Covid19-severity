# An Explainable Model of Host Genetic Interactions Linked to Covid-19 Severity
## Introduction
This project focus on the mapping of host-genetics factors determining COVID-19 severity using Machine learning approaches (supervised, unsupervised Machine learning methods, Pathway signaling processes, and Open Targets web-based Platform) at the bioinformatics group of the Biology Lab, Scuola Normale Superiore di-Pisa, Italy.

This project is strongly motivated by the scholarly works done so far using Geno-Wide Association studies (GWS) in identifying chromosome loci and genetic variants related to COVID-19 susceptibility severity among patients. However, an organic model explaining how these genetic factors concur in the establishment of susceptibility to severe infection when exposed to the SARS-CoV-2 coronavirus infection was not covered using the GWS approaches. Also, there is currently limited usage of Machine learning (ML) techniques available to experts working with Whole-exome sequencing (WES) data sets related to COVID-19. Our study utilized the whole-exome sequencing genome dataset of 2000 European descent patients collected from the GEN-COVID Multicenter Study group (https://clinicaltrials.gov/ct2/show/NCT04549831) coordinated by the University of Siena. The whole-exome genome sequencing dataset contained 1.057M genetic variants of the patients. We used the 2000 patients’ original phenotype information to filter only patients with severity and asymptomatic across all classification criteria (841 patients).

This project is organized as follows:

1. Performed stratified k-fold to split phenotype dataset into training and testing;
2. screening variants on the training set from each fold;
3. training the model using the stratified k-fold and 5-fold GridSearchCV;
4. testing the models on the corresponding testing sets of each fold;
5. aggregate the results from each ML model;
6. perform the variant interpretation using PCA, UMAP, K-Means clustering and Pathway analysis;
7. remapping procedure if needed (whenever the training has been done on a different genome assembly);
8. Final testing on external dataset from new cohort ( 3rd, 4th wave, etc..).
9. 
**Authors**: Anthony Onoja, Nicola Picchiotti,GEN-COVID Multicenter Study, Francesca Colombo, Francesca Chiaromonte, Alessandra Renieri, Simone Furini, Francesco Raimondi.

**Date**: 15/10/2021
