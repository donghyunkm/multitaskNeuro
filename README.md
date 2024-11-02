# Auxiliary objectives improve generalization performance but reduce model specification for low-data neuroimaging-based brain age prediction

Data scarcity challenges in healthcare applications impede the ability of machine learning models to generalize effectively. In this work, we propose to add an auxiliary objective to a brain age prediction model that significantly improves model performance and generalization under low-data regimes. We evaluate the impact of the auxiliary objective on model specification and particularly quantify how much random variations in the training process affect a model's representations and predictions. Our results show that while auxiliary objectives enhance generalization and performance, especially in data-limited settings, they also reduce model specification. These findings underscore the trade-off between improving generalization with added constraints such as auxiliary losses, and their reduction in model specification in low-data neuroimaging applications.

Paper accepted to the Extended Abstract Track at NeurIPS Workshop (UniReps), 2024 

### Environment Setup 

#### Using pyproject.toml
pip install -e . 

#### one time install and manual run of pre commit
pip install pre-commit  
pre-commit install  
pre-commit run --all-files 
