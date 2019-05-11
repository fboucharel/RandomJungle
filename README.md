# RandomJungle
A way to deal with unbalanced dataset :

- 00 - Training dataset with unbalanced classes ( target variable )
- 01 - Random sample of majority class stacked with entire minority class
- 02 - Fitting of a model on that subset
- 03 - Iterating on - 01 - and - 02 -
- 04 - Voting of all the classifiers on validation dataset
