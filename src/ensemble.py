from sklearn.ensemble import VotingClassifier

def create_voting_classifier(models):
    voting_clf = VotingClassifier(estimators=[
        ('lr', models['Logistic Regression']),
        ('rf', models['Random Forest']),
        ('svm', models['SVM (RBF Kernel)'])
    ], voting='hard')
    return voting_clf
