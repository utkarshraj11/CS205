import numpy as np
import sys
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import LeaveOneOut

def load_data(filename):
    data = np.loadtxt(filename)
    labels = data[:, 0].astype(int) # first column = class label, convert to int
    features = data[:, 1:]  # rest of columns = features, keep as float
    return labels, features


# Baseline accuracy function: leave-one-out cross-validation
def loo_accuracy(labels, features, feature_subset):
    """
    # https://scikit-learn.org/stable/modules/cross_validation.html
    # https://medium.com/@pacosun/one-out-all-in-leave-one-out-cross-validation-explained-409df5ff6385

    X = features[:, list(feature_subset)]   # select only the features in the subset
    n = len(labels)
    loo = LeaveOneOut()
    correct = 0
    for train_index, test_index in loo.split(X):
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        if y_pred[0] == y_test[0]:
            correct += 1
    """ 
    X = features[:, list(feature_subset)]
    n = len(labels)
    correct = 0
    for i in range(n):
        diffs = X - X[i]    # compute difference to all other points
        dists = np.einsum('ij,ij->i', diffs, diffs) # squared L2 distance
        dists[i] = np.inf   # ignore self by setting distance to infinity
        if labels[np.argmin(dists)] == labels[i]:   # check if nearest neighbor has the same label
            correct += 1

    return correct / n


def forward_selection(labels, features):    # Start with an empty set of features and iteratively add the feature that gives the best accuracy until no improvement is observed.
    # https://www.geeksforgeeks.org/machine-learning/sequential-feature-selection/

    total_features = features.shape[1]
    current_set = set()
    best_overall_set = set()
    best_overall_acc = 0.0

    print(f"\nBeginning Forward Selection search.\n")

    for _ in range(total_features):
        best_acc_level = -1
        best_feat_level = None

        for f in range(total_features):
            if f in current_set:    # already included, skip
                continue
            candidate = current_set | {f}   # Add this feature
            acc = loo_accuracy(labels, features, candidate)
            feat_display = sorted(f + 1 for f in candidate) # 1-indexed for display
            print(f"        Using feature(s) {{{','.join(map(str, feat_display))}}} accuracy is {acc * 100:.1f}%")

            if acc > best_acc_level:
                best_acc_level = acc
                best_feat_level = f

        current_set = current_set | {best_feat_level}   # Add the best feature of this level
        feat_display = sorted(f + 1 for f in current_set)

        if best_acc_level > best_overall_acc:
            best_overall_acc = best_acc_level
            best_overall_set = set(current_set) # copy current set
        else:
            print(f"\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)")

        print(f"\nFeature set {{{','.join(map(str, feat_display))}}} was best, accuracy is {best_acc_level * 100:.1f}%\n")

    best_display = sorted(f + 1 for f in best_overall_set)
    print(f"\nFinished search!! The best feature subset is {{{','.join(map(str, best_display))}}}, "
          f"which has an accuracy of {best_overall_acc * 100:.1f}%")
    return best_overall_set, best_overall_acc


def backward_elimination(labels, features): # The same as forward selection, but start with the full set and remove features instead of adding them.
    # https://www.geeksforgeeks.org/machine-learning/sequential-feature-selection/
    
    total_features = features.shape[1]
    current_set = set(range(total_features))
    best_overall_set = set(current_set)
    best_overall_acc = loo_accuracy(labels, features, current_set)

    print(f"\nBeginning Backward Elimination search.\n")

    for _ in range(total_features - 1):
        best_acc_level = -1
        worst_feat = None

        for f in current_set:
            candidate = current_set - {f}   # Remove this feature
            acc = loo_accuracy(labels, features, candidate)
            feat_display = sorted(x + 1 for x in candidate)
            print(f"        Using feature(s) {{{','.join(map(str, feat_display))}}} accuracy is {acc * 100:.1f}%")

            if acc > best_acc_level:
                best_acc_level = acc
                worst_feat = f

        current_set = current_set - {worst_feat}
        feat_display = sorted(f + 1 for f in current_set)

        if best_acc_level > best_overall_acc:
            best_overall_acc = best_acc_level
            best_overall_set = set(current_set)
        else:
            print(f"\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)")

        print(f"\nFeature set {{{','.join(map(str, feat_display))}}} was best, accuracy is {best_acc_level * 100:.1f}%\n")

    best_display = sorted(f + 1 for f in best_overall_set)
    print(f"\nFinished search!! The best feature subset is {{{','.join(map(str, best_display))}}}, "
          f"which has an accuracy of {best_overall_acc * 100:.1f}%")
    return best_overall_set, best_overall_acc


def main():
    print("Welcome to Feature Selection Algorithm.")
    filename = input("Type in the name of the file to test: ").strip()  # e.g. "CS170_Small_DataSet__11.txt"

    labels, features = load_data(filename)
    n, num_features = features.shape

    print(f"\nThis dataset has {num_features} features (not including the class attribute), "
          f"with {n} instances.\n")

    all_acc = loo_accuracy(labels, features, set(range(num_features)))  # baseline accuracy with all features
    print(f'Running nearest neighbor with all {num_features} features, using "leaving-one-out" '
          f'evaluation, I get an accuracy of {all_acc * 100:.1f}%\n')

    print("Type the number of the algorithm you want to run.")
    print("  1) Forward Selection")
    print("  2) Backward Elimination")
    choice = input("\nYour choice: ").strip()

    if choice == "1":
        best_set, best_acc = forward_selection(labels, features)
    elif choice == "2":
        best_set, best_acc = backward_elimination(labels, features)
    else:
        print("Invalid choice.")
        sys.exit(1)


if __name__ == "__main__":
    main()