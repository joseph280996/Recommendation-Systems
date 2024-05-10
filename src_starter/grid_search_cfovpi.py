import itertools
import json
from CollabFilterOneVectorPerItem import CollabFilterOneVectorPerItem
from train_valid_test_loader import load_train_valid_test_datasets


def save_best_combination(batch_size, step_size, alpha):
    with open("./best_combination.json", "w") as file:
        parameters = {"batch_size": batch_size, "step_size": step_size, "alpha": alpha}
        json.dump(parameters, file, indent=2)


def grid_search_cfovpi():
    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = (
        load_train_valid_test_datasets()
    )

    # Set the number of factors to try
    alphas = [0.0, 0.0001, 0.001, 0.1]
    batch_sizes = [16, 32, 128, 1016]
    step_sizes = [0.1, 0.3, 0.9, 2.7]

    best_maes_per_epoch = {}

    for alpha, batch_size, step_size in itertools.product(
        alphas, batch_sizes, step_sizes
    ):
        print(
            f"-----------------------------------------------------------------------------\n"
            f"Training with alpha={alpha}, batch_size={batch_size}, step_size={step_size}"
            f"-----------------------------------------------------------------------------\n"
        )
        # Create the model and initialize its parameters
        model = CollabFilterOneVectorPerItem(
            n_epochs=2000,
            batch_size=batch_size,
            step_size=step_size,
            n_factors=50,
            alpha=alpha,
        )
        model.init_parameter_dict(n_users, n_items, train_tuple)

        # Fit the model with SGD
        try:
            model.fit(train_tuple, valid_tuple)
        except ValueError as error:
            print("Failed to compute loss and error due to overfitting. Continue.")

        best_mae = min(model.trace_mae_valid)

        best_maes_per_epoch[(batch_size, step_size, alpha)] = best_mae

    # Get the combination that achieve lowest MAE
    best_combination = min(best_maes_per_epoch, key=best_maes_per_epoch.get)
    save_best_combination(best_combination[0], best_combination[1], best_combination[2])
    best_mae = best_maes_per_epoch[best_combination]
    return best_combination


if __name__ == "__main__":
    grid_search_cfovpi()
