from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .common import make_output_dir, save_json, seed_everything


def load_lfw(data_home: str | Path, min_faces: int = 70):
    dataset = fetch_lfw_people(
        min_faces_per_person=min_faces,
        resize=0.4,
        data_home=str(data_home),
    )
    return dataset.images, dataset.data, dataset.target, dataset.target_names


def fit_numpy_pca(
    training_data: np.ndarray,
    components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean_face = training_data.mean(axis=0)
    centred = training_data - mean_face
    _, singular_values, right_vectors = np.linalg.svd(centred, full_matrices=False)
    component_count = min(components, right_vectors.shape[0])
    eigenfaces = right_vectors[:component_count]
    explained_variance = singular_values**2 / (len(training_data) - 1)
    explained_ratio = explained_variance / explained_variance.sum()
    return mean_face, eigenfaces, singular_values, explained_ratio


def plot_eigenfaces(
    eigenfaces: np.ndarray,
    image_shape: tuple[int, int],
    output_path: Path,
    count: int = 15,
) -> None:
    count = min(count, len(eigenfaces))
    columns = 5
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 2.8 * rows))
    for index, axis in enumerate(np.asarray(axes).flat):
        axis.axis("off")
        if index < count:
            axis.imshow(eigenfaces[index].reshape(image_shape), cmap="gray")
            axis.set_title(f"Eigenface {index + 1}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_compactness(explained_ratio: np.ndarray, output_path: Path) -> None:
    cumulative = np.cumsum(explained_ratio)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(np.arange(1, len(cumulative) + 1), cumulative)
    axis.axhline(0.9, color="tab:red", linestyle="--", label="90% variance")
    axis.set_xlabel("Number of principal components")
    axis.set_ylabel("Cumulative explained variance")
    axis.set_ylim(0, 1.01)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_confusion_matrix(
    matrix: np.ndarray,
    target_names: np.ndarray,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(target_names)), target_names, rotation=45, ha="right")
    axis.set_yticks(range(len(target_names)), target_names)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Eigenfaces + Random Forest confusion matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 2: Eigenfaces and Random Forest")
    parser.add_argument("--data-home", default="data")
    parser.add_argument("--components", type=int, default=150)
    parser.add_argument("--trees", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--max-features", type=int, default=150)
    parser.add_argument("--min-faces", type=int, default=70)
    parser.add_argument("--output-dir", default="results/part2_eigenfaces")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = make_output_dir(args.output_dir)
    images, flattened, labels, target_names = load_lfw(args.data_home, args.min_faces)
    train_x, test_x, train_y, test_y = train_test_split(
        flattened,
        labels,
        test_size=0.25,
        random_state=args.seed,
        stratify=labels,
    )
    mean_face, eigenfaces, singular_values, explained_ratio = fit_numpy_pca(
        train_x, args.components
    )
    transformed_train = (train_x - mean_face) @ eigenfaces.T
    transformed_test = (test_x - mean_face) @ eigenfaces.T
    classifier = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=args.max_depth,
        max_features=min(args.max_features, transformed_train.shape[1]),
        random_state=args.seed,
        n_jobs=-1,
    )
    classifier.fit(transformed_train, train_y)
    predictions = classifier.predict(transformed_test)
    accuracy = accuracy_score(test_y, predictions)
    report = classification_report(
        test_y,
        predictions,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(test_y, predictions)

    plot_eigenfaces(eigenfaces, images.shape[1:], output_dir / "eigenfaces.png")
    plot_compactness(explained_ratio, output_dir / "compactness.png")
    plot_confusion_matrix(matrix, target_names, output_dir / "confusion_matrix.png")
    np.savez_compressed(
        output_dir / "pca.npz",
        mean_face=mean_face,
        eigenfaces=eigenfaces,
        singular_values=singular_values,
        explained_ratio=explained_ratio,
        image_shape=np.asarray(images.shape[1:]),
    )
    joblib.dump(classifier, output_dir / "random_forest.joblib")
    save_json(
        {
            "accuracy": float(accuracy),
            "data_home": str(Path(args.data_home).resolve()),
            "total_samples": len(flattened),
            "train_samples": len(train_x),
            "test_samples": len(test_x),
            "classes": len(target_names),
            "target_names": target_names.tolist(),
            "components": len(eigenfaces),
            "trees": args.trees,
            "max_depth": args.max_depth,
            "max_features": min(args.max_features, transformed_train.shape[1]),
            "variance_explained": float(explained_ratio[: len(eigenfaces)].sum()),
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
        },
        output_dir / "metrics.json",
    )
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
