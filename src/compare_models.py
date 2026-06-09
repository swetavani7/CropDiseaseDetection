import matplotlib.pyplot as plt

models = [
    "Custom CNN",
    "MobileNetV2"
]

accuracies = [
    87.2,
    95.1
]

plt.figure(figsize=(6,4))

plt.bar(
    models,
    accuracies
)

plt.ylabel("Accuracy (%)")
plt.title("CNN vs Transfer Learning")

plt.savefig(
    "outputs/model_comparison.png"
)

plt.show()