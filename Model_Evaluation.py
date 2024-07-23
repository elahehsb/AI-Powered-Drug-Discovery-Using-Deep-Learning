import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_class))
print(f'ROC-AUC Score: {roc_auc_score(y_test, y_pred)}')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
