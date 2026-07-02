# In[43]:


# evaluate
from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(actual, pred)

rmse = np.sqrt(mse)

mape = np.mean(
    np.abs((actual - pred) / actual)
) * 100

print("MSE :", mse)
print("RMSE:", rmse)
print("MAPE:", mape)

plt.figure(figsize=(15,5))

plt.plot(
    actual[:500],
    label="Actual"
)

plt.plot(
    pred[:500],
    label="ANN Prediction"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Observations")
plt.ylabel("Sales")

plt.legend()

plt.show()
