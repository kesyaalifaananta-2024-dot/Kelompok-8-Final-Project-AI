# In[35]:


# main GA loop
# initialize population
population = create_population()

best_individual = None
best_fitness = -float("inf")

for generation in range(GENERATIONS):

    print(f"\nGeneration {generation+1}")

    # evaluate every chromosome
    fitness_values = []

    for chromosome in population:

        fitness = evaluate(chromosome)

        fitness_values.append(fitness)

    # find best chromosome in this generation
    generation_best = max(fitness_values)
    generation_best_index = fitness_values.index(generation_best)
    print("Best Fitness:", generation_best)
    print("Best Individual:", population[generation_best_index])

    # update global best
    if generation_best > best_fitness:

        best_fitness = generation_best

        best_individual = (
            population[generation_best_index].copy()
        )

    # selection
    selected = roulette_selection(
        population,
        fitness_values
    )

    # next generation
    new_population = []

    for i in range(0, POP_SIZE, 2):

        parent1 = selected[i]
        parent2 = selected[i + 1]

        child1, child2 = crossover(
            parent1,
            parent2
        )

        child1 = mutate(child1)
        child2 = mutate(child2)

        new_population.append(child1)
        new_population.append(child2)

    # elitism
    new_population[0] = best_individual.copy()

    population = new_population

print("Best Chromosome:")
print(best_individual)
print("Best Fitness:")
print(best_fitness)


# In[36]:


# build final model
import numpy as np

X_train_final = np.vstack((X_train, X_val))
y_train_final = np.vstack((y_train, y_val))

final_model = build_model(best_individual)


# In[37]:


# train final model
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

batch_size = best_individual[3]

history = final_model.fit(
    X_train_final,
    y_train_final,
    validation_split=0.1,
    epochs=100,
    batch_size=batch_size,
    callbacks=[early_stop],
    verbose=1
)


# In[38]:


# predict test set
pred_scaled = final_model.predict(X_test)

# convert back to sales unit
# calculate metrics
from sklearn.metrics import mean_squared_error
import numpy as np

pred_scaled = final_model.predict(X_test)

pred = y_scaler.inverse_transform(pred_scaled)
actual = y_scaler.inverse_transform(y_test)

pred = np.expm1(pred)
actual = np.expm1(actual)

mse = mean_squared_error(actual, pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((actual - pred) / actual)) * 100

print("MSE :", mse)
print("RMSE:", rmse)
print("MAPE:", mape)

# plot predictions
import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))

plt.plot(actual[:500], label="Actual")
plt.plot(pred[:500], label="GANN Prediction")

plt.title("Actual vs Predicted Sales")
plt.xlabel("Observation")
plt.ylabel("Sales")
plt.legend()

plt.show()
