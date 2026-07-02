# In[24]:


# move to GA
# initialise population
import random

POP_SIZE = 20
GENERATIONS = 20

NEURONS1 = [32, 64, 128]

NEURONS2 = [16, 32, 64]

LEARNING_RATE = [
    0.0005,
    0.001,
    0.005
]

BATCH_SIZE = [
    16,
    32,
    64
]

DROPOUT = [
    0.0,
    0.1,
    0.2,
    0.3
]

# population: neurons1, neurons2, learning_rate, batch_size, dropout
def create_population():
    population = []
    for _ in range(POP_SIZE):
        chromosome = [
            random.choice(NEURONS1),
            random.choice(NEURONS2),
            random.choice(LEARNING_RATE),
            random.choice(BATCH_SIZE),
            random.choice(DROPOUT)
        ]
        population.append(chromosome)

    return population


# In[25]:


# build ANN from chromosome
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(chromosome):

    n1, n2, lr, batch, drop = chromosome

    model = Sequential([

        Input(shape=(X_train.shape[1],)),

        Dense(
            n1,
            activation="relu"
        ),

        Dropout(drop),

        Dense(
            n2,
            activation="relu"
        ),

        Dropout(drop),

        Dense(1)

    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mse"
    )

    return model


# In[31]:


# fitness function
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.callbacks import EarlyStopping

# Use only a subset for GA to speed up optimization
X_train_ga = X_train[:100000]
y_train_ga = y_train[:100000]

X_val_ga = X_val[:20000]
y_val_ga = y_val[:20000]

def evaluate(chromosome):

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    n1, n2, lr, batch_size, drop = chromosome

    model = build_model(chromosome)

    model.fit(
        X_train_ga,
        y_train_ga,
        validation_data=(X_val_ga, y_val_ga),
        epochs=10,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )

    pred = model.predict(
        X_val_ga,
        verbose=0
    )

    pred = y_scaler.inverse_transform(pred)
    actual = y_scaler.inverse_transform(y_val_ga)

    mape = mean_absolute_percentage_error(
        actual,
        pred
    ) * 100

    fitness = 1 / (mape + 1e-6)

    return fitness


# In[32]:


# roulette wheel selection
def roulette_selection(
    population,
    fitness_values
):

    total_fitness = sum(fitness_values)

    probs = [
        fit / total_fitness
        for fit in fitness_values
    ]

    selected = []

    for _ in range(POP_SIZE):

        r = random.random()

        cumulative = 0

        for i in range(len(population)):

            cumulative += probs[i]

            if r <= cumulative:

                selected.append(
                    population[i]
                )

                break

    return selected


# In[33]:


# crossover
def crossover(parent1, parent2):

    child1 = []
    child2 = []

    for g1, g2 in zip(parent1, parent2):

        if random.random() < 0.5:

            child1.append(g1)
            child2.append(g2)

        else:

            child1.append(g2)
            child2.append(g1)

    return child1, child2


# In[34]:


# mutation
PM = 0.10

def mutate(chromosome):

    if random.random() < PM:
        chromosome[0] = random.choice(NEURONS1)

    if random.random() < PM:
        chromosome[1] = random.choice(NEURONS2)

    if random.random() < PM:
        chromosome[2] = random.choice(LEARNING_RATE)

    if random.random() < PM:
        chromosome[3] = random.choice(BATCH_SIZE)

    if random.random() < PM:
        chromosome[4] = random.choice(DROPOUT)

    return chromosome
