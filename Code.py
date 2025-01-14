import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
import subprocess

x_min, x_max = 0.0, 1.0  
y_min, y_max = 0.0, 1.0  
eps = 1e-4  

layers = [2, 30, 1]  
lr_initial = 0.01   
epochs = 1000      
beta = 1.0           


N_interior = 4000    
N_boundary = 400     


test_path = " USER_LOCATION "
submission_path = " USER LOCATION "


def f(x, y):
    return 2*eps*(-x + np.exp(2*(x - 1)/eps)) + x*y**2 + 6*x*y - x*np.exp(3*(y - 1)/eps) \
           - y**2*np.exp(2*(x - 1)/eps) + 2*y**2 - 6*y*np.exp(2*(x - 1)/eps) \
           - 2*np.exp(3*(y - 1)/eps) + np.exp((2*x + 3*y - 5)/eps)


def u_bc(x, y):
    return np.zeros_like(x)


def generate_boundary_points(xmin, xmax, ymin, ymax, num_points):
    num_points_per_edge = num_points // 4
    x_left = np.full((num_points_per_edge, 1), xmin)
    y_left = np.random.uniform(ymin, ymax, (num_points_per_edge, 1))
    x_right = np.full((num_points_per_edge, 1), xmax)
    y_right = np.random.uniform(ymin, ymax, (num_points_per_edge, 1))
    x_bottom = np.random.uniform(xmin, xmax, (num_points_per_edge, 1))
    y_bottom = np.full((num_points_per_edge, 1), ymin)
    x_top = np.random.uniform(xmin, xmax, (num_points_per_edge, 1))
    y_top = np.full((num_points_per_edge, 1), ymax)
    
    x_boundary = np.vstack([x_left, x_right, x_bottom, x_top])
    y_boundary = np.vstack([y_left, y_right, y_bottom, y_top])
    return x_boundary, y_boundary


def pde_loss(model, train_interior, eps):
    x = tf.expand_dims(train_interior[:, 0], axis=1)
    y = tf.expand_dims(train_interior[:, 1], axis=1)
    
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(x)
        tape1.watch(y)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)
            tape2.watch(y)
            u = model(tf.concat([x, y], axis=1))
        grad_x = tape2.gradient(u, x)
        grad_y = tape2.gradient(u, y)
    grad2_x = tape1.gradient(grad_x, x)
    grad2_y = tape1.gradient(grad_y, y)
    
    residual = -eps * (grad2_x + grad2_y) + 2 * grad_x + 3 * grad_y - f(x, y)
    return tf.reduce_mean(tf.square(residual))


def bc_loss(model, x_bd):
    u_pred = model(x_bd)
    x = tf.expand_dims(x_bd[:, 0], axis=1)
    y = tf.expand_dims(x_bd[:, 1], axis=1)
    u_exact = u_bc(x, y)
    return tf.reduce_mean(tf.square(u_pred - u_exact))


def train(model, epochs, train_interior, train_boundary, eps, beta, lr_initial):
    optimizer = tf.optimizers.Adam(learning_rate=lr_initial)
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss_pde = pde_loss(model, train_interior, eps)
            loss_bc = bc_loss(model, train_boundary)
            total_loss = loss_pde + beta * loss_bc
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Total Loss: {total_loss.numpy()}, PDE Loss: {loss_pde.numpy()}, BC Loss: {loss_bc.numpy()}")


def create_submission(model, test_path, output_path):
    test_data = pd.read_csv(test_path)
    xy = test_data[['x', 'y']].values
    predictions = model.predict(xy)
    test_data['u'] = predictions
    test_data.to_csv(output_path, index=False)


x_train_interior = np.random.uniform(x_min, x_max, (N_interior, 1))
y_train_interior = np.random.uniform(y_min, y_max, (N_interior, 1))
train_interior = tf.convert_to_tensor(np.hstack([x_train_interior, y_train_interior]), dtype=tf.float32)
x_boundary, y_boundary = generate_boundary_points(x_min, x_max, y_min, y_max, N_boundary)
train_boundary = tf.convert_to_tensor(np.hstack([x_boundary, y_boundary]), dtype=tf.float32)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(2,)),
    *[tf.keras.layers.Dense(units, activation='tanh') for units in layers[1:-1]],
    tf.keras.layers.Dense(1)
])

train(model, epochs, train_interior, train_boundary, eps, beta, lr_initial)
create_submission(model, test_path, submission_path)

def f(x, y):
    return 2*eps*(-x + np.exp(2*(x - 1)/eps)) + x*y**2 + 6*x*y - x*np.exp(3*(y - 1)/eps) \
           - y**2*np.exp(2*(x - 1)/eps) + 2*y**2 - 6*y*np.exp(2*(x - 1)/eps) \
           - 2*np.exp(3*(y - 1)/eps) + np.exp((2*x + 3*y - 5)/eps)


def u_bc(x, y):
    return np.zeros_like(x)


def generate_boundary_points(xmin, xmax, ymin, ymax, num_points):
    num_points_per_edge = num_points // 4
    x_left = np.full((num_points_per_edge, 1), xmin)
    y_left = np.random.uniform(ymin, ymax, (num_points_per_edge, 1))
    x_right = np.full((num_points_per_edge, 1), xmax)
    y_right = np.random.uniform(ymin, ymax, (num_points_per_edge, 1))
    x_bottom = np.random.uniform(xmin, xmax, (num_points_per_edge, 1))
    y_bottom = np.full((num_points_per_edge, 1), ymin)
    x_top = np.random.uniform(xmin, xmax, (num_points_per_edge, 1))
    y_top = np.full((num_points_per_edge, 1), ymax)
    
    x_boundary = np.vstack([x_left, x_right, x_bottom, x_top])
    y_boundary = np.vstack([y_left, y_right, y_bottom, y_top])
    return x_boundary, y_boundary


def pde_loss(model, train_interior, eps):
    x = tf.expand_dims(train_interior[:, 0], axis=1)
    y = tf.expand_dims(train_interior[:, 1], axis=1)
    
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(x)
        tape1.watch(y)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)
            tape2.watch(y)
            u = model(tf.concat([x, y], axis=1))
        grad_x = tape2.gradient(u, x)
        grad_y = tape2.gradient(u, y)
    grad2_x = tape1.gradient(grad_x, x)
    grad2_y = tape1.gradient(grad_y, y)
    
    residual = -eps * (grad2_x + grad2_y) + 2 * grad_x + 3 * grad_y - f(x, y)
    return tf.reduce_mean(tf.square(residual))


def bc_loss(model, x_bd):
    u_pred = model(x_bd)
    x = tf.expand_dims(x_bd[:, 0], axis=1)
    y = tf.expand_dims(x_bd[:, 1], axis=1)
    u_exact = u_bc(x, y)
    return tf.reduce_mean(tf.square(u_pred - u_exact))

# Train the model
def train(model, epochs, train_interior, train_boundary, eps, beta, lr_initial):
    optimizer = tf.optimizers.Adam(learning_rate=lr_initial)
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss_pde = pde_loss(model, train_interior, eps)
            loss_bc = bc_loss(model, train_boundary)
            total_loss = loss_pde + beta * loss_bc
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Total Loss: {total_loss.numpy()}, PDE Loss: {loss_pde.numpy()}, BC Loss: {loss_bc.numpy()}")


def create_submission(model, test_path, output_path):
    test_data = pd.read_csv(test_path)
    xy = test_data[['x', 'y']].values
    predictions = model.predict(xy)
    test_data['u'] = predictions
    test_data.to_csv(output_path, index=False)


x_train_interior = np.random.uniform(x_min, x_max, (N_interior, 1))
y_train_interior = np.random.uniform(y_min, y_max, (N_interior, 1))
train_interior = tf.convert_to_tensor(np.hstack([x_train_interior, y_train_interior]), dtype=tf.float32)
x_boundary, y_boundary = generate_boundary_points(x_min, x_max, y_min, y_max, N_boundary)
train_boundary = tf.convert_to_tensor(np.hstack([x_boundary, y_boundary]), dtype=tf.float32)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(2,)),
    *[tf.keras.layers.Dense(units, activation='tanh') for units in layers[1:-1]],
    tf.keras.layers.Dense(1)
])

train(model, epochs, train_interior, train_boundary, eps, beta, lr_initial)
create_submission(model, test_path, submission_path)
