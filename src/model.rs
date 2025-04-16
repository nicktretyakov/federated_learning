use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

// Simple neural network model with two layers
#[derive(Clone, Serialize, Deserialize)]
pub struct SimpleModel {
    // First layer weights: [input_size, hidden_size]
    pub w1: Array2<f32>,
    // First layer bias: [hidden_size]
    pub b1: Array1<f32>,
    // Second layer weights: [hidden_size, output_size]
    pub w2: Array2<f32>,
    // Second layer bias: [output_size]
    pub b2: Array1<f32>,
    // Learning rate
    pub learning_rate: f32,
}

impl SimpleModel {
    // Create a new model with random initialization
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization for weights
        let w1_bound = (6.0 / (input_size + hidden_size) as f32).sqrt();
        let w2_bound = (6.0 / (hidden_size + output_size) as f32).sqrt();

        // Initialize weights and biases
        let mut w1 = Array2::zeros((input_size, hidden_size));
        let mut w2 = Array2::zeros((hidden_size, output_size));
        let b1 = Array1::zeros(hidden_size);
        let b2 = Array1::zeros(output_size);

        // Random initialization of weights
        for i in 0..input_size {
            for j in 0..hidden_size {
                w1[[i, j]] = rng.gen_range(-w1_bound..w1_bound);
            }
        }

        for i in 0..hidden_size {
            for j in 0..output_size {
                w2[[i, j]] = rng.gen_range(-w2_bound..w2_bound);
            }
        }

        SimpleModel {
            w1,
            b1,
            w2,
            b2,
            learning_rate: 0.01,
        }
    }

    // Forward pass
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // First layer
        let z1 = x.dot(&self.w1) + &self.b1;
        let a1 = z1.mapv(|v| if v > 0.0 { v } else { 0.0 }); // ReLU activation

        // Second layer
        let z2 = a1.dot(&self.w2) + &self.b2;

        z2
    }

    // Backward pass and weights update
    pub fn train(&mut self, x: &Array2<f32>, y: &Array2<f32>, epochs: usize) {
        for _ in 0..epochs {
            // Forward pass
            let z1 = x.dot(&self.w1) + &self.b1;
            let a1 = z1.mapv(|v| if v > 0.0 { v } else { 0.0 }); // ReLU activation

            let z2 = a1.dot(&self.w2) + &self.b2;

            // Backward pass
            // Compute gradients
            let dz2 = &z2 - y;
            let dw2 = a1.t().dot(&dz2) * (1.0 / x.nrows() as f32);
            let db2 = dz2.sum_axis(Axis(0)) * (1.0 / x.nrows() as f32);

            let da1 = dz2.dot(&self.w2.t());
            let dz1 = da1 * z1.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }); // ReLU derivative
            let dw1 = x.t().dot(&dz1) * (1.0 / x.nrows() as f32);
            let db1 = dz1.sum_axis(Axis(0)) * (1.0 / x.nrows() as f32);

            // Update parameters
            self.w2 = &self.w2 - &(&dw2 * self.learning_rate);
            self.b2 = &self.b2 - &(&db2 * self.learning_rate);
            self.w1 = &self.w1 - &(&dw1 * self.learning_rate);
            self.b1 = &self.b1 - &(&db1 * self.learning_rate);
        }
    }

    // Convert model parameters to a vector for transmission
    pub fn to_params_vec(&self) -> Vec<f32> {
        let mut params = Vec::new();

        // Add w1
        params.extend(self.w1.iter());

        // Add b1
        params.extend(self.b1.iter());

        // Add w2
        params.extend(self.w2.iter());

        // Add b2
        params.extend(self.b2.iter());

        params
    }

    // Update model from parameters vector
    pub fn from_params_vec(&mut self, params: &[f32]) -> Result<()> {
        let mut offset = 0;

        // Update w1
        let w1_size = self.w1.len();
        for i in 0..self.w1.shape()[0] {
            for j in 0..self.w1.shape()[1] {
                let idx = i * self.w1.shape()[1] + j;
                if idx < w1_size && offset + idx < params.len() {
                    self.w1[[i, j]] = params[offset + idx];
                }
            }
        }
        offset += w1_size;

        // Update b1
        let b1_size = self.b1.len();
        for i in 0..b1_size {
            if offset + i < params.len() {
                self.b1[i] = params[offset + i];
            }
        }
        offset += b1_size;

        // Update w2
        let w2_size = self.w2.len();
        for i in 0..self.w2.shape()[0] {
            for j in 0..self.w2.shape()[1] {
                let idx = i * self.w2.shape()[1] + j;
                if idx < w2_size && offset + idx < params.len() {
                    self.w2[[i, j]] = params[offset + idx];
                }
            }
        }
        offset += w2_size;

        // Update b2
        let b2_size = self.b2.len();
        for i in 0..b2_size {
            if offset + i < params.len() {
                self.b2[i] = params[offset + i];
            }
        }

        Ok(())
    }
}

// Type alias for a thread-safe model
pub type SharedModel = Arc<Mutex<SimpleModel>>;

// Create a new shared model
pub fn build_model() -> SharedModel {
    Arc::new(Mutex::new(SimpleModel::new(10, 64, 1)))
}

// Extract parameters from a model
pub fn extract_params(model: &SharedModel) -> Result<Vec<f32>> {
    let model_lock = model
        .lock()
        .map_err(|e| anyhow::anyhow!("Failed to lock model: {}", e))?;
    Ok(model_lock.to_params_vec())
}

// Update model with parameters
pub fn update_model(model: &SharedModel, params: &[f32]) -> Result<()> {
    let mut model_lock = model
        .lock()
        .map_err(|e| anyhow::anyhow!("Failed to lock model: {}", e))?;
    model_lock.from_params_vec(params)
}

// Generate some synthetic data for training (only for demo purposes)
pub fn generate_data(samples: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();

    let mut data = Vec::with_capacity(samples * 10);
    let mut labels = Vec::with_capacity(samples);

    for _ in 0..samples {
        // Generate 10 features
        for _ in 0..10 {
            data.push(rng.gen_range(-1.0..1.0));
        }

        // Simple function to generate label: sum of first 3 features
        let x = &data[data.len() - 10..];
        labels.push(x[0] + x[1] + x[2]);
    }

    (data, labels)
}

// Convert raw data vectors to ndarray format
pub fn prepare_data(data: &[f32], labels: &[f32]) -> (Array2<f32>, Array2<f32>) {
    let batch_size = labels.len();
    let features = 10;

    // Reshape data to [batch_size, features]
    let x = Array2::from_shape_vec((batch_size, features), data.to_vec())
        .expect("Failed to reshape data");

    // Reshape labels to [batch_size, 1]
    let y =
        Array2::from_shape_vec((batch_size, 1), labels.to_vec()).expect("Failed to reshape labels");

    (x, y)
}
