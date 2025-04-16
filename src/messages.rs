use actix::prelude::*;
use anyhow::Result;
use serde::{Deserialize, Serialize};

// Messages for communication between nodes and server
#[derive(Serialize, Deserialize, Message, Clone, Debug)]
#[rtype(result = "Result<(), String>")]
pub enum NodeMessage {
    Train { data: Vec<f32>, labels: Vec<f32> }, // Request to train on data
    Predict { data: Vec<f32> },                 // Request for prediction
    UpdateModel { params: Vec<f32> },           // Update model parameters
    RegisterNode { addr: String },              // Register node with server
}

// Message to request information about connected nodes
#[derive(Message)]
#[rtype(result = "Vec<crate::network::NodeStatus>")]
pub struct GetNodesRequest;

// Message to request current model parameters
#[derive(Message)]
#[rtype(result = "Result<Vec<f32>, String>")]
pub struct GetModelParams;

// Message for central server
#[derive(Message, Clone)]
#[rtype(result = "Result<(), String>")]
pub struct ServerMessage {
    pub node_addr: String,
    pub params: Vec<f32>,
}
