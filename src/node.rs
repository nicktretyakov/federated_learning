use crate::messages::{NodeMessage, ServerMessage};
use crate::model::{build_model, extract_params, prepare_data, update_model, SharedModel};
use actix::prelude::*;
use anyhow::Result;
use log::{error, info};
use ndarray::Array2;

pub struct NodeActor {
    model: SharedModel,
    server_addr: String,
    node_addr: String,
}

impl Actor for NodeActor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("Node {} started", self.node_addr);

        // Register node with server
        let msg = NodeMessage::RegisterNode {
            addr: self.node_addr.clone(),
        };

        let server_addr = self.server_addr.clone();
        ctx.run_later(std::time::Duration::from_secs(1), move |act, _| {
            act.send_to_server(&server_addr, msg.clone());
        });
    }
}

impl Handler<NodeMessage> for NodeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: NodeMessage, _ctx: &mut Self::Context) -> Self::Result {
        match msg {
            NodeMessage::Train { data, labels } => {
                info!("Training on node {}", self.node_addr);

                // Convert data to ndarray format
                let (x, y) = prepare_data(&data, &labels);

                // Train model
                match self.model.lock() {
                    Ok(mut model) => {
                        // Train for 10 epochs
                        model.train(&x, &y, 10);
                        info!("Node {} - Training completed", self.node_addr);
                    }
                    Err(e) => return Err(format!("Failed to lock model for training: {}", e)),
                }

                // Send updated parameters to server
                match extract_params(&self.model) {
                    Ok(params) => {
                        let server_msg = ServerMessage {
                            node_addr: self.node_addr.clone(),
                            params,
                        };
                        self.send_to_server(
                            &self.server_addr,
                            NodeMessage::UpdateModel {
                                params: server_msg.params.clone(),
                            },
                        );
                        Ok(())
                    }
                    Err(e) => Err(format!("Failed to extract model parameters: {}", e)),
                }
            }
            NodeMessage::Predict { data } => {
                info!("Prediction on node {}", self.node_addr);

                // Reshape data to [batch_size, features]
                let batch_size = data.len() / 10;
                let x = Array2::from_shape_vec((batch_size, 10), data.clone())
                    .map_err(|e| format!("Failed to reshape data: {}", e))?;

                match self.model.lock() {
                    Ok(model) => {
                        let predictions = model.forward(&x);
                        info!("Prediction on node {}: {:?}", self.node_addr, predictions);
                        Ok(())
                    }
                    Err(e) => Err(format!("Failed to lock model for prediction: {}", e)),
                }
            }
            NodeMessage::UpdateModel { params } => match update_model(&self.model, &params) {
                Ok(_) => {
                    info!("Model updated on node {}", self.node_addr);
                    Ok(())
                }
                Err(e) => Err(format!("Failed to update model: {}", e)),
            },
            NodeMessage::RegisterNode { .. } => Ok(()), // Ignore, this is for server
        }
    }
}

impl NodeActor {
    pub fn new(server_addr: String, node_addr: String) -> Self {
        let model = build_model();

        Self {
            model,
            server_addr,
            node_addr,
        }
    }

    fn send_to_server(&self, server_addr: &str, msg: NodeMessage) {
        let server_addr = format!("{}/message", server_addr);
        let msg_clone = msg.clone();

        // Use actix_web::rt::spawn instead of tokio::spawn
        actix_web::rt::spawn(async move {
            let client = awc::Client::default();
            match client.post(&server_addr).send_json(&msg_clone).await {
                Ok(_) => info!("Message sent to server successfully"),
                Err(e) => error!("Failed to send message to server: {}", e),
            }
        });
    }
}
