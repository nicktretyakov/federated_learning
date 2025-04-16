use crate::messages::{GetModelParams, GetNodesRequest, NodeMessage, ServerMessage};
use crate::model::{build_model, extract_params, update_model, SharedModel};
use crate::network::NodeStatus;
use actix::prelude::*;
use anyhow::Result;
use log::{error, info};

pub struct CentralServer {
    nodes: Vec<String>,
    aggregated_params: Option<Vec<f32>>,
    model: SharedModel,
    updates_received: usize,
    total_nodes: usize,
}

impl Actor for CentralServer {
    type Context = Context<Self>;

    fn started(&mut self, _: &mut Self::Context) {
        info!(
            "Central server started, expecting {} nodes",
            self.total_nodes
        );
    }
}

impl Handler<ServerMessage> for CentralServer {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: ServerMessage, _: &mut Self::Context) -> Self::Result {
        info!("Received update from node: {}", msg.node_addr);

        // Add node if not already registered
        if !self.nodes.contains(&msg.node_addr) {
            info!("Registering new node: {}", msg.node_addr);
            self.nodes.push(msg.node_addr.clone());
        }

        self.updates_received += 1;

        // Aggregate parameters
        if let Some(ref mut aggregated) = self.aggregated_params {
            for (a, b) in aggregated.iter_mut().zip(msg.params.iter()) {
                *a += *b;
            }
        } else {
            self.aggregated_params = Some(msg.params);
        }

        info!(
            "Received {}/{} updates",
            self.updates_received, self.total_nodes
        );

        // If we have updates from all nodes, perform FedAvg and broadcast
        if self.updates_received >= self.total_nodes {
            match self.aggregate_and_broadcast() {
                Ok(_) => {
                    info!("Aggregated and broadcasted model updates successfully");
                    self.updates_received = 0;
                    self.aggregated_params = None;
                }
                Err(e) => error!("Failed to aggregate and broadcast: {}", e),
            }
        }

        Ok(())
    }
}

impl Handler<NodeMessage> for CentralServer {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: NodeMessage, _: &mut Self::Context) -> Self::Result {
        match msg {
            NodeMessage::RegisterNode { addr } => {
                if !self.nodes.contains(&addr) {
                    info!("Registering new node: {}", addr);
                    self.nodes.push(addr);
                }
                Ok(())
            }
            NodeMessage::UpdateModel { params } => {
                // Create a server message and handle it
                let server_msg = ServerMessage {
                    node_addr: "direct".to_string(),
                    params,
                };
                self.handle(server_msg, &mut Context::new())
            }
            _ => Ok(()), // Ignore other messages
        }
    }
}

impl Handler<GetNodesRequest> for CentralServer {
    type Result = Vec<NodeStatus>;

    fn handle(&mut self, _: GetNodesRequest, _: &mut Self::Context) -> Self::Result {
        info!("Handling request for node information");

        // Convert nodes to NodeStatus objects
        self.nodes
            .iter()
            .filter(|&addr| addr != "ping") // Filter out ping addresses
            .map(|addr| NodeStatus {
                address: addr.clone(),
                status: "active".to_string(),
            })
            .collect()
    }
}

impl Handler<GetModelParams> for CentralServer {
    type Result = Result<Vec<f32>, String>;

    fn handle(&mut self, _: GetModelParams, _: &mut Self::Context) -> Self::Result {
        match extract_params(&self.model) {
            Ok(params) => Ok(params),
            Err(e) => Err(format!("Failed to extract model parameters: {}", e)),
        }
    }
}

impl CentralServer {
    pub fn new(total_nodes: usize) -> Self {
        let model = build_model();

        Self {
            nodes: Vec::new(),
            aggregated_params: None,
            model,
            updates_received: 0,
            total_nodes,
        }
    }

    fn aggregate_and_broadcast(&mut self) -> Result<(), String> {
        if let Some(ref mut aggregated) = self.aggregated_params {
            // Apply FedAvg algorithm (simple averaging)
            for param in aggregated.iter_mut() {
                *param /= self.total_nodes as f32;
            }

            // Update central model
            match update_model(&self.model, aggregated) {
                Ok(_) => {
                    info!("Central model updated successfully");
                }
                Err(e) => return Err(format!("Failed to update central model: {}", e)),
            }

            // Broadcast to all nodes
            let msg = NodeMessage::UpdateModel {
                params: aggregated.clone(),
            };

            for node in &self.nodes {
                if node != "ping" && !node.is_empty() && node != "direct" {
                    let node_addr = format!("{}/message", node);
                    let msg_clone = msg.clone();

                    // Use actix_web::rt::spawn instead of tokio::spawn
                    actix_web::rt::spawn(async move {
                        let client = awc::Client::default();
                        match client.post(&node_addr).send_json(&msg_clone).await {
                            Ok(_) => info!("Broadcast to {} successful", node_addr),
                            Err(e) => error!("Failed to broadcast to {}: {}", node_addr, e),
                        }
                    });
                }
            }

            Ok(())
        } else {
            Err("No parameters to aggregate".to_string())
        }
    }
}
